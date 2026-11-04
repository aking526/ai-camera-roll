"""Bundle the existing saved pilot for a local demo; never generate new content."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml
from PIL import Image

from ai_camera_roll.viewer import image_metadata, parse_result


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PREP = PROJECT / "evaluation"
DIST = HERE / "dist"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_images(root, paths, destination):
    """Copy only named, verified image assets into the demo's public directory."""
    copied = {}
    for relative in sorted(paths):
        path = (root / relative).resolve()
        assert path.is_relative_to(root.resolve()) and path.is_file(), relative
        assert path.suffix.lower() in {".png", ".jpg"}, relative
        with Image.open(path) as image:
            image.verify()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assert digest(path) == digest(target), relative
        copied[relative] = digest(target)
    return copied


def build():
    source = PROJECT / "output/camera_roll.yaml"
    image_root = PROJECT / "output/images"
    result = parse_result(source.read_text())
    manifest_path = image_root / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    metadata = image_metadata(image_root, result)
    assert metadata["notice"] is None, metadata["notice"]
    assert len(metadata["photos"]) == len(result["camera_roll"]["photos"])
    assert all(item["complete"] and item["final_url"] for item in metadata["photos"])
    assert all(item["completed"] and item["url"] for item in metadata["references"])

    image_paths = set()
    for reference in manifest["references"]:
        image_paths.update([reference["image"]["path"], reference["reference_path"]])
    for scene in manifest["photos"]:
        image_paths.add(scene["final_path"])
        for step in scene["passes"]:
            assert step["completed"] and step["prompt"]
            image_paths.add(step["path"])
            image_paths.update(item["path"] for item in step["inputs"])
        assert digest(image_root / scene["final_path"]) == digest(image_root / scene["passes"][-1]["path"])

    full_scores = json.loads((PREP / "results/scores.json").read_text())
    for filename, expected in full_scores["input_sha256"].items():
        path = PROJECT / filename
        assert path.resolve().is_relative_to(PROJECT / "output")
        assert digest(path) == expected, "Saved scores no longer match their source inputs"
    # Source hashes are retained in evaluation/results, separate from browser data.
    score_keys = ("purpose", "identity_threshold", "interpretation", "settings", "references",
                  "matrix_row_order", "reference_cosine_matrix", "scenes")
    scores = {key: full_scores[key] for key in score_keys}
    review = json.loads((PREP / "visual_review.json").read_text())
    review.pop("source_root", None)
    evaluation_paths = {reference["aligned_crop"] for reference in scores["references"]}
    assert scores["matrix_row_order"] == [person["id"] for person in result["camera_roll"]["people"]]
    assert len(scores["scenes"]) == len(manifest["photos"]) == len(review["scenes"])
    assert [scene["photo_index"] for scene in review["scenes"]] == [scene["photo_index"] for scene in manifest["photos"]]
    for scored, generated, photo in zip(scores["scenes"], manifest["photos"], result["camera_roll"]["photos"]):
        assert scored["photo_index"] == generated["photo_index"]
        assert scored["final_source"] == generated["final_path"]
        assert scored["expected_people"] == [person["person_id"] for person in photo["people"]]
        assert len(scored["passes"]) == len(generated["passes"])
        for score, step in zip(scored["passes"], generated["passes"]):
            assert score["source"] == step["path"]
            face_indices = {face["face_index"] for face in score["faces"]}
            paired = [item["face_index"] for item in score["candidates"] if item["face_index"] is not None]
            assert len(paired) == len(set(paired)) and set(paired) <= face_indices
            assert len(score["cosine_matrix"]) == len(scores["references"])
            assert all(len(row) == len(face_indices) for row in score["cosine_matrix"])
            evaluation_paths.add(score["detections_image"])
            evaluation_paths.update(face["aligned_crop"] for face in score["faces"])

    image_hashes = copy_images(image_root, image_paths, DIST / "assets/images")
    evaluation_hashes = copy_images(PREP / "results", evaluation_paths, DIST / "assets/evaluation")
    templates = {name: (PROJECT / "src/ai_camera_roll/prompts" / f"{name}.md").read_text()
                 for name in ("story", "photos", "image_scene", "image_character", "image_object")}
    commit = subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True).strip()
    provenance = {
        "source_repository": "https://github.com/aking526/ai-camera-roll",
        "source_commit": commit,
        "source_sha256": digest(source),
        "manifest_sha256": digest(manifest_path),
        "structured_source_sha256": manifest["settings"]["source_sha256"],
        "copied_source_images": len(image_hashes),
        "copied_evaluation_images": len(evaluation_hashes),
        "no_api_calls": True,
        "generation_calls_during_bundle": 0,
        "attribution": "The original repository supplied the generator, story and saved images. Codex assisted with the separate local demo, evaluation code and draft visual annotations.",
        "prompt_template_note": "Templates are from the inspected repository checkout. They are not exact historical language-model requests; the compiled image prompts are preserved verbatim in the saved manifest.",
        "evaluation_note": "Existing saved scores and assistant annotations are exploratory. No independent human ratings, calibrated identity cutoff or measured generation improvement are claimed.",
        "generation_note": "Viewing saved albums makes no model calls. Explicit story or photograph creation uses the configured providers. Existing extra passes add references; they are not score-triggered retries.",
        "model_provenance": full_scores["model_provenance"],
    }
    data = {**result, "manifest": manifest, "review": review, "scores": scores,
            "prompt_templates": templates, "provenance": provenance}
    serialized = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    assert "/Users/" not in serialized
    (DIST / "data.js").write_text("window.CAMERA_ROLL_DATA = " + serialized + ";\n", encoding="utf-8")
    # Verify the written payload rather than only the pre-serialization Python data.
    saved = (DIST / "data.js").read_text().removeprefix("window.CAMERA_ROLL_DATA = ").removesuffix(";\n")
    assert json.loads(saved) == data
    checks = {
        "source_schema_and_manifest_valid": True,
        "photos": len(manifest["photos"]),
        "references": len(manifest["references"]),
        "scene_passes": sum(len(photo["passes"]) for photo in manifest["photos"]),
        "source_images_copied_and_hash_checked": len(image_hashes),
        "evaluation_images_copied_and_hash_checked": len(evaluation_hashes),
        "all_assets_decoded": True,
        "all_final_images_match_last_pass": True,
        "score_sources_and_candidate_mappings_valid": True,
        "data_js_round_trip_valid": True,
        "no_absolute_local_paths_in_bundle": True,
        "january_input_entity_ids": [[item.get("entity_id") for item in step["inputs"]]
                                     for step in manifest["photos"][0]["passes"]],
        "generation_calls": 0,
        "source_image_sha256": image_hashes,
        "evaluation_image_sha256": evaluation_hashes,
    }
    verification = HERE / "verification"
    verification.mkdir(exist_ok=True)
    (verification / "data_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({key: value for key, value in checks.items() if not key.endswith("_sha256")}, indent=2))


if __name__ == "__main__":
    build()
