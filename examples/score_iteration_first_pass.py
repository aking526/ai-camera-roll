"""Score shared first passes with the frozen object evaluator used for final photos."""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from score_objects import Models, SETTINGS, cosine, crop_box, digest, read, summarize, write


def run(experiment, model_root):
    output = experiment / "object_evaluation"
    final_path = output / "scores.json"
    final_hash = digest(final_path)
    final = read(final_path)
    targets = read(output / "targets.json")
    if SETTINGS != final["settings"] or digest(output / "targets.json") != final["target_sha256"]:
        raise ValueError("First-pass settings or targets differ from final scoring")
    if digest(experiment / "frozen_source.yaml") != final["source_sha256"]:
        raise ValueError("Frozen source changed")
    if digest(Path(__file__).resolve().parents[1] / "evaluation/score_objects.py") != final["scorer_sha256"]:
        raise ValueError("Base object evaluator changed since final scoring")
    for package, expected in final["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("Object evaluation package changed: " + package)
    for model in final["model_provenance"].values():
        folder = model_root / model["model_id"].split("/")[-1]
        for name, expected in model["files_sha256"].items():
            if digest(folder / name) != expected:
                raise ValueError("Object model file changed: " + name)

    final_cache = read(output / "inference_cache.json")
    expected_signature = final["target_sha256"] + json.dumps(
        [SETTINGS, final["model_provenance"], final["packages"]], sort_keys=True
    )
    if final_cache["signature"] != expected_signature:
        raise ValueError("Final inference cache configuration differs")
    cache_path = output / "first_pass_inference_cache.json"
    cache = read(cache_path) if cache_path.exists() else {"signature": expected_signature, "items": {}}
    if cache["signature"] != expected_signature:
        raise ValueError("First-pass inference cache configuration differs")
    references = {r["object_id"]: r for r in final["references"]}
    embeddings = {}
    for obj in targets["objects"]:
        ref = references[obj["id"]]
        key = digest(experiment / ref["image"]) + ":" + obj["query"]
        candidates = final_cache["items"][key]
        if not candidates or candidates[0]["bbox"] != ref["bbox"]:
            raise ValueError("Canonical reference selection differs")
        embeddings[obj["id"]] = candidates[0]["embedding"]

    source = experiment / "control/images"
    manifest = yaml.safe_load((source / "manifest.yaml").read_text())
    treatment = yaml.safe_load((experiment / "portrait_reference/images/manifest.yaml").read_text())
    if digest(source / "manifest.yaml") != final["manifest_sha256"]["control"]:
        raise ValueError("Control manifest changed")
    assets = output / "first_pass_assets"
    assets.mkdir(exist_ok=True)
    input_hashes = {}
    models = None
    inferred = 0

    def asset(path):
        relative = path.relative_to(experiment).as_posix()
        input_hashes[relative] = digest(path)
        return relative

    def infer(path, query):
        nonlocal models, inferred
        key = digest(path) + ":" + query
        if key in final_cache["items"]:
            return copy.deepcopy(final_cache["items"][key])
        if key not in cache["items"]:
            if models is None:
                models = Models(model_root)
            image = Image.open(path).convert("RGB")
            found = models.detect(image, query)
            for candidate in found:
                candidate["crop_box"] = list(crop_box(candidate["bbox"], image.size))
                candidate["embedding"] = models.embed(image.crop(candidate["crop_box"]))
            cache["items"][key] = found
            write(cache_path, cache)
            inferred += 1
        return copy.deepcopy(cache["items"][key])

    object_by_id = {o["id"]: o for o in targets["objects"]}
    scenes = []
    for target, photo, paired in zip(targets["scenes"], manifest["photos"], treatment["photos"], strict=True):
        index = target["photo_index"]
        if index != photo["photo_index"] or index != paired["photo_index"]:
            raise ValueError("Scene ordering differs")
        original = source / photo["passes"][0]["path"]
        paired_image = experiment / "portrait_reference/images" / paired["passes"][0]["path"]
        if digest(original) != digest(paired_image):
            raise ValueError("Paired first passes differ")
        copied = assets / f"scene_{index}_pass_1.png"
        shutil.copyfile(original, copied)
        image = Image.open(copied).convert("RGB")
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        objects = []
        for oid in target["objects"]:
            obj = object_by_id[oid]
            candidates = infer(original, obj["query"])
            for rank, candidate in enumerate(candidates):
                candidate["similarity"] = cosine(candidate.pop("embedding"), embeddings[oid])
                crop = assets / f"scene_{index}_{oid}_{rank}.png"
                image.crop(candidate["crop_box"]).save(crop)
                candidate["crop"] = asset(crop)
                draw.rectangle(candidate["crop_box"], outline="#ffcc00" if rank == 0 else "#33aaff", width=3)
                draw.text((candidate["crop_box"][0], max(0, candidate["crop_box"][1] - 14)),
                          f"{obj['name']} #{rank + 1}", fill="#ffcc00", stroke_width=1, stroke_fill="black")
            objects.append({"object_id": oid, "name": obj["name"], "query": obj["query"], "expected_count": 1,
                            "requirement": target["requirements"][oid], "detections": candidates,
                            "candidate_count": len(candidates), "selected_index": 0 if candidates else None,
                            "similarity": candidates[0]["similarity"] if candidates else None,
                            "status": "detected" if candidates else "not_detected"})
        overlay_path = assets / f"scene_{index}_detections.jpg"
        overlay.save(overlay_path, quality=92)
        scene = {"photo_index": index, "month": target["month"], "pass_index": 0,
                 "shared_with_treatment": True, "has_second_pass": len(photo["passes"]) > 1,
                 "image": asset(copied), "sha256": digest(original),
                 "detections_image": asset(overlay_path), "objects": objects}
        if not scene["has_second_pass"]:
            final_scene = next(s for s in final["conditions"]["control"]["scenes"] if s["photo_index"] == index)
            if scene["sha256"] != final_scene["sha256"]:
                raise ValueError("Single-pass final image differs")
            for before, after in zip(scene["objects"], final_scene["objects"], strict=True):
                if (before["object_id"], before["similarity"], before["candidate_count"]) != (after["object_id"], after["similarity"], after["candidate_count"]):
                    raise ValueError("Single-pass scores differ from final scores")
        scenes.append(scene)
        print("Scene", index + 1, [(o["object_id"], o["candidate_count"], o["similarity"]) for o in objects], flush=True)

    if digest(final_path) != final_hash:
        raise ValueError("Existing final scores changed during this run")
    report = {"schema_version": 1, "condition": "shared_first_pass", "method": final["method"],
              "settings": SETTINGS, "model_provenance": final["model_provenance"], "packages": final["packages"],
              "source_sha256": final["source_sha256"], "target_sha256": final["target_sha256"],
              "final_score_sha256": final_hash, "base_scorer_sha256": final["scorer_sha256"],
              "scorer_sha256": digest(__file__), "references": final["references"],
              "scenes": scenes, "summary": summarize(scenes), "input_sha256": input_hashes,
              "verification": {"shared_first_passes_identical": True, "single_pass_scores_match_finals": True,
                               "settings_models_packages_match_finals": True, "original_final_scores_unchanged": True},
              "api_calls": 0, "new_local_image_query_inferences": inferred,
              "interpretation": "Scored against the final scene's object targets. Some objects are intentionally introduced only in pass 2. May has one pass; no second-pass result is invented.",
              "limits": final["limits"]}
    write(output / "first_pass_scores.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    args = parser.parse_args()
    run(args.experiment.resolve(), args.models.resolve())
