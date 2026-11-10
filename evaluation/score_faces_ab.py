"""A/B one frozen camera roll with two face embedders on identical aligned crops.

Detection and alignment run once per image; the ArcFace and AdaFace arms then
embed the same 112x112 crops. Only the recognition weights and their input
normalisation change between arms, so any score difference is attributable to
the embedder rather than to the detector, crop, pairing rule or reference set.
"""

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import yaml
from scipy.stats import spearmanr

from recognizers import align_face, load_adaface, load_arcface
from score_faces import cosine_scores, pair_candidates, sha256
from score_faces import parse_args as parse_score_args

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ARMS = ("arcface", "adaface")
# Inspection band quoted by the frozen pilot review; not a decision threshold.
MARGIN_BAND = 0.07
# Preprocessing candidates tested for an unknown export: gross errors (raw 0-255
# input) collapse every cosine to ~0.9, which the sanity checks below catch.
PREPROCESSING_CANDIDATES = {
    "rgb_symmetric_127.5": ("RGB", 127.5, 127.5),
    "bgr_symmetric_127.5": ("BGR", 127.5, 127.5),
    "rgb_imagenet": ("RGB", 123.675, 58.395),
    "rgb_unit": ("RGB", 0.0, 255.0),
    "rgb_unscaled": ("RGB", 0.0, 1.0),
}


def spearman(left, right):
    """Rank correlation between two similarity vectors; scipy returns a result object."""
    return cast(float, spearmanr(left, right)[0])


def rank1_margin(scores, row, column):
    """Assigned similarity minus the best similarity any other reference has to that face.

    Negative means another character's portrait matches this face better. The
    Hungarian pairing still assigns it, because a pairing is not an identity verdict.
    """
    scores = np.asarray(scores, dtype=float)
    others = [scores[other, column] for other in range(scores.shape[0]) if other != row]
    return None if not others else float(scores[row, column] - max(others))


def parse_args(argv=None):
    """Reuse the baseline CLI, but default the output to results_adaface, never results/."""
    argv = list(sys.argv[1:] if argv is None else argv)
    sweep = "--sweep-preprocessing" in argv
    argv = [argument for argument in argv if argument != "--sweep-preprocessing"]
    if not any(a == "--output" or a.startswith("--output=") for a in argv):
        argv += ["--output", str(HERE / "results_adaface")]
    args = parse_score_args(argv)
    args.sweep_preprocessing = sweep
    return args


def detect_and_align(app, image_path):
    """One detection pass and one alignment per face; both arms consume its output."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    faces = sorted(app.get(image), key=lambda face: float(face.bbox[0]))
    crops = [align_face(image, face.kps) for face in faces]
    return faces, crops


def build_models(args):
    from insightface.app import FaceAnalysis

    baseline = json.loads((HERE / "model-provenance.json").read_text())
    adaface = json.loads((HERE / "adaface-provenance.json").read_text())
    detector_dir = HERE / "model-cache/models/buffalo_l"
    for filename in ("det_10g.onnx", "w600k_r50.onnx"):
        if sha256(detector_dir / filename) != baseline["extracted_models_sha256"][filename]:
            raise ValueError(f"Model checksum mismatch: {filename}")
    adaface_path = HERE / adaface["file"]
    if sha256(adaface_path) != adaface["sha256"]:
        raise ValueError(f"Model checksum mismatch: {adaface_path.name}")
    app = FaceAnalysis(name=str(detector_dir), allowed_modules=["detection"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    recognizers = {
        "arcface": load_arcface(detector_dir / "w600k_r50.onnx"),
        "adaface": load_adaface(adaface_path),
    }
    return app, recognizers, baseline, adaface, detector_dir, adaface_path


def embed_crop(recognizers, crop):
    return {arm: recognizers[arm].embed(crop) for arm in ARMS}


def separation_stats(recognize, reference_crops, pass_crops):
    """Same-candidate-index vs different-candidate-index cosines across two passes.

    A candidate index is not an identity label, but for a working embedder the same
    position in two passes of one photo must score far above two different positions.
    """
    same, different = [], []
    for crops_by_pass in pass_crops:
        if len(crops_by_pass) < 2:
            continue
        first = [recognize(crop) for crop in crops_by_pass[0]]
        second = [recognize(crop) for crop in crops_by_pass[1]]
        for i, left in enumerate(first):
            for j, right in enumerate(second):
                (same if i == j else different).append(float(left @ right))
    references = [recognize(crop) for crop in reference_crops]
    inter = [float(references[a] @ references[b])
             for a in range(len(references)) for b in range(a + 1, len(references))]
    return {
        "same_index_across_passes_mean": float(np.mean(same)) if same else None,
        "same_index_across_passes_n": len(same),
        "different_index_across_passes_mean": float(np.mean(different)) if different else None,
        "different_index_across_passes_n": len(different),
        "reference_inter_identity_cosines": inter,
    }


def sweep_preprocessing(adaface_path, reference_crops, pass_crops):
    """Score preprocessing candidates for the AdaFace export on the identical crops."""
    from recognizers import OnnxEmbedder

    rows = []
    for name, (channel_order, mean, std) in PREPROCESSING_CANDIDATES.items():
        embedder = OnnxEmbedder(name, adaface_path, channel_order, mean, std)
        stats = separation_stats(embedder.embed, reference_crops, pass_crops)
        rows.append({"candidate": name, "channel_order": channel_order,
                     "mean": mean, "std": std, **stats})
    return {"note": "Two failure modes are visible here: an unscaled 0-255 input collapses every "
                    "cosine towards 0.9, and a wrong channel order or scale lowers the "
                    "same-index boost. The selected candidate is the AdaFace training "
                    "convention (RGB, (x - 127.5) / 127.5), not the best-scoring row, because "
                    "ranking normalisations against these few evaluation pairs would be "
                    "tuning on the evaluation set.",
            "selected": "rgb_symmetric_127.5", "candidates": rows}


def main(argv=None):
    started = time.perf_counter()
    args = parse_args(argv)
    project, source, images, output = args.project_root, args.source, args.images, args.output
    roll = yaml.safe_load(source.read_text())["camera_roll"]
    manifest_path = images / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    crops_dir = output / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    app, recognizers, baseline, adaface, detector_dir, adaface_path = build_models(args)
    inputs = {source.relative_to(project).as_posix(): sha256(source),
              manifest_path.relative_to(project).as_posix(): sha256(manifest_path)}
    assets = {asset["entity_id"]: asset for asset in manifest["references"]}

    reference_rows, reference_crops, reference_ids = [], [], []
    for person in roll["people"]:
        relative = assets[person["id"]]["image"]["path"]
        path = images / relative
        inputs[path.relative_to(project).as_posix()] = sha256(path)
        faces, crops = detect_and_align(app, path)
        if len(faces) != 1:
            raise ValueError(f"Reference {person['id']} has {len(faces)} faces; expected one.")
        crop_path = crops_dir / f"{person['id']}_face_0.png"
        if not cv2.imwrite(str(crop_path), crops[0]):
            raise OSError(f"Cannot write {crop_path}")
        reference_ids.append(person["id"])
        reference_crops.append(crops[0])
        reference_rows.append({"id": person["id"], "name": person["name"], "source": relative,
                               "face_index": 0, "bbox": faces[0].bbox.tolist(),
                               "detection_confidence": float(faces[0].det_score),
                               "aligned_crop": str(crop_path.relative_to(output))})
    reference_embeddings = {arm: [recognizers[arm].embed(crop) for crop in reference_crops]
                            for arm in ARMS}
    reference_matrices = {arm: cosine_scores(reference_embeddings[arm],
                                             reference_embeddings[arm]).tolist() for arm in ARMS}

    scenes, pass_crops_all = [], []
    for item in manifest["photos"]:
        photo = roll["photos"][item["photo_index"]]
        expected = [person["person_id"] for person in photo["people"]]
        expected_rows = [reference_ids.index(person) for person in expected]
        passes, drops = [], []
        for step_index, step in enumerate(item["passes"]):
            if not step["completed"]:
                raise ValueError("This pilot requires completed saved images.")
            label = f"scene_{item['photo_index']}_pass_{step_index}"
            path = images / step["path"]
            inputs[path.relative_to(project).as_posix()] = sha256(path)
            faces, crops = detect_and_align(app, path)
            drops.append((faces, crops))
            # One aligned crop, embedded by both arms: the only differing input is the network.
            embeddings = [{arm: recognizers[arm].embed(crop) for arm in ARMS} for crop in crops]
            records = []
            for index, (face, crop, embedding) in enumerate(zip(faces, crops, embeddings,
                                                                strict=True)):
                crop_path = crops_dir / f"{label}_face_{index}.png"
                if not cv2.imwrite(str(crop_path), crop):
                    raise OSError(f"Cannot write {crop_path}")
                records.append({
                    "face_index": index, "bbox": face.bbox.tolist(),
                    "detection_confidence": float(face.det_score),
                    "aligned_crop": str(crop_path.relative_to(output)),
                    "cosine_to_reference": {
                        reference_id: {arm: float(reference_embeddings[arm][row] @ embedding[arm])
                                       for arm in ARMS}
                        for row, reference_id in enumerate(reference_ids)}})
            arms = {}
            for arm in ARMS:
                scores = cosine_scores(reference_embeddings[arm],
                                       [embedding[arm] for embedding in embeddings])
                paired = pair_candidates(scores[expected_rows])
                arms[arm] = {
                    "cosine_matrix": scores.tolist(),
                    "candidates": [
                        {"person_id": person, "face_index": face_index,
                         "similarity": None if face_index is None else float(scores[row, face_index]),
                         "rank1_margin": None if face_index is None
                         else rank1_margin(scores, row, face_index)}
                        for person, row, face_index in zip(expected, expected_rows, paired,
                                                           strict=True)],
                    "unassigned_face_indices": [i for i in range(len(faces)) if i not in paired]}
            passes.append({"pass_index": step_index, "source": step["path"],
                           "faces": records, "arms": arms})
            print(f"Photo {item['photo_index'] + 1}, pass {step_index + 1}: "
                  f"{len(faces)} detected faces / {len(expected)} expected people", flush=True)
        pass_crops_all.append([crops for _, crops in drops])
        final = images / item["final_path"]
        inputs[final.relative_to(project).as_posix()] = sha256(final)
        if sha256(final) != sha256(images / item["passes"][-1]["path"]):
            raise ValueError("Final photo differs from last saved pass.")
        scenes.append({"photo_index": item["photo_index"], "month": photo["month"],
                       "expected_people": expected, "final_source": item["final_path"],
                       "passes": passes})

    comparison = compare_arms(scenes)
    comparison["detection_control"] = detection_control(scenes)
    comparison["baseline_check"] = compare_with_baseline(scenes, reference_matrices, inputs)
    validation = {
        "method": "Same alignment and the same crops feed both arms; a candidate index is a "
                  "position in a photo, not an identity label, so the across-pass comparison is "
                  "a separation check, not ground truth.",
        "arcface": separation_stats(recognizers["arcface"].embed, reference_crops,
                                    pass_crops_all),
        "adaface": separation_stats(recognizers["adaface"].embed, reference_crops,
                                    pass_crops_all),
        "raw_output_norms": {arm: float(np.mean([np.linalg.norm(recognizers[arm].forward(crop))
                                                 for crop in reference_crops])) for arm in ARMS},
    }
    result = {
        "purpose": "Controlled swap of the face embedder on frozen generated images.",
        "identity_threshold": None,
        "interpretation": ("Cosine similarity between embeddings is not a probability and no "
                           "calibrated identity threshold is used. Candidate pairings maximise "
                           "total cosine similarity; weak matches are still assigned. Both arms "
                           "see identical detections and identical 112x112 aligned crops."),
        "controlled_variable": {
            "changed": "recognition network weights and their input normalisation",
            "held_constant": ["SCRFD-10GF detection weights", "det_size", "detection threshold",
                              "face sort order", "landmark alignment and 112x112 crop",
                              "reference portraits", "linear-assignment pairing rule",
                              "provider (CPU)"]},
        "settings": {"detector": "SCRFD-10GF", "detector_file": "det_10g.onnx",
                     "provider": "CPUExecutionProvider", "det_size": [640, 640],
                     "detection_threshold": 0.5, "aligned_crop_size": [112, 112],
                     "pairing": "scipy.optimize.linear_sum_assignment(-cosine)",
                     "margin_band": MARGIN_BAND},
        "packages": {name: version(name) for name in
                     ["insightface", "onnxruntime", "numpy", "opencv-python", "scipy"]},
        "provenance": {
            "detector": {"file": "model-cache/models/buffalo_l/det_10g.onnx",
                         "sha256": sha256(detector_dir / "det_10g.onnx"),
                         "source": baseline["source"], "license": baseline["license"]},
            "recognizers": {
                "arcface": {"model": "ArcFace ResNet50 @ WebFace600K (buffalo_l)",
                            "file": "model-cache/models/buffalo_l/w600k_r50.onnx",
                            "sha256": sha256(detector_dir / "w600k_r50.onnx"),
                            "source": baseline["source"],
                            "license": baseline["license"],
                            "preprocessing": recognizers["arcface"].preprocessing,
                            "output": "512-d, L2-normalised here from the raw graph output"},
                "adaface": {"model": "AdaFace IR-101 @ WebFace12M",
                            "file": adaface["file"], "sha256": adaface["sha256"],
                            "upstream_url": adaface["upstream_url"],
                            "derived_from": adaface["derived_from"],
                            "paper": adaface["paper"], "license": adaface["license"],
                            "preprocessing": recognizers["adaface"].preprocessing,
                            "output": "512-d, unit length from the graph (ReduceL2 + Div); "
                                      "re-normalised here as a no-op check"}}},
        "input_sha256": inputs,
        "references": reference_rows, "matrix_row_order": reference_ids,
        "reference_cosine_matrix": reference_matrices,
        "scenes": scenes, "preprocessing_validation": validation,
        "comparison": comparison,
        "inference_seconds": time.perf_counter() - started, "generation_calls": 0,
    }
    if args.sweep_preprocessing:
        result["preprocessing_sweep"] = sweep_preprocessing(
            adaface_path, reference_crops, pass_crops_all)
    destination = output / "ab_scores.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(comparison, indent=2)[:4000], flush=True)
    print(f"Saved {destination}", flush=True)


def compare_arms(scenes):
    """Side-by-side comparison of the two arms on the identical detections."""
    changed, per_pair = [], []
    for scene in scenes:
        for item in scene["passes"]:
            for arc_candidate, ada_candidate in zip(item["arms"]["arcface"]["candidates"],
                                                    item["arms"]["adaface"]["candidates"],
                                                    strict=True):
                case = {"photo_index": scene["photo_index"], "month": scene["month"],
                        "pass_index": item["pass_index"],
                        "person_id": arc_candidate["person_id"]}
                if arc_candidate["face_index"] != ada_candidate["face_index"]:
                    changed.append({**case,
                                    "arcface_face_index": arc_candidate["face_index"],
                                    "adaface_face_index": ada_candidate["face_index"],
                                    "arcface_similarity": arc_candidate["similarity"],
                                    "adaface_similarity": ada_candidate["similarity"]})
                if arc_candidate["similarity"] is None or ada_candidate["similarity"] is None:
                    continue
                per_pair.append({**case,
                                 "arcface_face_index": arc_candidate["face_index"],
                                 "adaface_face_index": ada_candidate["face_index"],
                                 "arcface_similarity": arc_candidate["similarity"],
                                 "adaface_similarity": ada_candidate["similarity"],
                                 "arcface_rank1_margin": arc_candidate["rank1_margin"],
                                 "adaface_rank1_margin": ada_candidate["rank1_margin"]})
    arc_vectors = np.concatenate([np.asarray(item["arms"]["arcface"]["cosine_matrix"]).ravel()
                                  for scene in scenes for item in scene["passes"]])
    ada_vectors = np.concatenate([np.asarray(item["arms"]["adaface"]["cosine_matrix"]).ravel()
                                  for scene in scenes for item in scene["passes"]])
    all_cell_correlation = spearman(arc_vectors, ada_vectors)
    assigned_correlation = spearman([row["arcface_similarity"] for row in per_pair],
                                    [row["adaface_similarity"] for row in per_pair])
    deltas = np.array([row["adaface_similarity"] - row["arcface_similarity"] for row in per_pair])
    return {
        "scored_pairs": per_pair,
        "scored_pairs_note": "Only appearances that received a face under both arms; the "
                             "October second pass drops a different character in each arm.",
        "scored_pair_count": len(per_pair),
        "per_arm": {arm: arm_summary(scenes, arm) for arm in ARMS},
        "similarity_delta_adaface_minus_arcface": {
            "mean": float(deltas.mean()), "median": float(np.median(deltas)),
            "min": float(deltas.min()), "max": float(deltas.max())},
        "spearman_all_cosine_cells": float(all_cell_correlation),
        "spearman_all_cosine_cells_n": int(arc_vectors.size),
        "spearman_assigned_similarities": float(assigned_correlation),
        "assignment_changes": changed, "assignment_change_count": len(changed),
        "pass_to_pass_transitions": pass_to_pass_transitions(scenes),
    }


def arm_summary(scenes, arm):
    """Assigned similarities, rank-1 margins and unassigned appearances for one arm."""
    assigned, unassigned = [], []
    for scene in scenes:
        for item in scene["passes"]:
            for candidate in item["arms"][arm]["candidates"]:
                if candidate["similarity"] is None:
                    unassigned.append({"photo_index": scene["photo_index"],
                                       "month": scene["month"],
                                       "pass_index": item["pass_index"],
                                       "person_id": candidate["person_id"]})
                    continue
                assigned.append({"photo_index": scene["photo_index"],
                                 "month": scene["month"], "pass_index": item["pass_index"],
                                 "person_id": candidate["person_id"],
                                 "face_index": candidate["face_index"],
                                 "similarity": candidate["similarity"],
                                 "rank1_margin": candidate["rank1_margin"]})
    similarities = [row["similarity"] for row in assigned]
    margins = [row["rank1_margin"] for row in assigned]
    return {"assigned_count": len(assigned),
            "assigned_similarity_range": [min(similarities), max(similarities)],
            "rank1_margin_range": [min(margins), max(margins)],
            "rank1_margins_below_band": sum(1 for margin in margins if margin < MARGIN_BAND),
            "negative_rank1_margins": [row for row in assigned if row["rank1_margin"] < 0],
            "unassigned_appearances": unassigned,
            "assigned": assigned}


def pass_to_pass_transitions(scenes):
    """Per person, the same-candidate-index similarity in pass 1 and pass 2 of each photo."""
    transitions = []
    for scene in scenes:
        passes = scene["passes"]
        if len(passes) < 2:
            continue
        for arm in ARMS:
            for person in scene["expected_people"]:
                series = []
                for item in passes:
                    candidate = next(c for c in item["arms"][arm]["candidates"]
                                     if c["person_id"] == person)
                    series.append({"pass_index": item["pass_index"],
                                   "face_index": candidate["face_index"],
                                   "similarity": candidate["similarity"]})
                if any(point["similarity"] is None for point in series):
                    dropped = None
                else:
                    dropped = bool(series[-1]["similarity"] < series[0]["similarity"])
                transitions.append({"photo_index": scene["photo_index"],
                                    "month": scene["month"], "person_id": person, "arm": arm,
                                    "series": series, "dropped": dropped})
    return transitions


def detection_control(scenes):
    """Detection is shared by construction, so only the pairing can differ between arms."""
    counts, per_arm = [], {arm: [] for arm in ARMS}
    for scene in scenes:
        for item in scene["passes"]:
            counts.append({"photo_index": scene["photo_index"], "pass_index": item["pass_index"],
                           "detected_faces": len(item["faces"]),
                           "expected_people": len(scene["expected_people"])})
            for arm in ARMS:
                unassigned = [candidate["person_id"] for candidate
                              in item["arms"][arm]["candidates"] if candidate["face_index"] is None]
                per_arm[arm].append({"photo_index": scene["photo_index"],
                                     "pass_index": item["pass_index"],
                                     "unassigned_people": unassigned})
    return {"method": "One SCRFD detection pass per image feeds both arms, so per-image face "
                       "counts cannot differ; only which expected character a row is paired "
                       "with, or left without a face, can change.",
            "detected_faces_per_pass": counts,
            "total_detected_faces": sum(row["detected_faces"] for row in counts),
            "total_expected_appearances": sum(row["expected_people"] for row in counts),
            "unassigned_appearances": per_arm}


def compare_with_baseline(scenes, reference_matrices, inputs):
    """The ArcFace arm must reproduce evaluation/results/scores.json, or this script has a bug."""
    path = HERE / "results/scores.json"
    if not path.exists():
        return {"baseline": None, "note": "No frozen baseline found; reproduction unchecked."}
    baseline = json.loads(path.read_text())
    shared = {key: value for key, value in inputs.items() if key in baseline["input_sha256"]}
    if not shared or any(baseline["input_sha256"][key] != value for key, value in shared.items()):
        return {"baseline": str(path.relative_to(HERE)),
                "note": "Inputs differ from the frozen baseline; reproduction unchecked."}
    deviations, detection_mismatches = [], []
    for scene, base_scene in zip(scenes, baseline["scenes"], strict=False):
        for item, base_pass in zip(scene["passes"], base_scene["passes"], strict=False):
            deviations.append(np.max(np.abs(np.asarray(item["arms"]["arcface"]["cosine_matrix"])
                                            - np.asarray(base_pass["cosine_matrix"]))))
            if len(item["faces"]) != len(base_pass["faces"]):
                detection_mismatches.append((scene["photo_index"], item["pass_index"]))
                continue
            for face, base_face in zip(item["faces"], base_pass["faces"], strict=False):
                if (np.max(np.abs(np.asarray(face["bbox"]) - np.asarray(base_face["bbox"]))) > 1e-6
                        or abs(face["detection_confidence"] - base_face["detection_confidence"])
                        > 1e-9):
                    detection_mismatches.append((scene["photo_index"], item["pass_index"]))
    reference_deviation = max(
        np.max(np.abs(np.asarray(reference_matrices["arcface"]) - np.asarray(
            baseline["reference_cosine_matrix"]))), np.max(deviations))
    if reference_deviation > 1e-9 or detection_mismatches:
        raise ValueError(
            f"ArcFace arm does not reproduce the frozen baseline "
            f"(max cosine deviation {reference_deviation:.3e}, "
            f"detection mismatches {detection_mismatches}).")
    return {"baseline": str(path.relative_to(HERE)),
            "max_abs_cosine_deviation": float(reference_deviation),
            "detections_identical": True,
            "note": "The ArcFace arm reproduces the frozen baseline, so both arms share its "
                    "detections, crops and pairing rule."}


if __name__ == "__main__":
    main()
