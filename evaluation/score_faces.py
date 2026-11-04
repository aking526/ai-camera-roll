"""Score saved synthetic faces locally; no generation or identity pass/fail gate."""

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def cosine_scores(references, faces):
    """Rows are reference characters; columns are detected scene faces."""
    references = np.asarray(references, dtype=float)
    if len(faces) == 0:
        return np.empty((len(references), 0))
    faces = np.asarray(faces, dtype=float)
    return (references / np.linalg.norm(references, axis=1, keepdims=True)) @ (
        faces / np.linalg.norm(faces, axis=1, keepdims=True)
    ).T


def pair_candidates(scores):
    """Best total one-to-one pairing. A pairing is NOT an identity decision."""
    paired = [None] * scores.shape[0]
    rows, columns = linear_sum_assignment(-scores)
    for row, column in zip(rows, columns):
        paired[row] = int(column)
    return paired


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT,
                        help="Root used for portable input hashes and default input paths.")
    parser.add_argument("--source", type=Path,
                        help="Saved camera-roll YAML (default: PROJECT_ROOT/output/camera_roll.yaml).")
    parser.add_argument("--images", type=Path,
                        help="Image directory with manifest.yaml (default: PROJECT_ROOT/output/images).")
    parser.add_argument("--output", type=Path, default=HERE / "results",
                        help="Directory for scores and face crops (default: evaluation/results).")
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.source = (args.source or args.project_root / "output/camera_roll.yaml").resolve()
    args.images = (args.images or args.project_root / "output/images").resolve()
    args.output = args.output.resolve()
    for name in ("source", "images"):
        if not getattr(args, name).is_relative_to(args.project_root):
            parser.error(f"--{name} must be inside --project-root so input hashes remain portable.")
    return args


def main(argv=None):
    args = parse_args(argv)
    from insightface.app import FaceAnalysis
    from insightface.utils.face_align import norm_crop

    started = time.perf_counter()
    project = args.project_root
    source = args.source
    images = args.images
    manifest_path = images / "manifest.yaml"
    roll = yaml.safe_load(source.read_text())["camera_roll"]
    manifest = yaml.safe_load(manifest_path.read_text())
    output = args.output
    crops = output / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    models = HERE / "model-cache/models/buffalo_l"
    provenance = json.loads((HERE / "model-provenance.json").read_text())
    # Require verified local weights so inference never triggers a download.
    for filename, digest in provenance["extracted_models_sha256"].items():
        if sha256(models / filename) != digest:
            raise ValueError(f"Model checksum mismatch: {filename}")
    app = FaceAnalysis(
        name=str(models), allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    references, embeddings, inputs = [], [], {source.relative_to(project).as_posix(): sha256(source)}
    inputs[manifest_path.relative_to(project).as_posix()] = sha256(manifest_path)

    def detect(relative, label):
        path = images / relative
        inputs[path.relative_to(project).as_posix()] = sha256(path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        faces = sorted(app.get(image), key=lambda face: float(face.bbox[0]))
        records = []
        annotated = image.copy()
        for index, face in enumerate(faces):
            crop = norm_crop(image, landmark=face.kps, image_size=112)
            crop_path = crops / f"{label}_face_{index}.png"
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Cannot write {crop_path}")
            box = np.round(face.bbox).astype(int)
            cv2.rectangle(annotated, tuple(box[:2]), tuple(box[2:]), (0, 200, 255), 2)
            cv2.putText(annotated, str(index), (int(box[0]), max(20, int(box[1])-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            records.append({"face_index": index, "bbox": face.bbox.tolist(),
                            "detection_confidence": float(face.det_score),
                            "aligned_crop": str(crop_path.relative_to(output))})
        marked_path = output / f"{label}_detections.jpg"
        if not cv2.imwrite(str(marked_path), annotated):
            raise OSError(f"Cannot write {marked_path}")
        return faces, records, str(marked_path.relative_to(output))

    assets = {a["entity_id"]: a for a in manifest["references"]}
    for person in roll["people"]:
        relative = assets[person["id"]]["image"]["path"]
        faces, records, _ = detect(relative, person["id"])
        if len(faces) != 1:
            raise ValueError(f"Reference {person['id']} has {len(faces)} faces; expected one.")
        embeddings.append(faces[0].normed_embedding)
        references.append({"id": person["id"], "name": person["name"],
                           "source": relative, **records[0]})
    reference_ids = [r["id"] for r in references]
    scenes = []
    for item in manifest["photos"]:
        photo = roll["photos"][item["photo_index"]]
        expected = [p["person_id"] for p in photo["people"]]
        expected_rows = [reference_ids.index(person) for person in expected]
        passes = []
        for step_index, step in enumerate(item["passes"]):
            if not step["completed"]:
                raise ValueError("This pilot requires completed saved images.")
            label = f"scene_{item['photo_index']}_pass_{step_index}"
            faces, records, marked = detect(step["path"], label)
            scores = cosine_scores(embeddings, [f.normed_embedding for f in faces])
            paired = pair_candidates(scores[expected_rows])
            candidates = [
                {"person_id": person, "face_index": face_index,
                 "similarity": None if face_index is None else float(scores[row, face_index])}
                for person, row, face_index in zip(expected, expected_rows, paired)
            ]
            passes.append({"pass_index": step_index, "source": step["path"],
                           "detections_image": marked, "faces": records,
                           "cosine_matrix": scores.tolist(), "candidates": candidates,
                           "unassigned_face_indices": [i for i in range(len(faces)) if i not in paired]})
            print(f"Photo {item['photo_index'] + 1}, pass {step_index + 1}: "
                  f"{len(faces)} detected faces / {len(expected)} expected people", flush=True)
        # The displayed final file must really be the last scored pass.
        final = images / item["final_path"]
        inputs[final.relative_to(project).as_posix()] = sha256(final)
        if sha256(final) != sha256(images / item["passes"][-1]["path"]):
            raise ValueError("Final photo differs from last saved pass.")
        scenes.append({"photo_index": item["photo_index"], "month": photo["month"],
                       "expected_people": expected, "final_source": item["final_path"],
                       "passes": passes})
    result = {
        "purpose": "Exploratory face similarity for fictional generated characters.",
        "identity_threshold": None,
        "interpretation": "Candidate pairings maximize total cosine similarity; weak matches are still assigned. Scores are not probabilities or identity verdicts. No detected face does not establish absence.",
        "settings": {"model": "buffalo_l", "detector": "SCRFD-10GF",
                     "recognition": "ArcFace ResNet50@WebFace600K", "provider": "CPUExecutionProvider",
                     "det_size": [640, 640], "detection_threshold": 0.5, "aligned_crop_size": [112, 112]},
        "packages": {name: version(name) for name in ["insightface", "onnxruntime", "numpy", "opencv-python", "scipy"]},
        "model_provenance": provenance, "input_sha256": inputs,
        "references": references, "matrix_row_order": reference_ids,
        "reference_cosine_matrix": cosine_scores(embeddings, embeddings).tolist(),
        "scenes": scenes, "inference_seconds": time.perf_counter() - started,
        "generation_calls": 0,
    }
    (output / "scores.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output / 'scores.json'}", flush=True)


if __name__ == "__main__":
    main()
