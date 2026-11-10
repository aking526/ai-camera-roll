#!/usr/bin/env python3
"""Assemble a static view of the saved evaluation inputs and their scores.

This reads only committed artifacts: the raw scorer output in
``experiments/iteration_01_portrait_retention_2026-11-04/results/raw`` and the
crops, overlays and reference images in ``evidence/``. It runs no models, makes
no API calls, and writes nothing back into the source tree.

Output (``dist/``, ignored by Git):

    dist/index.html  dist/app.js  dist/style.css
    dist/data.js     window.EVAL_DATA = {...}
    dist/img/...     only the crops and overlays the page references

Usage::

    python3 evaluation_site/build.py
    open evaluation_site/dist/index.html
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EXPERIMENT = PROJECT / "experiments/iteration_01_portrait_retention_2026-11-04"
RAW = EXPERIMENT / "results/raw"
EVIDENCE = EXPERIMENT / "evidence"
DIST = HERE / "dist"

CONDITIONS = [
    ("control", "Control", "Fresh replay of the original method. This is the primary condition."),
    ("portrait_reference", "Portrait reference", "Same first pass, second pass also given Maya's portrait."),
    ("historical", "Historical pilot", "Earlier run with different starting images. Context only."),
]
DEFAULT_CONDITION = "control"

# The packaged experiment stores byte-identical files once, so a condition's own
# evidence directory can be incomplete. records/asset_manifest.json maps the
# original scorer paths onto the packaged copies.
FACE_SCORES = {
    "historical": "experiment/historical_scores/",
    "control": "experiment/control/scores/",
    "portrait_reference": "experiment/portrait_reference/scores/",
}
OBJECT_ASSETS = "experiment/object_evaluation/assets/"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"Unreadable saved result {path}: {error}") from error


@cache
def manifest() -> dict[str, dict]:
    return read(EXPERIMENT / "records/asset_manifest.json")["assets"]


def face_asset(condition: str, relative: str) -> Path:
    """Aligned crop or detection overlay, tolerating de-duplicated packaging."""
    direct = EVIDENCE / f"face_{condition}" / relative
    if direct.is_file():
        return direct
    packaged = manifest().get(FACE_SCORES[condition] + relative)
    if packaged:
        return EXPERIMENT / packaged["path"]
    raise FileNotFoundError(f"No saved face asset for {condition}: {relative}")


def object_asset(relative: str) -> Path:
    """Candidate crop, reference crop or overlay from the object evaluation."""
    name = Path(relative).name
    direct = EVIDENCE / "objects" / name
    if direct.is_file():
        return direct
    packaged = manifest().get(OBJECT_ASSETS + name)
    if packaged:
        return EXPERIMENT / packaged["path"]
    raise FileNotFoundError(f"No saved object asset: {relative}")


class Publisher:
    """Copy one source image per destination path and record where it came from."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.by_source: dict[Path, str] = {}
        self.by_dest: dict[str, Path] = {}

    def __call__(self, source: Path, dest_rel: str) -> str:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing evaluation asset: {source}")
        if dest_rel in self.by_dest:
            if self.by_dest[dest_rel] != source:
                raise ValueError(f"Two sources claim {dest_rel}")
            return dest_rel
        if source in self.by_source:
            return self.by_source[source]
        target = self.root / dest_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        self.by_source[source] = dest_rel
        self.by_dest[dest_rel] = source
        return dest_rel


def scene_table() -> dict[int, dict]:
    scenes = read(EXPERIMENT / "results/comparison_data.json")["scenes"]
    return {s["photo_index"]: s for s in scenes}


def face_rows(publish: Publisher, scenes: dict[int, dict], references: dict[str, dict]) -> tuple[list, dict]:
    """One row per requested character per pass, plus unmatched detections."""
    rows: list[dict] = []
    methods: dict = {}
    for condition, _, _ in CONDITIONS:
        data = read(RAW / f"faces_{condition}.json")
        methods = {
            "detector": data["settings"]["detector"],
            "recognition": data["settings"]["recognition"],
            "det_size": data["settings"]["det_size"],
            "detection_threshold": data["settings"]["detection_threshold"],
            "crop": data["settings"]["aligned_crop_size"],
            "threshold_note": data["identity_threshold"],
        }
        for scene in data["scenes"]:
            name = scenes[scene["photo_index"]]["name"]
            for step in scene["passes"]:
                overlay = publish(
                    face_asset(condition, step["detections_image"]),
                    f"img/faces/{condition}/{Path(step['detections_image']).name}",
                )
                by_index = {face["face_index"]: face for face in step["faces"]}
                for candidate in step["candidates"]:
                    face = by_index.get(candidate["face_index"]) if candidate["face_index"] is not None else None
                    rows.append({
                        "condition": condition,
                        "photo_index": scene["photo_index"],
                        "scene": name,
                        "month": scene["month"],
                        "pass": step["pass_index"] + 1,
                        "person_id": candidate["person_id"],
                        "person": references[candidate["person_id"]]["name"],
                        "reference_image": references[candidate["person_id"]]["image"],
                        "detected_image": publish(
                            face_asset(condition, face["aligned_crop"]),
                            f"img/faces/{condition}/{Path(face['aligned_crop']).name}",
                        ) if face else None,
                        "similarity": candidate["similarity"],
                        "face_index": candidate["face_index"],
                        "detector_confidence": face["detection_confidence"] if face else None,
                        "bbox": face["bbox"] if face else None,
                        "detections_image": overlay,
                        "unassigned": list(step["unassigned_face_indices"]),
                        "detected_faces": len(step["faces"]),
                        "expected_faces": len(step["candidates"]),
                    })
                # Detected faces the assignment did not pair with a requested character.
                for face_index in step["unassigned_face_indices"]:
                    face = by_index[face_index]
                    rows.append({
                        "condition": condition,
                        "photo_index": scene["photo_index"],
                        "scene": name,
                        "month": scene["month"],
                        "pass": step["pass_index"] + 1,
                        "person_id": None,
                        "person": "Unassigned face",
                        "reference_image": None,
                        "detected_image": publish(
                            face_asset(condition, face["aligned_crop"]),
                            f"img/faces/{condition}/{Path(face['aligned_crop']).name}",
                        ),
                        "similarity": None,
                        "face_index": face_index,
                        "detector_confidence": face["detection_confidence"],
                        "bbox": face["bbox"],
                        "detections_image": overlay,
                        "unassigned": list(step["unassigned_face_indices"]),
                        "detected_faces": len(step["faces"]),
                        "expected_faces": len(step["candidates"]),
                    })
    return rows, methods


def object_rows(publish: Publisher, scenes: dict[int, dict], targets: dict) -> tuple[list, dict]:
    data = read(RAW / "objects_final.json")
    reference_image = {
        ref["object_id"]: publish(
            object_asset(ref["crop"]),
            f"img/objects/{Path(ref['crop']).name}",
        )
        for ref in data["references"]
    }
    rows: list[dict] = []
    for condition, _, _ in CONDITIONS:
        for scene in data["conditions"][condition]["scenes"]:
            name = scenes[scene["photo_index"]]["name"]
            overlay = publish(
                object_asset(scene["detections_image"]),
                f"img/objects/{Path(scene['detections_image']).name}",
            )
            for item in scene["objects"]:
                candidates = [
                    {
                        "image": publish(
                            object_asset(det["crop"]),
                            f"img/objects/{Path(det['crop']).name}",
                        ),
                        "confidence": det["confidence"],
                        "similarity": det["similarity"],
                    }
                    for det in item["detections"]
                ]
                selected = candidates[item["selected_index"]] if item["selected_index"] is not None else None
                rows.append({
                    "condition": condition,
                    "photo_index": scene["photo_index"],
                    "scene": name,
                    "month": scene["month"],
                    "object_id": item["object_id"],
                    "object": item["name"],
                    "query": item["query"],
                    "requirement": item["requirement"],
                    "reference_image": reference_image[item["object_id"]],
                    "detected_image": selected["image"] if selected else None,
                    "detector_confidence": selected["confidence"] if selected else None,
                    "similarity": item["similarity"],
                    "candidate_count": item["candidate_count"],
                    "candidates": candidates,
                    "detections_image": overlay,
                    "comparison": targets["objects"],
                })
    settings = data["settings"]
    methods = {
        "detector": data["model_provenance"]["detector"]["model_id"],
        "encoder": data["model_provenance"]["encoder"]["model_id"],
        "box_threshold": settings["box_threshold"],
        "text_threshold": settings["text_threshold"],
        "nms_iou": settings["nms_iou"],
        "selection": settings["selection"],
        "resize": settings["resize"],
    }
    return rows, methods


def main() -> None:
    if not EXPERIMENT.is_dir():
        raise SystemExit(f"Missing saved experiment: {EXPERIMENT}")
    # Output paths are deterministic for a given experiment, so images are
    # overwritten in place rather than cleared first.
    DIST.mkdir(parents=True, exist_ok=True)
    publish = Publisher(DIST)

    scenes = scene_table()
    targets = read(EXPERIMENT / "records/object_targets.json")

    # Reference portraits are the same canonical images in every condition, so
    # their aligned crops are shared and resolved through the asset manifest.
    faces_control = read(RAW / "faces_control.json")
    for other in ("portrait_reference", "historical"):
        if read(RAW / f"faces_{other}.json")["references"] != faces_control["references"]:
            raise ValueError("Reference portraits differ between conditions")
    references = {}
    for ref in faces_control["references"]:
        references[ref["id"]] = {
            "name": ref["name"],
            "image": publish(
                face_asset("control", ref["aligned_crop"]),
                f"img/faces/reference/{Path(ref['aligned_crop']).name}",
            ),
        }

    faces, face_methods = face_rows(publish, scenes, references)
    objects, object_methods = object_rows(publish, scenes, targets)

    sources = sorted(set(RAW.glob("*.json"))) + [
        EXPERIMENT / "results/comparison_data.json",
        EXPERIMENT / "records/object_targets.json",
    ]
    data = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "experiment": EXPERIMENT.relative_to(PROJECT).as_posix(),
        "sources": [
            {"path": p.relative_to(PROJECT).as_posix(), "sha256": sha256(p)} for p in sources
        ],
        "scenes": [scenes[k] for k in sorted(scenes)],
        "conditions": [
            {"key": key, "label": label, "note": note} for key, label, note in CONDITIONS
        ],
        "default_condition": DEFAULT_CONDITION,
        "face": {
            "method": face_methods,
            "references": references,
            "rows": sorted(faces, key=lambda r: (r["photo_index"], r["pass"], r["person_id"] is None, str(r["person_id"]))),
        },
        "object": {
            "method": object_methods,
            "rows": sorted(objects, key=lambda r: (r["photo_index"], r["object"])),
        },
    }
    (DIST / "data.js").write_text("window.EVAL_DATA = " + json.dumps(data, indent=1) + ";\n")
    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(HERE / name, DIST / name)

    images = len(publish.by_dest)
    size = sum((DIST / rel).stat().st_size for rel in publish.by_dest)
    print(f"rows: {len(faces)} face, {len(objects)} object")
    print(f"images: {images} ({size / 1e6:.1f} MB) -> {DIST.relative_to(PROJECT)}/")
    print(f"open {DIST.relative_to(PROJECT)}/index.html")


if __name__ == "__main__":
    main()
