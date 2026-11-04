"""Object-report checks use synthetic records and never run models or APIs."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "portrait_report", Path(__file__).resolve().parents[1] / "examples/report_portrait_experiment.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


@pytest.fixture
def object_scores(tmp_path):
    root = tmp_path / "experiment"
    directory = root / "object_evaluation"
    directory.mkdir(parents=True)
    (root / "frozen_source.yaml").write_text("source")
    (directory / "targets.json").write_text("{}")
    (root / "image.png").write_bytes(b"hash fixture only")
    digest = report.sha(root / "image.png")

    def obj(name, similarity):
        detections = [] if similarity is None else [
            {"bbox": [0, 0, 1, 1], "confidence": .8, "similarity": similarity, "crop": "image.png"}
        ]
        return {"object_id": name, "name": name, "query": name, "expected_count": 1,
                "detections": detections, "selected_index": None if similarity is None else 0,
                "similarity": similarity, "candidate_count": len(detections),
                "status": "not_detected" if similarity is None else "detected"}

    conditions = {}
    for name, values in (("historical", (.1, .2)), ("control", (0., None)), ("portrait_reference", (.5, .4))):
        scenes = []
        for index, month, objects in ((0, 1, [obj("mug", values[0]), obj("cap", values[1])]),
                                      (2, 5, [obj("mug", .9)])):
            scenes.append({"photo_index": index, "month": month, "image": "image.png",
                           "sha256": digest, "detections_image": "image.png", "objects": objects})
        conditions[name] = {"label": name, "summary": {}, "scenes": scenes}
    data = {"schema_version": 1, "method": "Grounding DINO + DINOv2", "settings": {},
            "model_provenance": {}, "source_sha256": report.sha(root / "frozen_source.yaml"),
            "target_sha256": report.sha(directory / "targets.json"),
            "input_sha256": {"image.png": digest},
            "references": [{"object_id": name, "crop": "image.png"} for name in ("mug", "cap")],
            "conditions": conditions}
    (directory / "scores.json").write_text(json.dumps(data))
    return root, data


def test_common_detection_mean_keeps_real_zero_and_excludes_missing_and_may(object_scores):
    root, _ = object_scores
    result = report.object_comparison(root, {0})
    assert result["paired_edited_scenes"] == {
        "eligible_targets": 2, "common_detected_targets": 1,
        "control_mean": 0., "treatment_mean": .5, "mean_paired_delta": .5,
    }
    assert result["conditions"]["control"]["detected_count"] == 2
    assert result["conditions"]["control"]["expected_count"] == 3
    assert next(r for r in result["objects"] if r["object_id"] == "cap")["delta"] is None


def test_changed_scored_image_is_rejected(object_scores):
    root, _ = object_scores
    (root / "image.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Object-scored input changed"):
        report.object_comparison(root, {0})


def test_changed_scene_hash_is_rejected(object_scores):
    root, data = object_scores
    data["conditions"]["control"]["scenes"][0]["sha256"] = "wrong"
    (root / "object_evaluation/scores.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="scene image changed"):
        report.object_comparison(root, {0})


def test_evidence_outside_experiment_is_rejected(object_scores):
    root, data = object_scores
    (root.parent / "outside.png").write_bytes(b"outside")
    data["references"][0]["crop"] = "../outside.png"
    (root / "object_evaluation/scores.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="outside the experiment"):
        report.object_comparison(root, {0})


def test_face_report_can_exist_before_object_scores(tmp_path):
    assert report.object_comparison(tmp_path, {0}) is None


def test_blank_image_false_positive_control_remains_visible(object_scores):
    root, data = object_scores
    data["sanity_checks"] = {"blank_image_detection_counts": {"mug": 1, "cap": 0}}
    (root / "object_evaluation/scores.json").write_text(json.dumps(data))
    result = report.object_comparison(root, {0})
    assert result["blank_image_false_positive_targets"] == ["mug"]
    assert result["sanity_checks"] == data["sanity_checks"]
