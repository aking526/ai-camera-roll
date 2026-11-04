"""Offline checks for object crops, selection, missingness and provenance."""

import math

from PIL import Image
import pytest
import yaml

import score_objects as scorer


def test_crop_encloses_fractional_box_and_clamps_image_boundaries():
    assert scorer.crop_box((-2.2, -1.1, 22.8, 12.1), (20, 10)) == (0, 0, 20, 10)
    assert scorer.crop_box((4.9, 1.2, 8.1, 4.1), (20, 10)) == (4, 1, 9, 5)
    for box in ((30, 0, 40, 5), (5, 5, 5, 9), (9, 4, 3, 8)):
        with pytest.raises(ValueError, match="Invalid detector crop"):
            scorer.crop_box(box, (20, 10))


def test_letterbox_preserves_both_edges_instead_of_center_cropping():
    image = Image.new("RGB", (200, 100), "red")
    image.paste("blue", (180, 0, 200, 100))
    prepared = scorer.prepare_crop(image)
    assert prepared.size == (224, 224)
    assert prepared.getpixel((112, 0)) == (124, 116, 104)
    assert prepared.getpixel((112, 223)) == (124, 116, 104)
    assert prepared.getpixel((4, 112)) == (255, 0, 0)
    assert prepared.getpixel((220, 112)) == (0, 0, 255)


def test_nms_keeps_highest_confidence_and_spatially_distinct_candidates():
    high = {"confidence": 0.9, "bbox": [0, 0, 100, 100]}
    overlapping = {"confidence": 0.8, "bbox": [1, 1, 101, 101]}
    separate = {"confidence": 0.6, "bbox": [200, 0, 300, 100]}
    assert scorer.suppress([overlapping, separate, high]) == [high, separate]
    assert scorer.iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0
    assert scorer.suppress([]) == []


def test_cosine_retains_negative_and_zero_scores_for_normalized_features():
    assert scorer.cosine([1, 0], [-1, 0]) == -1
    assert scorer.cosine([1, 0], [0, 1]) == 0
    assert scorer.cosine([1 + 1e-8, 0], [1, 0]) == 1
    with pytest.raises(ValueError, match="dimensions differ"):
        scorer.cosine([1, 0], [1])


def test_summary_retains_missing_denominator_without_treating_it_as_zero():
    summary = scorer.summarize([{"objects": [
        {"similarity": None, "candidate_count": 0},
        {"similarity": 0.0, "candidate_count": 2},
        {"similarity": -0.4, "candidate_count": 1},
    ]}])
    assert summary == {"expected_count": 3, "detected_count": 2,
                       "multiple_candidate_count": 1, "mean_similarity_detected": -0.2}
    assert scorer.summarize([{"objects": [{"similarity": None, "candidate_count": 0}]}])["mean_similarity_detected"] is None


@pytest.fixture
def saved_case(tmp_path):
    project, experiment, models = (tmp_path / name for name in ("project", "experiment", "models"))
    (project / "evaluation").mkdir(parents=True)
    experiment.mkdir()
    models.mkdir()
    source = experiment / "frozen_source.yaml"
    source.write_text("story: frozen\n")
    scorer.write(project / "evaluation/object_targets.json", {
        "source_sha256": scorer.digest(source),
        "objects": [{"id": "object_mug", "reference_id": "object_mug", "name": "Mug", "query": "mug."}],
        "scenes": [{"photo_index": 0, "month": 1, "objects": ["object_mug"],
                    "requirements": {"object_mug": "On the desk."}}],
    })
    downloads = {}
    for name in scorer.MODEL_IDS.values():
        folder = models / name.split("/")[-1]
        folder.mkdir()
        (folder / "test.bin").write_bytes(b"offline placeholder")
        downloads[name] = {"revision": "test-revision"}
    scorer.write(models / "downloads.json", downloads)
    roots = {"historical": project / "output/images", "control": experiment / "control/images",
             "portrait_reference": experiment / "portrait_reference/images"}
    for condition, root in roots.items():
        root.mkdir(parents=True)
        Image.new("RGB", (20, 10), "red").save(root / "reference.png")
        image = Image.new("RGB", (20, 10), "white" if condition == "historical" else "blue")
        if condition != "historical":
            image.paste("red", (10, 0, 20, 10))
        image.save(root / "photo.png")
        (root / "manifest.yaml").write_text(yaml.safe_dump({
            "references": [{"entity_id": "object_mug", "image": {"path": "reference.png"}}],
            "photos": [{"photo_index": 0, "final_path": "photo.png"}],
        }))
    return project, experiment, models, roots


def test_full_run_selects_detector_confidence_and_preserves_hashed_evidence(saved_case, monkeypatch):
    project, experiment, model_root, roots = saved_case
    calls = []

    class FakeModels:
        def __init__(self, root):
            pass

        def detect(self, image, query):
            color = image.getpixel((0, 0))
            calls.append((color, query))
            if color in ((255, 255, 255), (128, 128, 128)):
                return []
            if color == (255, 0, 0):
                return [{"bbox": [0, 0, 20, 10], "confidence": 0.95}]
            return scorer.suppress([
                {"bbox": [10, 0, 20, 10], "confidence": 0.6},
                {"bbox": [0, 0, 10, 10], "confidence": 0.9},
            ])

        def embed(self, image):
            feature = image.getpixel((0, 0))
            norm = math.sqrt(sum(v * v for v in feature))
            return [v / norm for v in feature]

    monkeypatch.setattr(scorer, "Models", FakeModels)
    monkeypatch.setattr(scorer.importlib.metadata, "version", lambda _: "test")
    result = scorer.run(project, experiment, model_root)
    missing = result["conditions"]["historical"]["scenes"][0]["objects"][0]
    assert missing["similarity"] is None and missing["selected_index"] is None
    assert missing["status"] == "not_detected"
    control = result["conditions"]["control"]["scenes"][0]["objects"][0]
    assert control["similarity"] == 0.0
    assert control["detections"][1]["similarity"] == 1.0
    assert control["selected_index"] == 0 and control["candidate_count"] == 2
    assert result["conditions"]["portrait_reference"]["scenes"][0]["objects"] == [
        {**control, "detections": [
            {**d, "crop": d["crop"].replace("control_", "portrait_reference_")}
            for d in control["detections"]
        ]}
    ]
    assert sum(color == (0, 0, 255) for color, _ in calls) == 1
    for relative, expected in result["input_sha256"].items():
        assert scorer.digest(experiment / relative) == expected
    for condition, root in roots.items():
        scene = result["conditions"][condition]["scenes"][0]
        assert scene["sha256"] == scorer.digest(root / "photo.png")
        assert scene["sha256"] == scorer.digest(experiment / scene["image"])
        assert result["manifest_sha256"][condition] == scorer.digest(root / "manifest.yaml")
    assert result["reference_sha256"]["historical"] == result["reference_sha256"]["portrait_reference"]
    assert result["sanity_checks"]["repeated_reference_max_embedding_difference"] == 0


def test_reference_mismatch_stops_before_loading_models(saved_case, monkeypatch):
    project, experiment, model_root, roots = saved_case
    Image.new("RGB", (20, 10), "green").save(roots["portrait_reference"] / "reference.png")
    monkeypatch.setattr(scorer, "Models", lambda _: pytest.fail("Models must not load on mismatched inputs"))
    with pytest.raises(ValueError, match="Canonical object references differ"):
        scorer.run(project, experiment, model_root)


def test_changed_source_stops_before_loading_models(saved_case, monkeypatch):
    project, experiment, model_root, _ = saved_case
    (experiment / "frozen_source.yaml").write_text("story: changed\n")
    monkeypatch.setattr(scorer, "Models", lambda _: pytest.fail("Models must not load on mismatched inputs"))
    with pytest.raises(ValueError, match="targets do not match"):
        scorer.run(project, experiment, model_root)
