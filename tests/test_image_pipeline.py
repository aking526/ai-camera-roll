import io
import json

import pytest
import yaml
from PIL import Image

from ai_camera_roll import (
    GenerationResult,
    ImageGenerationError,
    LLMResponseError,
    generate_images,
)
from ai_camera_roll.image_prompts import (
    generate_character_prompt,
    generate_object_prompt,
    generate_scene_prompts,
    reference_passes,
)
from ai_camera_roll.images import ImageModel
from ai_camera_roll.image_pipeline import _manifest


class FakeImages(ImageModel):
    model = "test-image-model"

    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def generate(self, prompt, *, references=(), width=1024, height=1024, seed=None):
        self.calls.append({"prompt": prompt, "references": references, "seed": seed})
        if len(self.calls) == self.fail_at:
            raise TimeoutError("Provider timed out")
        stream = io.BytesIO()
        Image.new("RGB", (width, height), "white").save(stream, format="PNG")
        return stream.getvalue()


@pytest.fixture
def sample(story_payload, roll_payload):
    roll = roll_payload(1, 1)
    roll["people"] = [{**roll["people"][0], "id": f"person_{i}"} for i in range(3)]
    roll["objects"] = [{**roll["objects"][0], "id": f"object_{i}"} for i in range(3)]
    roll["protagonist_id"] = "person_0"
    roll["photos"][0]["people"] = [
        {"person_id": p["id"], "appearance": "Standing together."}
        for p in roll["people"]
    ]
    roll["photos"][0]["objects"] = [
        {"object_id": o["id"], "appearance": "On a table."} for o in roll["objects"]
    ]
    return GenerationResult.model_validate(
        {"story": {"seed": "Maya", **story_payload(1)}, "camera_roll": roll}
    )


def responses():
    return [{"prompt": f"Reference {i}"} for i in range(6)] + [
        {"prompts": ["Initial scene", "Add remaining objects"]}
    ]


def test_prompt_compilation_includes_source_and_exact_slot_assignments(
    sample, fake_llm
):
    llm = fake_llm(
        [{"prompt": "Portrait"}, {"prompt": "Object"}, {"prompts": ["Scene", "Edit"]}]
    )
    generate_character_prompt(sample.camera_roll.people[0], story=sample.story, llm=llm)
    generate_object_prompt(sample.camera_roll.objects[0], story=sample.story, llm=llm)
    plan = generate_scene_prompts(sample, 0, llm=llm)
    assert plan.prompts == ["Scene", "Edit"]
    assert (
        json.loads(llm.calls[0]["prompt"])["person"]
        == sample.camera_roll.people[0].model_dump()
    )
    assert (
        json.loads(llm.calls[1]["prompt"])["object"]
        == sample.camera_roll.objects[0].model_dump()
    )
    assert "waist-up" in llm.calls[0]["system"]
    assert "pure white" in llm.calls[1]["system"]
    payload = json.loads(llm.calls[2]["prompt"])
    assert payload["photo"] == sample.camera_roll.photos[0].model_dump()
    assert payload["story"]["seed"] == sample.story.seed
    first, edit = payload["passes"]
    assert [entry["slot"] for entry in first["inputs"]] == [
        f"input_image_{i}" for i in range(4)
    ]
    assert edit["inputs"][0] == {
        "slot": "input_image_0",
        "role": "current scene from the previous pass",
    }
    assert edit["already_established_ids"] == [
        "person_0",
        "person_1",
        "person_2",
        "object_0",
    ]
    assert [entry["entity"]["id"] for entry in edit["inputs"][1:]] == [
        "object_1",
        "object_2",
    ]


def test_wrong_pass_count_and_empty_prompts_rejected(sample, fake_llm):
    for payload in [{"prompts": ["Only one pass"]}, {"prompts": [" ", "Edit"]}]:
        with pytest.raises(LLMResponseError):
            generate_scene_prompts(sample, 0, llm=fake_llm([payload]))


def test_reference_chunking_covers_every_entity_without_duplicates(sample):
    photo = sample.camera_roll.photos[0]
    photo.objects.append(photo.objects[0])
    groups = reference_passes(sample, 0)
    assert groups == [
        ["person_0", "person_1", "person_2", "object_0"],
        ["object_1", "object_2"],
    ]


def test_retained_portrait_preserves_initial_inputs_and_existing_edit_slots(sample):
    original = reference_passes(sample, 0)
    anchored = reference_passes(sample, 0, retain_protagonist_reference=True)
    assert anchored[0] == original[0]
    assert anchored[1] == [*original[1], "person_0"]
    assert len(anchored[1]) + 1 == 4  # Previous scene plus three references.


def test_retained_portrait_batches_new_entities_within_four_input_limit(sample):
    for index in range(3, 6):
        obj = sample.camera_roll.objects[0].model_copy(update={"id": f"object_{index}"})
        sample.camera_roll.objects.append(obj)
        sample.camera_roll.photos[0].objects.append(
            sample.camera_roll.photos[0].objects[0].model_copy(update={"object_id": obj.id})
        )
    groups = reference_passes(sample, 0, retain_protagonist_reference=True)
    assert groups == [
        ["person_0", "person_1", "person_2", "object_0"],
        ["object_1", "object_2", "person_0"],
        ["object_3", "object_4", "person_0"],
        ["object_5", "person_0"],
    ]
    assert all(len(group) <= 3 for group in groups[1:])


def test_retained_portrait_does_not_add_absent_protagonist_or_extra_pass(sample):
    sample.camera_roll.photos[0].people = sample.camera_roll.photos[0].people[1:]
    assert reference_passes(sample, 0, retain_protagonist_reference=True) == reference_passes(sample, 0)
    sample.camera_roll.photos[0].objects = []
    assert reference_passes(sample, 0, retain_protagonist_reference=True) == [["person_1", "person_2"]]
    sample.camera_roll.photos[0].people.insert(
        0, sample.camera_roll.photos[0].people[0].model_copy(update={"person_id": "person_0"})
    )
    assert reference_passes(sample, 0, retain_protagonist_reference=True) == [["person_0", "person_1", "person_2"]]


def test_retained_portrait_waits_until_protagonist_is_introduced(sample):
    photo = sample.camera_roll.photos[0]
    protagonist = photo.people.pop(0)
    for index in range(3, 5):
        person = sample.camera_roll.people[0].model_copy(update={"id": f"person_{index}"})
        sample.camera_roll.people.append(person)
        photo.people.append(protagonist.model_copy(update={"person_id": person.id}))
    photo.people.append(protagonist)
    groups = reference_passes(sample, 0, retain_protagonist_reference=True)
    assert groups == [
        ["person_1", "person_2", "person_3", "person_4"],
        ["person_0", "object_0", "object_1"],
        ["object_2", "person_0"],
    ]


def test_retained_portrait_prompt_marks_anchor_as_established_not_new(sample, fake_llm):
    llm = fake_llm([{"prompts": ["Scene", "Edit"]}])
    generate_scene_prompts(sample, 0, llm=llm, retain_protagonist_reference=True)
    payload = json.loads(llm.calls[0]["prompt"])
    first, edit = payload["passes"]
    assert first["new_ids"] == ["person_0", "person_1", "person_2", "object_0"]
    assert edit["new_ids"] == ["object_1", "object_2"]
    assert edit["inputs"][0]["role"] == "current scene from the previous pass"
    assert edit["inputs"][-1]["entity"]["id"] == "person_0"
    assert edit["inputs"][-1]["slot"] == "input_image_3"
    assert edit["inputs"][-1]["role"] == "identity reference for an already present person"
    assert "not a new person to add" in llm.calls[0]["system"]


def test_retained_portrait_run_resumes_and_rejects_policy_switch(sample, fake_llm, tmp_path):
    llm = fake_llm(responses())
    images = FakeImages()
    original = _manifest(sample, llm, images, 42)
    assert "retain_protagonist_reference" not in original.settings
    manifest = generate_images(
        sample, llm=llm, image_model=images, output_dir=tmp_path,
        retain_protagonist_reference=True, progress=lambda _: None,
    )
    assert manifest.settings["retain_protagonist_reference"] is True
    assert manifest.signature != original.signature
    assert [step.seed for step in manifest.photos[0].passes] == [
        step.seed for step in original.photos[0].passes
    ]
    assert [len(call["references"]) for call in images.calls[-2:]] == [4, 4]
    anchor = manifest.photos[0].passes[1].inputs[-1]
    assert anchor.entity_id == "person_0"
    assert anchor.path == manifest.references[0].reference_path
    generate_images(
        sample, llm=llm, image_model=images, output_dir=tmp_path,
        retain_protagonist_reference=True, progress=lambda _: None,
    )
    assert len(images.calls) == 8
    assert len(llm.calls) == 7
    with pytest.raises(ValueError, match="fresh output"):
        generate_images(sample, llm=llm, image_model=images, output_dir=tmp_path)
    assert len(images.calls) == 8
    assert len(llm.calls) == 7


def test_run_saves_all_outputs_and_resume_makes_no_provider_calls(
    sample, fake_llm, tmp_path
):
    llm = fake_llm(responses())
    images = FakeImages()
    before = sample.model_dump()
    manifest = generate_images(
        sample,
        llm=llm,
        image_model=images,
        output_dir=tmp_path,
        progress=lambda _: None,
    )
    assert len(llm.calls) == 7
    assert len(images.calls) == 8
    assert [len(call["references"]) for call in images.calls[-2:]] == [4, 3]
    for asset in manifest.references:
        assert Image.open(tmp_path / asset.image.path).size == (1024, 1024)
        assert Image.open(tmp_path / asset.reference_path).size == (448, 448)
    assert (tmp_path / manifest.photos[0].final_path).read_bytes() == (
        tmp_path / manifest.photos[0].passes[-1].path
    ).read_bytes()
    assert (
        yaml.safe_load((tmp_path / "manifest.yaml").read_text())["photos"][0]["passes"][
            1
        ]["inputs"][0]["entity_id"]
        is None
    )
    assert sample.model_dump() == before
    generate_images(
        sample,
        llm=llm,
        image_model=images,
        output_dir=tmp_path,
        progress=lambda _: None,
    )
    assert len(llm.calls) == 7
    assert len(images.calls) == 8


def test_failed_edit_resumes_without_regenerating_references_or_prompts(
    sample, fake_llm, tmp_path
):
    llm = fake_llm(responses())
    images = FakeImages(fail_at=8)
    with pytest.raises(ImageGenerationError, match="photo 0, pass 1"):
        generate_images(
            sample,
            llm=llm,
            image_model=images,
            output_dir=tmp_path,
            progress=lambda _: None,
        )
    payload = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert payload["photos"][0]["passes"][0]["completed"]
    assert not payload["photos"][0]["passes"][1]["completed"]
    assert payload["photos"][0]["passes"][1]["prompt"] == "Add remaining objects"
    images.fail_at = None
    generate_images(
        sample,
        llm=llm,
        image_model=images,
        output_dir=tmp_path,
        progress=lambda _: None,
    )
    assert len(llm.calls) == 7
    assert len(images.calls) == 9
    assert len(images.calls[-1]["references"]) == 3


@pytest.mark.parametrize("change", ["source", "seed", "model"])
def test_incompatible_resume_rejected_before_paid_calls(
    sample, fake_llm, tmp_path, change
):
    llm = fake_llm(responses())
    images = FakeImages()
    generate_images(
        sample,
        llm=llm,
        image_model=images,
        output_dir=tmp_path,
        progress=lambda _: None,
    )
    options = {}
    if change == "source":
        sample.story.seed = "Another story"
    elif change == "seed":
        options["seed"] = 123
    else:
        images.model = "other-model"
    with pytest.raises(ValueError, match="fresh output"):
        generate_images(
            sample,
            llm=llm,
            image_model=images,
            output_dir=tmp_path,
            progress=lambda _: None,
            **options,
        )
    assert len(images.calls) == 8
    assert len(llm.calls) == 7


def test_objects_only_scene_supported(sample, fake_llm, tmp_path):
    sample.camera_roll.photos[0].people = []
    llm = fake_llm(responses()[:6] + [{"prompts": ["Objects on a table"]}])
    images = FakeImages()
    manifest = generate_images(
        sample,
        llm=llm,
        image_model=images,
        output_dir=tmp_path,
        progress=lambda _: None,
    )
    assert len(manifest.photos[0].passes) == 1
    assert len(images.calls[-1]["references"]) == 3


def test_completed_reference_is_not_silently_replaced_when_damaged(
    sample, fake_llm, tmp_path
):
    llm = fake_llm(responses())
    images = FakeImages()
    manifest = generate_images(
        sample,
        llm=llm,
        image_model=images,
        output_dir=tmp_path,
        progress=lambda _: None,
    )
    (tmp_path / manifest.references[0].image.path).write_bytes(b"corrupted")
    with pytest.raises(ImageGenerationError, match="reference person_0"):
        generate_images(
            sample,
            llm=llm,
            image_model=images,
            output_dir=tmp_path,
            progress=lambda _: None,
        )
    assert len(images.calls) == 8
