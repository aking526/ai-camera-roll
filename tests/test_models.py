import pytest
from pydantic import ValidationError

from ai_camera_roll import CameraRoll, LifeStory


def test_photo_can_show_only_objects(roll_payload):
    roll = CameraRoll.model_validate(roll_payload(1, 1))
    assert roll.photos[0].people == []
    assert roll.photos[0].objects[0].object_id == roll.objects[0].id


@pytest.mark.parametrize("catalog", ["people", "objects"])
def test_duplicate_catalog_ids_rejected(catalog, roll_payload):
    payload = roll_payload(1, 1)
    payload[catalog].append(payload[catalog][0].copy())
    with pytest.raises(ValidationError, match="unique IDs"):
        CameraRoll.model_validate(payload)


@pytest.mark.parametrize("catalog,reference", [("people", "person_id"), ("objects", "object_id")])
def test_unknown_references_rejected(catalog, reference, roll_payload):
    payload = roll_payload(1, 1)
    payload["photos"][0][catalog] = [{reference: "missing", "appearance": "In frame."}]
    with pytest.raises(ValidationError, match="unknown IDs"):
        CameraRoll.model_validate(payload)


def test_unknown_protagonist_rejected(roll_payload):
    payload = roll_payload(1, 1)
    payload["protagonist_id"] = "missing"
    with pytest.raises(ValidationError, match="Unknown protagonist"):
        CameraRoll.model_validate(payload)


def test_every_photo_requires_an_object(roll_payload):
    payload = roll_payload(1, 1)
    payload["photos"][0]["objects"] = []
    with pytest.raises(ValidationError, match="at least 1 item"):
        CameraRoll.model_validate(payload)


def test_photos_must_be_chronological(roll_payload):
    payload = roll_payload(2, 2)
    payload["photos"].reverse()
    with pytest.raises(ValidationError, match="ordered by year and month"):
        CameraRoll.model_validate(payload)


@pytest.mark.parametrize("month", [0, 13])
def test_month_range(month, roll_payload):
    payload = roll_payload(1, 1)
    payload["photos"][0]["month"] = month
    with pytest.raises(ValidationError, match="month"):
        CameraRoll.model_validate(payload)


@pytest.mark.parametrize("years", [[1, 1], [1, 3], [2, 3], [2, 1]])
def test_story_years_are_complete_and_ordered(years, story_payload):
    payload = story_payload(2)
    for chapter, year in zip(payload["years"], years):
        chapter["year"] = year
    with pytest.raises(ValidationError, match="consecutive"):
        LifeStory(seed="Maya", **payload)
