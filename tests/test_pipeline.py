import json
from collections import Counter

import pytest
from pydantic import ValidationError

from ai_camera_roll import (
    GenerationResult,
    LLMResponseError,
    LifeStory,
    generate_camera_roll,
    generate_photos,
    generate_story,
)


@pytest.mark.parametrize("options", [{}, {"years": 2, "photos_per_year": 3}])
def test_two_pass_generation(options, fake_llm, story_payload, roll_payload):
    years = options.get("years", 5)
    count = options.get("photos_per_year", 10)
    llm = fake_llm([story_payload(years), roll_payload(years, count)])
    seed = ' Zoë says "hello".\nShe lives in Montréal. '

    result = generate_camera_roll(seed, llm=llm, **options)

    assert len(llm.calls) == 2
    assert result.story.seed == seed
    assert Counter(photo.year for photo in result.camera_roll.photos) == {
        year: count for year in range(1, years + 1)
    }
    assert json.loads(llm.calls[0]["prompt"]) == {"seed": seed, "years": years}
    assert json.loads(llm.calls[1]["prompt"]) == {
        "story": result.story.model_dump(),
        "photos_per_year": count,
    }
    schema = json.loads(llm.calls[1]["system"].split("JSON schema:\n")[1])
    assert "photos" in schema["properties"]
    assert "PhotoIdea" in schema["$defs"]
    assert GenerationResult.model_validate_json(result.model_dump_json()) == result


def test_each_stage_is_independent_and_story_can_be_edited(
    fake_llm, story_payload, roll_payload
):
    storyteller = fake_llm([story_payload(1)])
    story = generate_story("Maya loves cycling.", llm=storyteller, years=1)
    story = LifeStory.model_validate_json(story.model_dump_json())
    story.years[0].narrative = "In June, Maya starts cycling to her pottery class."
    photographer = fake_llm([roll_payload(1, 2)])

    roll = generate_photos(story, llm=photographer, photos_per_year=2)

    assert len(storyteller.calls) == len(photographer.calls) == 1
    assert len(roll.photos) == 2
    assert json.loads(photographer.calls[0]["prompt"])["story"] == story.model_dump()


def test_invalid_edited_story_is_rejected_before_call(fake_llm, story_payload):
    story = LifeStory(seed="Maya", **story_payload(1))
    story.years[0].year = 2
    llm = fake_llm([])
    with pytest.raises(ValidationError, match="consecutive"):
        generate_photos(story, llm=llm)
    assert llm.calls == []


@pytest.mark.parametrize("field", ["years", "photos_per_year"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_counts_never_call_provider(field, value, fake_llm):
    llm = fake_llm([])
    with pytest.raises(ValueError, match="positive integer"):
        generate_camera_roll("Maya", llm=llm, **{field: value})
    assert llm.calls == []


def test_empty_seed_never_calls_provider(fake_llm):
    llm = fake_llm([])
    with pytest.raises(ValueError, match="seed"):
        generate_story(" \n ", llm=llm)
    assert llm.calls == []


def test_wrong_story_length_stops_before_extraction(fake_llm, story_payload):
    llm = fake_llm([story_payload(1)])
    with pytest.raises(LLMResponseError, match="Expected 5 story years"):
        generate_camera_roll("Maya", llm=llm)
    assert len(llm.calls) == 1


@pytest.mark.parametrize("case", ["missing_photo", "wrong_distribution", "extra_year"])
def test_wrong_photo_counts_rejected(case, fake_llm, story_payload, roll_payload):
    payload = roll_payload(2, 2)
    if case == "missing_photo":
        payload["photos"].pop()
    elif case == "wrong_distribution":
        payload["photos"][2]["year"] = 1
        payload["photos"][2]["month"] = 12
    else:
        payload["photos"][-1]["year"] = 3
    llm = fake_llm([payload])
    story = LifeStory(seed="Maya", **story_payload(2))
    with pytest.raises(LLMResponseError, match="photo counts by year"):
        generate_photos(story, llm=llm, photos_per_year=2)
    assert len(llm.calls) == 1


@pytest.mark.parametrize("response", ["not JSON", "```json\n{}\n```", "{}"])
def test_invalid_output_fails_without_retry(response, fake_llm):
    llm = fake_llm([response])
    with pytest.raises(LLMResponseError, match="Invalid"):
        generate_story("Maya", llm=llm)
    assert len(llm.calls) == 1


def test_prompts_load_outside_repo(monkeypatch, tmp_path, fake_llm, story_payload, roll_payload):
    monkeypatch.chdir(tmp_path)
    llm = fake_llm([story_payload(1), roll_payload(1, 1)])
    result = generate_camera_roll("Maya", llm=llm, years=1, photos_per_year=1)
    assert len(result.camera_roll.photos) == 1
    assert "believable fictional lives" in llm.calls[0]["system"]
    assert "personal camera roll" in llm.calls[1]["system"]
