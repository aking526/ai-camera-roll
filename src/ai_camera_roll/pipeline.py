"""Two independently callable stages, or one convenience function for both."""

import json
from collections import Counter
from importlib.resources import files

from .llm import LLM, LLMResponseError
from .models import CameraRoll, GenerationResult, LifeStory, _StoryContent


def _prompt(name: str) -> str:
    return files("ai_camera_roll").joinpath("prompts", f"{name}.md").read_text(
        encoding="utf-8"
    )


def _check_count(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def generate_story(seed: str, *, llm: LLM, years: int = 5) -> LifeStory:
    """Generate a coherent story while preserving the caller's original seed."""
    if not seed.strip():
        raise ValueError("seed must not be empty.")
    _check_count("years", years)
    content = llm.generate(
        system=_prompt("story"),
        prompt=json.dumps({"seed": seed, "years": years}, ensure_ascii=False),
        response_model=_StoryContent,
    )
    if len(content.years) != years:
        raise LLMResponseError(
            f"Expected {years} story years, received {len(content.years)}."
        )
    return LifeStory(seed=seed, **content.model_dump())


def generate_photos(
    story: LifeStory, *, llm: LLM, photos_per_year: int = 10
) -> CameraRoll:
    """Extract a shared catalog and photo ideas from an existing life story."""
    _check_count("photos_per_year", photos_per_year)
    # Revalidate stories that callers may have edited in place.
    story = LifeStory.model_validate(story.model_dump())
    roll = llm.generate(
        system=_prompt("photos"),
        prompt=json.dumps(
            {"story": story.model_dump(), "photos_per_year": photos_per_year},
            ensure_ascii=False,
        ),
        response_model=CameraRoll,
    )
    expected = {chapter.year: photos_per_year for chapter in story.years}
    actual = dict(Counter(photo.year for photo in roll.photos))
    if actual != expected:
        raise LLMResponseError(
            f"Expected photo counts by year {expected}, received {actual}."
        )
    return roll


def generate_camera_roll(
    seed: str, *, llm: LLM, years: int = 5, photos_per_year: int = 10
) -> GenerationResult:
    """Generate the story and camera roll with exactly two successful LLM calls."""
    _check_count("photos_per_year", photos_per_year)
    story = generate_story(seed, llm=llm, years=years)
    camera_roll = generate_photos(story, llm=llm, photos_per_year=photos_per_year)
    return GenerationResult(story=story, camera_roll=camera_roll)
