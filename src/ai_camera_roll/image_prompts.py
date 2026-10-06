"""Compile catalog records and story moments into image-model instructions."""

import json
from importlib.resources import files
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from .llm import LLM, LLMResponseError
from .models import GenerationResult, LifeStory, Person, PersonalObject

PromptText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ImagePrompt(BaseModel):
    prompt: PromptText


class ScenePromptPlan(BaseModel):
    prompts: list[PromptText] = Field(min_length=1)


def instruction(name: str) -> str:
    return (
        files("ai_camera_roll")
        .joinpath("prompts", f"image_{name}.md")
        .read_text(encoding="utf-8")
    )


def generate_character_prompt(
    person: Person, *, story: LifeStory, llm: LLM
) -> ImagePrompt:
    return llm.generate(
        system=instruction("character"),
        prompt=json.dumps(
            {"story": story.model_dump(), "person": person.model_dump()},
            ensure_ascii=False,
        ),
        response_model=ImagePrompt,
    )


def generate_object_prompt(
    obj: PersonalObject, *, story: LifeStory, llm: LLM
) -> ImagePrompt:
    return llm.generate(
        system=instruction("object"),
        prompt=json.dumps(
            {"story": story.model_dump(), "object": obj.model_dump()},
            ensure_ascii=False,
        ),
        response_model=ImagePrompt,
    )


def reference_passes(result: GenerationResult, photo_index: int) -> list[list[str]]:
    """Four assets initially; later passes reserve slot zero for the current scene."""
    photo = result.camera_roll.photos[photo_index]
    ids = list(
        dict.fromkeys(
            [p.person_id for p in photo.people]
            + [obj.object_id for obj in photo.objects]
        )
    )
    return [ids[:4]] + [ids[start : start + 3] for start in range(4, len(ids), 3)]


def generate_scene_prompts(
    result: GenerationResult, photo_index: int, *, llm: LLM
) -> ScenePromptPlan:
    photo = result.camera_roll.photos[photo_index]
    people = {p.id: p for p in result.camera_roll.people}
    objects = {obj.id: obj for obj in result.camera_roll.objects}
    appearances = {p.person_id: p.appearance for p in photo.people}
    appearances.update({obj.object_id: obj.appearance for obj in photo.objects})
    passes = []
    established = []
    for index, ids in enumerate(reference_passes(result, photo_index)):
        inputs = []
        if index:
            inputs.append(
                {
                    "slot": "input_image_0",
                    "role": "current scene from the previous pass",
                }
            )
        for position, identifier in enumerate(ids, start=1 if index else 0):
            record = people.get(identifier) or objects[identifier]
            inputs.append(
                {
                    "slot": f"input_image_{position}",
                    "entity": record.model_dump(),
                    "kind": "person" if identifier in people else "object",
                    "appearance": appearances[identifier],
                }
            )
        passes.append(
            {
                "inputs": inputs,
                "already_established_ids": established.copy(),
                "new_ids": ids,
            }
        )
        established.extend(ids)
    plan = llm.generate(
        system=instruction("scene"),
        prompt=json.dumps(
            {
                "story": {
                    "seed": result.story.seed,
                    "background": result.story.background,
                    "years": [
                        chapter.model_dump()
                        for chapter in result.story.years
                        if chapter.year <= photo.year
                    ],
                },
                "photo": photo.model_dump(),
                "passes": passes,
            },
            ensure_ascii=False,
        ),
        response_model=ScenePromptPlan,
    )
    if len(plan.prompts) != len(passes):
        raise LLMResponseError(
            f"Expected {len(passes)} scene prompts, received {len(plan.prompts)}."
        )
    return plan
