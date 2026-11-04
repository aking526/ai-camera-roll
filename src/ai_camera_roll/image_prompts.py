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


def reference_passes(
    result: GenerationResult,
    photo_index: int,
    *,
    retain_protagonist_reference: bool = False,
) -> list[list[str]]:
    """Four assets initially; optionally retain an established protagonist on edits."""
    photo = result.camera_roll.photos[photo_index]
    ids = list(
        dict.fromkeys(
            [p.person_id for p in photo.people]
            + [obj.object_id for obj in photo.objects]
        )
    )
    groups = [ids[:4]]
    remaining = ids[4:]
    established = set(groups[0])
    protagonist = result.camera_roll.protagonist_id
    while remaining:
        anchor = retain_protagonist_reference and protagonist in established
        capacity = 2 if anchor else 3  # Slot zero holds the previous scene.
        new_ids, remaining = remaining[:capacity], remaining[capacity:]
        groups.append(new_ids + ([protagonist] if anchor else []))
        established.update(new_ids)
    return groups


def generate_scene_prompts(
    result: GenerationResult,
    photo_index: int,
    *,
    llm: LLM,
    retain_protagonist_reference: bool = False,
) -> ScenePromptPlan:
    photo = result.camera_roll.photos[photo_index]
    people = {p.id: p for p in result.camera_roll.people}
    objects = {obj.id: obj for obj in result.camera_roll.objects}
    appearances = {p.person_id: p.appearance for p in photo.people}
    appearances.update({obj.object_id: obj.appearance for obj in photo.objects})
    passes = []
    established = []
    groups = reference_passes(
        result, photo_index, retain_protagonist_reference=retain_protagonist_reference
    )
    for index, ids in enumerate(groups):
        inputs = []
        if index:
            inputs.append(
                {
                    "slot": "input_image_0",
                    "role": "current scene from the previous pass",
                }
            )
        new_ids = [identifier for identifier in ids if identifier not in established]
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
            if identifier in established:
                inputs[-1]["role"] = "identity reference for an already present person"
        passes.append(
            {
                "inputs": inputs,
                "already_established_ids": established.copy(),
                "new_ids": new_ids,
            }
        )
        established.extend(new_ids)
    system = instruction("scene")
    if retain_protagonist_reference:
        system += (
            "\nAn input marked as an identity reference for an already present person "
            "repeats their canonical portrait. Use it to preserve that existing person's "
            "facial identity, while retaining the scene's pose and appearance. It is not "
            "a new person to add. Only new_ids should introduce entities."
        )
    plan = llm.generate(
        system=system,
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
