"""Generate reusable reference assets and scenes, checkpointing every provider call."""

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .image_prompts import (
    generate_character_prompt,
    generate_object_prompt,
    generate_scene_prompts,
    instruction,
    reference_passes,
)
from .images import ImageModel, png_bytes
from .llm import LLM
from .models import GenerationResult


class ImageGenerationError(RuntimeError):
    """Generation stopped; the manifest retains all completed work."""


class ImageInput(BaseModel):
    slot: int
    path: str
    entity_id: str | None = None  # None denotes the previous scene image.


class ImageArtifact(BaseModel):
    path: str
    seed: int
    prompt: str | None = None
    inputs: list[ImageInput] = Field(default_factory=list)
    completed: bool = False


class ReferenceAsset(BaseModel):
    entity_id: str
    kind: Literal["people", "objects"]
    image: ImageArtifact
    reference_path: str


class PhotoImages(BaseModel):
    photo_index: int
    year: int
    month: int
    passes: list[ImageArtifact]
    final_path: str


class ImageGenerationManifest(BaseModel):
    version: int = 1
    signature: str
    settings: dict
    references: list[ReferenceAsset]
    photos: list[PhotoImages]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _seed(seed: int, name: str) -> int:
    return int(_digest([seed, name])[:8], 16)


def _manifest(
    result: GenerationResult, llm: LLM, image_model: ImageModel, seed: int
) -> ImageGenerationManifest:
    settings = {
        "source_sha256": _digest(result.model_dump(mode="json")),
        "llm_adapter": type(llm).__name__,
        "llm_model": getattr(llm, "model", type(llm).__name__),
        "image_adapter": type(image_model).__name__,
        "image_model": image_model.model,
        "instructions_sha256": _digest(
            [instruction(name) for name in ("character", "object", "scene")]
        ),
        "seed": seed,
        "width": 1024,
        "height": 1024,
        "reference_size": 448,
        "pipeline_version": 1,
    }
    assets = []
    for kind in ("people", "objects"):
        for index, record in enumerate(getattr(result.camera_roll, kind)):
            slug = re.sub(r"[^A-Za-z0-9_-]", "_", record.id)[:80] or "asset"
            name = f"{index:03d}_{slug}"
            assets.append(
                ReferenceAsset(
                    entity_id=record.id,
                    kind=kind,
                    image=ImageArtifact(
                        path=f"references/{kind}/{name}.png",
                        seed=_seed(seed, f"{kind}:{record.id}"),
                    ),
                    reference_path=f"references/{kind}/{name}_input.png",
                )
            )
    by_id = {asset.entity_id: asset for asset in assets}
    photos = []
    for index, photo in enumerate(result.camera_roll.photos):
        steps = []
        for step_index, ids in enumerate(reference_passes(result, index)):
            inputs = []
            if step_index:
                inputs.append(ImageInput(slot=0, path=steps[-1].path))
            inputs.extend(
                ImageInput(
                    slot=position,
                    path=by_id[identifier].reference_path,
                    entity_id=identifier,
                )
                for position, identifier in enumerate(ids, start=1 if step_index else 0)
            )
            steps.append(
                ImageArtifact(
                    path=f"photos/{index:03d}/pass_{step_index:02d}.png",
                    seed=_seed(seed, f"photo:{index}:pass:{step_index}"),
                    inputs=inputs,
                )
            )
        photos.append(
            PhotoImages(
                photo_index=index,
                year=photo.year,
                month=photo.month,
                passes=steps,
                final_path=f"photos/{index:03d}_year_{photo.year}_month_{photo.month:02d}.png",
            )
        )
    return ImageGenerationManifest(
        signature=_digest(settings), settings=settings, references=assets, photos=photos
    )


def _path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(
            f"Artifact path must stay within the output directory: {relative}"
        )
    return path


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def generate_images(
    result: GenerationResult,
    *,
    llm: LLM,
    image_model: ImageModel,
    output_dir: Path | str,
    seed: int = 42,
    progress: Callable[[str], None] = print,
) -> ImageGenerationManifest:
    """Generate all references, then photos; rerunning resumes the same manifest."""
    result = GenerationResult.model_validate(result.model_dump())
    years = {chapter.year for chapter in result.story.years}
    if any(photo.year not in years for photo in result.camera_roll.photos):
        raise ValueError("Every photo must belong to a year present in the story.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    root = Path(output_dir)
    expected = _manifest(result, llm, image_model, seed)
    manifest_path = root / "manifest.yaml"
    if manifest_path.exists():
        manifest = ImageGenerationManifest.model_validate(
            yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        )
        if (
            manifest.signature != expected.signature
            or manifest.settings != expected.settings
        ):
            raise ValueError(
                "This output belongs to different source data, models, instructions, or seed. Choose a fresh output directory."
            )

        # Validate that checkpoint paths, seeds, and reference assignments still match this run.
        def layout(value: ImageGenerationManifest) -> dict:
            payload = value.model_dump()
            for asset in payload["references"]:
                asset["image"].pop("prompt")
                asset["image"].pop("completed")
            for photo in payload["photos"]:
                for step in photo["passes"]:
                    step.pop("prompt")
                    step.pop("completed")
            return payload

        if layout(manifest) != layout(expected):
            raise ValueError(
                "Manifest layout was changed. Choose a fresh output directory."
            )
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(
                "Choose an empty output directory, or one with a matching manifest.yaml."
            )
        manifest = expected

    def checkpoint() -> None:
        _write(
            manifest_path,
            yaml.safe_dump(
                manifest.model_dump(mode="json"),
                allow_unicode=True,
                sort_keys=False,
                width=100,
            ).encode("utf-8"),
        )

    def render(artifact: ImageArtifact, label: str) -> None:
        path = _path(root, artifact.path)
        if artifact.completed:
            # Never silently replace an established identity if its saved asset is damaged.
            png_bytes(path.read_bytes())
            progress(f"Reusing {label}: {artifact.path}")
            return
        progress(f"Generating {label}...")
        inputs = [_path(root, item.path).read_bytes() for item in artifact.inputs]
        content = image_model.generate(
            artifact.prompt, references=inputs, seed=artifact.seed
        )
        _write(path, png_bytes(content))
        artifact.completed = True
        checkpoint()
        progress(f"Saved {artifact.path}")

    checkpoint()
    label = "initialization"
    try:
        for asset in manifest.references:
            label = f"{asset.kind} reference {asset.entity_id}"
            if asset.image.prompt is None:
                progress(f"Writing prompt for {label}...")
                record = next(
                    item
                    for item in getattr(result.camera_roll, asset.kind)
                    if item.id == asset.entity_id
                )
                compiler = (
                    generate_character_prompt
                    if asset.kind == "people"
                    else generate_object_prompt
                )
                asset.image.prompt = compiler(
                    record, story=result.story, llm=llm
                ).prompt
                checkpoint()
            render(asset.image, label)
            _write(
                _path(root, asset.reference_path),
                png_bytes(_path(root, asset.image.path).read_bytes(), reference=True),
            )

        for photo in manifest.photos:
            label = f"photo {photo.photo_index} prompts"
            if any(step.prompt is None for step in photo.passes):
                progress(f"Writing prompt plan for photo {photo.photo_index}...")
                plan = generate_scene_prompts(result, photo.photo_index, llm=llm)
                for step, prompt in zip(photo.passes, plan.prompts, strict=True):
                    step.prompt = prompt
                checkpoint()
            for index, step in enumerate(photo.passes):
                label = f"photo {photo.photo_index}, pass {index}"
                render(step, label)
            _write(
                _path(root, photo.final_path),
                _path(root, photo.passes[-1].path).read_bytes(),
            )
            progress(f"Final photo: {photo.final_path}")
    except Exception as error:
        raise ImageGenerationError(
            f"Stopped at {label}: {error}. Completed work is saved in {manifest_path}; rerun to resume."
        ) from error
    progress(
        f"Finished: {len(manifest.references)} references and {len(manifest.photos)} photos in {root}"
    )
    return manifest
