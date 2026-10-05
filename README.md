# AI Camera Roll

A small Python library that turns a seed about a fictional person into a life
story, then extracts a structured camera roll. Defaults: five years, ten photo
ideas per year, and two LLM requests. It produces text and JSON, not images.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
export OPENCODE_GO_API_KEY='your-key'
export OPENCODE_GO_MODEL='glm-5.2'
uv run python examples/generate.py
```

Choose a model listed for **Chat Completions** in the
[OpenCode Go documentation](https://opencode.ai/docs/go/#endpoints). Supply the
bare model ID, not the OpenCode CLI's `opencode-go/` prefix. Models that require
the Messages or Responses endpoints are not supported by this adapter.

Go currently describes its intended usage as coding-agent traffic. Creative
generation support is unverified. This library identifies itself as
`ai-camera-roll/0.1.0`, sends a stable session header per LLM instance, and surfaces
service errors without disguising the application or switching providers.

## Python API

```python
from ai_camera_roll import OpenCodeGoLLM, generate_camera_roll

llm = OpenCodeGoLLM(model="glm-5.2")  # Reads OPENCODE_GO_API_KEY.
result = generate_camera_roll(
    "Maya, 24, lives in Philadelphia, rides a blue bicycle, and loves cooking.",
    llm=llm,
    years=5,
    photos_per_year=10,
)

print(result.story.years[0].narrative)
print(result.camera_roll.photos[0].description)
print(result.model_dump_json(indent=2))
```

`OpenCodeGoLLM` also accepts `api_key`, `timeout` (seconds, default 180), and
`max_tokens` (default 32768). Use a new instance for an unrelated generation to
start a new provider session. Larger stories or photo counts may require a larger
output budget within the selected model's limits.

Both stages are independently callable, so you can save or edit the story first:

```python
from pathlib import Path
from ai_camera_roll import LifeStory, generate_photos, generate_story

story = generate_story("Maya, a student who loves cycling and cooking.", llm=llm)
Path("story.json").write_text(story.model_dump_json(indent=2), encoding="utf-8")

# Edit the saved narrative if desired, then validate and load it again.
story = LifeStory.model_validate_json(Path("story.json").read_text(encoding="utf-8"))
roll = generate_photos(story, llm=llm, photos_per_year=10)
```

Load a complete saved result with `GenerationResult.model_validate_json(...)`.
The original seed is preserved at `result.story.seed`; generation does not ask
the LLM to reproduce it. The library returns Pydantic objects and leaves file
storage to the caller. The example writes to `output/camera_roll.json`.

## Output and continuity

- **LifeStory:** original seed, background, and consecutive yearly narratives.
- **CameraRoll:** protagonist ID, shared people and object catalogs, and photos
  ordered by relative year and month.
- **Person:** stable ID, name, relationship, age at the beginning of year 1, and
  baseline visual appearance.
- **PersonalObject:** stable ID, name, visual appearance, and personal significance.
- **PhotoIdea:** scene description, story context, and references to visible people
  and objects with scene-specific appearance notes.

For example, `object_green_cap` has one canonical description: faded green canvas
with a mountain patch. One photo shows it hanging by the door; another shows it
rain-soaked on a hike. These are appearances of the same object. A replacement
cap receives a new ID.

Every photo includes at least one personal object. Photos can have no visible
people: a bicycle outside a cafe or a sketchbook on a kitchen table can belong in
the protagonist's own camera roll. Recurring identities are consistent within a
generated roll; regenerating the second stage creates a new catalog and may
choose different IDs or previously unspecified visual details.

Pydantic checks structure, year/month ranges, unique catalog IDs, references, and
chronological ordering. The pipeline checks the requested year and photo counts.
Prompts address semantic continuity such as aging, ownership, and relationships;
those properties still require human review and are not guaranteed by JSON
validation. The models intentionally do not implement a full object lifecycle
or relationship history system.

## Prompts and providers

The editable writing instructions live in
[`story.md`](src/ai_camera_roll/prompts/story.md) and
[`photos.md`](src/ai_camera_roll/prompts/photos.md). They are packaged with the
library and loaded with `importlib.resources`, independent of the working
directory. The pipeline supplies input as JSON; the LLM wrapper appends the
Pydantic-generated output schema. Schema definitions have one source of truth.

To add a provider, subclass `LLM` and implement only
`_complete(self, *, system: str, prompt: str) -> str`. Return the completion's
text, and raise on transport errors or incomplete output. The inherited
`generate(..., response_model=SomePydanticModel)` handles schema instructions and
JSON validation. No provider SDK types enter the pipeline or domain models.

There are no automatic retries or JSON repair calls. Successful end-to-end
generation makes exactly two requests. Invalid JSON, invalid references,
unexpected counts, or incomplete output raise `LLMResponseError`. Input errors
raise `ValueError` or Pydantic `ValidationError`; HTTP failures propagate HTTPX
exceptions. An `httpx.HTTPStatusError` exposes the service's response through
`error.response`. If extraction fails, you can rerun `generate_photos` on a saved
story without regenerating its narrative.

## Development

```bash
uv run pytest
uv build
```

Tests use a fake LLM and mocked HTTP transport; they do not require credentials or
make paid calls. The example above is the live smoke test. Review its output for
believable everyday moments, object recurrence, visual consistency, and chronology
in addition to the structural checks.
