# AI Camera Roll

A small Python library that turns a seed about a fictional person into a life
story, then extracts a structured camera roll. Defaults: five years, ten photo
ideas per year, and two LLM requests for the text stages. The text example saves
readable YAML; a separate image pipeline generates references and scene images.

## Web visualizer

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From the project root,
install the dependencies and start the local viewer:

```bash
uv sync
uv run python -m ai_camera_roll.viewer
```

Open **http://127.0.0.1:8000**. The included example loads from
`output/camera_roll.yaml`, with its images in `output/images/`. Viewing saved output
requires no API keys. Press **Ctrl+C** in the terminal to stop the server.

The horizontal timeline connects each photo event to people above and objects
below. Timeline cards show generated photo thumbnails. Select a moment to open its
photo, full description, and scene-specific appearances in a modal, with previous/next
controls. Photos with multiple generation passes open on the final image; select an
earlier pass in the thumbnail strip to inspect how the image evolved. Incomplete runs
show the latest available pass and mark remaining passes as pending. Each pass lists
its input references in slot order, marking new and reused people/object references.
Later passes also show the previous scene input and the references carried forward
through it. You can select a pending pass to inspect its planned references. Expand
**Show prompt** to read the selected pass's generation prompt; switching passes updates
the prompt and preserves whether it is expanded. Reference popups also include their
generation prompts. Popups use a wider layout on desktop and fit smaller screens.
Close it with Escape, the close button,
or a click outside. The timeline uses the full page width;
click a person or object to explore its shared description and connected moments.
The **People** and **Objects** tabs provide searchable catalogs with canonical
reference images. References also appear beside scene-specific appearances and in
each person's or object's detail popup. Use the year and
connection filters to narrow the timeline, or **Read the story** for the narrative.
Records display exact field names, value types, and zero-based paths into the saved
data. Photo details distinguish `PersonAppearance` / `ObjectAppearance` entries
from shared `Person` / `PersonalObject` records; click `person_id` or `object_id`
to follow the reference. Catalogs show every field, including IDs and significance.

Use **Open YAML** to preview another complete result (`.yaml`, `.yml`, or `.json`)
without changing the saved file. Images are loaded from `images/` beside the source
file and shown when the manifest's source fingerprint matches the loaded result,
including uploaded previews. Rolls without matching images remain browsable.
The viewer runs locally and makes no LLM calls.
You can also choose a source file and port at startup:

```bash
uv run python -m ai_camera_roll.viewer path/to/result.yml --port 8001
```

If you generated images in a different directory, select it explicitly:

```bash
uv run python -m ai_camera_roll.viewer path/to/result.yml --images path/to/images
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Set `OPENCODE_GO_API_KEY` in `.env` to your OpenCode Go API key, then run:

```bash
uv run python examples/generate.py
```

The adapter automatically loads `.env` from the current working directory, so run
the example from the project root. `.env.example` is only a template; `.env` is
already ignored by Git. Explicit `api_key=` values take priority over environment
variables, which take priority over `.env` values. The example also reads
`OPENCODE_GO_MODEL` from `.env` or the environment before constructing the adapter.

Choose a model listed for **Chat Completions** in the
[OpenCode Go documentation](https://opencode.ai/docs/go/#endpoints). Supply the
bare model ID, not the OpenCode CLI's `opencode-go/` prefix. Models that require
the Messages or Responses endpoints are not supported by this adapter.

Go currently describes its intended usage as coding-agent traffic. Creative
generation support is unverified. This library identifies itself as
`ai-camera-roll/0.1.0`, sends a stable session header per LLM instance, and surfaces
service errors without disguising the application or switching providers.

## Image generation

Set `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, and `CLOUDFLARE_IMAGE_MODEL`
in `.env` alongside the OpenCode Go credentials. The token needs Cloudflare
Workers AI Read and Edit permissions. The image adapter currently supports
`@cf/black-forest-labs/flux-2-klein-4b`.

Generate images from an existing complete result without regenerating its story:

```bash
uv run python examples/generate_images.py \
  --source output/camera_roll.yaml --output output/images --seed 42
```

The pipeline makes one LLM call per person, object, and photo to write image
prompts. It first generates photorealistic, waist-up character portraits and
isolated objects on blank white backgrounds. It then uses those canonical assets
to generate the photo scenes. Photos that need more than four references are
completed through additional edits: the current scene occupies the first input
slot, followed by up to three new asset references. Only entities listed as
visible in the photo are supplied. Prompts describe their scene-specific poses,
clothes, placement, and condition, and ask subsequent edits to preserve existing
subjects and composition. The image model may still introduce visual drift.

Outputs live under the selected directory:

- `manifest.yaml`: prompts, source/model settings, entity IDs, deterministic seeds,
  ordered input-slot mappings, image paths, and completion checkpoints.
- `references/people/` and `references/objects/`: 1024×1024 canonical assets and
  white-padded 448×448 `_input.png` copies for Cloudflare's reference input limit.
- `photos/`: final 1024×1024 photos, with intermediate passes in numbered folders.

Generated data and images can be committed to Git to share the full run. Credentials
in `.env` files, virtual environments, and local caches remain ignored; `.env.example`
is safe to share.

Rerun the same command to resume: completed prompts and image requests are reused.
Provider failures stop the run and identify the failing item while preserving
completed work. A completed image that is missing or corrupted raises an error
instead of silently changing an established identity. If source data, models,
prompt instructions, or seed change, choose a fresh output directory. Existing
story and camera-roll YAML files are not edited. The web viewer reads the generated
manifest and images alongside the camera roll.

The stages are also callable independently:

```python
import yaml
from pathlib import Path
from ai_camera_roll import (
    CloudflareImageModel, GenerationResult, OpenCodeGoLLM,
    generate_character_prompt, generate_object_prompt,
    generate_scene_prompts, generate_images,
)

result = GenerationResult.model_validate(
    yaml.safe_load(Path("output/camera_roll.yaml").read_text(encoding="utf-8"))
)
llm = OpenCodeGoLLM(model="deepseek-v4.1-flash", timeout=300, max_tokens=8192)
portrait_prompt = generate_character_prompt(
    result.camera_roll.people[0], story=result.story, llm=llm
)
manifest = generate_images(
    result, llm=llm, image_model=CloudflareImageModel(),
    output_dir="output/images", seed=42,
)
```

To add an image provider, subclass `ImageModel` and implement
`generate(prompt, *, references=(), width=1024, height=1024, seed=None) -> bytes`,
where references are image bytes in the specified slot order. Provider adapters
own any input resizing. The prompt instructions are packaged as
`image_character.md`, `image_object.md`, and `image_scene.md` alongside the text prompts.

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
import yaml
from ai_camera_roll import LifeStory, generate_photos, generate_story

story = generate_story("Maya, a student who loves cycling and cooking.", llm=llm)
Path("story.yaml").write_text(
    yaml.safe_dump(story.model_dump(mode="json"), allow_unicode=True, sort_keys=False, width=100),
    encoding="utf-8",
)

# Edit the saved narrative if desired, then validate and load it again.
story = LifeStory.model_validate(yaml.safe_load(Path("story.yaml").read_text(encoding="utf-8")))
roll = generate_photos(story, llm=llm, photos_per_year=10)
```

Load a complete saved result with `GenerationResult.model_validate(yaml.safe_load(...))`.
The original seed is preserved at `result.story.seed`; generation does not ask
the LLM to reproduce it. The library returns Pydantic objects and leaves file
storage to the caller. The example writes to `output/camera_roll.yaml`, preserving
field order and Unicode text and wrapping long prose for readability. The LLM
still returns JSON internally for schema validation before the result is saved as YAML.

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
text generation makes exactly two requests. Invalid JSON, invalid references,
unexpected counts, or incomplete output raise `LLMResponseError`. Input errors
raise `ValueError` or Pydantic `ValidationError`; HTTP failures propagate HTTPX
exceptions. An `httpx.HTTPStatusError` exposes the service's response through
`error.response`. If extraction fails, you can rerun `generate_photos` on a saved
story without regenerating its narrative.

## Development

```bash
uv run pytest
node --test tests/test_viewer_web.cjs
uv build
```

Tests use a fake LLM and mocked HTTP transport; they do not require credentials or
make paid calls. The example above is the live smoke test. Review its output for
believable everyday moments, object recurrence, visual consistency, and chronology
in addition to the structural checks.

The viewer's JavaScript tests use Node.js's built-in test runner with a simulated
document; they do not open a browser or make network requests.
