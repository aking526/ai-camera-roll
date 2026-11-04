# A life in frames — local presentation website

This website wraps the existing generator with a story-first camera roll,
inspectable generation records, saved face comparisons, and a human evaluation
workflow. Maya's included example has five final photos, eight references, and
nine scene passes. These are the repository's existing synthetic images.

## Run

Requires Python 3.12+ and uv. From the repository root:

```bash
uv sync
uv run python demo_site/server.py
```

Open <http://127.0.0.1:5812/>. Keep the server running during the demonstration;
Ctrl+C stops it. On macOS, `Open_Demo.command` also starts the server after
`uv sync` has created the repository's `.venv`.

Viewing Maya, inspecting saved scores, and rating photos need no credentials or
face-model downloads. The server binds only to this computer's loopback address;
this is a local development application, not a public hosting setup.

## Demonstration route

1. **Photos:** show Maya's story and camera roll.
2. **Pipeline → Generation passes:** choose January and compare both passes.
   The first uses portraits, a cap, and a sketchbook. The second uses the previous
   scene plus the mug reference; the portraits are not supplied directly again.
3. **Scores & results:** compare the reference, first-pass, and final-pass face
   crops. January's assigned Maya similarity changes from 0.4456 to 0.1453.
   Expand the method explanation, then inspect the five-photo table.
4. **Rate photos:** set up the local Maya round, then have three actual people
   review it independently. [The rating guide](RATING_GUIDE.txt) explains each
   judgment, draft recovery, submission, exports, and the iteration plan.

The committed bundle contains saved model results and explicitly labeled
assistant observations. It contains no reviewer submissions. A fresh checkout
must create its own rating round; actual local reviews are never prefilled.

## Optional generation

The **New story** flow calls the existing Python pipeline. Story creation and
photograph creation are separate explicit actions. Review the planned reference
and scene-pass counts before generating images: one final photograph can require
several image requests, and prompt compilation also makes text requests.

To configure providers locally:

```bash
mkdir -p demo_site/private
cp demo_site/credentials.env.example demo_site/private/credentials.env
chmod 600 demo_site/private/credentials.env
```

Fill the local file with your own provider settings. It is ignored by Git and is
never served to the browser. Environment variables also work when running the
Python command directly; the macOS launcher uses the private file instead.
The checked-in example contains no keys. Valid credentials and provider access
are required; successful live generation through this website has not yet been
verified. The complete orchestration has been tested with fake providers.

New runs preserve checkpoints, prompts, input slots, request attempts, and output
artifacts under `demo_site/private/runs/`. A stop request takes effect after the
current request finishes. Resuming is explicit; there are no automatic retries.
Failed image attempts count toward the run's image-request limit. Only one run
executes at a time. The local ceiling is 40 image attempts, not a currency budget.

## Evaluation boundaries

- SCRFD detects faces; ArcFace compares aligned face embeddings with canonical
  portraits using cosine similarity. Assignment is one-to-one and can include
  weak matches. An assignment does not confirm identity.
- Scores are not percentages or probabilities. No calibrated identity cutoff
  exists. Pose, visibility, resolution, and assignment can affect the result.
- Four of four comparable multi-pass Maya photos have a lower final assigned
  protagonist similarity. This motivates investigation; it does not establish
  that editing caused identity loss.
- Twelve final-scene face detections against thirteen requested appearances are
  counts, not an identity accuracy measure. May has one pass and no change score.
- Human ratings cover protagonist identity, visible people, required objects,
  setting, and action. They remain independent of the displayed model scores.
- Automatic face scoring for newly generated albums, CLIP scene scoring, and a
  score-triggered corrective retry are not connected. Existing generation passes
  accommodate reference limits; they are not corrective retries.
- Building this workflow starts measurement. An improvement iteration still
  needs an evidence-driven change and comparable reevaluation.

## Checks and rebuilding the saved bundle

From the repository root:

```bash
uv run pytest tests demo_site
node --check demo_site/dist/app.js
node --check demo_site/dist/ratings.js
uv run python demo_site/build_demo_data.py
```

Tests use fake providers and temporary records, without credentials or generation
requests. The builder reads `output/` and the committed `evaluation/` results,
checks source hashes, and copies their images into `dist/assets/`. It does not
run models. To rerun face inference, see [evaluation setup](../evaluation/README.md).

Only `dist/` is served as static content. Credentials, new runs, rating records,
backups, and verification outputs are excluded from Git. Preserve `private/`
when updating code if you want to keep your local work. Rating exports include
reviewer names and observations; review them before sharing.

## Attribution

The original `aking526/ai-camera-roll` project supplies the generator, Maya's
story, reference assets, saved photographs, and generation manifests. Codex
assisted with this website, integration, offline evaluation, tests, documentation,
and draft visual annotations. Those annotations are not independent human votes.
The team must review and understand this AI-assisted code and report its use.

Story-first visual inspiration: <https://1in8b.life/en/>. Face tooling uses
InsightFace/SCRFD and ArcFace; library code is MIT, while the supplied pretrained
weights are restricted to non-commercial research. Model provenance is included
with the evaluation; weights are not redistributed here.
