# Evaluation inputs viewer

A single static page that shows *what each similarity score compared*: the
reference image, the crop taken from the generated photo, and the detector's
view of the scene. Rows are grouped into two tables — face consistency and
object appearance — and the score is the rightmost column.

Nothing here re-runs a model. Every image and number is read from the saved
record in `experiments/iteration_01_portrait_retention_2026-11-04/`.

## Build and open

```sh
python3 evaluation_site/build.py
open evaluation_site/dist/index.html
```

The build uses only the Python standard library: no dependencies, no model
weights, no API keys, and no server. `dist/` is generated and ignored by Git.

## What the page shows

**Face consistency** — one row per requested character per generation pass:

1. SCRFD detects faces in the saved photo.
2. Landmarks align each face to a 112×112 crop.
3. ArcFace embeds the crop and the canonical portrait.
4. Cosine similarity between the two embeddings is the score.

The pairing is a maximum-total one-to-one assignment across the characters
requested for that photo, so a weak match is still assigned. A row named
"Unassigned face" is a detection left over after pairing.

**Object appearance** — one row per requested object per photo:

1. Grounding DINO receives a generic noun query (`mug.`, `bicycle.`) and finds
   candidate boxes; the query never contains the colour or description.
2. Overlapping boxes are removed with NMS, and the **highest-confidence** box
   is kept — not the most similar one.
3. The crop is fitted into 224×224 with neutral padding.
4. DINOv2 embeds it; cosine similarity against the canonical reference crop is
   the score.

Clicking any thumbnail opens a modal with both images large, the detector
details (confidence, box, query, requested state), and the scene photo with the
detector's own annotations. Arrow keys step through rows.

## Reading the numbers

Similarity is raw cosine similarity from −1 to 1. It is **not** a probability or
a percentage, the scale is not calibrated, and no threshold is recorded. A
missing detection is shown as *not detected*, which is not a score of zero.
Object crops keep their background, so resemblance is not proof that the right
object is in the right place.

## Files

| Path | Role |
| --- | --- |
| `build.py` | Reads saved results, copies referenced crops, writes `dist/data.js` |
| `index.html` | Page structure |
| `app.js` | Table, condition switch and modal; no framework |
| `style.css` | Styling |
| `dist/` | Generated output; open `dist/index.html` |

## Inputs

| Source | Used for |
| --- | --- |
| `results/raw/faces_{control,portrait_reference,historical}.json` | Face rows, matrices and detector settings |
| `results/raw/objects_final.json` | Object rows, candidates and settings |
| `results/comparison_data.json` | Scene names |
| `records/object_targets.json` | Detector queries and requested object states |
| `records/asset_manifest.json` | Resolves crops stored once for byte-identical files |
| `evidence/face_*`, `evidence/objects` | Aligned crops, candidate crops, detection overlays |

`data.js` records a SHA-256 of every JSON input and the build timestamp, and the
page shows them in the footer.

## Scope

The page covers **final saved photos** for three conditions: the fresh control,
the portrait-reference change, and the historical pilot. First-pass object
scores, the face cosine matrix, and human ratings are not shown. This viewer
adds no new measurement and no new claim; `evaluation/README.md`,
`evaluation/OBJECT_SCORING.md` and the experiment's `ANALYSIS.md` remain the
authorities on method and limitations.
