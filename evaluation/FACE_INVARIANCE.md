# Face-similarity noise floor

`score_faces.py` reports a raw ArcFace cosine between a canonical portrait and a
detected scene face. The number has no calibrated identity threshold, so a change
between two runs cannot currently be attributed to identity change. This measures
how far that cosine moves when the identity provably stays the same.

Every perturbation below is applied to a canonical portrait. No generated face is
compared with another generated face, no label changes, and no model is trained.
Any movement in the cosine is therefore metric sensitivity, not identity drift.

## What was varied

Two regimes, reported separately because they answer different questions:

- **end to end**: perturb the 1024-px portrait, then re-run SCRFD detection →
  5-point `norm_crop` alignment → ArcFace. Includes detector and alignment.
- **embed only**: perturb the already-aligned 112 × 112 crop, then embed. Isolates
  the embedder.

The grid was written into `face_invariance.py` before the first run and is copied
verbatim into `results_invariance/invariance.json` under `grid`: 73 levels ×
3 identities × 2 regimes = 438 measurements, of which 37 levels per identity are
inside the realistic tier (111 scored cells). All 219 photo-regime cells were
detected at the unchanged threshold of 0.5, so no cell is missing because of a
detection failure.

| Family | Levels |
| --- | --- |
| pose | yaw, pitch, roll at ±10, ±20, ±30, ±40, ±50°, planar perspective warp |
| illumination | γ ∈ {0.5, 0.7, 1.0, 1.4, 2.0}; gain/offset pairs; 3000/4500/9000 K; directional shading ramps |
| resolution | downscale-and-restore to 224/160/112/80/56/40 px; Gaussian σ ∈ {0.5, 1, 2, 3}; JPEG q ∈ {30, 50, 70, 90} |
| occlusion | seeded rectangles over the eye and mouth regions at 5/10/20/30 % of face area; sunglass band; mask band |

Pose deserves a caveat. It is a homography induced by rotating the portrait
plane, i.e. a 2D approximation: no self-occlusion of the far side of the face, no
change in shading, and the detector still sees two eyes and roughly frontal
geometry at 50°. A real 50° head turn would lose more. Pose numbers here are
therefore optimistic, which makes the pose component of the floor a lower bound.

The pose focal length is a single camera property (0.7 × image width, a 71° phone
field). The 112-px crop inherits the focal-to-face-width ratio that face had in
its photo, so both regimes turn the face by the same amount; otherwise the crop
regime would distort twice as hard simply because it is zoomed in.

Occlusion is filled with flat grey (128) for every variant so that the measurement
is about a region becoming uninformative, not about a particular fabric or lens.
Occlusion in the embed-only regime uses the landmark positions `norm_crop` produces
inside the aligned crop.

## Results

Mean cosine drop across the three identities at the worst level inside the
realistic tier, and the same level in the embed-only regime:

| Sub-family | Level | End to end | Worst identity | Embed only |
| --- | --- | ---: | ---: | ---: |
| pose / yaw | +20° | 0.073 | maya 0.091 | 0.081 |
| pose / pitch | +20° | 0.168 | maya 0.236 | 0.093 |
| pose / roll | +10° | 0.024 | maya 0.032 | 0.132 |
| illumination / gamma | γ 1.4 | 0.006 | maya 0.008 | 0.003 |
| illumination / gain-offset | 1.25, +20 | 0.035 | maya 0.056 | 0.031 |
| illumination / temperature | 3000 K | 0.054 | maya 0.072 | 0.043 |
| illumination / shading | 0.5 at 90° | 0.024 | maya 0.030 | 0.091 |
| resolution / downscale | 160 px | 0.153 | jonah 0.250 | 0.000 |
| resolution / blur | σ 1 | 0.018 | maya 0.022 | 0.058 |
| resolution / JPEG | q50 | 0.023 | jonah 0.034 | 0.086 |
| occlusion / eyes | 10 % | 0.236 | priya 0.300 | 0.223 |
| occlusion / mouth | 10 % | 0.219 | maya 0.261 | 0.171 |

Outside the realistic tier the same grid reaches much larger changes: 40-px
downscale 0.805 mean (jonah −0.020, a negative same-identity cosine), 50° pitch
0.463, 30 % eye occlusion 0.426, mask band 0.388, sunglass band 0.194. Those levels
destroy the cues the embedder uses, which is exactly why they are excluded from
the floor below.

Three comparisons are worth stating plainly:

- **Photometric changes are almost free.** Gamma and gain/offset inside the
  realistic tier cost at most 0.056 for any identity. Colour temperature is the
  largest illumination effect at 0.072, and it does not come near the pose or
  occlusion numbers.
- **The aligner absorbs roll.** Rolling the photo by 20° costs 0.026 end to end
  because `norm_crop` re-estimates the pose; embedding an already-rolled crop
  without re-alignment costs 0.329, more than twelve times as much. This is the
  clearest case where the two regimes disagree, and it is a property of the
  pipeline, not of the embedder.
- **Small opaque patches are expensive.** A grey rectangle over both eyes covering
  10 % of the face box costs 0.22 in the embedder alone. On the contact sheets the
  face is still plainly the same person; that is my own reading of the tiles, not
  an observer study, but the gap between it and the metric is the point.

## The floor T

`T` is defined as the largest cosine drop that any single identity showed under a
**realistic** perturbation in the **end-to-end** regime. It is a bound, not an
average: it says no nuisance realisation in the grid moved any portrait further
than this while identity was unchanged.

| Quantity | Value |
| --- | ---: |
| `T` (realistic, end to end, worst cell) | 0.2997 |
| `S_floor` = 1 − `T` (lowest same-identity cosine seen) | 0.7003 |
| Driving cell | priya, 10 % eye occlusion |
| `T` without the occlusion family | 0.2496 |
| `T` at the 90th percentile of the 111 realistic cells | 0.1557 |
| `T` at the median of the same cells | 0.0241 |
| `T_all` over all 216 cells (harsh tier included) | 1.0195 |

"Realistic" was fixed in code before the run: pose within ±20°, γ within 0.7–1.4,
gain within 0.8–1.25, any of the three colour temperatures, shading strength ≤ 0.5,
short side ≥ 160 px, blur ≤ σ 1, JPEG ≥ q50, occlusion ≤ 10 % of face area.
The criterion is that a human observer would still name the same person, because
the level matches an ordinary photographic variation: a casual head turn, ordinary
indoor or outdoor light, a preview-sized rendering, soft focus, social-media
recompression, or a small patch over one facial region.

Two judgement calls are worth exposing rather than burying. First, the eye and
mouth rectangles at 10 % are not obviously milder than the sunglass band, which
is excluded; treating one as realistic and the other as not is a defensible but
arbitrary line, so `T_without_occlusion = 0.2496` is reported alongside. Second,
the floor is built from one nuisance at a time. A generated scene face differs
from a canonical portrait in pose *and* lighting *and* expression *and* age at
once, so the combined effect is larger than any single cell here. `T` is a lower
bound on nuisance, and any verdict of "inside the envelope" is correspondingly
permissive.

Between-identity gap, for scale: the most similar pair of different identities is
maya–priya at cosine 0.2588, so a full identity swap moves the score by 0.7412.
Nuisance therefore consumes up to 0.2997 / 0.7412 = **40 %** of the drop produced
by swapping to the hardest impostor; signal-to-noise is 2.47. The same-identity
envelope [0.7003, 1.0] and the between-identity level (0.060–0.259) do not
overlap, so a single global threshold can separate *these three* portraits under
*one* nuisance each. That is a two-class result on three subjects, not a calibrated
identity threshold, and it does not transfer to scene faces, which sit far lower.

## The 22 saved assignments against T

`results_invariance/assignment_verdicts.csv` holds every saved candidate. A cosine
of 1.0 is the only identity-equal anchor available, so each score is compared with
`S_floor`: below it, the reference and the face are further apart than any
same-identity pair the grid produced.

All 22 scored assignments fall below `S_floor = 0.7003`; the highest is 0.4456.
Two further slots have no detected face and are null, not zero. That result is
expected and should not be read as 22 identity failures: the grid moves one thing
at a time, whereas a re-generated scene face differs in everything at once, so no
single-score reading of "same person" or "different person" is available at this
scale. What the table does establish is that **the absolute level of these scores
is not interpretable**, and that anything resembling an identity-threshold gate
would reject every face in the pilot set.

The rank-1 margin (assigned similarity minus the best other identity for the same
face) is more robust, because the face-side nuisance cancels. Twenty of 22 margins
are positive, but 19 of 22 are smaller than `T`, and 14 are smaller than `T`/2.
The two negative margins are priya on photo 2 pass 2 (−0.061) and priya on photo 4
pass 2 (−0.049): the assigned identity is not even the best-scoring one there.

| Photo (month) | Pass | Person | Similarity | Best other | Margin |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 (Jan) | 1 | maya | 0.4456 | 0.1187 | 0.3269 |
| 1 (Jan) | 1 | jonah | 0.1519 | 0.0271 | 0.1249 |
| 1 (Jan) | 2 | maya | 0.1453 | 0.0486 | 0.0966 |
| 1 (Jan) | 2 | jonah | 0.1216 | −0.0138 | 0.1354 |
| 2 (Feb) | 1 | maya | 0.3081 | 0.0397 | 0.2684 |
| 2 (Feb) | 1 | priya | 0.0957 | 0.0881 | 0.0077 |
| 2 (Feb) | 1 | jonah | 0.1880 | 0.0506 | 0.1374 |
| 2 (Feb) | 2 | maya | 0.0618 | 0.0594 | 0.0024 |
| 2 (Feb) | 2 | priya | −0.0874 | −0.0267 | −0.0608 |
| 2 (Feb) | 2 | jonah | 0.1376 | 0.0293 | 0.1084 |
| 3 (May) | 1 | maya | 0.3098 | 0.1499 | 0.1598 |
| 3 (May) | 1 | jonah | 0.3027 | 0.1045 | 0.1982 |
| 4 (Aug) | 1 | maya | 0.3418 | 0.1403 | 0.2015 |
| 4 (Aug) | 1 | priya | 0.2019 | 0.1336 | 0.0683 |
| 4 (Aug) | 1 | jonah | 0.0072 | −0.0212 | 0.0284 |
| 4 (Aug) | 2 | maya | 0.2231 | 0.0452 | 0.1779 |
| 4 (Aug) | 2 | priya | 0.0008 | 0.0499 | −0.0490 |
| 4 (Aug) | 2 | jonah | 0.0234 | −0.0364 | 0.0598 |
| 5 (Oct) | 1 | maya | 0.1091 | 0.0544 | 0.0547 |
| 5 (Oct) | 1 | priya | not detected | — | — |
| 5 (Oct) | 1 | jonah | 0.1223 | 0.0373 | 0.0851 |
| 5 (Oct) | 2 | maya | 0.0575 | 0.0148 | 0.0427 |
| 5 (Oct) | 2 | priya | not detected | — | — |
| 5 (Oct) | 2 | jonah | 0.0111 | −0.0008 | 0.0119 |

## Iteration 01: did Maya's similarity really fall?

`experiments/iteration_01_portrait_retention_2026-11-04/ANALYSIS.md` states:
"In the historical pilot, Maya's assigned ArcFace similarity fell after the second
pass in all four multi-pass scenes." Checking that claim
(`results_invariance/iteration01_check.csv`):

| Pilot | Month | Pass 1 | Pass 2 | Drop | vs `T` 0.2997 | vs 90th pct 0.1557 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| historical | 1 | 0.4456 | 0.1453 | +0.3003 | exceeds by 0.0006 | exceeds |
| historical | 2 | 0.3081 | 0.0618 | +0.2463 | inside | exceeds |
| historical | 8 | 0.3418 | 0.2231 | +0.1188 | inside | inside |
| historical | 10 | 0.1091 | 0.0575 | +0.0516 | inside | inside |
| fresh control | 1 | 0.3119 | 0.1554 | +0.1565 | inside | exceeds |
| fresh control | 2 | 0.1973 | 0.0910 | +0.1063 | inside | inside |
| fresh control | 8 | 0.1002 | 0.0726 | +0.0276 | inside | inside |
| fresh control | 10 | 0.1277 | 0.1284 | −0.0007 | rose, inside | inside |
| portrait reference | 1 | 0.3119 | 0.2496 | +0.0622 | inside | inside |
| portrait reference | 2 | 0.1973 | 0.1743 | +0.0230 | inside | inside |
| portrait reference | 8 | 0.1002 | 0.1106 | −0.0104 | rose, inside | inside |
| portrait reference | 10 | 0.1277 | 0.2039 | −0.0762 | rose, assignment changed face | — |

Reading this plainly:

- The direction of the claim reproduces in the historical pilot: all four
  similarities are lower on the second pass.
- The **magnitude** is not evidence of identity loss. Three of the four drops
  (0.2463, 0.1188, 0.0516) are inside the floor. January's 0.3003 exceeds `T` by
  0.0006, which is a tie at the precision of the model, not a separation; and it
  is well inside the 2`T` = 0.5994 that a difference between two independent
  renders can reach. No drop exceeds 2`T`.
- The two largest historical drops (0.3003, 0.2463) sit above the 90th percentile
  of the realistic cells, as does the fresh control's January (0.1565 against
  0.1557, which is inside the same percentile edge). They are upper-tail events
  rather than typical nuisance. That is the strongest form of the claim the data
  supports: the score change is unusual for nuisance, not beyond it.
- The claim does not even reproduce in direction in the fresh control, where
  October rose by 0.0007. That is inside the floor by any measure.
- Where the portrait-reference condition improved after October's matcher switched
  Maya from face 0 to face 1, the "drop" is confounded by the assignment, so the
  comparison is not measuring the same face.

The defensible summary is that the second-pass change is real but small relative
to metric noise, and that the iteration-01 evidence does not support reading it as
identity drift. Nothing here contradicts the partial improvement the iteration
reports; it changes what the score drop can be claimed to mean.

## How to rerun

```sh
evaluation/.venv/bin/python evaluation/face_invariance.py
evaluation/.venv/bin/python -m pytest evaluation/test_face_invariance.py -q
```

One CPU run takes about 25 s and writes `results_invariance/`:
`invariance.json`, `curves.csv`, `assignment_verdicts.csv`,
`iteration01_check.csv`, `crops/`, and `contact_sheets/` (about 22 MB of PNG — the
tiles are the exact 112 × 112 arrays the embedder sees, so they can be checked
against the numbers). `--seed` is refused at any value but the frozen one, so a
second run is comparable. The JSON records model and input SHA-256 hashes,
package versions, the frozen grid and its realistic tier, and the baseline check.

Reproducibility: the harness matches `results/scores.json` on unperturbed input.
Reference cosine matrix 1.2e-7, scene cosine matrices 2.5e-8, and the embed-only
baseline against the end-to-end baseline 1.3e-8 — float32 storage in the frozen
file, not a pipeline difference.

## Limits

- Cosine similarity is not a probability, a percentage, or a calibrated identity
  decision. No calibrated identity threshold exists for these weights.
- This measures metric sensitivity, not human-perceived identity. The realism
  judgement behind `T` is reasoning about ordinary photography, not a human study;
  no observer rated the perturbed images.
- Three identities, all synthetic portraits from one generator, all on plain white
  studio backgrounds. Nothing here establishes behaviour on real photographs.
- One nuisance at a time, and only the families listed above. Expression change,
  age, true 3D pose, partial occlusion by hair or hands, and camera-to-subject
  distance are not modelled. `T` is a lower bound, so a drop classified as inside
  the envelope may still be a real change that this grid could not reproduce.
- Pose is a planar warp and the detector re-aligns every variant, so pose costs
  are optimistic and no profile view was tested.
- The floor is a per-score bound. A difference between two runs needs two
  realisations, so 2`T` is the matching bound for the iteration-01 comparison and
  is reported next to `T`.

## Attribution

Detection is SCRFD-10GF and recognition is ArcFace ResNet50 trained on
WebFace600K, both from the InsightFace `buffalo_l` package; weights are
non-commercial research only and stay outside the repository. Perturbations use
OpenCV. Codex assisted with the harness, tests and this writeup; the team should
understand and be able to defend the realism tier, which is the only judgement in
the analysis.
