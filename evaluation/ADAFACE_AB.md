# AdaFace vs ArcFace: a one-component swap

## What was swapped and what was frozen

Only the face embedder changed. One image is detected once by SCRFD-10GF, each
detected face is aligned once onto a 112×112 crop, and **both** embedders read
that identical crop. There is a single alignment path (`align_face` in
`recognizers.py`) and a single detection result, so no score difference can come
from detection, crop, roll angle, face ordering or reference choice.

| Held constant | Value |
| --- | --- |
| Detector | SCRFD-10GF, `det_10g.onnx`, det_size 640×640, threshold 0.5 |
| Face ordering | ascending `bbox[0]` |
| Alignment | `insightface.utils.face_align.norm_crop`, 5 landmarks, 112×112 |
| Reference portraits | the three canonical `output/images/references/people/*.png` |
| Pairing rule | `scipy.optimize.linear_sum_assignment(-cosine)` |
| Similarity | cosine on L2-normalised 512-d embeddings |
| Provider | `CPUExecutionProvider`, no API calls, 0 generation calls |

| Swapped | ArcFace arm | AdaFace arm |
| --- | --- | --- |
| Weights | `w600k_r50.onnx`, R50 @ WebFace600K | `adaface_ir_101.onnx`, IR-101 @ WebFace12M |
| SHA-256 | `4c06341c…619e43` | `f2eb07d0…7e5a43e` |
| Upstream | `buffalo_l.zip` (deepinsight/insightface releases) | `yakhyo/uniface-weights` (exported official AdaFace weights) |
| Raw output | 512-d, not normalised (norm ≈ 16.1) | 512-d, already unit length (norm ≈ 1.000) |
| Preprocessing | RGB, `(x − 127.5) / 127.5`, NCHW | RGB, `(x − 127.5) / 127.5`, NCHW (validated, see below) |

Both weight files are checksum-verified before inference; nothing is downloaded.
AdaFace provenance and the research-use-only restriction are recorded in
`evaluation/adaface-provenance.json`.

Run: `evaluation/.venv/bin/python evaluation/score_faces_ab.py` (~6 s CPU, writes
`evaluation/results_adaface/ab_scores.json`).

## The ArcFace arm must reproduce the frozen baseline

`score_faces_ab.py` re-scores the ArcFace arm through its own code path and
compares it against `evaluation/results/scores.json` on every run. On matched
inputs the maximum absolute cosine deviation is **6.7e-16** and every detection
box and confidence is identical. If that check fails the script raises instead of
writing a report, so the numbers below are not a second, drifted pipeline.

## Validating AdaFace preprocessing instead of assuming it

The exported graph has no input `Sub`/`Mul` node (its first node is a `Conv`) and
it ends in `ReduceL2 → Div`, so all input normalisation happens outside the model
and the output is unit length by construction. The preprocessing therefore had to
be shown, not assumed. Five candidates were run on the same crops
(`--sweep-preprocessing`, results in `ab_scores.json → preprocessing_sweep`):

| Candidate | Same index across passes | Different index across passes | Reference inter-identity | Verdict |
| --- | ---: | ---: | ---: | --- |
| RGB, `(x−127.5)/127.5` | 0.385 | 0.019 | 0.248 / 0.212 / 0.069 | **selected** |
| BGR, `(x−127.5)/127.5` | 0.375 | 0.007 | 0.301 / 0.156 / 0.110 | plausible, not the training convention |
| RGB, x/255 | 0.394 | 0.027 | 0.239 / 0.206 / 0.063 | plausible |
| RGB, ImageNet mean/std | 0.497 | 0.056 | 0.169 / 0.148 / 0.084 | plausible |
| RGB, raw 0–255 | **0.975** | **0.836** | 0.928 / 0.891 / 0.888 | **degenerate — every cosine ≈ 0.9** |

"Same index across passes" compares the face at one position in pass 1 with the
face at the same position in pass 2 of the same photo (n=10); "different index"
compares the other pairs (n=16). A candidate index is a position, not an identity
label, so this is a separation check, not ground truth.

What the evidence supports:

- **Not degenerate.** The scaled candidates spread cosines from −0.13 to +0.38
  (sd ≈ 0.10) and separate same-position from different-position pairs by roughly
  20×. The unscaled 0–255 input collapses everything to ≈ 0.9 and would make every
  character look like every other character; that failure mode is caught.
- **Not a channel or scale accident.** Reference inter-identity cosines of
  0.248 / 0.212 / 0.069 land within ~0.01 of ArcFace's 0.259 / 0.208 / 0.060, which
  is what a working face embedder should produce on three distinct people.

What it does not support, stated plainly: the model is **not** very sensitive to
normalisation. BGR, x/255 and even ImageNet mean/std all produce usable
embeddings, and ImageNet normalisation actually scores *higher* on the
same-position proxy (0.497 vs 0.385). We did **not** select on that ranking,
because choosing a normalisation by how it scores on these 22 pairs would be
tuning on the evaluation set. The selected candidate is the convention the AdaFace
weights were trained with. The honest read is that this validation detects gross
errors and confirms plausible behaviour; it cannot distinguish fine preprocessing
differences on a pilot this small.

## Results

### Reference inter-identity cosine

| Pair | ArcFace | AdaFace |
| --- | ---: | ---: |
| maya–priya | 0.259 | 0.248 |
| maya–jonah | 0.208 | 0.212 |
| priya–jonah | 0.060 | 0.069 |

The three canonical portraits are about as separable under both embedders.

### Assigned similarities (all scored appearances)

`--` means no face was available for that character under that arm. Sim = assigned
cosine; margin = rank-1 margin defined below.

| Photo (month) | Pass | Person | ArcFace face | ArcFace sim | ArcFace margin | AdaFace face | AdaFace sim | AdaFace margin |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (Jan) | 1 | maya | 0 | 0.446 | 0.327 | 0 | 0.383 | 0.318 |
| 1 (Jan) | 1 | jonah | 1 | 0.152 | 0.125 | 1 | 0.090 | 0.036 |
| 1 (Jan) | 2 | maya | 0 | 0.145 | 0.097 | 0 | 0.098 | 0.115 |
| 1 (Jan) | 2 | jonah | 1 | 0.122 | 0.135 | 1 | 0.111 | 0.086 |
| 2 (Feb) | 1 | maya | 0 | 0.308 | 0.268 | 0 | 0.230 | 0.216 |
| 2 (Feb) | 1 | priya | 1 | 0.096 | 0.008 | 1 | 0.022 | −0.048 |
| 2 (Feb) | 1 | jonah | 2 | 0.188 | 0.137 | 2 | 0.162 | 0.116 |
| 2 (Feb) | 2 | maya | 0 | 0.062 | 0.002 | 0 | 0.021 | 0.037 |
| 2 (Feb) | 2 | priya | 1 | −0.087 | −0.061 | 1 | −0.069 | −0.039 |
| 2 (Feb) | 2 | jonah | 2 | 0.138 | 0.108 | 2 | 0.088 | 0.057 |
| 3 (May) | 1 | maya | 0 | 0.310 | 0.160 | 0 | 0.330 | 0.240 |
| 3 (May) | 1 | jonah | 1 | 0.303 | 0.198 | 1 | 0.264 | 0.088 |
| 4 (Aug) | 1 | maya | 0 | 0.342 | 0.201 | 0 | 0.305 | 0.203 |
| 4 (Aug) | 1 | priya | 1 | 0.202 | 0.068 | 1 | 0.163 | 0.027 |
| 4 (Aug) | 1 | jonah | 2 | 0.007 | 0.028 | 2 | 0.052 | 0.016 |
| 4 (Aug) | 2 | maya | 0 | 0.223 | 0.178 | 0 | 0.134 | 0.169 |
| 4 (Aug) | 2 | priya | 1 | 0.001 | −0.049 | 1 | 0.031 | 0.002 |
| 4 (Aug) | 2 | jonah | 2 | 0.023 | 0.060 | 2 | −0.047 | −0.005 |
| 5 (Oct) | 1 | maya | 0 | 0.109 | 0.055 | 0 | 0.073 | 0.017 |
| 5 (Oct) | 1 | jonah | 1 | 0.122 | 0.085 | 1 | 0.108 | 0.082 |
| 5 (Oct) | 2 | maya | 0 | 0.057 | 0.043 | `–` | `–` | `–` |
| 5 (Oct) | 2 | jonah | 1 | 0.011 | 0.012 | 0 | −0.041 | 0.010 |
| 5 (Oct) | 2 | priya | `–` | `–` | `–` | 1 | 0.018 | 0.017 |

- ArcFace: 22 scored assignments, range −0.087 … 0.446.
- AdaFace: 22 scored assignments, range −0.069 … 0.383.
- Δ (AdaFace − ArcFace) on the 21 appearances scored by both: mean **−0.034**,
  median −0.039, range −0.089 … +0.045.

AdaFace is systematically *lower* on this pilot, by about 0.03 for the same pairs.
Absolute cosines are therefore not interchangeable between embedders; only
within-arm comparisons or rankings are meaningful here.

### Rank-1 margin

Margin = assigned similarity minus the strongest similarity any **other**
reference portrait has to that same face. Negative means another character's
portrait matches the assigned face better; the pairing still assigns it, because a
pairing is not an identity verdict. On the frozen baseline this definition
reproduces the two documented cases (priya −0.061 in February pass 2, −0.049 in
August pass 2) and the count of 10 assignments below the 0.07 inspection band.

| | ArcFace | AdaFace |
| --- | ---: | ---: |
| Assignments with margin < 0.07 | 10 / 22 | 12 / 22 |
| Negative margins | 2 | 3 |
| Negative margin cases | priya Feb p2 (−0.061), priya Aug p2 (−0.049) | priya Feb p1 (−0.048), priya Feb p2 (−0.039), jonah Aug p2 (−0.005) |

AdaFace does not resolve the negative-margin problem; it reproduces it for priya
in February and adds one more (a near-zero negative for jonah in August pass 2).
Both arms say the same thing about these faces: the assigned character is no
better matched than a competing character's portrait.

### Did the Hungarian pairing change?

Yes, in one photo. **3 of 22 rows** differ, all in October (photo 5) pass 2, and
they are one reshuffle of a 3-people / 2-faces case:

| Row | ArcFace | AdaFace |
| --- | --- | --- |
| maya | face 0 (0.057) | unassigned |
| priya | unassigned | face 1 (0.018) |
| jonah | face 1 (0.011) | face 0 (−0.041) |

Every other photo keeps exactly the same character→face mapping under both
embedders, including the contested February passes. Note that the October case was
already flagged in `experiments/iteration_01_*/ANALYSIS.md` as a documented
matcher failure, and every similarity in it is weak (|cos| < 0.06) under both
arms: the reshuffle is a tie between bad options, not a correction.

### Agreement between the two embedders

| Quantity | n | Spearman ρ |
| --- | ---: | ---: |
| All reference×face cosine cells | 66 | 0.821 |
| Assigned similarities (both arms scored) | 21 | 0.939 |

The two embedders rank these faces similarly but not identically, and they do not
agree on absolute magnitudes.

### The iteration_01 claim: does Maya's similarity fall from pass 1 to pass 2?

The claim in `experiments/iteration_01_*/ANALYSIS.md` is that Maya's assigned
ArcFace similarity fell after the second pass in every multi-pass scene. May has
only one saved pass in the frozen pilot, so four photos are testable:

| Photo (month) | ArcFace p1 → p2 | Fell? | AdaFace p1 → p2 | Fell? |
| --- | --- | --- | --- | --- |
| 1 (Jan) | 0.446 → 0.145 | yes | 0.383 → 0.098 | yes |
| 2 (Feb) | 0.308 → 0.062 | yes | 0.230 → 0.021 | yes |
| 4 (Aug) | 0.342 → 0.223 | yes | 0.305 → 0.134 | yes |
| 5 (Oct) | 0.109 → 0.057 | yes | 0.073 → unassigned | no value |

Under both embedders Maya's assigned similarity falls in every two-pass photo the
claim covers, and the size of the fall is nearly identical: over January,
February and August the mean drops 0.365 → 0.143 under ArcFace (−0.222) and
0.306 → 0.084 under AdaFace (−0.222). **The conclusion does not flip.** October is
the only qualitative difference: AdaFace does not produce a lower number for
Maya, it stops pairing her with a face at all — consistent with the weak
covering-set interpretation above, not with a competing claim.

The claim survives the swap, but it is a statement about *assigned* similarity:
the matcher picks the pairing, and supporting-character directions are not stable
across embedders (Jonah rises 0.090 → 0.111 in January under AdaFace while falling
0.152 → 0.122 under ArcFace, and falls 0.052 → −0.047 in August under AdaFace
while rising under ArcFace). Identity drift of one character is the only pattern
both embedders agree on.

### Detection control

Detection is shared by construction: one SCRFD pass per image produces 22 faces
for 24 expected appearances under both arms, and no bbox or detection confidence
can differ. The count of unassigned appearances is also equal (2 per arm), but
they are not the same two: October pass 2 leaves priya unassigned under ArcFace
and maya under AdaFace. That is a pairing difference, not a detection difference,
and it is the mechanism behind the 3 changed rows above.

## Limits

- **Cosine is not a probability.** There is no calibrated identity threshold in
  either arm. A similarity of 0.30 does not mean "30% likely the same person", and
  no detected face does not establish absence.
- **22 scored assignments, 3 identities, 5 photos.** This is a pilot. One changed
  photo and a ρ of 0.94 are interesting, not decisive; revisit before generalising
  to the wider camera roll.
- **Absolute scores are not comparable across embedders.** AdaFace sits ~0.03
  lower on the same pairs. Compare directions and rankings within an arm.
- **The assignment step is part of the measurement.** A different embedder can
  change which face a character is paired with, so the two arms are not scoring
  identical pairs in October. Those rows are reported rather than repaired.
- **Preprocessing validation is coarse.** It excludes gross normalisation errors
  and confirms plausible separation; it cannot rank fine preprocessing variants on
  a pilot this size, and three of the tested variants produced usable embeddings.
- **Weights are research-only.** AdaFace pretrained weights and the buffalo_l
  weights are non-commercial research artefacts; see
  `evaluation/adaface-provenance.json` and `evaluation/model-provenance.json`.
  These scores are not an identity verification system.

## What the swap says

Swapping the embedder changed nothing about the qualitative conclusions of the
frozen pilot: the same faces are paired with the same characters except in the one
photo already known to be broken, Maya's pass-2 similarity still falls everywhere
it can be measured, and the negative-margin cases persist. A controlled swap that
changes little is a useful result — it means the earlier conclusions were not an
artefact of a particular embedder's quirks, and the remaining weakness (weak
absolute similarity, ties between characters, a pairing failure in October) is a
property of the generated images and the scoring design rather than of ArcFace.

## Reproducing

```
evaluation/.venv/bin/python evaluation/score_faces_ab.py                   # main run
evaluation/.venv/bin/python evaluation/score_faces_ab.py --sweep-preprocessing
evaluation/.venv/bin/python -m pytest evaluation/test_recognizers.py -q     # logic checks
```

Outputs land in `evaluation/results_adaface/`: `ab_scores.json` (all cosines,
per-face reference similarities, margins, detection record, provenance) and
`crops/` (the shared aligned crops both arms consumed).
