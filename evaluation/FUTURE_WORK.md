# Future work: measuring face and object similarity

Backlog for the two visual-consistency metrics. Nothing here is implemented unless marked
**done**. This file exists so the reasoning behind each option survives longer than a chat
thread; it is a menu with recommendations, not a plan of record.

Companion documents:

- [`./README.md`](README.md) / [`OBJECT_SCORING.md`](OBJECT_SCORING.md) — what the metrics do today.
- [`FACE_INVARIANCE.md`](FACE_INVARIANCE.md) — measured nuisance envelope of the ArcFace cosine.
- [`ADAFACE_AB.md`](ADAFACE_AB.md) — the controlled embedder swap.
- [`face_metric_results.ipynb`](face_metric_results.ipynb) — renders both result sets as tables and plots.

## What exists today

| | Face | Object |
| --- | --- | --- |
| Detector | SCRFD-10GF | Grounding DINO tiny (fixed noun query) |
| Normalisation | 5-point similarity transform → 112² | letterbox → 224² |
| Embedding | ArcFace R50 @ WebFace600K | DINOv2-small **CLS** |
| Distance | cosine + Hungarian one-to-one pairing | cosine, single best box |

## 0. What the measurements already rule out

Worth recording so nobody re-litigates these:

1. **Illumination normalisation is not the bottleneck.** Gamma, shading, gain and a 3000 K
   colour cast each cost ≤ 0.072 cosine inside the realistic tier. Relighting work,
   Retinex, intrinsic decomposition for *lighting* — all low value here.
2. **A better embedder does not fix it.** AdaFace (quality-adaptive margin, built for
   low-quality inputs) reproduces the same conclusions, with slightly *worse* rank-1
   margins. ρ = 0.94 against ArcFace on assigned pairs.
3. **The bottleneck is measurement design.** Nuisance consumes ≈ 40 % of an identity swap
   (`T` ≈ 0.30 against a 0.74 signal). No embedder swap closes that gap; the design has to.

## A. Fix the measurement before changing the model

### A1. Matched generated-vs-generated baseline

**This is the top priority** — it repairs a real defect. `assignment_verdicts.csv` labels all
22 scene faces `beyond_envelope` using `1 − cosine`, but `T` was measured on *reference vs a
perturbed copy of the same reference*, whose baseline is 1.0. A generated scene face is a
different rendering and was never at 1.0, so the absolute gate is guaranteed to fire and
carries no information. `T` is valid only as a **delta budget between conditions**.

- *Do:* compute cosine between the aligned 112² crops of the **same character in two passes**
  (same identity, different generation) and between **different characters in the same pass**
  (impostor). The crops already exist in `results_adaface/crops/`.
- *Gives:* a properly matched floor and a real separation statistic (AUC / CMC over
  generated pairs) instead of an absolute cosine.
- *Effort:* **S**. *First step:* add a `--pairwise-generated` mode that reads saved crops.
- *Risk:* if same-identity generated pairs score below impostor pairs, the honest finding is
  that the generator does not preserve identity — do not tune the metric to hide it.

### A2. Joint nuisances

`T` perturbs one family at a time, which makes it a **lower bound** and makes every
"inside envelope" verdict permissive. Real scenes combine pose + occlusion + resolution.

- *Do:* a factorial sweep over a small subset (e.g. yaw × downscale × eye occlusion, three
  levels each). Report the worst cell and an interaction term.
- *Effort:* **M** (grows as *kⁿ*; keep the grid small). *First step:* re-use the existing
  perturbation functions composed in sequence.

### A3. Quality gating

Stop averaging unwinnable comparisons into a mean.

- *Options:* **MagFace** embedding norm (free — the norm is a learned quality proxy and the
  model already emits a vector), or **CR-FIQA** (CVPR 2023, predicts sample relative
  classifiability).
- *Do:* report a per-appearance quality score next to every similarity, and every mean with
  its detection/quality coverage. Down-weight or exclude low-quality comparisons explicitly
  rather than silently.
- *Effort:* **S** for MagFace-norm; **M** for CR-FIQA. *Payoff:* medium-high.

### A4. Report margin and rank, not just raw cosine

Cheapest real improvement: the number that matters is *how far the assigned identity is ahead
of the alternative*, not the absolute level.

- *Do:* add `rank1_margin` and `rank` to every assignment row. Already computed ad hoc here:
  10 of 22 baseline assignments have a margin below 0.07 and two are negative.
- *Effort:* **S** (a few lines in `score_faces.py`). *Payoff:* high, zero new dependencies.
- *Note:* this changes the schema of `results/`, which is frozen for `iteration_01`. Emit a new
  file rather than rewriting the frozen one.

### A5. Calibrate cosine into a probability

The README correctly says the score is not a probability and no cutoff is calibrated. That can
be fixed rather than disclaimed, using data already being collected.

- *Do:* fit same/different score distributions on the human-rating set in the demo site, or on
  the generated-pair labels from A1; convert cosine to a likelihood ratio (z-norm/t-norm style
  cohort normalisation, or a simple 1-D logistic fit); report **AUC and CMC** rather than a mean.
- *Effort:* **M**. *Payoff:* high — retires the biggest caveat in the writeup.
- *Risk:* with ~22 appearances, a fitted calibration will be badly underpowered. Report
  bootstrap CIs, and say so.

## B. Faces

### B1. AdaFace swap — **done**

Controlled single-variable A/B. Verdict: **lateral, not a fix.** Lower mean assigned similarity
(−0.034), worse margins (12/22 below 0.07 vs 10/22), identical conclusions, three changed
assignments all in October pass 2. Kept as a second opinion because agreement between two
independent embedders (ρ = 0.94 on Maya) is evidence the *Maya* signal is real, while their
disagreement on supporting characters is evidence it is not. See [`ADAFACE_AB.md`](ADAFACE_AB.md).

### B2. Multi-reference gallery

Currently one canonical portrait per character, so all reference-side variance lands in every
comparison.

- *Do:* 2–3 canonicals per character (different pose/lighting), average their embeddings into
  one template. Standard practice in face recognition (template adaptation).
- *Effort:* **S** (code) + **S** (generate 1–2 more portraits per character).
- *Payoff:* reduces a real, currently unmodelled error term. Do this before any more model work.

### B3. Test-time augmentation

- *Do:* embed the crop and its horizontal flip, average, renormalise. Also try ±5 % scale.
- *Effort:* **S**. *Payoff:* small but essentially free variance reduction; note in the
  writeup that the invariance sweep shows roll is already absorbed by the aligner.

### B4. Pose-aware alignment

`norm_crop` uses a 5-point **similarity** transform: no perspective, and under yaw the
landmarks themselves are biased. The sweep shows pitch is ~2.5× costlier than yaw at equal
angle, which is consistent with the aligner fighting pose rather than normalising it.

- *Do:* compare 5-point `norm_crop` against the 106-point (`2d106det.onnx`) and 3D
  (`1k3d68.onnx`) landmarks that ship in the same buffalo_l archive, and against frontalisation.
- *Effort:* **M**. *Payoff:* medium — potentially the largest face-side gain after B2.

### B5. Disentangled decomposition

The highest-ceiling idea, and the only one that turns "is it invariant?" into a measurement
rather than an assumption.

- *Do:* fit a 3D face model (DECA / EMOCA) to reference and scene face. Report **three**
  numbers per appearance instead of one: identity (shape code distance), geometry (pose and
  expression coefficients), appearance (albedo/shading). Then a score change can be
  *attributed*: "identity fell 0.30, but yaw moved 25°, which is inside the measured envelope."
- *Effort:* **L**. *Payoff:* very high for the final writeup — it makes a causal claim possible
  that a single cosine never can.
- *Risk:* a second model brings its own failure modes and its own provenance burden.

### B6. Larger gallery models

`buffalo_l` is the small bundle. `antelopev2` (ResNet100, Glint360K) and CVLFace/WebFace12M
checkpoints are drop-in-ish. Expected: a small absolute lift, no change in conclusion — treat as
low priority given B1's result.

## C. Objects

The object metric has the *opposite* problem to faces: DINOv2 CLS is a semantic summary trained
with self-supervised invariance objectives, so it is **too** invariant — it deliberately discards
instance-level detail. That is exactly the documented failure where an identical mug moved from a
radiator to a hand still scores well.

### C1. Patch tokens instead of CLS

Highest-value object change.

- *Do:* pool DINOv2 **patch tokens** (mean, GeM, or attention-weighted) instead of the CLS
  token; optionally concatenate CLS. DINOv2's patch features are the instance-discriminative
  ones and its own paper demonstrates dense matching from them.
- *Effort:* **S**. *Payoff:* high. *Caveat:* patch pooling is more sensitive to background —
  which motivates C2.

### C2. Foreground masking

- *Do:* segment the object with SAM/SAM 2, then pool only foreground patch tokens. `OBJECT_SCORING.md`
  already cites CUTE for exactly this motivation but does not implement it.
- *Effort:* **M** (+ a new model dependency).
- *Payoff:* high; directly attacks "the cropped region can include background" in the limitations.

### C3. Visual-prompt detection

Detection is currently a fixed generic noun (e.g. `mug.`) with highest-confidence selection — a
brittle step and a whole error class.

- *Do:* use **T-Rex2** (ECCV 2024), which accepts a *visual* prompt: hand it the canonical object
  crop and let it find *that* object rather than any mug. Report detector agreement with the
  current text-prompt arm.
- *Effort:* **M**. *Payoff:* high — removes a specifiable failure mode rather than averaging it.

### C4. Local-feature matching with geometric verification

- *Do:* for rigid objects, match local features (LightGlue/LoFTR) between reference and crop and
  require a geometric consensus. This is the classical, strong answer to *instance* verification:
  near-invariant to pose and scale.
- *Effort:* **M–L**. *Payoff:* high precision, but it fails on deformable/textureless objects and
  is sensitive to lighting — so it belongs as a second axis, not a replacement.

### C5. Whitening

- *Do:* PCA-whiten embeddings fitted on a background corpus before cosine.
- *Effort:* **S**. *Payoff:* small-but-free retrieval boost; well established in image retrieval.

### C6. Attribute-level checks (orthogonal axis)

Similarity cannot verify *count, placement, ownership, or fine attributes* — a mug in the wrong
hand scores identically.

- *Do:* an explicit attribute checklist per object (colour, material, count, held/worn,
  relative position), scored separately from similarity. CLIP zero-shot per attribute, or plain
  human annotation on the frozen set.
- *Effort:* **M**. *Payoff:* high, but it measures a different thing — label it clearly as
  attribute accuracy, **not** as similarity, and keep it out of the similarity column.

## D. Evaluation design

### D1. DreamSim

Human-aligned *mid-level* perceptual similarity (layout, pose, semantics) from synthetic
triplet data, NeurIPS 2023. Arguably closer to "does this look like the same photo of Maya"
than CLIP or DINOv2. *Effort:* **S** (pretrained, pip-installable-ish). Use as a **third
opinion**, not a replacement: it is not identity-specific.

### D2. Whole-image envelope

DINO-I and CLIP-I are the de-facto subject-fidelity metrics in the personalisation literature
(DreamBooth). Computing them on the *whole generated image* alongside the crop-level object
score gives a sanity envelope that catches gross crop or detector failures. *Effort:* **S**.

### D3. Human study and a power calculation

n = 22 appearances cannot support fine distinctions. Before claiming any improvement, state the
effect size the design can detect. The demo site's rating workflow already collects independent
ratings — that is the asset to spend. *Effort:* **M**.

Note DreamBench++'s finding that DINO- and CLIP-style scores often diverge from human judgement:
human validation is not optional garnish here, it is the calibration set for A5.

### D4. VLM judging

Tempting for identity and attribute questions, and the project has deliberately avoided it.
If it is ever added, keep it in a **separately labelled** column that never mixes into the
similarity metric, and record the model and prompt — otherwise the metric stops being
reproducible and starts being an opinion.

## E. Threats to validity to state plainly

- Cosine is not a probability; no calibrated identity threshold exists.
- The invariance sweep covers illumination, planar pose, resolution, blur, JPEG and occlusion
  with **three synthetic identities**. Expression, true 3D pose, age and non-frontal profiles are
  unmodelled, which makes `T` a lower bound.
- Planar pose warping is optimistic: no far-cheek self-occlusion, unchanged shading, and SCRFD
  found landmarks even at ±50°.
- One nuisance at a time (A2).
- The reference gallery is tightly packed (inter-identity cosine up to 0.26), so a "correct"
  assignment can be a near-tie. 10 of 22 baseline assignments have rank-1 margin < 0.07.
- Resolution sensitivity scales with how large a character happens to be: Jonah (91×118 px)
  reaches a *negative* same-identity cosine at 40 px.
- Ground truth is a prompt, not an annotated image. A missing detection is null, not zero.

## F. References

- ArcFace: Deng et al., *Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019. <https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf>
- AdaFace: Kim, Jain, Liu, *Quality Adaptive Margin for Face Recognition*, CVPR 2022. <https://arxiv.org/abs/2204.00964>
- MagFace: Meng et al., AAAI 2021 — embedding magnitude as a quality proxy.
- CR-FIQA: Boutros et al., CVPR 2023. <https://openaccess.thecvf.com/content/CVPR2023/html/Boutros_CR-FIQA_Face_Image_Quality_Assessment_by_Learning_Sample_Relative_Classifiability_CVPR_2023_paper.html>
- Forensic FR evaluation — published >95 % accuracy falling to as low as 65 % under low resolution and occlusion. <https://arxiv.org/abs/2311.06145>
- DINOv2: Oquab et al., *Learning Robust Visual Features without Supervision*. <https://arxiv.org/abs/2304.07193>
- T-Rex2: Jiang et al., *Text-Visual Prompt Synergy*, ECCV 2024. <https://arxiv.org/abs/2403.14610>
- DreamBooth: Ruiz et al., CVPR 2023 — establishes CLIP-I / DINO-I as subject-fidelity metrics. <https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf>
- DreamBench++: human-aligned personalisation benchmark. <https://arxiv.org/html/2406.16855v1>
- DreamSim: Fu et al., NeurIPS 2023. <https://arxiv.org/abs/2306.09344>
- F?D: on the role of deep feature spaces in face generation evaluation. <https://arxiv.org/html/2305.20048v3>
- CUTE: object intrinsic comparison, foreground/background separation. <https://github.com/s-tian/CUTE>

## Suggested order

1. **A1** matched baseline — repairs a defect in current reporting.
2. **A4** margin and rank — trivial, immediately useful.
3. **B2** multi-reference gallery — cheapest real variance reduction.
4. **C1** patch tokens — the object metric is the weaker of the two.
5. **A3** quality gating, then **A5** calibration.
6. **A2** joint nuisances.
7. **B5** disentangled decomposition — most work, best final story.
