# Object visual similarity

The object counterpart to SCRFD → ArcFace is **Grounding DINO → DINOv2**:

1. Grounding DINO finds candidate boxes for a fixed generic noun query, such as
   `mug.` or `baseball cap.`. The detector does not receive the desired color or
   the generation condition.
2. Keep boxes above 0.30 detector confidence, with a 0.25 text threshold.
   Remove overlapping boxes using per-query non-maximum suppression at IoU 0.50.
3. The candidate with the highest detector confidence is the primary crop.
   Other candidates remain visible; choosing the best similarity could reward
   an image containing several competing objects.
4. Detect and crop each saved canonical reference with the same procedure.
5. Fit each complete crop into 224 × 224 with neutral RGB (124,116,104) padding.
   This preserves the ends of wide objects such as bicycles and skillet handles.
   The standard encoder's subsequent resizing and center crop are disabled.
6. Extract DINOv2-small's 384-dimensional CLS feature, normalize it, and compute
   cosine similarity to the canonical object's feature.

This is an algorithmic feature comparison, with no visual question answering,
language-model judgments, or human ratings used to calculate the score.

## Meaning and limits

Higher cosine means greater visual resemblance in this feature space. It is not
percentage accuracy, a probability, or a calibrated same-object decision. The
numerical scale is not interchangeable with ArcFace. DINOv2 was trained for
general visual representations rather than this project's object identities.

The cropped region can include background and can depict the wrong candidate.
Pose, lighting, occlusion, small object size, and detector errors affect scores.
Correct placement, ownership, exact count, and fine attributes are not verified.
In particular, an identical mug moved from a radiator to a hand can score well.
An open sketchbook may look less like a closed canonical reference despite
better satisfying the scene. These are limits of the metric, not reasons to
silently alter its answers.

Missing detections are null, not zero similarity. Always show detection coverage
next to any average. The primary before/after mean uses the same jointly detected
object appearances in the four edited scenes; May's shared single-pass photo is
excluded. Historical original scores are context, separate from the fresh paired
control. Multiple candidate boxes flag ambiguity, not confirmed duplication.

The first targets/settings were frozen before object-model inference. Future
changes to thresholds, crops, queries, models or selection rules constitute a new
evaluation version and require rescoring every compared condition.

## Local setup and rerun

Install `object-requirements.txt` into a separate Python environment. Download
only the public configuration/tokenizer files and `model.safetensors` for:

- `IDEA-Research/grounding-dino-tiny`
- `facebook/dinov2-small`

The local model directory contains subdirectories `grounding-dino-tiny` and
`dinov2-small`, plus `downloads.json` mapping each model ID to its exact Hugging
Face revision. The evaluator loads only these local files, never remote code.
The saved score file records revisions, weight/config hashes, package versions,
input and crop hashes, queries, thresholds, boxes, and candidate scores.

From the repository, using the separate environment's Python:

```bash
python evaluation/score_objects.py \
  --experiment ../Final_Project_Local/experiments/portrait_reference_2026-11-04 \
  --models ../Final_Project_Local/object_eval/models
python examples/report_portrait_experiment.py \
  ../Final_Project_Local/experiments/portrait_reference_2026-11-04
```

Inference runs on CPU without API keys. Repeated image/query inputs share a cache;
changed settings or models reject that cache. Reference self-comparison, repeated
embedding extraction, and blank-image detection checks are diagnostics, not
same-category identity calibration. The report preserves all unmodified scores.

## Sources

- [Meta's DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md): visual features and nearest-neighbor retrieval.
- [Grounding DINO documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino): text-conditioned boxes and postprocessing.
- [CUTE / object intrinsic comparison](https://github.com/s-tian/CUTE): motivates foreground/background separation. Our tight-box CLS baseline does not reproduce its foreground-mask feature averaging method.

Both selected checkpoints are distributed under Apache-2.0. Weights stay outside
the source repository. Codex assisted with the evaluator, tests and report; the
team should understand and attribute this code in the final submission.
