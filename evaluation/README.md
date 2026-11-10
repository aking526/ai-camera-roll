# Local face-consistency evaluation

This evaluates the saved synthetic Maya example in `output/`. It uses InsightFace's SCRFD detector to find faces, aligns 112 × 112 crops, and compares ArcFace embeddings with each canonical character portrait using cosine similarity. Expected characters receive a maximum-total, one-to-one candidate assignment.

The committed `results/` contains the existing five-photo, nine-pass measurements, face crops, and numbered detection previews. `results/scores.json` includes model settings, package versions, model provenance, and SHA-256 hashes of the source inputs. Hash keys are relative to the repository root. These saved results can be inspected in the [demo website](../demo_site/) without installing the scoring dependencies or model weights.

## Interpretation

- Similarity ranges from −1 to 1; higher means a closer model match. It is not a percentage or probability.
- There is no calibrated identity threshold. Weak or negative matches may still be assigned. An unassigned character does not establish who is absent.
- The detector threshold of 0.5 controls face detection, not identity acceptance.
- The pilot has 12 final-photo face detections for 13 requested person appearances. This is a detection count, not identity accuracy.
- Maya's assigned similarity decreased in all four multi-pass photos. Pose, visibility, alignment, and assignment can affect this pattern; it does not prove that edits caused identity loss.
- `visual_review.json` contains assistant observations, not independent human labels. The website's human-rating workflow collects those separately. No improvement in generation quality or score-triggered retry is claimed.

## Recompute locally

Use a separate Python 3.12 environment for the pinned scoring dependencies; they are independent of the generator's environment.

```sh
python3.12 -m venv evaluation/.venv
evaluation/.venv/bin/python -m pip install -r evaluation/requirements.txt
```

Obtain the official [InsightFace buffalo_l archive](https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip). Verify its SHA-256 against `model-provenance.json`, then place only these two model files under `evaluation/model-cache/models/buffalo_l/`:

- `det_10g.onnx`
- `w600k_r50.onnx`

The pretrained weights are for non-commercial research; InsightFace library code is MIT licensed. Model weights are not included in this repository. The script checks both local model files against the recorded checksums before loading them, so a scoring run does not download weights.

From the repository root:

```sh
evaluation/.venv/bin/python evaluation/score_faces.py
evaluation/.venv/bin/python -m pytest evaluation/test_score_faces.py -q
```

Scoring uses the CPU and overwrites the derived files in `evaluation/results/`. It makes no story/image-generation API calls. The tests cover cosine similarity, empty detections, rectangular assignments, globally optimal matching, and weak matches; they do not require model weights.

After recomputing scores, rebuild the website's saved data with the demo setup command documented in [demo_site](../demo_site/). Model inference and human identity validation remain separate steps.

## How much the score moves when identity does not

A raw cosine has no calibrated threshold, so a change between two runs cannot be
attributed to identity without knowing the metric's own sensitivity. Two follow-up
measurements cover that, and both run offline on CPU with no generation calls:

- [FACE_INVARIANCE.md](FACE_INVARIANCE.md) perturbs canonical portraits along pose,
  illumination, resolution, blur, JPEG and occlusion — identity is unchanged by
  construction, so every movement is metric sensitivity. Result: `T` ≈ 0.30, about 40 %
  of a full identity swap.
- [ADAFACE_AB.md](ADAFACE_AB.md) swaps the embedder for AdaFace while holding detection,
  alignment and pairing fixed. Result: lateral, not a fix; identical conclusions.
- [face_metric_results.ipynb](face_metric_results.ipynb) renders both as tables and plots.
  Requires `notebook-requirements.txt`; reads committed results and needs no weights.

The backlog of un-implemented options (matched baseline, quality gating, patch-token
object features, disentangled decomposition) is in [FUTURE_WORK.md](FUTURE_WORK.md).

## Attribution

The generator, source story, and saved synthetic images are from this repository. Codex assisted with the evaluator, tests, and draft visual observations. Recognition uses the existing ArcFace model bundled with InsightFace; this project does not train a new recognition model.
