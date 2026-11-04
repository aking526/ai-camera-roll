# Iteration 01: portrait retention

## What we changed

The original pipeline adds references through successive image-editing passes.
Later passes receive the previous scene and newly introduced references, but do
not directly receive Maya's portrait again. In the historical pilot, Maya's
assigned ArcFace similarity fell after the second pass in all four multi-pass
scenes. That observation motivated a specific hypothesis: resupplying her
portrait might reduce facial drift during those edits.

We appended Maya's canonical portrait to each of the four second-pass requests,
with a short instruction identifying it as the facial reference for the person
already present. This is the pipeline change. Adding evaluation software is a
measurement step, not itself the generation improvement.

## What stayed constant

- Saved story, scene requirements and eight canonical reference images.
- Existing prompt bodies and per-pass seeds.
- Cloudflare `@cf/black-forest-labs/flux-2-klein-4b` image model.
- Original 448-pixel input preprocessing and 1024 × 1024 outputs.
- Identical first-pass image bytes for control and modified conditions.
- Face-scoring models/settings and object-scoring models/settings within their comparisons.

The two changes were the added portrait input and its identifying instruction.
The older pilot was not reproduced exactly despite reused seeds, so we generated
a fresh control and shared its first passes with the modified condition. The
historical pilot remains useful context and is preserved unchanged.

Thirteen Cloudflare image requests succeeded: nine control passes and four
modified second passes. There were no new text-generation or reference-generation
calls and no provider errors. All face/object evaluation ran locally without API
calls. Actual provider billing was not queried.

## Main numerical result

All face values below are raw ArcFace cosine similarities, not accuracy percentages.

SCRFD locates faces and facial landmarks. Those landmarks align each face crop
before ArcFace converts it to a numerical identity representation. We compare
that representation with the character's canonical portrait using cosine
similarity. These pretrained models give us a repeatable detector and face-specific
comparison without training a new model. In group photos, the assignment step
matches detected faces to expected characters; that assignment can be wrong,
as October demonstrates.

| Scene | Shared first pass | Original second pass | Modified second pass | Second-pass difference |
| --- | ---: | ---: | ---: | ---: |
| January | 0.3119 | 0.1554 | 0.2496 | +0.0943 |
| February | 0.1973 | 0.0910 | 0.1743 | +0.0834 |
| May | 0.1611 | No second pass | No second pass | Shared control |
| August | 0.1002 | 0.0726 | 0.1106 | +0.0381 |
| October | 0.1277 | 0.1284 | **0.2039*** | **+0.0755*** |

For January, February and August, Maya keeps the same candidate index. Their
mean is **0.2031 in pass 1 → 0.1063 in the original pass 2 → 0.1782 in the modified
pass 2**. The modified second pass gains **0.0719**, recovering some of the measured
similarity loss on this subset. The final mean remains below the first-pass mean.
This subset is identified after inspecting assignments; it is not a preregistered
success rate. Matching candidate indices alone does not prove identity.

**October is a failure case, not a fourth validated improvement.** The matcher
switches Maya from the left-hand cook to the center person in scrubs. The intended
left-hand cook's similarity falls **0.1284 → 0.0696**. This is a post-hoc
position-based diagnostic, separate from the fixed assignment metric. Keep that
warning beside the October photos and score.

Supporting-character assigned mean across edited scenes falls **0.1180 → 0.0934**.
Their assignments also change in some scenes, so this is an additional diagnostic,
not proof of supporting-character identity loss. Both final sets have 13 face
detections for 13 requested appearances; detection counts do not establish identity.

## Object performance

The algorithm is **Grounding DINO localization → DINOv2 cropped-reference cosine**.
It does not use human ratings, VQA answers or a generative model to assign scores.
The highest detector-confidence candidate is selected after overlap filtering;
we do not choose whichever candidate produces the highest similarity. Other
candidates and all crops remain available in the evidence.

| Final-photo set | Targets with candidates | Mean selected-candidate similarity |
| --- | ---: | ---: |
| Historical pilot | 13/13 | 0.3452 (n=13) |
| Fresh control | 12/13 | 0.3199 (n=12) |
| Modified pipeline | 12/13 | 0.3183 (n=12) |

The fair edited-scene comparison uses the same **10 of 11 eligible appearances**
with candidates in both conditions and excludes shared May. Mean similarity is
**0.3064 → 0.3045**, a **−0.0020** change. There is no average object-resemblance
improvement in this run. Individual changes differ: the February mug falls
0.5153 → 0.3317, while August's cap and bicycle scores rise.

First-pass objects are now scored with exactly the same model weights, crop rule,
queries and thresholds. Some objects are introduced only in pass 2, so a missing
first-pass candidate is not necessarily a failure. Do not compare stage averages
with different detected-object denominators as if they were a matched set.

Object similarity measures resemblance to one canonical view. It does not verify
correct placement, ownership, count or exact state. The February mug moves from
the requested radiator to Maya's hand. A closed reference sketchbook can make a
correctly opened scene sketchbook score lower. Bicycle crops retain background
and people, so composition can affect their score.

The blank-image sanity check produced false-positive book and pan detections.
Some mug candidates are indistinct regions, bowls or glass cups. Candidate
coverage is therefore not verified object presence. No handpicked crops or
manual ratings replaced the algorithm's results.

## What this iteration demonstrates

We evaluated a failure, made one targeted pipeline change, and reevaluated it on
matched inputs. The numerical evidence suggests partial facial-similarity
recovery in three scenes, but also reveals wrong-person editing and object
tradeoffs. This is a documented evaluation/change/reevaluation cycle with mixed
results, not proof of overall improvement or completion of the course's required
iteration count.

The practical decision is to **keep portrait retention experimental**. The next
generation change should address an observed failure, such as directing an edit
to the intended person's region. Background-reduced object features and better
candidate selection are possible evaluator improvements. Those are hypotheses
to test one at a time; no future outcomes have been decided.

This is one five-scene run, with four treated scenes. There are no repeated
generation trials, confidence intervals, calibrated identity cutoffs, or completed
independent human study. These limits should remain in the presentation.

## Suggested PowerPoint sequence

1. Explain the second-pass identity-drift hypothesis and the single input change.
2. Show January's three-column figure: original second-pass loss and partial recovery.
3. Show the face-score table, then October's figure and its wrong-person caveat.
4. Show the object comparison: essentially unchanged average resemblance, with the
   February mug as a concrete tradeoff.
5. State the evidence-based decision and the next question to test. Do not present
   planned future iterations as completed results.

The historical pilot is preserved for a background slide. Do not pair its first
pass with the modified second pass and describe that as a controlled intervention.

## Sources and attribution

- [Original project](https://github.com/aking526/ai-camera-roll)
- [InsightFace / SCRFD / ArcFace](https://github.com/deepinsight/insightface)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Meta DINOv2](https://github.com/facebookresearch/dinov2)

Saved JSON contains exact checkpoint hashes, versions, image hashes, assignments,
detection boxes and numerical values. `records/asset_manifest.json` maps source
paths to packaged evidence. Codex assisted with code, evaluation and writing;
assistant visual observations are not independent human ratings.
