"""Build a local, exploratory report for the paired portrait-reference experiment."""

import argparse
import calendar
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import mean
from urllib.parse import quote

import yaml


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, signed=False):
    return "unassigned" if value is None else format(value, "+.4f" if signed else ".4f")


def candidate(step, person):
    return next(c for c in step["candidates"] if c["person_id"] == person)


def summarize(rows):
    paired = [r for r in rows if r["delta"] is not None]
    same = [r for r in paired if not r["assignment_changed"]]
    return {"expected_pairs": len(rows), "paired_assignments": len(paired),
            "higher_assigned_scores": sum(r["delta"] > 0 for r in paired),
            "lower_assigned_scores": sum(r["delta"] < 0 for r in paired),
            "unchanged_assigned_scores": sum(r["delta"] == 0 for r in paired),
            "mean_paired_delta": mean(r["delta"] for r in paired) if paired else None,
            "candidate_index_changed": sum(r["assignment_changed"] for r in paired),
            "same_index_pairs": len(same),
            "same_index_higher_scores": sum(r["delta"] > 0 for r in same),
            "same_index_mean_delta": mean(r["delta"] for r in same) if same else None}


def object_comparison(root, edited):
    path = root / "object_evaluation/scores.json"
    if not path.exists():
        return None
    data = read(path)
    for key in ("schema_version", "method", "settings", "model_provenance", "references", "conditions", "input_sha256"):
        if key not in data:
            raise ValueError("Missing object-score field: " + key)
    def checked(relative):
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError("Object evidence path is missing or outside the experiment: " + relative)
        return target
    if data["source_sha256"] != sha(root / "frozen_source.yaml") or data["target_sha256"] != sha(root / "object_evaluation/targets.json"):
        raise ValueError("Object-scoring source or target specification changed.")
    for relative, digest in data["input_sha256"].items():
        if sha(checked(relative)) != digest:
            raise ValueError("Object-scored input changed: " + relative)
    for ref in data["references"]:
        checked(ref["crop"])
    summaries, by_condition = {}, {}
    for name in ("historical", "control", "portrait_reference"):
        condition = data["conditions"][name]
        by_condition[name] = {}
        for scene in condition["scenes"]:
            if sha(checked(scene["image"])) != scene["sha256"]:
                raise ValueError("Object-scored scene image changed.")
            checked(scene["detections_image"])
            for obj in scene["objects"]:
                if obj["candidate_count"] != len(obj["detections"]):
                    raise ValueError("Object candidate count disagrees with saved detections.")
                for detection in obj["detections"]:
                    checked(detection["crop"])
                selected = obj["selected_index"]
                if selected is None:
                    if obj["similarity"] is not None:
                        raise ValueError("Unselected object has a similarity score.")
                elif not 0 <= selected < len(obj["detections"]) or obj["similarity"] != obj["detections"][selected]["similarity"]:
                    raise ValueError("Object selected candidate and similarity disagree.")
                if obj["similarity"] is not None and not math.isfinite(obj["similarity"]):
                    raise ValueError("Object similarity must be finite or null.")
                key = (scene["photo_index"], obj["object_id"])
                if key in by_condition[name]:
                    raise ValueError("Duplicate scene/object score.")
                by_condition[name][key] = obj
        items = list(by_condition[name].values())
        detected = [o["similarity"] for o in items if o["similarity"] is not None]
        summaries[name] = {"label": condition["label"], "expected_count": len(items), "detected_count": len(detected),
                           "multiple_candidate_count": sum(o["candidate_count"] > 1 for o in items),
                           "mean_similarity_detected": mean(detected) if detected else None}
    if not by_condition["historical"].keys() == by_condition["control"].keys() == by_condition["portrait_reference"].keys():
        raise ValueError("Object targets differ between conditions.")
    rows = []
    for (index, object_id), before in by_condition["control"].items():
        old, after = by_condition["historical"][(index, object_id)], by_condition["portrait_reference"][(index, object_id)]
        rows.append({"photo_index": index, "object_id": object_id, "name": before["name"],
                     "historical": old["similarity"], "control": before["similarity"], "treatment": after["similarity"],
                     "delta": None if before["similarity"] is None or after["similarity"] is None else after["similarity"] - before["similarity"],
                     "candidate_counts": [old["candidate_count"], before["candidate_count"], after["candidate_count"]]})
    eligible = [r for r in rows if r["photo_index"] in edited]
    paired = [r for r in eligible if r["delta"] is not None]
    summary = {"eligible_targets": len(eligible), "common_detected_targets": len(paired),
               "control_mean": mean(r["control"] for r in paired) if paired else None,
               "treatment_mean": mean(r["treatment"] for r in paired) if paired else None,
               "mean_paired_delta": mean(r["delta"] for r in paired) if paired else None}
    sanity = data.get("sanity_checks", {})
    blank_counts = sanity.get("blank_image_detection_counts")
    false_positive_targets = None if blank_counts is None else [key for key, count in blank_counts.items() if count > 0]
    return {"method": data["method"], "settings": data["settings"], "model_provenance": data["model_provenance"],
            "verification": "All recorded hashes and displayed evidence paths verified.",
            "conditions": summaries, "paired_edited_scenes": summary, "objects": rows,
            "sanity_checks": sanity, "blank_image_false_positive_targets": false_positive_targets,
            "evidence": data}


def build(root):
    conditions = ("control", "portrait_reference")
    scores = {name: read(root / name / "scores/scores.json") for name in conditions}
    historical = read(root / "historical_scores/scores.json")
    plan, requests = read(root / "plan.json"), read(root / "requests.json")
    observations = read(root / "assistant_observations.json") if (root / "assistant_observations.json").exists() else {}
    manifests = {name: yaml.safe_load((root / name / "images/manifest.yaml").read_text()) for name in conditions}
    control, treatment = (scores[name] for name in conditions)
    for data in scores.values():
        for relative, digest in data["input_sha256"].items():
            if sha(root / relative) != digest:
                raise ValueError("Input changed after scoring: " + relative)
    for key in ("settings", "model_provenance", "packages"):
        if not control[key] == treatment[key] == historical[key]:
            raise ValueError(f"Scoring configuration differs: {key}")
    source_hash = sha(root / "frozen_source.yaml")
    if any(s["input_sha256"]["frozen_source.yaml"] != source_hash for s in scores.values()):
        raise ValueError("Frozen source differs from the scored source.")
    names = {r["id"]: r["name"] for r in control["references"]}
    for ref in control["references"]:
        paths = [root / name / "images" / ref["source"] for name in conditions]
        if sha(paths[0]) != sha(paths[1]):
            raise ValueError("Canonical portraits differ between conditions.")
    edited = {change["photo_index"] for change in plan["changes"]}
    scenes, rows = [], []
    old_scenes = {s["photo_index"]: s for s in historical["scenes"]}
    for before, after in zip(control["scenes"], treatment["scenes"], strict=True):
        index = before["photo_index"]
        if index != after["photo_index"] or before["expected_people"] != after["expected_people"]:
            raise ValueError("Paired scene identities differ.")
        first_hashes = []
        for name, scene in zip(conditions, (before, after), strict=True):
            relative = f"{name}/images/{scene['passes'][0]['source']}"
            digest = sha(root / relative)
            if scores[name]["input_sha256"][relative] != digest:
                raise ValueError("First-pass image changed after scoring.")
            first_hashes.append(digest)
        if first_hashes[0] != first_hashes[1]:
            raise ValueError("Paired first passes are not identical.")
        if before["passes"][0]["cosine_matrix"] != after["passes"][0]["cosine_matrix"]:
            raise ValueError("Identical first passes received different scores.")
        if index not in edited and sha(root / "control/images" / before["final_source"]) != sha(root / "portrait_reference/images" / after["final_source"]):
            raise ValueError("Shared single-pass scene changed.")
        a, b = before["passes"][-1], after["passes"][-1]
        scene_rows = []
        for person in before["expected_people"]:
            ca, cb = candidate(a, person), candidate(b, person)
            old = candidate(old_scenes[index]["passes"][-1], person)
            row = {"photo_index": index, "month": before["month"], "person_id": person,
                   "person_name": names[person], "treatment_applied": index in edited,
                   "historical_final": old["similarity"], "control_final": ca["similarity"],
                   "treatment_final": cb["similarity"], "control_face_index": ca["face_index"],
                   "treatment_face_index": cb["face_index"],
                   "assignment_changed": ca["face_index"] != cb["face_index"],
                   "delta": None if ca["similarity"] is None or cb["similarity"] is None else cb["similarity"] - ca["similarity"]}
            scene_rows.append(row)
            rows.append(row)
        scenes.append({"photo_index": index, "month": before["month"], "treatment_applied": index in edited,
                       "shared_first_pass_sha256": first_hashes[0], "expected_people": len(before["expected_people"]),
                       "control_detections": len(a["faces"]), "treatment_detections": len(b["faces"]), "characters": scene_rows})
    protagonist_rows = [r for r in rows if r["person_id"] == plan["protagonist_id"] and r["treatment_applied"]]
    report = {"status": "Exploratory automatic scores; independent human review pending.",
              "comparison": "Fresh paired control versus portrait reference. Historical output is context only.",
              "warning": "Higher assigned scores may select a different person. Candidate indices are horizontal detection order, not tracked identities. Object and role regressions can coexist with higher similarity.",
              "verification": {"all_paired_scored_inputs_unchanged": True, "shared_first_passes_identical": True, "frozen_source_identical": True,
                               "canonical_portraits_identical": True, "scoring_settings_models_packages_identical": True},
              "protagonist_edited_scenes": summarize(protagonist_rows),
              "supporting_characters_edited_scenes": summarize([r for r in rows if r["person_id"] != plan["protagonist_id"] and r["treatment_applied"]]),
              "requests": {"attempts": len(requests), "completed": sum(r["status"] == "completed" for r in requests),
                           "summed_request_seconds": sum(r.get("seconds", 0) for r in requests),
                           "text_calls": plan["text_calls"], "reference_generation_calls": plan["reference_generation_calls"]},
              "scoring_settings": control["settings"], "scenes": scenes, "assistant_observations": observations}
    objects = object_comparison(root, edited)
    if objects:
        report["algorithmic_objects"] = {key: value for key, value in objects.items() if key != "evidence"}
    (root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    esc = html.escape
    def picture(relative, label, small=False):
        return f'<figure><img class="{"crop" if small else "photo"}" src="{quote(str(relative))}" alt="{esc(label)}"><figcaption>{esc(label)}</figcaption></figure>'
    def table(items):
        body = "".join(f'<tr><td>{esc(r["person_name"])}</td><td>{number(r["historical_final"])}</td><td>{number(r["control_final"])}</td><td>{number(r["treatment_final"])}</td><td>{number(r["delta"], True)}</td><td>{r["control_face_index"]} → {r["treatment_face_index"]}{" (changed)" if r["assignment_changed"] else ""}</td></tr>' for r in items)
        return '<div class="scroll"><table><thead><tr><th>Character</th><th>Historical final¹</th><th>Fresh control</th><th>Portrait reference</th><th>Paired change</th><th>Face index</th></tr></thead><tbody>' + body + '</tbody></table></div>'
    def object_number(value, signed=False):
        return "no candidate" if value is None else number(value, signed)
    summary = report["protagonist_edited_scenes"]
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Portrait-reference experiment</title>',
             '<style>body{font:17px/1.5 system-ui,sans-serif;background:#f4f1e9;color:#24382d;margin:0}main{max-width:1180px;margin:auto;padding:36px 24px}h1{font-size:38px;line-height:1.15}h2{margin-top:12px}h3{font-size:20px}a{color:#365c45}section{border-top:1px solid #c8ccbf;padding:28px 0;margin-top:28px}.notice{border-left:5px solid #af573d;padding:12px 20px;background:#fff8ed}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}.faces{display:flex;gap:24px;flex-wrap:wrap}.faces figure{width:220px}figure{margin:0}img.photo{width:100%;height:auto}img.crop{width:180px;height:180px;object-fit:contain;background:#e8e6dd}figcaption{font-size:15px;margin:7px 0 18px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:15px;margin:16px 0}th,td{text-align:left;padding:10px;border-bottom:1px solid #c8ccbf}th{background:#e6e8dd}pre{white-space:pre-wrap;font:14px/1.55 system-ui}details{margin:18px 0}summary{cursor:pointer;font-weight:600}.muted{color:#62685d;font-size:15px}@media(max-width:700px){.pair{grid-template-columns:1fr}h1{font-size:30px}}@media print{details{display:none}section{break-before:page}}</style><main>',
             '<p>CIS 5810 · one controlled pipeline experiment</p><h1>Resupplying Maya’s portrait during later edits</h1>',
             '<p>The control and modified albums share exactly the same newly generated first passes, canonical references, seeds, image model and scene requirements. The treatment appends Maya’s portrait and its slot instruction to four editing passes. May is a shared single-pass control.</p>',
             f'<div class="notice"><strong>Human validation is pending.</strong> Maya’s assigned score is higher in {summary["higher_assigned_scores"]}/{summary["paired_assignments"]} edited scenes, but one comparison switches to a different face candidate in October. This is not evidence of four successful identity improvements. Visual review also identifies possible object and role regressions.</div>',
             f'<p><strong>Same-index Maya comparisons:</strong> {summary["same_index_higher_scores"]}/{summary["same_index_pairs"]} higher, mean paired change {number(summary["same_index_mean_delta"], True)}. All assigned comparisons: {summary["paired_assignments"]}/{summary["expected_pairs"]} available, mean {number(summary["mean_paired_delta"], True)} including the candidate switch. Matching indices alone do not validate identity.</p>',
             f'<p>{report["requests"]["completed"]}/{report["requests"]["attempts"]} image requests completed. No new story calls or reference-image generations. Scoring settings, model hashes and package versions match across all three score files.</p>',
             '<p><a href="RUN_SUMMARY.txt">Run summary</a> · <a href="new_photo_set.zip">Download the new photos</a> · <a href="comparison.json">Comparison data</a> · <a href="plan.json">Frozen experiment plan</a> · <a href="requests.json">Request record</a> · <a href="assistant_observations.json">Assistant observations</a></p>',
             '<p class="muted">¹ Historical results provide context, not the primary comparator: the provider’s newly generated control differs from the older outputs despite reused seeds. Similarity is not a probability or pass/fail judgment. Missing assignments remain missing.</p>']
    if objects:
        paired = objects["paired_edited_scenes"]
        parts.append('<section id="object-scores"><h2>Algorithmic object similarity</h2><p><strong>Grounding DINO</strong> locates candidates using a fixed text query for each required object. After overlap filtering, the highest detector-confidence candidate is selected. <strong>DINOv2</strong> converts that crop and the canonical reference crop into visual features. Their cosine similarity measures resemblance, from −1 to 1.</p><p>These numbers come from local detection and image embeddings. No VQA answers, language-model judgments or human ratings enter this score. Crops include their background; this is not segmentation.</p><p class="muted">Method sources: <a href="https://github.com/IDEA-Research/GroundingDINO">Grounding DINO</a> and <a href="https://github.com/facebookresearch/dinov2">Meta DINOv2</a>. The paired fresh-control comparison is primary. Historical images provide context only.</p>')
        parts.append('<div class="scroll"><table><thead><tr><th>Condition</th><th>Targets with candidates / expected</th><th>Targets with multiple candidates / expected</th><th>Mean selected-candidate similarity</th></tr></thead><tbody>')
        for item in objects["conditions"].values():
            parts.append(f'<tr><td>{esc(item["label"])}</td><td>{item["detected_count"]}/{item["expected_count"]}</td><td>{item["multiple_candidate_count"]}/{item["expected_count"]}</td><td>{object_number(item["mean_similarity_detected"])} (n={item["detected_count"]})</td></tr>')
        parts.append('</tbody></table></div>')
        parts.append(f'<p><strong>Paired comparison on the four edited scenes:</strong> {paired["common_detected_targets"]}/{paired["eligible_targets"]} targets have candidates in both conditions. On this same subset, control mean {object_number(paired["control_mean"])}, portrait-reference mean {object_number(paired["treatment_mean"])}, paired change {object_number(paired["mean_paired_delta"], True)}. May is excluded because its image is shared.</p>')
        parts.append('<div class="notice">Object similarity measures appearance. It does not determine correct placement, ownership or scene role. Detector confidence is not object accuracy, and multiple candidates are not verified duplicates. Missing detections remain missing instead of becoming zero scores. This metric has not been calibrated as an object-identity verdict.</div><p><a href="object_evaluation/scores.json">Raw object scores and model provenance</a> · <a href="object_evaluation/targets.json">Fixed object queries</a></p></section>')
        if objects["blank_image_false_positive_targets"]:
            refs = {r["object_id"]: r["name"] for r in objects["evidence"]["references"]}
            false_positives = ', '.join(refs.get(key, key) for key in objects["blank_image_false_positive_targets"])
            parts.append(f'<div class="notice"><strong>Blank-image check found false positives.</strong> The detector returned candidates for {esc(false_positives)} on a uniform blank image. Coverage therefore means candidate coverage, not proven object presence. Inspect the crops before interpreting an appearance score.</div>')
        if objects["sanity_checks"]:
            parts.append('<details><summary>Raw detector and embedding sanity checks</summary><pre>' + esc(json.dumps(objects["sanity_checks"], indent=2)) + '</pre></details>')
    observation_map = {s["month"]: s["observations"] for s in observations.get("scenes", [])}
    for scene, before, after in zip(scenes, control["scenes"], treatment["scenes"], strict=True):
        index, month = scene["photo_index"], scene["month"]
        parts.append(f'<section><h2>{calendar.month_name[month]}{ " — shared single-pass control" if not scene["treatment_applied"] else ""}</h2><p>Detected faces: control {scene["control_detections"]}/{scene["expected_people"]} requested appearances; treatment {scene["treatment_detections"]}/{scene["expected_people"]}. Counts do not establish identities.</p>')
        parts.append('<div class="pair">' + picture(f'control/images/{before["final_source"]}', 'Fresh control final photo') + picture(f'portrait_reference/images/{after["final_source"]}', 'Portrait-reference final photo') + '</div>')
        parts.append(table(scene["characters"]))
        parts.append('<div class="faces">')
        for name, s in zip(conditions, (before, after), strict=True):
            step = s["passes"][-1]
            match = candidate(step, plan["protagonist_id"])
            if match["face_index"] is not None:
                face = next(f for f in step["faces"] if f["face_index"] == match["face_index"])
                parts.append(picture(f'{name}/scores/{face["aligned_crop"]}', f'{name.replace("_", " ")}: assigned Maya candidate {match["face_index"]}, {number(match["similarity"])}', True))
        parts.append('</div>')
        if objects:
            object_rows = [r for r in objects["objects"] if r["photo_index"] == index]
            parts.append('<h3>Object appearance scores</h3><div class="scroll"><table><thead><tr><th>Object</th><th>Historical original¹</th><th>Fresh control</th><th>Portrait reference</th><th>Paired change</th><th>Candidates H / C / T</th></tr></thead><tbody>')
            for item in object_rows:
                counts = ' / '.join(map(str, item["candidate_counts"]))
                parts.append(f'<tr><td>{esc(item["name"])}</td><td>{object_number(item["historical"])}</td><td>{object_number(item["control"])}</td><td>{object_number(item["treatment"])}</td><td>{object_number(item["delta"], True)}</td><td>{counts}</td></tr>')
            parts.append('</tbody></table></div><details><summary>Inspect object reference crops, candidates and detections</summary>')
            references = {r["object_id"]: r for r in objects["evidence"]["references"]}
            object_scenes = {name: next(s for s in condition["scenes"] if s["photo_index"] == index) for name, condition in objects["evidence"]["conditions"].items()}
            for item in object_rows:
                parts.append(f'<h3>{esc(item["name"])}</h3><div class="faces">')
                parts.append(picture(references[item["object_id"]]["crop"], 'Canonical object reference', True))
                for name in ("historical", "control", "portrait_reference"):
                    obj = next(o for o in object_scenes[name]["objects"] if o["object_id"] == item["object_id"])
                    for position, detection in enumerate(obj["detections"]):
                        chosen = 'selected' if position == obj["selected_index"] else 'other candidate'
                        label = f'{objects["conditions"][name]["label"]}, {chosen}: similarity {object_number(detection["similarity"])}, detector confidence {detection["confidence"]:.3f}'
                        parts.append(picture(detection["crop"], label, True))
                    if not obj["detections"]:
                        parts.append(f'<p>{esc(objects["conditions"][name]["label"])}: no candidate.</p>')
                parts.append('</div>')
            for name, object_scene in object_scenes.items():
                parts.append(picture(object_scene["detections_image"], objects["conditions"][name]["label"] + ': object detections'))
            parts.append('</details>')
        if month in observation_map:
            parts.append('<div class="notice"><strong>Assistant visual observations, not independent ratings</strong><ul>' + ''.join(f'<li>{esc(item)}</li>' for item in observation_map[month]) + '</ul></div>')
        parts.append('<details><summary>Saved prompts, reference slots and seeds</summary>')
        for name in conditions:
            photo = next(p for p in manifests[name]["photos"] if p["photo_index"] == index)
            parts.append(f'<h3>{name.replace("_", " ")}</h3>')
            for step_index, step in enumerate(photo["passes"]):
                slots = ', '.join(f'{i["slot"]}: {i["entity_id"] or "previous scene"}' for i in step["inputs"])
                parts.append(f'<p><strong>Pass {step_index + 1}</strong> · seed {step["seed"]} · {esc(slots)}</p><pre>{esc(step["prompt"])}</pre>')
        parts.append('</details></section>')
    parts.append('<section><h2>Decision pending</h2><p>Keep portrait retention experimental until independent reviewers assess identity, character roles and object/scene correctness. Use these results to choose the next change. Higher assigned similarity alone is insufficient to adopt the treatment.</p></section></main></html>')
    (root / "comparison.html").write_text('\n'.join(parts))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    report = build(args.experiment.resolve())
    print(json.dumps({"scenes": len(report["scenes"]), "maya": report["protagonist_edited_scenes"], "verification": report["verification"]}, indent=2))
