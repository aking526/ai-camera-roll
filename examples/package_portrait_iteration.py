"""Package saved iteration evidence without generation, credentials or model weights."""

import argparse
import calendar
import csv
import hashlib
import json
from pathlib import Path
import shutil

import yaml


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def build(project, experiment, destination):
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    mapping = {}

    def copy(source, suggested):
        digest = sha(source)
        if digest not in copied:
            target = destination / suggested
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            assert sha(target) == digest
            copied[digest] = suggested
        origin = ("experiment/" + source.relative_to(experiment).as_posix()
                  if source.is_relative_to(experiment) else "repository/" + source.relative_to(project).as_posix())
        mapping[origin] = {"path": copied[digest], "sha256": digest}
        return copied[digest]

    roots = {"historical": project / "output/images", "control": experiment / "control/images",
             "portrait_reference": experiment / "portrait_reference/images"}
    manifests = {k: yaml.safe_load((v / "manifest.yaml").read_text()) for k,v in roots.items()}
    faces = {"historical": read(experiment / "historical_scores/scores.json"),
             "control": read(experiment / "control/scores/scores.json"),
             "portrait_reference": read(experiment / "portrait_reference/scores/scores.json")}
    objects = read(experiment / "object_evaluation/scores.json")
    first_objects = read(experiment / "object_evaluation/first_pass_scores.json")
    if first_objects["final_score_sha256"] != sha(experiment / "object_evaluation/scores.json"):
        raise ValueError("First-pass scores do not match final-score version")
    for data in (objects, first_objects):
        for name, digest in data["input_sha256"].items():
            if sha(experiment / name) != digest:
                raise ValueError("Scored object artifact changed: " + name)
    for condition, data in faces.items():
        root = project if condition == "historical" else experiment
        for name, digest in data["input_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError("Scored face input changed: " + name)

    photo_paths = {}
    for condition, manifest in manifests.items():
        for ref in manifest["references"]:
            for key, value in (("original", ref["image"]["path"]), ("conditioning", ref["reference_path"])):
                copy(roots[condition] / value, f"images/references/{ref['entity_id']}_{key}.png")
        for scene in manifest["photos"]:
            month = calendar.month_name[faces[condition]["scenes"][scene["photo_index"]]["month"]].lower()
            for step_index, step in enumerate(scene["passes"]):
                image = roots[condition] / step["path"]
                photo_paths[(condition, scene["photo_index"], step_index)] = copy(image, f"images/{condition}/{month}_pass_{step_index+1}.png")
            # Final copies are byte-identical to a pass; preserve the provenance mapping without duplication.
            copy(roots[condition] / scene["final_path"], f"images/{condition}/{month}_final.png")
        copy(roots[condition] / "manifest.yaml", f"records/{condition}_generation_manifest.yaml")

    evidence_roots = [("face_historical", experiment / "historical_scores"),
                      ("face_control", experiment / "control/scores"),
                      ("face_portrait_reference", experiment / "portrait_reference/scores"),
                      ("objects", experiment / "object_evaluation/assets"),
                      ("objects_first_pass", experiment / "object_evaluation/first_pass_assets")]
    for label, root in evidence_roots:
        for image in sorted(root.rglob("*")):
            if image.is_file() and image.suffix.lower() in (".png", ".jpg"):
                copy(image, f"evidence/{label}/{image.relative_to(root).as_posix()}")
    for original, target in (
        ("historical_scores/scores.json", "results/raw/faces_historical.json"),
        ("control/scores/scores.json", "results/raw/faces_control.json"),
        ("portrait_reference/scores/scores.json", "results/raw/faces_portrait_reference.json"),
        ("object_evaluation/scores.json", "results/raw/objects_final.json"),
        ("object_evaluation/first_pass_scores.json", "results/raw/objects_first_pass.json"),
        ("object_evaluation/targets.json", "records/object_targets.json"),
        ("plan.json", "records/experiment_plan.json"),
        ("requests.json", "records/image_requests.json"),
        ("frozen_source.yaml", "records/frozen_source.yaml"),
        ("assistant_observations.json", "records/assistant_observations.json"),
        ("object_evaluation/verification.json", "records/object_verification.json"),
    ):
        copy(experiment / original, target)
    copy(project / "output/camera_roll.yaml", "records/historical_source.yaml")

    source = yaml.safe_load((experiment / "frozen_source.yaml").read_text())["camera_roll"]
    names = {p["id"]: p["name"] for p in source["people"]}
    face_csv, object_csv = [], []
    for condition, data in faces.items():
        for scene in data["scenes"]:
            for step in scene["passes"]:
                for candidate in step["candidates"]:
                    face_csv.append({"condition": condition, "month": scene["month"], "pass": step["pass_index"]+1,
                                     "person": names[candidate["person_id"]], "similarity": candidate["similarity"],
                                     "face_index": candidate["face_index"],
                                     "image": photo_paths[(condition,scene["photo_index"],step["pass_index"])]})

    def column(condition, index, step_index, key, label, object_scene):
        step = faces[condition]["scenes"][index]["passes"][step_index]
        object_rows = [] if object_scene is None else object_scene["objects"]
        return {"key": key, "label": label, "image": photo_paths[(condition,index,step_index)],
                "face_scores": [{**c, "name": names[c["person_id"]]} for c in step["candidates"]],
                "object_scores": [{k: obj[k] for k in ("object_id", "name", "similarity", "candidate_count")} for obj in object_rows],
                "object_note": "Historical first-pass object scores were not computed." if object_scene is None else ""}

    scenes, historical_scenes = [], []
    for index, scene in enumerate(faces["control"]["scenes"]):
        month = scene["month"]
        initial = first_objects["scenes"][index]
        control = objects["conditions"]["control"]["scenes"][index]
        modified = objects["conditions"]["portrait_reference"]["scenes"][index]
        old = objects["conditions"]["historical"]["scenes"][index]
        multi = len(scene["passes"]) > 1
        columns = [column("control",index,0,"original_first_pass","Original method · Pass 1",initial)]
        if multi:
            columns += [column("control",index,1,"original_second_pass","Original method · Pass 2",control),
                        column("portrait_reference",index,1,"modified_second_pass","Portrait retention · Pass 2",modified)]
        for condition, step_index, scored in [("control",0,initial), ("control",1,control), ("portrait_reference",1,modified)]:
            if not multi and step_index:
                continue
            for obj in scored["objects"]:
                object_csv.append({"condition":condition,"month":month,"pass":step_index+1,"object":obj["name"],
                                   "similarity":obj["similarity"],"candidate_count":obj["candidate_count"],
                                   "image":photo_paths[(condition,index,step_index)]})
        for obj in old["objects"]:
            object_csv.append({"condition":"historical","month":month,"pass":2 if multi else 1,"object":obj["name"],
                               "similarity":obj["similarity"],"candidate_count":obj["candidate_count"],
                               "image":photo_paths[("historical",index,1 if multi else 0)]})
        warning = ""
        if month == 10:
            warning = "Wrong-person warning: assigned Maya switches from the left cook to the center person. The left cook falls from 0.1284 to 0.0696 (post-hoc check)."
        note = ("Identical first pass for both methods. Only the later portrait input and its instruction change."
                if multi else "Single-pass control. This same image is used by both methods; there is no second pass or treatment gain.")
        if month == 2:
            note += " The mug moves from the requested radiator to Maya’s hand."
        if month == 8:
            note += " Supporting-character face assignments also switch."
        scenes.append({"photo_index":index,"month":month,"name":calendar.month_name[month],"columns":columns,"note":note,"warning":warning})
        old_columns = [column("historical",index,0,"historical_first_pass","Historical pilot · Pass 1",old if not multi else None)]
        if multi:
            old_columns.append(column("historical",index,1,"historical_second_pass","Historical pilot · Pass 2",old))
        historical_scenes.append({"photo_index":index,"month":month,"name":calendar.month_name[month],"columns":old_columns,
                                  "note":"Earlier pilot preserved unchanged. Its starting image differs from the fresh controlled experiment.","warning":""})
    for filename, rows in (("face_scores.csv",face_csv),("object_scores.csv",object_csv)):
        path = destination / "results" / filename
        with path.open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    ledger = read(experiment / "requests.json")
    paired = [(before, after) for scene in scenes if len(scene["columns"]) == 3
              for before, after in zip(scene["columns"][1]["object_scores"], scene["columns"][2]["object_scores"], strict=True)
              if before["similarity"] is not None and after["similarity"] is not None]
    data = {"title":"Iteration 01 · Retaining Maya’s portrait", "subtitle":"A controlled comparison of the original method and one proposed change.",
            "summary":{"decision":"Mixed results; keep experimental.",
                       "object_paired_control_mean":sum(a["similarity"] for a,b in paired)/len(paired),
                       "object_paired_modified_mean":sum(b["similarity"] for a,b in paired)/len(paired),
                       "object_paired_count":len(paired),"image_requests":len(ledger),"text_requests":0},
            "scenes":scenes,"historical_scenes":historical_scenes,
            "sources":{"faces":"SCRFD + ArcFace cosine","objects":"Grounding DINO + DINOv2 cosine",
                       "record":"records/experiment_plan.json","data":"results/raw/","analysis":"ANALYSIS.md"}}
    # Derive face means from the same three candidate-index-stable scenes, not hand-entered summaries.
    for pos, key in enumerate(("first", "control", "modified")):
        selected = [next(c for c in s["columns"][pos]["face_scores"] if c["person_id"] == "person_maya")["similarity"] for s in scenes if s["month"] in (1,2,8)]
        data["summary"][f"maya_same_candidate_{key}_mean"] = sum(selected)/len(selected)
    write_json(destination / "results/comparison_data.json",data)
    write_json(destination / "records/asset_manifest.json",{"description":"Original paths map to byte-identical packaged files; identical images are stored once.","assets":mapping})
    print(json.dumps({"scenes":len(scenes),"historical_scenes":len(historical_scenes),"unique_copied_files":len(copied),"source_paths":len(mapping)},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    build(Path(__file__).resolve().parents[1],args.experiment.resolve(),args.output.resolve())
