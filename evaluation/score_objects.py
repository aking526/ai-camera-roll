"""Local Grounding DINO boxes + DINOv2 reference similarity; no generative judge."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
from statistics import mean

from PIL import Image, ImageDraw, ImageOps
import yaml


SETTINGS = {
    "box_threshold": 0.30, "text_threshold": 0.25, "nms_iou": 0.50,
    "selection": "highest detector confidence after per-query NMS; not highest similarity",
    "embedding": "L2-normalized DINOv2-small CLS token, 384 dimensions",
    "crop": "tight clamped bounding box; no segmentation or extra padding",
    "resize": "aspect-preserving fit into 224x224, RGB (124,116,104) padding; no center crop",
    "similarity": "raw cosine, not accuracy or probability; no calibrated cutoff",
    "device": "cpu", "dtype": "float32", "threads": 4,
}
MODEL_IDS = {"detector": "IDEA-Research/grounding-dino-tiny", "encoder": "facebook/dinov2-small"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    union = area(a) + area(b) - overlap
    return overlap / union if union else 0.0


def suppress(candidates, threshold=0.5):
    kept = []
    for candidate in sorted(candidates, key=lambda c: (-c["confidence"], c["bbox"])):
        if all(iou(candidate["bbox"], prior["bbox"]) <= threshold for prior in kept):
            kept.append(candidate)
    return kept


def crop_box(box, size):
    w, h = size
    x1, y1, x2, y2 = box
    result = (max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2)))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("Invalid detector crop")
    return result


def prepare_crop(image):
    return ImageOps.pad(image.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC, color=(124, 116, 104))


def summarize(scenes):
    objects = [obj for scene in scenes for obj in scene["objects"]]
    scores = [obj["similarity"] for obj in objects if obj["similarity"] is not None]
    return {"expected_count": len(objects), "detected_count": len(scores),
            "multiple_candidate_count": sum(obj["candidate_count"] > 1 for obj in objects),
            "mean_similarity_detected": mean(scores) if scores else None}


class Models:
    def __init__(self, root):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoImageProcessor, AutoModel

        self.torch = torch
        torch.set_num_threads(SETTINGS["threads"])
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        detector = root / "grounding-dino-tiny"
        encoder = root / "dinov2-small"
        self.detector_processor = AutoProcessor.from_pretrained(detector, local_files_only=True, trust_remote_code=False)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector, local_files_only=True, use_safetensors=True).eval()
        self.encoder_processor = AutoImageProcessor.from_pretrained(encoder, local_files_only=True, trust_remote_code=False)
        self.encoder = AutoModel.from_pretrained(encoder, local_files_only=True, use_safetensors=True).eval()

    def detect(self, image, query):
        inputs = self.detector_processor(images=image, text=query, return_tensors="pt")
        with self.torch.inference_mode():
            output = self.detector(**inputs)
        result = self.detector_processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=SETTINGS["box_threshold"],
            text_threshold=SETTINGS["text_threshold"], target_sizes=[image.size[::-1]],
        )[0]
        candidates = [{"bbox": box.tolist(), "confidence": score.item()}
                      for box, score in zip(result["boxes"], result["scores"], strict=True)]
        return suppress(candidates, SETTINGS["nms_iou"])

    def embed(self, image):
        inputs = self.encoder_processor(images=prepare_crop(image), return_tensors="pt", do_resize=False, do_center_crop=False)
        with self.torch.inference_mode():
            feature = self.encoder(**inputs).last_hidden_state[:, 0, :]
            feature = self.torch.nn.functional.normalize(feature, p=2, dim=-1)
        return feature[0].tolist()


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError("Embedding dimensions differ")
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b, strict=True))))


def run(project, experiment, model_root):
    target_path = project / "evaluation/object_targets.json"
    targets = read(target_path)
    if digest(experiment / "frozen_source.yaml") != targets["source_sha256"]:
        raise ValueError("Object targets do not match frozen source")
    sources = (
        ("historical", project / "output/images", "Historical original"),
        ("control", experiment / "control/images", "Fresh control"),
        ("portrait_reference", experiment / "portrait_reference/images", "Portrait reference"),
    )
    manifests = {name: yaml.safe_load((root / "manifest.yaml").read_text()) for name, root, _ in sources}
    reference_hashes = {}
    for name, root, _ in sources:
        references = {ref["entity_id"]: ref for ref in manifests[name]["references"]}
        reference_hashes[name] = {obj["id"]: digest(root / references[obj["reference_id"]]["image"]["path"])
                                 for obj in targets["objects"]}
    if any(hashes != reference_hashes["control"] for hashes in reference_hashes.values()):
        raise ValueError("Canonical object references differ between conditions")
    out = experiment / "object_evaluation"
    out.mkdir(exist_ok=True)
    shutil.copyfile(target_path, out / "targets.json")
    downloads = read(model_root / "downloads.json")
    provenance = {}
    for kind, name in MODEL_IDS.items():
        folder = model_root / name.split("/")[-1]
        provenance[kind] = {"model_id": name, "revision": downloads[name]["revision"], "license": "Apache-2.0",
                            "files_sha256": {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    packages = {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "transformers", "numpy", "pillow", "safetensors")}
    signature = digest(target_path) + json.dumps([SETTINGS, provenance, packages], sort_keys=True)
    cache_path = out / "inference_cache.json"
    cache = read(cache_path) if cache_path.exists() else {"signature": signature, "items": {}}
    if cache["signature"] != signature:
        raise ValueError("Scoring configuration changed: use a separate output experiment")
    models = Models(model_root)
    hashes = {}

    def asset(path):
        relative = path.relative_to(experiment).as_posix()
        hashes[relative] = digest(path)
        return relative

    def infer(image_path, obj):
        key = digest(image_path) + ":" + obj["query"]
        if key not in cache["items"]:
            image = Image.open(image_path).convert("RGB")
            found = models.detect(image, obj["query"])
            for candidate in found:
                candidate["crop_box"] = list(crop_box(candidate["bbox"], image.size))
                candidate["embedding"] = models.embed(image.crop(candidate["crop_box"]))
            cache["items"][key] = found
            write(cache_path, cache)
        return copy.deepcopy(cache["items"][key])

    manifest = manifests["control"]
    reference_manifest = {r["entity_id"]: r for r in manifest["references"]}
    refs = []
    ref_embeddings = {}
    assets = out / "assets"
    assets.mkdir(exist_ok=True)
    for obj in targets["objects"]:
        reference_path = experiment / "control/images" / reference_manifest[obj["id"]]["image"]["path"]
        found = infer(reference_path, obj)
        if not found:
            raise ValueError("Canonical reference not detected: " + obj["id"])
        selected = found[0]
        crop_path = assets / (obj["id"] + "_reference.png")
        Image.open(reference_path).convert("RGB").crop(selected["crop_box"]).save(crop_path)
        ref_embeddings[obj["id"]] = selected["embedding"]
        refs.append({"object_id": obj["id"], "name": obj["name"], "query": obj["query"],
                     "image": asset(reference_path), "crop": asset(crop_path), "bbox": selected["bbox"],
                     "confidence": selected["confidence"], "candidate_count": len(found)})
        print("Reference:", obj["name"], "candidates", len(found), flush=True)

    object_by_id = {o["id"]: o for o in targets["objects"]}
    conditions = {}
    for condition, image_root, label in sources:
        source_manifest = manifests[condition]
        scene_records = []
        for target, photo in zip(targets["scenes"], source_manifest["photos"], strict=True):
            index = target["photo_index"]
            if index != photo["photo_index"]:
                raise ValueError("Scene order differs")
            original = image_root / photo["final_path"]
            copied = assets / f"{condition}_{index}_photo.png"
            shutil.copyfile(original, copied)
            image = Image.open(copied).convert("RGB")
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            objects = []
            for oid in target["objects"]:
                obj = object_by_id[oid]
                candidates = infer(original, obj)
                for rank, candidate in enumerate(candidates):
                    candidate["similarity"] = cosine(candidate.pop("embedding"), ref_embeddings[oid])
                    crop_path = assets / f"{condition}_{index}_{oid}_{rank}.png"
                    image.crop(candidate["crop_box"]).save(crop_path)
                    candidate["crop"] = asset(crop_path)
                    draw.rectangle(candidate["crop_box"], outline="#ffcc00" if rank == 0 else "#33aaff", width=3)
                    draw.text((candidate["crop_box"][0], max(0, candidate["crop_box"][1]-14)), f"{obj['name']} #{rank+1}", fill="#ffcc00", stroke_width=1, stroke_fill="black")
                objects.append({"object_id": oid, "name": obj["name"], "query": obj["query"],
                                "expected_count": 1, "requirement": target["requirements"][oid],
                                "detections": candidates, "candidate_count": len(candidates),
                                "selected_index": 0 if candidates else None,
                                "similarity": candidates[0]["similarity"] if candidates else None,
                                "status": "detected" if candidates else "not_detected"})
            overlay_path = assets / f"{condition}_{index}_detections.jpg"
            overlay.save(overlay_path, quality=92)
            scene_records.append({"photo_index": index, "month": target["month"], "image": asset(copied),
                                  "sha256": digest(original), "detections_image": asset(overlay_path), "objects": objects})
            print(condition, "scene", index + 1, ":", [(o["object_id"], o["candidate_count"], None if o["similarity"] is None else round(o["similarity"], 4)) for o in objects], flush=True)
        conditions[condition] = {"label": label, "scenes": scene_records, "summary": summarize(scene_records)}

    reference_matrix = [[cosine(ref_embeddings[a["id"]], ref_embeddings[b["id"]]) for b in targets["objects"]] for a in targets["objects"]]
    blank = Image.new("RGB", (1024, 1024), "#808080")
    blank_results = {obj["id"]: len(models.detect(blank, obj["query"])) for obj in targets["objects"]}
    repeated_ref = targets["objects"][0]["id"]
    repeat_embedding = models.embed(Image.open(experiment / refs[0]["crop"]))
    sanity = {"reference_cosine_matrix": reference_matrix, "reference_order": [o["id"] for o in targets["objects"]],
              "repeated_reference_max_embedding_difference": max(abs(a-b) for a,b in zip(repeat_embedding,ref_embeddings[repeated_ref],strict=True)),
              "blank_image_detection_counts": blank_results,
              "note": "Sanity checks only; different object categories do not calibrate same-instance identity."}
    report = {"schema_version": 1, "method": "Grounding DINO detection + DINOv2 cropped-reference cosine similarity",
              "settings": SETTINGS, "model_provenance": provenance, "packages": packages,
              "source_sha256": targets["source_sha256"], "target_sha256": digest(target_path),
              "reference_sha256": reference_hashes,
              "manifest_sha256": {name: digest(root / "manifest.yaml") for name, root, _ in sources},
              "scorer_sha256": digest(__file__), "references": refs, "conditions": conditions, "sanity_checks": sanity,
              "input_sha256": hashes, "api_calls": 0,
              "limits": "No VQA, human ratings or generative judge. Similarity is not exact-object accuracy. Detection can miss small/occluded objects or confuse categories. Boxes retain background. Location, ownership, counts and fine attributes are not established by cosine similarity."}
    write(out / "scores.json", report)
    print(json.dumps({"summaries": {k: v["summary"] for k,v in conditions.items()}, "sanity": sanity}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.experiment.resolve(), args.models.resolve())
