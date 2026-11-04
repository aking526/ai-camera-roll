"""Paired full-album experiment using frozen prompts/assets and the real pipeline.

Preparation is offline. --run explicitly authorizes the bounded Cloudflare run.
Completed requests are checkpointed; failed requests are never retried implicitly.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml
from dotenv import dotenv_values

from ai_camera_roll.image_pipeline import _manifest, generate_images
from ai_camera_roll.images import CloudflareImageModel, ImageModel, png_bytes
from ai_camera_roll.llm import LLM
from ai_camera_roll.models import GenerationResult


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class FrozenPrompts(LLM):
    def __init__(self, model):
        self.model = model

    def _complete(self, **kwargs):
        raise RuntimeError("This experiment forbids text calls; all prompts must be frozen.")


class RecordedImages(ImageModel):
    def __init__(self, model, folder, delegate=None, secrets=()):
        self.model, self.folder, self.delegate = model, folder, delegate
        self.secrets = [s for s in secrets if s]
        self.condition = None
        self.ledger_path = folder / "requests.json"
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else []

    def generate(self, prompt, *, references=(), width=1024, height=1024, seed=None):
        if self.delegate is None:
            raise RuntimeError("Offline preparation cannot generate images.")
        plan = json.loads((self.folder / "plan.json").read_text())
        if len(self.ledger) >= plan["maximum_image_attempts"]:
            raise RuntimeError("Experiment image-attempt budget exhausted.")
        if any(item["status"] != "completed" for item in self.ledger):
            raise RuntimeError("A previous request failed or was interrupted; inspect requests.json before any retry.")
        entry = {"condition": self.condition, "started_at": datetime.now(timezone.utc).isoformat(),
                 "model": self.model, "seed": seed, "width": width, "height": height,
                 "prompt_sha256": digest(prompt.encode()),
                 "input_sha256": [digest(png_bytes(value, reference=True)) for value in references],
                 "status": "attempted"}
        self.ledger.append(entry)
        save_json(self.ledger_path, self.ledger)
        started = time.perf_counter()
        try:
            result = self.delegate.generate(prompt, references=references, width=width, height=height, seed=seed)
            result = png_bytes(result)
        except Exception as error:
            details = {"type": type(error).__name__}
            if isinstance(error, httpx.HTTPStatusError):
                details["http_status"] = error.response.status_code
                details["retry_after"] = error.response.headers.get("Retry-After")
                try:
                    payload = error.response.json()
                    details["provider_errors"] = [
                        {"code": item.get("code"), "message": str(item.get("message", ""))[:800]}
                        for item in payload.get("errors", []) if isinstance(item, dict)
                    ]
                except (ValueError, AttributeError):
                    details["provider_errors"] = []
            encoded = json.dumps(details)
            for secret in self.secrets:
                encoded = encoded.replace(secret, "[redacted]")
            entry.update(status="failed", error=json.loads(encoded), seconds=time.perf_counter()-started)
            save_json(self.ledger_path, self.ledger)
            raise RuntimeError("Cloudflare request failed: " + encoded) from None
        entry.update(status="completed", seconds=time.perf_counter()-started, output_sha256=digest(result))
        save_json(self.ledger_path, self.ledger)
        return result


def prepare(folder, source, source_images):
    original = yaml.safe_load((source_images / "manifest.yaml").read_text())
    result = GenerationResult.model_validate(yaml.safe_load(source.read_text()))
    frozen = folder / "frozen_source.yaml"
    folder.mkdir(parents=True, exist_ok=True)
    if frozen.exists() and frozen.read_bytes() != source.read_bytes():
        raise ValueError("Source changed since preparation.")
    if not frozen.exists():
        shutil.copy2(source, frozen)
    llm = FrozenPrompts(original["settings"]["llm_model"])
    model = RecordedImages(original["settings"]["image_model"], folder)
    changes = []
    original_assets = {asset["entity_id"]: asset for asset in original["references"]}
    hashes = {"source": digest(source.read_bytes()), "manifest": digest((source_images / "manifest.yaml").read_bytes())}
    protagonist = result.camera_roll.protagonist_id
    person = next(p for p in result.camera_roll.people if p.id == protagonist)
    plans = {}
    for condition, retain in (("control", False), ("portrait_reference", True)):
        root = folder / condition / "images"
        expected = _manifest(result, llm, model, original["settings"]["seed"], retain_protagonist_reference=retain)
        if expected.settings["source_sha256"] != original["settings"]["source_sha256"]:
            raise ValueError("Parsed source differs from original generation.")
        for asset in expected.references:
            saved = original_assets[asset.entity_id]
            if asset.image.seed != saved["image"]["seed"] or asset.image.inputs:
                raise ValueError("Reference generation settings changed.")
            asset.image.prompt = saved["image"]["prompt"]
            asset.image.completed = True
            for relative in (saved["image"]["path"], saved["reference_path"]):
                content = (source_images / relative).read_bytes()
                hashes[relative] = digest(content)
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != content:
                    raise ValueError("Frozen reference changed: " + relative)
                if not target.exists():
                    target.write_bytes(content)
        for photo, saved in zip(expected.photos, original["photos"], strict=True):
            if len(photo.passes) != len(saved["passes"]):
                raise ValueError("This paired replay requires unchanged pass counts.")
            for index, (step, old) in enumerate(zip(photo.passes, saved["passes"], strict=True)):
                if step.seed != old["seed"] or step.path != old["path"]:
                    raise ValueError("Scene seed or path changed.")
                step.prompt = old["prompt"]
                baseline_inputs = old["inputs"]
                inputs = [item.model_dump() for item in step.inputs]
                if inputs != baseline_inputs:
                    if not (retain and index and inputs[:-1] == baseline_inputs and inputs[-1]["entity_id"] == protagonist):
                        raise ValueError("Only appending the protagonist reference is permitted.")
                    suffix = (f" Image {inputs[-1]['slot']} is the canonical portrait of {person.name}, "
                              "who is already present in image 0. Use it as the facial identity reference "
                              "for that existing person while carrying out the edit above; do not add "
                              "another person or change the requested scene.")
                    step.prompt += suffix
                    changes.append({"photo_index": photo.photo_index, "pass_index": index,
                                    "added_slot": inputs[-1]["slot"], "entity_id": protagonist,
                                    "prompt_appendix": suffix.strip()})
        manifest = root / "manifest.yaml"
        if not manifest.exists():
            manifest.write_text(yaml.safe_dump(expected.model_dump(mode="json"), allow_unicode=True, sort_keys=False, width=100))
        else:
            saved = yaml.safe_load(manifest.read_text())
            # Completed states may change, but experiment prompts/settings/input mapping must not.
            wanted = expected.model_dump(mode="json")
            for value in (saved, wanted):
                for asset in value["references"]:
                    asset["image"].pop("completed")
                for photo in value["photos"]:
                    for step in photo["passes"]:
                        step.pop("completed")
            if saved != wanted:
                raise ValueError("Prepared experiment manifest changed.")
        plans[condition] = expected
    plan = {"hypothesis": "Resupplying the protagonist portrait during later edits preserves identity.",
            "source_sha256": hashes, "protagonist_id": protagonist,
            "model": model.model, "source_settings": original["settings"],
            "photos_per_condition": len(original["photos"]),
            "maximum_image_attempts": sum(len(p["passes"]) for p in original["photos"]) + len(changes),
            "changes": changes, "text_calls": 0, "reference_generation_calls": 0,
            "comparison": "Fresh control and portrait-reference albums share identical newly generated first passes. Single-pass scenes are shared controls, not treatment improvements.",
            "evaluation": "Unchanged SCRFD/ArcFace scorer, all expected characters and missing assignments reported. Human identity and scene/object review remain required."}
    existing = folder / "plan.json"
    if existing.exists() and json.loads(existing.read_text()) != plan:
        raise ValueError("Frozen experiment plan changed.")
    save_json(existing, plan)
    return result, llm, model, plan


def run(folder, result, llm, model, plan, credentials):
    settings = dotenv_values(credentials)
    names = ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID")
    if not all(settings.get(name) for name in names):
        raise ValueError("Missing Cloudflare token or account ID in the specified private credentials file.")
    model.delegate = CloudflareImageModel(model=model.model, api_token=settings[names[0]], account_id=settings[names[1]])
    model.secrets = [settings[name] for name in names]
    seed = plan["source_settings"]["seed"]
    model.condition = "control"
    control = generate_images(result, llm=llm, image_model=model, output_dir=folder/"control/images", seed=seed)
    treatment_path = folder / "portrait_reference/images/manifest.yaml"
    treatment = yaml.safe_load(treatment_path.read_text())
    for photo, saved in zip(treatment["photos"], control.photos, strict=True):
        step = photo["passes"][0]
        source = folder / "control/images" / saved.passes[0].path
        target = folder / "portrait_reference/images" / step["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("Shared first-pass checkpoint differs.")
        if not target.exists():
            shutil.copy2(source, target)
        step["completed"] = True
    treatment_path.write_text(yaml.safe_dump(treatment, allow_unicode=True, sort_keys=False, width=100))
    model.condition = "portrait_reference"
    generate_images(result, llm=llm, image_model=model, output_dir=folder/"portrait_reference/images", seed=seed, retain_protagonist_reference=True)
    save_json(folder/"completion.json", {"status": "generated", "image_attempts": len(model.ledger), "text_calls": 0, "reference_generation_calls": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/camera_roll.yaml"))
    parser.add_argument("--source-images", type=Path, default=Path("output/images"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        result, llm, model, plan = prepare(args.output, args.source, args.source_images)
        print(json.dumps({"photos_per_condition": plan["photos_per_condition"], "image_attempt_limit": plan["maximum_image_attempts"], "modified_passes": len(plan["changes"]), "text_calls": 0}), flush=True)
        if args.run:
            if not args.credentials_file:
                raise ValueError("--run requires a private --credentials-file.")
            run(args.output, result, llm, model, plan, args.credentials_file)
    except Exception as error:
        # Provider exceptions are replaced by RecordedImages before reaching here.
        print(f"STOPPED: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
