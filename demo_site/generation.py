"""Local, checkpointed runs around the existing camera-roll pipeline."""

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

import httpx
import yaml
from dotenv import dotenv_values

from ai_camera_roll import CloudflareImageModel, OpenCodeGoLLM
from ai_camera_roll.image_pipeline import _manifest, generate_images
from ai_camera_roll.images import png_bytes
from ai_camera_roll.llm import LLM, LLMResponseError
from ai_camera_roll.models import GenerationResult, LifeStory
from ai_camera_roll.pipeline import generate_photos, generate_story

HERE = Path(__file__).resolve().parent
MAX_IMAGE_CALLS = 40
ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class AppError(Exception):
    def __init__(self, message, code="invalid_request", status=400):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def load_settings():
    """Secrets stay in server memory; only readiness booleans enter API responses."""
    local = dotenv_values(HERE / "private/credentials.env")
    names = ("OPENCODE_GO_API_KEY", "OPENCODE_GO_MODEL", "CLOUDFLARE_API_TOKEN",
             "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_IMAGE_MODEL")
    return {name: os.environ.get(name) or local.get(name) or "" for name in names}


def safe_error(error):
    """Never expose provider exception text, which can contain URLs or credentials."""
    chain = []
    while error and error not in chain:
        chain.append(error)
        error = error.__cause__
    for error in chain:
        if isinstance(error, AppError):
            return error.code, error.message
        if isinstance(error, httpx.HTTPStatusError):
            code = error.response.status_code
            if code == 401:
                if "this api key has expired" in error.response.text[:4096].lower():
                    return "provider_key_expired", "The provider API key has expired. Create a new key in the provider console and replace the local credential; signing in again does not renew it. Completed work is saved."
                return "provider_credentials", "The provider rejected the API credentials. Check or replace the local provider key. Completed work is saved."
            if code == 403:
                return "provider_permissions", "The provider denied permission for this request. Check the account permissions and model access. Completed work is saved."
            if code == 429:
                return "provider_limit", "The provider reached a usage or rate limit. Completed work is saved; no automatic retry was made."
            return "provider_request", f"The provider rejected this request (HTTP {code}). Completed work is saved; no automatic retry or provider switch was made."
        if isinstance(error, (httpx.TimeoutException, TimeoutError)):
            return "provider_timeout", "The provider timed out. The attempt may have been billed. Completed work is saved; no automatic retry was made."
        if isinstance(error, LLMResponseError):
            return "invalid_model_output", "The text model returned an invalid or incomplete result. Completed work is saved; no automatic retry was made."
    return "generation_failed", "This step could not finish. Completed work is saved. Check provider access and the saved run before trying again."


class CurrentOpenCodeGoLLM(OpenCodeGoLLM):
    """Use the provider's current documented endpoint without changing the library."""
    def _complete(self, *, system, prompt):
        response = httpx.post(
            "https://opencode.ai/inference/go/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "User-Agent": "ai-camera-roll/0.1.0", "x-opencode-session": self.session_id},
            json={"model": self.model, "messages": [{"role": "system", "content": system},
                  {"role": "user", "content": prompt}], "max_tokens": self.max_tokens, "stream": False},
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
            reason = choice["finish_reason"]
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise LLMResponseError("Malformed text response.") from error
        if reason != "stop" or not isinstance(content, str) or not content.strip():
            raise LLMResponseError("The text response was incomplete.")
        return content


class RecordedLLM(LLM):
    def __init__(self, job, delegate):
        self.job, self.delegate, self.model = job, delegate, delegate.model
        self.entry = None

    def generate(self, *, system, prompt, response_model):
        request = {"system": system, "prompt": prompt, "schema": response_model.model_json_schema()}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        self.job.last_request_key = key
        cache = self.job.cache
        if key in cache:
            return response_model.model_validate(cache[key])
        self.job.check_before_call()
        try:
            result = super().generate(system=system, prompt=prompt, response_model=response_model)
        except Exception:
            if self.entry and self.entry["status"] == "received":
                self.entry["status"] = "invalid_output"
                self.job.save_trace()
            raise
        cache[key] = result.model_dump(mode="json")
        write_json(self.job.folder / "text_cache.json", cache)
        self.entry.update(status="completed", output=cache[key])
        self.job.state["usage"]["llm_completed"] += 1
        self.job.save_trace()
        self.job.save()
        return result

    def _complete(self, *, system, prompt):
        self.job.check_before_call()
        self.entry = {"kind": "text", "stage": self.job.state["progress"],
                      "started_at": now(), "model": self.model,
                      "request_sha256": self.job.last_request_key,
                      "system": system, "prompt": prompt, "status": "attempted"}
        self.job.trace.append(self.entry)
        self.job.state["usage"]["llm_attempts"] += 1
        self.job.save_trace()
        self.job.save()
        try:
            content = self.delegate._complete(system=system, prompt=prompt)
        except Exception as error:
            self.entry.update(status="failed", error_code=safe_error(error)[0])
            self.job.save_trace()
            raise
        self.entry["status"] = "received"
        self.job.save_trace()
        return content


class RecordedImages:
    def __init__(self, job, delegate):
        self.job, self.delegate, self.model = job, delegate, delegate.model

    def generate(self, prompt, *, references=(), width=1024, height=1024, seed=None):
        self.job.check_stop()
        state = self.job.state
        if state["usage"]["image_attempts"] >= state["max_image_calls"]:
            raise AppError("The run reached its image-request limit. No further request was sent. You can review the saved work and explicitly raise the limit to continue.", "image_budget", 409)
        entry = {"kind": "image", "stage": state["progress"], "started_at": now(),
                 "model": self.model, "prompt": prompt, "seed": seed,
                 "width": width, "height": height, "reference_count": len(references),
                 "status": "attempted"}
        self.job.trace.append(entry)
        state["usage"]["image_attempts"] += 1
        self.job.save_trace()
        self.job.save()
        try:
            content = self.delegate.generate(prompt, references=references, width=width, height=height, seed=seed)
            content = png_bytes(content)
        except Exception as error:
            entry.update(status="failed", error_code=safe_error(error)[0])
            self.job.save_trace()
            raise
        entry["status"] = "completed"
        state["usage"]["image_completed"] += 1
        self.job.save_trace()
        self.job.save()
        return content


class Job:
    def __init__(self, manager, state):
        self.manager, self.state = manager, state
        self.folder = manager.root / state["id"]
        self.trace = read_json(self.folder / "trace.json") if (self.folder / "trace.json").exists() else []
        self.cache = read_json(self.folder / "text_cache.json") if (self.folder / "text_cache.json").exists() else {}
        self.last_request_key = None

    def save(self):
        with self.manager.lock:
            self.state["updated_at"] = now()
            write_json(self.folder / "run.json", self.state)

    def save_trace(self):
        write_json(self.folder / "trace.json", self.trace)

    def check_stop(self):
        if self.state["stop_requested"]:
            raise AppError("Stopped after the current request. Completed work is saved and can be resumed.", "stopped", 409)

    def check_before_call(self):
        self.check_stop()
        if self.state["phase"] == "images" and self.state["usage"]["image_attempts"] >= self.state["max_image_calls"]:
            raise AppError("The run reached its image-request limit. No further request was sent. Review the saved work and explicitly raise the limit to continue.", "image_budget", 409)

    def discard_invalid_response(self):
        # The original pipeline also checks counts after schema validation.
        if self.last_request_key in self.cache:
            del self.cache[self.last_request_key]
            write_json(self.folder / "text_cache.json", self.cache)
            for entry in reversed(self.trace):
                if entry.get("request_sha256") == self.last_request_key and entry["status"] == "completed":
                    entry["status"] = "invalid_output"
                    self.state["usage"]["llm_completed"] -= 1
                    self.save_trace()
                    break

    def progress(self, message):
        if message.startswith("Finished:"):
            message = "All reference images and photographs are saved."
        elif message.startswith("Final photo:"):
            message = "Saved a final photograph."
        elif message.startswith("Saved "):
            message = "Saved an image checkpoint."
        elif message.startswith("Reusing "):
            message = "Reusing a saved image checkpoint."
        self.state["progress"] = message
        self.save()


class RunManager:
    def __init__(self, root=None, settings=None, llm_factory=None, image_factory=None, background=True):
        self.root = Path(root) if root else HERE / "private/runs"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.fixed_settings = settings is not None
        self.settings = settings if settings is not None else load_settings()
        self.models = {
            "text": self.settings.get("OPENCODE_GO_MODEL") or "deepseek-v4.1-flash",
            "image": self.settings.get("CLOUDFLARE_IMAGE_MODEL") or "@cf/black-forest-labs/flux-2-klein-4b",
        }
        self.llm_factory = llm_factory or (lambda model: CurrentOpenCodeGoLLM(model=model, api_key=self.settings["OPENCODE_GO_API_KEY"], timeout=180, max_tokens=8192))
        self.image_factory = image_factory or (lambda model: CloudflareImageModel(model=model, api_token=self.settings["CLOUDFLARE_API_TOKEN"], account_id=self.settings["CLOUDFLARE_ACCOUNT_ID"], timeout=300))
        self.background, self.lock, self.active = background, threading.RLock(), None
        self.thread = None
        for path in self.root.glob("*/run.json"):
            state = read_json(path)
            if state["status"] in ("planning", "generating"):
                state.update(status="interrupted", stop_requested=False, error="The server stopped during this run. Completed work is saved; unfinished work requires an explicit resume.", error_code="interrupted", progress="Run interrupted; saved work is available.")
                write_json(path, state)

    def readiness(self):
        if not self.fixed_settings:
            self.settings = load_settings()
            self.models = {"text": self.settings.get("OPENCODE_GO_MODEL") or "deepseek-v4.1-flash",
                           "image": self.settings.get("CLOUDFLARE_IMAGE_MODEL") or "@cf/black-forest-labs/flux-2-klein-4b"}
        return {"text_ready": bool(self.settings.get("OPENCODE_GO_API_KEY")),
                "images_ready": bool(self.settings.get("CLOUDFLARE_API_TOKEN") and self.settings.get("CLOUDFLARE_ACCOUNT_ID"))}

    def config(self):
        return {"credentials": self.readiness(), "models": self.models,
                "limits": {"years": 1, "max_photos": 5, "max_brief_length": 2000, "max_image_calls": MAX_IMAGE_CALLS},
                "active_run_id": self.active.state["id"] if self.active else None}

    def folder(self, identifier):
        if not ID_PATTERN.fullmatch(identifier):
            raise AppError("Run not found.", "not_found", 404)
        folder = self.root / identifier
        if not (folder / "run.json").is_file():
            raise AppError("Run not found.", "not_found", 404)
        return folder

    def get(self, identifier):
        return read_json(self.folder(identifier) / "run.json")

    def list(self):
        return sorted([read_json(path) for path in self.root.glob("*/run.json")], key=lambda s: s["created_at"], reverse=True)

    def create(self, payload):
        brief, years, count = payload.get("brief"), payload.get("years", 1), payload.get("photos_per_year", 3)
        if not isinstance(brief, str) or not 1 <= len(brief.strip()) <= 2000:
            raise AppError("Write a character brief between 1 and 2,000 characters.")
        if type(years) is not int or years != 1 or type(count) is not int or not 1 <= count <= 5:
            raise AppError("Choose one year and between one and five photographs.")
        if not self.readiness()["text_ready"]:
            raise AppError("Add the text provider key to the local credentials file before creating a story.", "credentials_missing", 409)
        state = {"id": uuid4().hex, "status": "planning", "phase": "planning", "brief": brief.strip(),
                 "years": years, "photos_per_year": count, "name": "New story", "created_at": now(), "updated_at": now(),
                 "models": dict(self.models), "progress": "Writing the story…", "error": None, "error_code": None,
                 "plan": None, "usage": {"llm_attempts": 0, "llm_completed": 0, "image_attempts": 0, "image_completed": 0},
                 "max_image_calls": 0, "stop_requested": False, "data_available": False}
        return self._launch(state, self._plan)

    def resume(self, identifier):
        with self.lock:
            state = self.get(identifier)
            if state["phase"] != "planning" or state["status"] not in ("failed", "interrupted"):
                raise AppError("This run is not waiting for story planning to resume.", "run_state", 409)
            if not self.readiness()["text_ready"]:
                raise AppError("The text provider key is missing.", "credentials_missing", 409)
            state.update(status="planning", progress="Resuming the saved story plan…")
            return self._launch(state, self._plan)

    def generate(self, identifier, payload):
        # Read and launch atomically so overlapping requests cannot restore old usage.
        with self.lock:
            state = self.get(identifier)
            if not state["data_available"] or state["status"] not in ("ready", "failed", "interrupted"):
                raise AppError("This album is not waiting for image generation.", "run_state", 409)
            if not all(self.readiness().values()):
                raise AppError("Both providers must be configured before generating photographs.", "credentials_missing", 409)
            ceiling = payload.get("max_image_calls")
            if type(ceiling) is not int or not 1 <= ceiling <= MAX_IMAGE_CALLS:
                raise AppError(f"Choose a total image-request limit between 1 and {MAX_IMAGE_CALLS}.")
            if ceiling <= state["usage"]["image_attempts"]:
                raise AppError("The total request limit must exceed the attempts already used.", "image_budget", 409)
            if ceiling < state["max_image_calls"]:
                raise AppError("A resumed run must preserve or increase its previous total request limit.", "image_budget", 409)
            state.update(status="generating", phase="images", max_image_calls=ceiling, progress="Preparing reference images…")
            return self._launch(state, self._images)

    def stop(self, identifier):
        with self.lock:
            if not self.active or self.active.state["id"] != identifier:
                raise AppError("This run has no active request to stop.", "run_state", 409)
            self.active.state.update(stop_requested=True, progress="Stopping after the current request finishes…")
            self.active.save()
            return dict(self.active.state)

    def _launch(self, state, work):
        with self.lock:
            if self.active:
                raise AppError("Another run is already in progress. Let it finish or stop it first.", "busy", 409)
            state.update(error=None, error_code=None, stop_requested=False)
            job = Job(self, state)
            job.save()
            self.active = job
        def execute():
            try:
                work(job)
            except Exception as error:
                code, message = safe_error(error)
                if code == "invalid_model_output":
                    job.discard_invalid_response()
                job.state.update(status="interrupted" if code == "stopped" else "failed", error_code=code, error=message, progress=message)
            finally:
                job.save()
                with self.lock:
                    self.active = None
        if self.background:
            self.thread = threading.Thread(target=execute, daemon=True, name="camera-roll-generation")
            self.thread.start()
        else:
            execute()
        return self.get(state["id"])

    def _providers(self, job, images=False):
        llm = RecordedLLM(job, self.llm_factory(job.state["models"]["text"]))
        image_model = RecordedImages(job, self.image_factory(job.state["models"]["image"])) if images else None
        return llm, image_model

    def _plan(self, job):
        llm, _ = self._providers(job)
        story_path = job.folder / "story.json"
        if story_path.exists():
            story = LifeStory.model_validate(read_json(story_path))
        else:
            story = generate_story(job.state["brief"], llm=llm, years=1)
            write_json(story_path, story.model_dump(mode="json"))
        job.check_stop()
        job.progress("Planning photographs and recurring characters…")
        roll = generate_photos(story, llm=llm, photos_per_year=job.state["photos_per_year"])
        result = GenerationResult(story=story, camera_roll=roll)
        write_json(job.folder / "camera_roll.json", result.model_dump(mode="json"))
        # The same wrapper names and configured models are used during actual generation.
        image_identity = type("ImageIdentity", (), {"model": job.state["models"]["image"]})()
        manifest = _manifest(result, llm, RecordedImages(job, image_identity), 42)
        write_json(job.folder / "planned_manifest.json", manifest.model_dump(mode="json"))
        references, passes = len(manifest.references), sum(len(p.passes) for p in manifest.photos)
        protagonist = next(p for p in roll.people if p.id == roll.protagonist_id)
        job.state.update(status="ready", progress="Your story and photo plan are ready. Review them before generating images.",
                         name=protagonist.name, data_available=True,
                         plan={"image_calls": references + passes, "reference_count": references, "photo_count": len(roll.photos), "scene_passes": passes})

    def _images(self, job):
        llm, image_model = self._providers(job, images=True)
        result = GenerationResult.model_validate(read_json(job.folder / "camera_roll.json"))
        generate_images(result, llm=llm, image_model=image_model, output_dir=job.folder / "images", seed=42, progress=job.progress)
        job.state.update(status="completed", stop_requested=False, progress="Your camera roll is ready.")

    def data(self, identifier):
        folder, state = self.folder(identifier), self.get(identifier)
        if not state["data_available"]:
            raise AppError("The story plan is not ready to inspect yet.", "not_ready", 409)
        result = read_json(folder / "camera_roll.json")
        manifest_path = folder / "images/manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else read_json(folder / "planned_manifest.json")
        for photo in manifest["photos"]:
            photo["final_available"] = all(step["completed"] for step in photo["passes"]) and (folder / "images" / photo["final_path"]).is_file()
        templates = {name: files("ai_camera_roll").joinpath("prompts", f"{name}.md").read_text() for name in ("story", "photos", "image_scene", "image_character", "image_object")}
        return {**result, "manifest": manifest, "scores": None, "review": None, "prompt_templates": templates,
                "image_base": f"/runs/{identifier}/images/", "evaluation_base": None, "run": state,
                "trace": read_json(folder / "trace.json") if (folder / "trace.json").exists() else [],
                "provenance": {"source_repository": "https://github.com/aking526/ai-camera-roll", "source_commit": "e1b084164c203f0a87793503aa7ddaebf83f859f",
                    "attribution": "Generated locally through the original ai-camera-roll pipeline. Codex assisted with this website integration.",
                    "prompt_template_note": "Source templates are shown separately from the actual recorded text requests in trace.",
                    "evaluation_note": "No face evaluation or independent ratings have been run on this new album.",
                    "generation_note": "Extra image passes accommodate reference inputs; they are not score-triggered retries."}}

    def image_file(self, identifier, relative):
        folder = self.folder(identifier)
        path = (folder / "images" / relative).resolve()
        if not path.is_relative_to((folder / "images").resolve()) or path.suffix != ".png":
            raise AppError("Image not found.", "not_found", 404)
        manifest_path = folder / "images/manifest.yaml"
        if not manifest_path.exists():
            raise AppError("Image not found.", "not_found", 404)
        manifest = yaml.safe_load(manifest_path.read_text())
        allowed = set()
        for ref in manifest["references"]:
            if ref["image"]["completed"]:
                allowed.update((ref["image"]["path"], ref["reference_path"]))
        for photo in manifest["photos"]:
            allowed.update(step["path"] for step in photo["passes"] if step["completed"])
            if all(step["completed"] for step in photo["passes"]):
                allowed.add(photo["final_path"])
        if relative not in allowed or not path.is_file():
            raise AppError("Image not found.", "not_found", 404)
        return path
