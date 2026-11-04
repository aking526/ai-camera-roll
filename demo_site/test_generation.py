"""Exercise real orchestration with fake providers; these tests never use credentials."""

import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from generation import AppError, CurrentOpenCodeGoLLM, RunManager, safe_error, write_json
from server import make_handler


STORY = {"background": "Ada repairs bicycles in Boston.", "years": [{"year": 1, "narrative": "Ada rides beside the river in spring."}]}
ROLL = {"protagonist_id": "ada", "people": [{"id": "ada", "name": "Ada", "relationship": "self", "age_at_start": 25, "visual_description": "Curly hair and glasses."}],
        "objects": [{"id": "bike", "name": "Blue bicycle", "visual_description": "A blue road bicycle.", "significance": "Her daily transport."}],
        "photos": [{"year": 1, "month": 4, "description": "Ada stands beside her blue bicycle by the river.", "story_context": "A spring ride.", "people": [{"person_id": "ada", "appearance": "Standing."}], "objects": [{"object_id": "bike", "appearance": "Parked by Ada."}]}]}
RESPONSES = [STORY, ROLL, {"prompt": "Ada portrait"}, {"prompt": "Blue bicycle"}, {"prompts": ["Ada beside the bicycle"]}]
SETTINGS = {"OPENCODE_GO_API_KEY": "fake-text-secret", "CLOUDFLARE_API_TOKEN": "fake-image-secret", "CLOUDFLARE_ACCOUNT_ID": "fake-account", "OPENCODE_GO_MODEL": "fake-text", "CLOUDFLARE_IMAGE_MODEL": "fake-image"}


class FakeLLM:
    model = "fake-text"
    def __init__(self, responses=RESPONSES):
        self.responses, self.calls = iter(responses), []
    def _complete(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)


class FakeImages:
    model = "fake-image"
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at
    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if len(self.calls) == self.fail_at:
            raise TimeoutError("secret fake-image-secret must not escape")
        stream = io.BytesIO()
        Image.new("RGB", (32, 32), "green").save(stream, format="PNG")
        return stream.getvalue()


def manager(tmp_path, llm=None, images=None, background=False):
    llm, images = llm or FakeLLM(), images or FakeImages()
    return RunManager(tmp_path / "runs", dict(SETTINGS), lambda model: llm, lambda model: images, background), llm, images


def plan(app):
    return app.create({"brief": "Ada repairs bicycles in Boston.", "years": 1, "photos_per_year": 1})


def test_plan_previews_exact_cost_without_image_calls(tmp_path):
    app, llm, images = manager(tmp_path)
    run = plan(app)
    assert run["status"] == "ready" and run["name"] == "Ada"
    assert run["plan"] == {"image_calls": 3, "reference_count": 2, "photo_count": 1, "scene_passes": 1}
    assert len(llm.calls) == 2 and not images.calls
    data = app.data(run["id"])
    assert data["story"]["seed"] == run["brief"] and data["scores"] is None
    assert data["manifest"]["photos"][0]["passes"][0]["inputs"][0]["entity_id"] == "ada"
    assert "JSON schema" in data["trace"][0]["system"]
    assert not data["manifest"]["references"][0]["image"]["completed"]


def test_complete_run_uses_original_pipeline_and_records_every_call(tmp_path):
    app, llm, images = manager(tmp_path)
    run = app.generate(plan(app)["id"], {"max_image_calls": 3})
    assert run["status"] == "completed"
    assert run["usage"] == {"llm_attempts": 5, "llm_completed": 5, "image_attempts": 3, "image_completed": 3}
    data = app.data(run["id"])
    assert len(data["trace"]) == 8
    assert all(r["status"] == "completed" for r in data["trace"])
    source = data["manifest"]["photos"][0]
    assert app.image_file(run["id"], source["final_path"]).read_bytes() == app.image_file(run["id"], source["passes"][-1]["path"]).read_bytes()
    assert len(images.calls[-1][1]["references"]) == 2
    with pytest.raises(AppError):
        app.generate(run["id"], {"max_image_calls": 3})
    assert len(images.calls) == 3


def test_budget_stops_before_next_paid_request_and_resume_reuses_checkpoints(tmp_path):
    app, llm, images = manager(tmp_path)
    run = app.generate(plan(app)["id"], {"max_image_calls": 2})
    assert run["status"] == "failed" and run["error_code"] == "image_budget"
    assert len(images.calls) == 2 and len(llm.calls) == 4
    saved = app.data(run["id"])["manifest"]["references"]
    assert all(r["image"]["completed"] for r in saved)
    run = app.generate(run["id"], {"max_image_calls": 3})
    assert run["status"] == "completed"
    assert len(images.calls) == 3 and len(llm.calls) == 5


def test_failed_image_attempt_counts_toward_budget_and_prompt_is_not_repeated(tmp_path):
    images = FakeImages(fail_at=1)
    app, llm, _ = manager(tmp_path, images=images)
    run = app.generate(plan(app)["id"], {"max_image_calls": 3})
    assert run["usage"]["image_attempts"] == 1 and run["usage"]["image_completed"] == 0
    assert "fake-image-secret" not in json.dumps(run)
    assert len(llm.calls) == 3
    run = app.generate(run["id"], {"max_image_calls": 4})
    assert run["status"] == "completed" and len(images.calls) == 4
    assert len(llm.calls) == 5


def test_overlapping_generate_requests_cannot_restore_stale_usage(tmp_path, monkeypatch):
    images = FakeImages(fail_at=2)
    app, _, _ = manager(tmp_path, images=images)
    identifier = plan(app)["id"]
    original_get = app.get
    snapshot_taken, release_snapshot, other_finished = threading.Event(), threading.Event(), threading.Event()
    held_once = False
    errors = []

    def delayed_get(run_id):
        nonlocal held_once
        state = original_get(run_id)
        if threading.current_thread().name == "delayed-request" and not held_once:
            held_once = True
            snapshot_taken.set()
            assert release_snapshot.wait(5)
        return state

    def generate(finished=None):
        try:
            app.generate(identifier, {"max_image_calls": 3})
        except Exception as error:
            errors.append(error)
        finally:
            if finished:
                finished.set()

    monkeypatch.setattr(app, "get", delayed_get)
    delayed = threading.Thread(target=generate, name="delayed-request")
    other = threading.Thread(target=generate, args=(other_finished,))
    delayed.start()
    assert snapshot_taken.wait(5)
    other.start()
    # Without the lock, this request finishes and the delayed snapshot resets usage.
    other_finished.wait(1)
    release_snapshot.set()
    delayed.join(5)
    other.join(5)
    assert not delayed.is_alive() and not other.is_alive() and not errors
    run = original_get(identifier)
    assert len(images.calls) == run["usage"]["image_attempts"] == 3
    assert run["status"] == "failed" and run["error_code"] == "image_budget"


def test_failed_photo_plan_keeps_successful_story(tmp_path):
    llm = FakeLLM([STORY, TimeoutError("secret"), ROLL])
    app, _, _ = manager(tmp_path, llm=llm)
    run = plan(app)
    assert run["status"] == "failed" and len(llm.calls) == 2
    run = app.resume(run["id"])
    assert run["status"] == "ready" and len(llm.calls) == 3
    assert run["usage"]["llm_attempts"] == 3


def test_wrong_photo_count_does_not_poison_retry_cache(tmp_path):
    wrong = {**ROLL, "photos": ROLL["photos"] * 2}
    llm = FakeLLM([STORY, wrong, ROLL])
    app, _, _ = manager(tmp_path, llm=llm)
    run = plan(app)
    assert run["status"] == "failed" and run["error_code"] == "invalid_model_output"
    run = app.resume(run["id"])
    assert run["status"] == "ready" and len(llm.calls) == 3


def test_wrong_scene_pass_count_can_be_retried_without_regenerating_references(tmp_path):
    llm = FakeLLM(RESPONSES[:4] + [{"prompts": ["One", "Too many"]}, RESPONSES[4]])
    app, _, images = manager(tmp_path, llm=llm)
    run = app.generate(plan(app)["id"], {"max_image_calls": 3})
    assert run["status"] == "failed" and run["error_code"] == "invalid_model_output"
    assert len(images.calls) == 2
    run = app.generate(run["id"], {"max_image_calls": 3})
    assert run["status"] == "completed" and len(images.calls) == 3 and len(llm.calls) == 6


def test_restart_marks_work_interrupted_without_issuing_requests(tmp_path):
    app, llm, images = manager(tmp_path)
    run = plan(app)
    run.update(status="generating", phase="images", max_image_calls=3)
    write_json(app.folder(run["id"]) / "run.json", run)
    restarted, _, _ = manager(tmp_path, llm=llm, images=images)
    assert restarted.get(run["id"])["status"] == "interrupted"
    assert len(llm.calls) == 2 and not images.calls


def test_single_flight_and_stop_after_current_request(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class SlowLLM(FakeLLM):
        def _complete(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return super()._complete(**kwargs)
    llm = SlowLLM()
    app, _, images = manager(tmp_path, llm=llm, background=True)
    run = plan(app)
    assert entered.wait(5)
    with pytest.raises(AppError, match="already in progress"):
        plan(app)
    app.stop(run["id"])
    release.set()
    app.thread.join(5)
    assert not app.thread.is_alive()
    run = app.get(run["id"])
    assert run["status"] == "interrupted" and run["error_code"] == "stopped"
    assert len(llm.calls) == 1 and not images.calls
    run = app.resume(run["id"])
    app.thread.join(5)
    assert app.get(run["id"])["status"] == "ready" and len(llm.calls) == 2


@pytest.mark.parametrize("payload", [{"brief": ""}, {"brief": "x" * 2001}, {"brief": "Ada", "years": True}, {"brief": "Ada", "years": 2}, {"brief": "Ada", "photos_per_year": 0}, {"brief": "Ada", "photos_per_year": 6}, {"brief": "Ada", "photos_per_year": True}])
def test_invalid_plans_make_no_calls(tmp_path, payload):
    app, llm, images = manager(tmp_path)
    with pytest.raises(AppError):
        app.create(payload)
    assert not llm.calls and not images.calls and not app.list()


def request(app, path, body=None, headers=None, static_root=None):
    class Connection:
        def __init__(self, raw):
            self.input, self.output = io.BytesIO(raw), io.BytesIO()
        def makefile(self, *args, **kwargs):
            return self.input
        def sendall(self, content):
            self.output.write(content)
    method, raw_body = ("GET", b"") if body is None else ("POST", json.dumps(body).encode())
    supplied = {"Host": "127.0.0.1:5812", "Content-Length": str(len(raw_body)), "Content-Type": "application/json", "X-Demo-Token": "test-token", **(headers or {})}
    raw = f"{method} {path} HTTP/1.0\r\n" + "".join(f"{key}: {value}\r\n" for key, value in supplied.items()) + "\r\n"
    connection = Connection(raw.encode() + raw_body)
    make_handler(app, "test-token", static_root)(connection, ("127.0.0.1", 1234), SimpleNamespace(server_address=("127.0.0.1", 5812)))
    head, content = connection.output.getvalue().split(b"\r\n\r\n", 1)
    return int(head.split()[1]), content


@pytest.mark.parametrize("headers", [{"Origin": "https://evil.example"}, {"Origin": "null"}, {"Host": "evil.example:5812"}, {"X-Demo-Token": "wrong"}, {"Content-Type": "text/plain"}])
def test_generation_api_rejects_cross_site_or_untrusted_requests(tmp_path, headers):
    app, llm, images = manager(tmp_path)
    status, _ = request(app, "/api/runs", {"brief": "Ada"}, headers)
    assert status in (403, 415)
    assert not llm.calls and not images.calls


def test_http_end_to_end_and_secret_safe_api(tmp_path):
    app, _, _ = manager(tmp_path)
    status, content = request(app, "/api/config")
    assert status == 200 and json.loads(content)["csrf_token"] == "test-token"
    assert b"fake-text-secret" not in content and b"fake-image-secret" not in content
    status, content = request(app, "/api/runs", {"brief": "Ada", "photos_per_year": 1})
    run = json.loads(content)["run"]
    assert status == 202 and run["status"] == "ready"
    assert request(app, f"/api/runs/{run['id']}/generate", {"max_image_calls": 3})[0] == 202
    status, content = request(app, f"/api/runs/{run['id']}/data")
    album = json.loads(content)
    assert status == 200 and album["run"]["status"] == "completed"
    assert b"fake-text-secret" not in content and b"fake-image-secret" not in content
    assert request(app, album["image_base"] + album["manifest"]["photos"][0]["final_path"])[0] == 200
    assert request(app, f"/runs/{run['id']}/images/../run.json")[0] == 404
    assert request(app, f"/runs/{run['id']}/images/%2e%2e%2frun.json")[0] == 404


@pytest.mark.parametrize("path", ["/../private/credentials.env", "/%2e%2e/private/credentials.env", "/.env", "/api/runs/../../private", "/private/runs", "/assets/../../private"])
def test_static_and_run_paths_cannot_expose_private_files(tmp_path, path):
    app, _, _ = manager(tmp_path)
    assert request(app, path)[0] == 404


def test_current_text_endpoint_preserves_honest_identity_and_model(monkeypatch):
    seen = []
    def fake_post(url, **kwargs):
        seen.append((url, kwargs))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}, request=httpx.Request("POST", url))
    monkeypatch.setattr("generation.httpx.post", fake_post)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    llm = CurrentOpenCodeGoLLM("fake-model", api_key="fake-only")
    assert llm._complete(system="system", prompt="prompt") == "{}"
    assert seen[0][0] == "https://opencode.ai/inference/go/openai/v1/chat/completions"
    assert seen[0][1]["headers"]["User-Agent"] == "ai-camera-roll/0.1.0"
    assert seen[0][1]["json"]["model"] == "fake-model"


@pytest.mark.parametrize("status,body,expected", [
    (401, "This API key has expired. Bearer super-secret", "provider_key_expired"),
    (401, "Invalid API key: super-secret", "provider_credentials"),
    (403, "This API key has expired. Bearer super-secret", "provider_permissions"),
    (500, "This API key has expired. Bearer super-secret", "provider_request"),
])
def test_provider_error_text_never_leaks_secrets(status, body, expected):
    request_obj = httpx.Request("POST", "https://example.com/secret-account")
    response = httpx.Response(status, json={"error": {"message": body}}, request=request_obj)
    error = httpx.HTTPStatusError("Bearer super-secret", request=request_obj, response=response)
    code, message = safe_error(error)
    assert code == expected
    assert "secret" not in message and "example.com" not in message and "Bearer" not in message
    if expected == "provider_key_expired":
        assert "signing in again does not renew it" in message
