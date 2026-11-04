"""Offline replay checks against the real saved five-scene experiment inputs."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import httpx
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "portrait_experiment", ROOT / "examples/compare_portrait_reference.py"
)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def hashes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class FakeImages:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, *, references=(), width=1024, height=1024, seed=None):
        self.calls.append({"prompt": prompt, "references": references, "seed": seed})
        fingerprint = hashlib.sha256(prompt.encode() + str(seed).encode()).digest()
        stream = io.BytesIO()
        Image.new("RGB", (width, height), tuple(fingerprint[:3])).save(stream, format="PNG")
        return stream.getvalue()


def test_full_saved_album_replay_is_paired_bounded_and_resumable(tmp_path, monkeypatch):
    source = ROOT / "output/camera_roll.yaml"
    source_images = ROOT / "output/images"
    original_paths = [source, *(p for p in source_images.rglob("*") if p.is_file())]
    original_hashes = hashes(original_paths)
    baseline = yaml.safe_load((source_images / "manifest.yaml").read_text())
    folder = tmp_path / "experiment"
    result, llm, model, plan = experiment.prepare(folder, source, source_images)
    assert plan["photos_per_condition"] == 5
    assert plan["maximum_image_attempts"] == 13
    assert plan["text_calls"] == plan["reference_generation_calls"] == 0
    assert len(plan["changes"]) == 4

    control = yaml.safe_load((folder / "control/images/manifest.yaml").read_text())
    treatment = yaml.safe_load((folder / "portrait_reference/images/manifest.yaml").read_text())
    for old_photo, a_photo, b_photo in zip(baseline["photos"], control["photos"], treatment["photos"], strict=True):
        for index, (old, a, b) in enumerate(zip(old_photo["passes"], a_photo["passes"], b_photo["passes"], strict=True)):
            assert a["prompt"] == old["prompt"]
            assert a["inputs"] == old["inputs"]
            assert a["seed"] == b["seed"] == old["seed"]
            if index == 0:
                assert a == b
            else:
                assert b["inputs"][:-1] == a["inputs"]
                assert b["inputs"][-1]["entity_id"] == "person_maya"
                assert len(b["inputs"]) <= 4
                assert b["prompt"].startswith(a["prompt"] + " ")
                appendix = b["prompt"][len(a["prompt"]):]
                assert "canonical portrait of Maya" in appendix
                assert "do not add another person" in appendix
                assert b["prompt"] == a["prompt"] + " " + next(
                    c["prompt_appendix"] for c in plan["changes"]
                    if c["photo_index"] == b_photo["photo_index"] and c["pass_index"] == index
                )

    fake = FakeImages()
    monkeypatch.setattr(experiment, "CloudflareImageModel", lambda **kwargs: fake)
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("Network calls are forbidden in offline tests"))
    credentials = tmp_path / "fake.env"
    credentials.write_text("CLOUDFLARE_API_TOKEN=dummy-test-token\nCLOUDFLARE_ACCOUNT_ID=dummy-test-account\n")
    experiment.run(folder, result, llm, model, plan, credentials)
    assert len(fake.calls) == 13
    ledger = json.loads((folder / "requests.json").read_text())
    assert len(ledger) == 13
    assert all(item["status"] == "completed" for item in ledger)
    assert [item["condition"] for item in ledger] == ["control"] * 9 + ["portrait_reference"] * 4
    for photo in control["photos"]:
        first = photo["passes"][0]["path"]
        assert (folder / "control/images" / first).read_bytes() == (folder / "portrait_reference/images" / first).read_bytes()
        for condition in ("control", "portrait_reference"):
            final = folder / condition / "images" / photo["final_path"]
            assert final.is_file()
            assert Image.open(final).size == (1024, 1024)
        if photo["month"] == 5:
            assert (folder / "control/images" / photo["final_path"]).read_bytes() == (folder / "portrait_reference/images" / photo["final_path"]).read_bytes()

    result, llm, model, plan = experiment.prepare(folder, source, source_images)
    experiment.run(folder, result, llm, model, plan, credentials)
    assert len(fake.calls) == 13
    assert len(json.loads((folder / "requests.json").read_text())) == 13
    assert hashes(original_paths) == original_hashes


def test_failed_request_is_recorded_redacted_and_blocks_retry(tmp_path):
    token, account = "dummy-secret-token", "dummy-secret-account"
    experiment.save_json(tmp_path / "plan.json", {"maximum_image_attempts": 13})

    class RejectedImages:
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            request = httpx.Request("POST", f"https://example.invalid/{account}", headers={"Authorization": f"Bearer {token}"})
            response = httpx.Response(429, request=request, headers={"Retry-After": "30"},
                                      json={"errors": [{"code": 999, "message": f"Limit for {account}; token {token}"}]})
            raise httpx.HTTPStatusError(f"Raw exception containing {token}", request=request, response=response)

    delegate = RejectedImages()
    model = experiment.RecordedImages("fake-model", tmp_path, delegate, secrets=(token, account))
    model.condition = "control"
    with pytest.raises(RuntimeError, match="Cloudflare request failed") as failure:
        model.generate("Test prompt", seed=1)
    saved = (tmp_path / "requests.json").read_text()
    assert token not in saved + str(failure.value)
    assert account not in saved + str(failure.value)
    assert "[redacted]" in saved
    row = json.loads(saved)[0]
    assert row["status"] == "failed"
    assert row["error"]["http_status"] == 429
    assert row["error"]["provider_errors"][0]["code"] == 999
    assert row["error"]["retry_after"] == "30"
    with pytest.raises(RuntimeError, match="previous request failed"):
        model.generate("Test prompt", seed=1)
    restarted = experiment.RecordedImages("fake-model", tmp_path, delegate)
    with pytest.raises(RuntimeError, match="previous request failed"):
        restarted.generate("Test prompt", seed=1)
    assert delegate.calls == 1
    assert len(json.loads((tmp_path / "requests.json").read_text())) == 1
