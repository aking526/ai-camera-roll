import json
import os
import runpy
from contextlib import ExitStack
from pathlib import Path

import httpx
import pytest
import yaml

from ai_camera_roll import GenerationResult, LLMResponseError, OpenCodeGoLLM, StoryYear


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in ("OPENCODE_GO_API_KEY", "OPENCODE_GO_MODEL", "PYTHON_DOTENV_DISABLED"):
        monkeypatch.delenv(name, raising=False)


def completion(content='{"year": 1, "narrative": "Maya cycles."}', reason="stop"):
    return httpx.Response(
        200, json={"choices": [{"finish_reason": reason, "message": {"content": content}}]}
    )


@pytest.fixture
def mock_go(monkeypatch):
    with ExitStack() as stack:
        def install(responses):
            responses = iter(responses)
            requests = []

            def handle(request):
                requests.append(request)
                response = next(responses)
                if isinstance(response, Exception):
                    raise response
                return response

            client = stack.enter_context(httpx.Client(transport=httpx.MockTransport(handle)))
            monkeypatch.setattr("ai_camera_roll.llm.httpx.post", client.post)
            return requests

        yield install


def generate(llm):
    return llm.generate(system="Write a story.", prompt="Maya cycles.", response_model=StoryYear)


def test_request_format_and_stable_session(mock_go):
    requests = mock_go([completion(), completion()])
    llm = OpenCodeGoLLM("test-model", api_key="test-key", timeout=42, max_tokens=1234)
    for _ in range(2):
        assert generate(llm).year == 1
    assert len(requests) == 2
    for request in requests:
        assert str(request.url) == "https://opencode.ai/zen/go/v1/chat/completions"
        assert request.method == "POST"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["user-agent"] == "ai-camera-roll/0.1.0"
        assert request.headers["x-opencode-session"] == llm.session_id
        assert request.extensions["timeout"]["read"] == 42
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["max_tokens"] == 1234
        assert body["stream"] is False
        assert body["messages"][0]["role"] == "system"
        assert "JSON schema:" in body["messages"][0]["content"]
        assert body["messages"][1] == {"role": "user", "content": "Maya cycles."}
    other = OpenCodeGoLLM("test-model", api_key="test-key")
    assert other.session_id != llm.session_id


def test_api_key_from_environment(monkeypatch, mock_go):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "environment-key")
    requests = mock_go([completion()])
    generate(OpenCodeGoLLM("test-model"))
    assert requests[0].headers["authorization"] == "Bearer environment-key"


def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENCODE_GO_API_KEY"):
        OpenCodeGoLLM("test-model")


@pytest.mark.parametrize(
    "environment_key,explicit_key,expected",
    [
        (None, None, "file-key"),
        ("environment-key", None, "environment-key"),
        ("environment-key", "explicit-key", "explicit-key"),
    ],
)
def test_dotenv_key_and_precedence(environment_key, explicit_key, expected, monkeypatch, mock_go):
    Path(".env").write_text("OPENCODE_GO_API_KEY=file-key\n", encoding="utf-8")
    if environment_key is not None:
        monkeypatch.setenv("OPENCODE_GO_API_KEY", environment_key)
    requests = mock_go([completion()])
    generate(OpenCodeGoLLM("test-model", api_key=explicit_key))
    assert requests[0].headers["authorization"] == f"Bearer {expected}"


def test_example_loads_key_and_model_from_dotenv(mock_go, story_payload, roll_payload):
    Path(".env").write_text(
        "OPENCODE_GO_API_KEY=file-key\nOPENCODE_GO_MODEL=file-model\n", encoding="utf-8"
    )
    requests = mock_go([
        completion(json.dumps(story_payload())),
        completion(json.dumps(roll_payload())),
    ])
    example = Path(__file__).resolve().parents[1] / "examples" / "generate.py"
    runpy.run_path(str(example), run_name="__main__")
    assert len(requests) == 2
    for request in requests:
        assert request.headers["authorization"] == "Bearer file-key"
        assert json.loads(request.content)["model"] == "file-model"
    result = yaml.safe_load(Path("output/camera_roll.yaml").read_text(encoding="utf-8"))
    assert len(result["camera_roll"]["photos"]) == 50
    validated = GenerationResult.model_validate(result)
    assert validated.camera_roll.model_dump() == roll_payload()
    assert validated.story.background == story_payload()["background"]


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_http_error_propagates_without_retry(status, mock_go):
    requests = mock_go([httpx.Response(status, json={"error": "Service rejected request."})])
    with pytest.raises(httpx.HTTPStatusError) as error:
        generate(OpenCodeGoLLM("test-model", api_key="test-key"))
    assert error.value.response.status_code == status
    assert error.value.response.json()["error"] == "Service rejected request."
    assert len(requests) == 1


def test_timeout_propagates_without_retry(mock_go):
    requests = mock_go([httpx.ReadTimeout("Timed out")])
    with pytest.raises(httpx.ReadTimeout):
        generate(OpenCodeGoLLM("test-model", api_key="test-key"))
    assert len(requests) == 1


@pytest.mark.parametrize(
    "response,message",
    [
        (completion(reason="length"), "truncated"),
        (completion(reason="content_filter"), "content_filter"),
        (completion(content=None), "no text"),
        (completion(content=""), "no text"),
        (httpx.Response(200, text="not JSON"), "Malformed"),
        (httpx.Response(200, json={"choices": []}), "Malformed"),
        (completion(content="not JSON"), "Invalid StoryYear"),
    ],
)
def test_invalid_completion(response, message, mock_go):
    requests = mock_go([response])
    with pytest.raises(LLMResponseError, match=message):
        generate(OpenCodeGoLLM("test-model", api_key="test-key"))
    assert len(requests) == 1
