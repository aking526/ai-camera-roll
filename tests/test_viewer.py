import io
import json
from pathlib import Path

import pytest
import yaml

from ai_camera_roll.viewer import make_handler


def request(source: Path, path="/api/result", *, body=None, headers=None):
    """Exercise the actual HTTP handler without binding a socket."""
    class Connection:
        def __init__(self, raw):
            self.input = io.BytesIO(raw)
            self.output = io.BytesIO()

        def makefile(self, *args, **kwargs):
            return self.input

        def sendall(self, content):
            self.output.write(content)

    method = "GET" if body is None else "POST"
    body = b"" if body is None else body
    headers = {"Host": "127.0.0.1:8000", "Content-Length": str(len(body)), **(headers or {})}
    raw = f"{method} {path} HTTP/1.0\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    connection = Connection(raw.encode() + b"\r\n" + body)
    make_handler(source)(connection, ("127.0.0.1", 1234), None)
    head, content = connection.output.getvalue().split(b"\r\n\r\n", 1)
    return int(head.split()[1]), content


@pytest.fixture
def result(story_payload, roll_payload):
    return {"story": {"seed": "Maya", **story_payload(1)}, "camera_roll": roll_payload(1, 2)}


def test_read_saved_result(tmp_path, result):
    source = tmp_path / "roll.yml"
    source.write_text(yaml.safe_dump(result))
    status, body = request(source)
    assert status == 200
    assert json.loads(body) == {"name": "roll.yml", "result": result}


@pytest.mark.parametrize("serialize", [yaml.safe_dump, json.dumps])
def test_preview_does_not_change_source(tmp_path, result, serialize):
    source = tmp_path / "roll.yaml"
    source.write_text("original file")
    status, body = request(source, "/api/preview", body=serialize(result).encode())
    assert status == 200
    assert json.loads(body)["result"] == result
    assert source.read_text() == "original file"


@pytest.mark.parametrize("body", [b"[bad yaml", b"story: {}", b"!!python/object/apply:os.system ['echo unsafe']"])
def test_invalid_upload_is_rejected(tmp_path, body):
    status, content = request(tmp_path / "missing.yaml", "/api/preview", body=body)
    assert status == 400
    assert "error" in json.loads(content)


def test_invalid_catalog_reference_is_rejected(tmp_path, result):
    result["camera_roll"]["photos"][0]["objects"][0]["object_id"] = "missing"
    status, _ = request(tmp_path / "missing.yaml", "/api/preview", body=yaml.safe_dump(result).encode())
    assert status == 400


def test_missing_source_can_still_open_viewer(tmp_path):
    source = tmp_path / "missing.yaml"
    assert request(source)[0] == 404
    status, body = request(source, "/")
    assert status == 200
    assert b"Open YAML" in body


@pytest.mark.parametrize("path", ["/.env", "/../.env", "/src/ai_camera_roll/llm.py"])
def test_only_viewer_assets_are_served(tmp_path, path):
    assert request(tmp_path / "missing.yaml", path)[0] == 404


def test_oversized_upload_is_rejected(tmp_path):
    status, _ = request(tmp_path / "missing.yaml", "/api/preview", body=b"x", headers={"Content-Length": "6000000"})
    assert status == 413


def test_cross_origin_upload_is_rejected(tmp_path):
    status, _ = request(tmp_path / "missing.yaml", "/api/preview", body=b"x", headers={"Origin": "https://example.com"})
    assert status == 403
