import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from ai_camera_roll.image_pipeline import _manifest
from ai_camera_roll.models import GenerationResult
from ai_camera_roll.viewer import make_handler


def request(
    source: Path, path="/api/result", *, body=None, headers=None, images_dir=None
):
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
    headers = {
        "Host": "127.0.0.1:8000",
        "Content-Length": str(len(body)),
        **(headers or {}),
    }
    raw = f"{method} {path} HTTP/1.0\r\n" + "".join(
        f"{k}: {v}\r\n" for k, v in headers.items()
    )
    connection = Connection(raw.encode() + b"\r\n" + body)
    make_handler(source, images_dir)(connection, ("127.0.0.1", 1234), None)
    head, content = connection.output.getvalue().split(b"\r\n\r\n", 1)
    return int(head.split()[1]), content


@pytest.fixture
def result(story_payload, roll_payload):
    return {
        "story": {"seed": "Maya", **story_payload(1)},
        "camera_roll": roll_payload(1, 2),
    }


def test_read_saved_result(tmp_path, result):
    source = tmp_path / "roll.yml"
    source.write_text(yaml.safe_dump(result))
    status, body = request(source)
    assert status == 200
    payload = json.loads(body)
    assert payload["name"] == "roll.yml"
    assert payload["result"] == result
    assert payload["images"]["photos"] == []
    assert payload["images"]["references"] == []
    assert "No generated images" in payload["images"]["notice"]


@pytest.mark.parametrize("serialize", [yaml.safe_dump, json.dumps])
def test_preview_does_not_change_source(tmp_path, result, serialize):
    source = tmp_path / "roll.yaml"
    source.write_text("original file")
    status, body = request(source, "/api/preview", body=serialize(result).encode())
    assert status == 200
    assert json.loads(body)["result"] == result
    assert source.read_text() == "original file"


@pytest.mark.parametrize(
    "body",
    [b"[bad yaml", b"story: {}", b"!!python/object/apply:os.system ['echo unsafe']"],
)
def test_invalid_upload_is_rejected(tmp_path, body):
    status, content = request(tmp_path / "missing.yaml", "/api/preview", body=body)
    assert status == 400
    assert "error" in json.loads(content)


def test_invalid_catalog_reference_is_rejected(tmp_path, result):
    result["camera_roll"]["photos"][0]["objects"][0]["object_id"] = "missing"
    status, _ = request(
        tmp_path / "missing.yaml", "/api/preview", body=yaml.safe_dump(result).encode()
    )
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
    status, _ = request(
        tmp_path / "missing.yaml",
        "/api/preview",
        body=b"x",
        headers={"Content-Length": "6000000"},
    )
    assert status == 413


def test_cross_origin_upload_is_rejected(tmp_path):
    status, _ = request(
        tmp_path / "missing.yaml",
        "/api/preview",
        body=b"x",
        headers={"Origin": "https://example.com"},
    )
    assert status == 403


def save_manifest(root, manifest):
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"))
    )


@pytest.fixture
def image_run(tmp_path, result):
    roll = result["camera_roll"]
    roll["objects"].extend(
        {**roll["objects"][0], "id": f"object_{i}"} for i in range(1, 4)
    )
    roll["photos"][0]["people"] = [
        {"person_id": "person_maya", "appearance": "Standing nearby."}
    ]
    roll["photos"][0]["objects"] = [
        {"object_id": item["id"], "appearance": "On the bench."}
        for item in roll["objects"]
    ]
    source = tmp_path / "roll.yaml"
    source.write_text(yaml.safe_dump(result))
    manifest = _manifest(
        GenerationResult.model_validate(result),
        SimpleNamespace(model="test-llm"),
        SimpleNamespace(model="test-images"),
        42,
    )
    root = tmp_path / "images"
    artifacts = [asset.image for asset in manifest.references] + [
        step for photo in manifest.photos for step in photo.passes
    ]
    for index, artifact in enumerate(artifacts):
        artifact.completed = True
        path = root / artifact.path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (index, 20, 40)).save(path)
    for photo in manifest.photos:
        (root / photo.final_path).write_bytes(
            (root / photo.passes[-1].path).read_bytes()
        )
    save_manifest(root, manifest)
    return source, root, manifest


def test_image_manifest_maps_photos_passes_and_canonical_references(image_run):
    source, root, manifest = image_run
    status, body = request(source)
    assert status == 200
    images = json.loads(body)["images"]
    assert images["notice"] is None
    assert [len(photo["passes"]) for photo in images["photos"]] == [2, 1]
    assert images["photos"][0] == {
        "photo_index": 0,
        "complete": True,
        "final_url": "/images/" + manifest.photos[0].final_path,
        "passes": [
            {"pass_index": i, "completed": True, "url": "/images/" + step.path}
            for i, step in enumerate(manifest.photos[0].passes)
        ],
    }
    assert images["references"] == [
        {
            "kind": asset.kind,
            "entity_id": asset.entity_id,
            "completed": True,
            "url": "/images/" + asset.image.path,
        }
        for asset in manifest.references
    ]
    # Only the canonical reference is exposed, not the padded provider input.
    asset = manifest.references[0]
    (root / asset.reference_path).write_bytes((root / asset.image.path).read_bytes())
    assert request(source, "/images/" + asset.reference_path)[0] == 404


@pytest.mark.parametrize("artifact_kind", ["reference", "pass", "final"])
def test_manifest_png_artifacts_are_served(image_run, artifact_kind):
    source, root, manifest = image_run
    relative = {
        "reference": manifest.references[0].image.path,
        "pass": manifest.photos[0].passes[0].path,
        "final": manifest.photos[0].final_path,
    }[artifact_kind]
    status, body = request(source, "/images/" + relative)
    assert status == 200
    assert body == (root / relative).read_bytes()


def test_custom_image_directory(image_run):
    source, root, manifest = image_run
    custom = root.with_name("custom-images")
    root.rename(custom)
    assert json.loads(request(source)[1])["images"]["photos"] == []
    status, body = request(source, images_dir=custom)
    assert status == 200
    assert json.loads(body)["images"]["photos"][0]["final_url"]
    assert (
        request(source, "/images/" + manifest.photos[0].final_path, images_dir=custom)[
            0
        ]
        == 200
    )


def test_matching_uploaded_preview_uses_images_without_changing_source(image_run):
    source, _, _ = image_run
    content = source.read_bytes()
    source.write_text("saved file unchanged")
    status, body = request(source, "/api/preview", body=content)
    assert status == 200
    assert json.loads(body)["images"]["photos"][0]["final_url"]
    assert source.read_text() == "saved file unchanged"


@pytest.mark.parametrize("upload", [False, True])
def test_different_source_does_not_attach_unrelated_images(image_run, upload):
    source, _, _ = image_run
    result = yaml.safe_load(source.read_text())
    result["story"]["seed"] = "Another story"
    content = yaml.safe_dump(result).encode()
    if upload:
        status, body = request(source, "/api/preview", body=content)
    else:
        source.write_bytes(content)
        status, body = request(source)
    assert status == 200
    images = json.loads(body)["images"]
    assert images["photos"] == images["references"] == []
    assert "different camera roll" in images["notice"]


@pytest.mark.parametrize(
    "problem",
    ["yaml", "schema", "version", "photo-index", "photo-date", "duplicate-reference"],
)
def test_invalid_image_manifest_preserves_text_details(image_run, problem):
    source, root, manifest = image_run
    if problem in {"yaml", "schema"}:
        (root / "manifest.yaml").write_text(
            "references: [broken" if problem == "yaml" else "references: []"
        )
    else:
        if problem == "version":
            manifest.version = 2
        elif problem == "photo-index":
            manifest.photos[0].photo_index = -1
        elif problem == "photo-date":
            manifest.photos[0].month = 12
        else:
            manifest.references.append(manifest.references[0])
        save_manifest(root, manifest)
    status, body = request(source)
    assert status == 200
    payload = json.loads(body)
    assert payload["result"]["story"]["seed"] == "Maya"
    assert payload["images"]["photos"] == []
    assert payload["images"]["notice"]


def test_incomplete_generation_exposes_only_completed_artifacts(image_run):
    source, root, manifest = image_run
    photo = manifest.photos[0]
    photo.passes[-1].completed = False
    manifest.references[0].image.completed = False
    save_manifest(root, manifest)
    images = json.loads(request(source)[1])["images"]
    assert images["photos"][0]["complete"] is False
    assert images["photos"][0]["final_url"] is None
    assert images["photos"][0]["passes"][0]["url"]
    assert images["photos"][0]["passes"][1] == {
        "pass_index": 1,
        "completed": False,
        "url": None,
    }
    assert images["references"][0]["url"] is None
    for relative in [
        photo.final_path,
        photo.passes[-1].path,
        manifest.references[0].image.path,
    ]:
        assert request(source, "/images/" + relative)[0] == 404


def test_missing_completed_images_and_final_copy_fallback(image_run):
    source, root, manifest = image_run
    (root / manifest.photos[0].passes[0].path).unlink()
    (root / manifest.photos[0].final_path).unlink()
    (root / manifest.references[0].image.path).unlink()
    images = json.loads(request(source)[1])["images"]
    assert images["photos"][0]["passes"][0]["url"] is None
    assert images["photos"][0]["passes"][0]["completed"] is True
    assert (
        images["photos"][0]["final_url"]
        == "/images/" + manifest.photos[0].passes[-1].path
    )
    assert images["references"][0]["url"] is None
    assert request(source, "/images/" + manifest.photos[0].passes[0].path)[0] == 404


@pytest.mark.parametrize(
    "relative",
    [
        "../secret.png",
        "/secret.png",
        "nested/../../secret.png",
        "secret.txt",
        "secret.png",
    ],
)
def test_image_serving_rejects_unsafe_paths_and_non_images(image_run, relative):
    source, root, manifest = image_run
    (root.parent / "secret.png").write_text("private data")
    (root / "secret.png").write_text("not an image")
    (root / "secret.txt").write_text("private data")
    manifest.references[0].image.path = relative
    save_manifest(root, manifest)
    status, _ = request(source, "/images/" + relative)
    assert status == 404


def test_image_serving_rejects_unlisted_and_encoded_traversal_paths(image_run):
    source, root, _ = image_run
    Image.new("RGB", (4, 4)).save(root / "unlisted.png")
    for path in [
        "/images/unlisted.png",
        "/images/manifest.yaml",
        "/images/%2e%2e/secret.png",
        "/images/%2Fsecret.png",
    ]:
        assert request(source, path)[0] == 404


def test_image_serving_rejects_symlinks_outside_image_directory(image_run):
    source, root, manifest = image_run
    external = root.parent / "external.png"
    Image.new("RGB", (4, 4)).save(external)
    path = root / manifest.references[0].image.path
    path.unlink()
    path.symlink_to(external)
    images = json.loads(request(source)[1])["images"]
    assert images["references"][0]["url"] is None
    assert request(source, "/images/" + manifest.references[0].image.path)[0] == 404


def test_corrupted_png_is_rejected_without_crashing_handler(image_run):
    source, root, manifest = image_run
    relative = manifest.photos[0].passes[0].path
    content = bytearray((root / relative).read_bytes())
    content[45] ^= 1  # Break the IDAT checksum while retaining a valid PNG header.
    (root / relative).write_bytes(content)
    assert request(source, "/images/" + relative)[0] == 404
