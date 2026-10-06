"""Local visualizer: uv run python -m ai_camera_roll.viewer [result.yaml]."""

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import yaml
from PIL import Image
from pydantic import ValidationError

from .image_pipeline import ImageGenerationManifest, _digest
from .models import GenerationResult

MAX_UPLOAD = 5 * 1024 * 1024


def parse_result(content: str | bytes) -> dict:
    """Load YAML (or JSON) safely and validate catalog references before display."""
    return GenerationResult.model_validate(yaml.safe_load(content)).model_dump(
        mode="json"
    )


def read_manifest(root: Path) -> ImageGenerationManifest:
    manifest = ImageGenerationManifest.model_validate(
        yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    )
    if manifest.version != 1:
        raise ValueError("Unsupported image manifest version")
    return manifest


def image_path(root: Path, relative: str) -> Path | None:
    """Resolve a PNG artifact without allowing traversal or escaping symlinks."""
    try:
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.suffix.lower() != ".png"
        ):
            return None
        path = (root / candidate).resolve()
        if path.is_relative_to(root.resolve()) and path.is_file():
            return path
    except (OSError, ValueError, RuntimeError):
        pass
    return None


def image_metadata(root: Path, result: dict) -> dict:
    images = {"notice": None, "photos": [], "references": []}
    try:
        manifest = read_manifest(root)
    except FileNotFoundError:
        images["notice"] = "No generated images found for this camera roll."
        return images
    except (OSError, ValueError, yaml.YAMLError, UnicodeError):
        images["notice"] = (
            "Generated images could not be loaded: the image manifest is invalid or unreadable."
        )
        return images
    if manifest.settings.get("source_sha256") != _digest(result):
        images["notice"] = (
            "The generated images belong to a different camera roll and are not shown."
        )
        return images

    # The source hash connects the runs; also validate the manifest's record mappings.
    roll = result["camera_roll"]
    reference_keys = {(asset.kind, asset.entity_id) for asset in manifest.references}
    assets_by_id = {asset.entity_id: asset for asset in manifest.references}
    known_keys = {
        (kind, item["id"]) for kind in ("people", "objects") for item in roll[kind]
    }
    photo_indices = {photo.photo_index for photo in manifest.photos}
    if (
        len(reference_keys) != len(manifest.references)
        or not reference_keys <= known_keys
        or len(photo_indices) != len(manifest.photos)
        or any(
            not 0 <= photo.photo_index < len(roll["photos"])
            or (photo.year, photo.month)
            != (
                roll["photos"][photo.photo_index]["year"],
                roll["photos"][photo.photo_index]["month"],
            )
            or not photo.passes
            or len({step.path for step in photo.passes}) != len(photo.passes)
            for photo in manifest.photos
        )
        or any(
            len({item.slot for item in step.inputs}) != len(step.inputs)
            or any(
                item.slot < 0
                or (
                    item.entity_id is not None
                    and (
                        item.entity_id not in assets_by_id
                        or item.path != assets_by_id[item.entity_id].reference_path
                    )
                )
                or (
                    item.entity_id is None
                    and item.path not in {prior.path for prior in photo.passes[:index]}
                )
                for item in step.inputs
            )
            for photo in manifest.photos
            for index, step in enumerate(photo.passes)
        )
    ):
        images["notice"] = (
            "Generated images could not be loaded: the image manifest has invalid record mappings."
        )
        return images

    def url(relative: str, completed: bool) -> str | None:
        return (
            f"/images/{quote(relative, safe='/')}"
            if completed and image_path(root, relative)
            else None
        )

    images["references"] = [
        {
            "kind": asset.kind,
            "entity_id": asset.entity_id,
            "completed": asset.image.completed,
            "url": url(asset.image.path, asset.image.completed),
            "prompt": asset.image.prompt,
        }
        for asset in manifest.references
    ]
    for photo in manifest.photos:
        complete = all(step.completed for step in photo.passes)
        passes = []
        seen_ids = set()
        scene_references = []
        for index, step in enumerate(photo.passes):
            inputs = []
            inherited = []
            ordered_inputs = sorted(step.inputs, key=lambda item: item.slot)
            current_references = [
                item.entity_id for item in ordered_inputs if item.entity_id is not None
            ]
            current_ids = set(current_references)
            for item in ordered_inputs:
                if item.entity_id is None:
                    source_index = next(
                        i
                        for i, prior in enumerate(photo.passes[:index])
                        if prior.path == item.path
                    )
                    inputs.append(
                        {
                            "slot": item.slot,
                            "kind": "scene",
                            "source_pass_index": source_index,
                        }
                    )
                    inherited.extend(scene_references[source_index])
                else:
                    asset = assets_by_id[item.entity_id]
                    inputs.append(
                        {
                            "slot": item.slot,
                            "kind": asset.kind,
                            "entity_id": item.entity_id,
                            "is_new": item.entity_id not in seen_ids,
                        }
                    )
            inherited = list(dict.fromkeys(inherited))
            carried_forward = [
                {"kind": assets_by_id[identifier].kind, "entity_id": identifier}
                for identifier in inherited
                if identifier not in current_ids
            ]
            passes.append(
                {
                    "pass_index": index,
                    "completed": step.completed,
                    "url": url(step.path, step.completed),
                    "prompt": step.prompt,
                    "inputs": inputs,
                    "carried_forward": carried_forward,
                }
            )
            scene_references.append(list(dict.fromkeys(inherited + current_references)))
            seen_ids.update(current_ids)
        final_url = url(photo.final_path, complete)
        if complete and not final_url:
            final_url = passes[-1]["url"]
        images["photos"].append(
            {
                "photo_index": photo.photo_index,
                "complete": complete,
                "final_url": final_url,
                "passes": passes,
            }
        )
    return images


def make_handler(
    source: Path, images_dir: Path | None = None
) -> type[BaseHTTPRequestHandler]:
    root = (
        images_dir if images_dir is not None else source.parent / "images"
    ).resolve()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, value: dict) -> None:
            self.respond(
                status, json.dumps(value).encode(), "application/json; charset=utf-8"
            )

        def result_response(self, content: str | bytes, name: str) -> None:
            try:
                result = parse_result(content)
            except (yaml.YAMLError, ValidationError, ValueError, UnicodeError):
                self.send_json(
                    400,
                    {
                        "error": "This file is not a valid camera roll. Open a complete generated result containing story and camera_roll, with valid people and object references."
                    },
                )
                return
            self.send_json(
                200,
                {
                    "name": name,
                    "result": result,
                    "images": image_metadata(root, result),
                },
            )

        def image_response(self, relative: str) -> None:
            try:
                manifest = read_manifest(root)
                allowed = {
                    asset.image.path
                    for asset in manifest.references
                    if asset.image.completed
                }
                for photo in manifest.photos:
                    allowed.update(step.path for step in photo.passes if step.completed)
                    if photo.passes and all(step.completed for step in photo.passes):
                        allowed.add(photo.final_path)
                path = image_path(root, relative) if relative in allowed else None
                if path is None:
                    self.send_error(404)
                    return
                content = path.read_bytes()
                # A listed file must actually be a PNG, not just have a PNG suffix.
                with Image.open(io.BytesIO(content)) as image:
                    if image.format != "PNG":
                        raise ValueError("Not a PNG")
                    image.verify()
            except (
                OSError,
                ValueError,
                SyntaxError,
                yaml.YAMLError,
                UnicodeError,
                Image.DecompressionBombError,
            ):
                self.send_error(404)
                return
            self.respond(200, content, "image/png")

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/result":
                try:
                    content = source.read_bytes()
                except OSError:
                    self.send_json(
                        404,
                        {
                            "error": "No camera roll found. Use Open YAML to choose a generated result."
                        },
                    )
                    return
                self.result_response(content, source.name)
                return
            if path.startswith("/images/"):
                self.image_response(unquote(path.removeprefix("/images/")))
                return
            assets = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if path not in assets:
                self.send_error(404)
                return
            name, mime = assets[path]
            self.respond(
                200,
                files("ai_camera_roll").joinpath("web", name).read_bytes(),
                f"{mime}; charset=utf-8",
            )

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/preview":
                self.send_error(404)
                return
            # Only same-origin browser requests may submit a preview.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_UPLOAD:
                self.send_json(413, {"error": "Choose a YAML file smaller than 5 MB."})
                return
            self.result_response(self.rfile.read(length), "Uploaded camera roll")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", type=Path, default=Path("output/camera_roll.yaml")
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--images",
        type=Path,
        help="Image output directory (default: images/ beside the source)",
    )
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(args.source, args.images)
    )
    print(f"Camera roll viewer: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Reading {args.source}. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
