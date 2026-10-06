"""Local visualizer: uv run python -m ai_camera_roll.viewer [result.yaml]."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import ValidationError

from .models import GenerationResult

MAX_UPLOAD = 5 * 1024 * 1024


def parse_result(content: str | bytes) -> dict:
    """Load YAML (or JSON) safely and validate catalog references before display."""
    return GenerationResult.model_validate(yaml.safe_load(content)).model_dump(mode="json")


def make_handler(source: Path) -> type[BaseHTTPRequestHandler]:
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
            self.respond(status, json.dumps(value).encode(), "application/json; charset=utf-8")

        def result_response(self, content: str | bytes, name: str) -> None:
            try:
                result = parse_result(content)
            except (yaml.YAMLError, ValidationError, ValueError, UnicodeError):
                self.send_json(400, {"error": "This file is not a valid camera roll. Open a complete generated result containing story and camera_roll, with valid people and object references."})
                return
            self.send_json(200, {"name": name, "result": result})

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/result":
                try:
                    content = source.read_bytes()
                except OSError:
                    self.send_json(404, {"error": "No camera roll found. Use Open YAML to choose a generated result."})
                    return
                self.result_response(content, source.name)
                return
            assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}
            if path not in assets:
                self.send_error(404)
                return
            name, mime = assets[path]
            self.respond(200, files("ai_camera_roll").joinpath("web", name).read_bytes(), f"{mime}; charset=utf-8")

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
    parser.add_argument("source", nargs="?", type=Path, default=Path("output/camera_roll.yaml"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.source))
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
