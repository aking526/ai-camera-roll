"""Loopback-only website and generation API. No secrets are served as static files."""

import hmac
import json
import mimetypes
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from generation import AppError, HERE, RunManager
from ratings import RatingStore, default_loader


def make_handler(manager, token, static_root=None, rating_store=None):
    static_root = (Path(static_root) if static_root else HERE / "dist").resolve()
    ratings = rating_store or RatingStore(manager.root.parent / "ratings", default_loader(manager, static_root))

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, content, content_type="application/json; charset=utf-8"):
            body = json.dumps(content, ensure_ascii=False).encode() if isinstance(content, (dict, list)) else content
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def check_host(self):
            port = self.server.server_address[1]
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in allowed:
                raise AppError("Use the local demo address to access this page.", "origin", 403)
            return allowed

        def do_GET(self):
            try:
                self.check_host()
                path = unquote(urlsplit(self.path).path)
                parts = path.strip("/").split("/")
                if path == "/api/ratings":
                    return self.respond(200, ratings.overview())
                if len(parts) in (4, 5) and parts[:3] == ["api", "ratings", "rounds"]:
                    if len(parts) == 4:
                        return self.respond(200, ratings.round(parts[3]))
                    if parts[4] == "results":
                        return self.respond(200, ratings.results(parts[3]))
                    if parts[4] == "export":
                        return self.respond(200, ratings.export_round(parts[3]))
                if len(parts) in (4, 5) and parts[:3] == ["api", "ratings", "sessions"]:
                    if len(parts) == 4:
                        return self.respond(200, ratings.session(parts[3]))
                    if parts[4] == "export":
                        return self.respond(200, ratings.export_session(parts[3]))
                if len(parts) == 4 and parts[:2] == ["ratings", "assets"]:
                    return self.respond(200, ratings.asset_file(parts[2], parts[3]).read_bytes(), "image/png")
                if path == "/api/config":
                    return self.respond(200, {**manager.config(), "csrf_token": token})
                if path == "/api/runs":
                    return self.respond(200, {"runs": manager.list()})
                if len(parts) in (3, 4) and parts[:2] == ["api", "runs"]:
                    if len(parts) == 3:
                        return self.respond(200, {"run": manager.get(parts[2])})
                    if parts[3] == "data":
                        return self.respond(200, manager.data(parts[2]))
                if len(parts) >= 4 and parts[0] == "runs" and parts[2] == "images":
                    return self.respond(200, manager.image_file(parts[1], "/".join(parts[3:])).read_bytes(), "image/png")
                if path.startswith("/api/") or path.startswith("/runs/") or path.startswith("/ratings/"):
                    raise AppError("Not found.", "not_found", 404)
                target = (static_root / (path.lstrip("/") or "index.html")).resolve()
                if not target.is_relative_to(static_root) or not target.is_file() or any(part.startswith(".") for part in Path(path).parts if part != "/"):
                    raise AppError("Not found.", "not_found", 404)
                return self.respond(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            except AppError as error:
                self.respond(error.status, {"error": error.message, "code": error.code})
            except (OSError, ValueError):
                self.respond(500, {"error": "The saved data could not be read.", "code": "read_failed"})

        def do_POST(self):
            try:
                allowed = self.check_host()
                origin = self.headers.get("Origin")
                if origin and origin not in {"http://" + host for host in allowed}:
                    raise AppError("Use this website's local page to start a run.", "origin", 403)
                if not hmac.compare_digest(self.headers.get("X-Demo-Token", ""), token):
                    raise AppError("Reload this local page before starting a run.", "token", 403)
                if self.headers.get_content_type() != "application/json":
                    raise AppError("A JSON request is required.", "content_type", 415)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise AppError("The request is too large or empty.", "request_size", 413)
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise AppError("The request must be a JSON object.")
                path = unquote(urlsplit(self.path).path)
                parts = path.strip("/").split("/")
                if path == "/api/ratings/rounds":
                    return self.respond(201, ratings.create_round(payload))
                if len(parts) == 5 and parts[:3] == ["api", "ratings", "rounds"] and parts[4] == "sessions":
                    return self.respond(201, ratings.start_session(parts[3], payload))
                if len(parts) == 5 and parts[:3] == ["api", "ratings", "sessions"]:
                    if parts[4] == "answer":
                        return self.respond(200, ratings.save_answer(parts[3], payload))
                    if parts[4] == "submit":
                        return self.respond(200, ratings.submit(parts[3]))
                if path == "/api/runs":
                    return self.respond(202, {"run": manager.create(payload)})
                if len(parts) == 4 and parts[:2] == ["api", "runs"]:
                    identifier, action = parts[2:]
                    if action == "generate":
                        return self.respond(202, {"run": manager.generate(identifier, payload)})
                    if action == "resume":
                        return self.respond(202, {"run": manager.resume(identifier)})
                    if action == "stop":
                        return self.respond(200, {"run": manager.stop(identifier)})
                raise AppError("Not found.", "not_found", 404)
            except AppError as error:
                self.respond(error.status, {"error": error.message, "code": error.code})
            except (ValueError, TypeError):
                self.respond(400, {"error": "The request could not be read.", "code": "invalid_request"})
            except Exception:
                self.respond(500, {"error": "The request could not be completed.", "code": "server_error"})

    return Handler


def main():
    # Bind first: a second launcher must not mark the running server's jobs interrupted.
    server = ThreadingHTTPServer(("127.0.0.1", 5812), BaseHTTPRequestHandler)
    manager = RunManager()
    server.RequestHandlerClass = make_handler(manager, secrets.token_urlsafe(32))
    print("A life in frames: http://127.0.0.1:5812/", flush=True)
    print("Saved albums stay local. Generation runs only after an explicit request in the website.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
