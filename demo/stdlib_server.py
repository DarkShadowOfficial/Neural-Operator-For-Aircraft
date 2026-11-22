"""Dependency-free HTTP server with the same routes as demo.server (used when FastAPI is not installed)."""
from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from demo.core import DemoApp

STATIC = Path(__file__).parent / "static"


def _default(o):
    try:
        return float(o)
    except Exception:
        return str(o)


def make_handler(core: DemoApp):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, payload, ctype="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, default=_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path.startswith("/static/"):
                f = (STATIC / u.path[len("/static/"):]).resolve()
                if STATIC.resolve() in f.parents and f.is_file():
                    return self._send(200, f.read_bytes(), mimetypes.guess_type(str(f))[0] or "application/octet-stream")
                return self._send(404, {"detail": "not found"})
            if u.path == "/api/status":
                return self._send(200, core.status())
            if u.path == "/api/config":
                return self._send(200, core.config())
            if u.path == "/api/random_case":
                q = parse_qs(u.query)
                seed = int(q["seed"][0]) if "seed" in q else None
                return self._send(200, core.random_case(seed))
            self._send(404, {"detail": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(422, {"detail": "invalid JSON"})
            route = {"/api/simulate": core.simulate, "/api/snapshot": core.snapshot, "/api/stress": core.stress}.get(self.path)
            if route is None:
                return self._send(404, {"detail": "not found"})
            try:
                self._send(200, route(body))
            except ValueError as e:
                self._send(422, {"detail": str(e)})
            except RuntimeError as e:
                self._send(503, {"detail": str(e)})
            except Exception as e:                                  # pragma: no cover
                self._send(500, {"detail": f"{type(e).__name__}: {e}"})
    return H


def serve(core: DemoApp, host="127.0.0.1", port=8000):
    core.start()
    srv = ThreadingHTTPServer((host, port), make_handler(core))
    srv.daemon_threads = True
    srv.serve_forever()
