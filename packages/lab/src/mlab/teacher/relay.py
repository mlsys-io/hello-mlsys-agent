"""A loopback teacher for one workspace, run by `./claude`'s launcher.

The launcher holds MLAB_TEACHER_API_KEY; the student's agent only gets MLAB_TEACHER_URL and a session token.
The relay reads the workspace itself and writes the teacher log and assessments itself, so neither the question
context nor the grade passes through the agent.
"""
import hmac
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..core.io import LabError

ENDPOINTS = ("/ask", "/assess")


def handle(root, client, path, body):
    from ..core.workspace import Lab
    from . import assess, coach
    lab = Lab(root, readonly=True)
    try:
        if path == "/ask":
            return coach.ask(lab, body.get("question"), client)
        return assess.assess(lab, client)
    finally:
        lab.close()


def serve(root, client, token):
    lock = threading.Lock()  # one teacher call at a time per workspace

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # no questions or grades in logs

        def reply(self, status, value):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            supplied = self.headers.get("Authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(supplied, token):
                return self.reply(401, {"ok": False, "error": {"code": "E_SCHEMA", "message": "teacher token required"}})
            if self.path not in ENDPOINTS:
                return self.reply(404, {"ok": False, "error": {"code": "E_SCHEMA", "message": "unknown teacher endpoint"}})
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}") if 0 <= length <= 64_000 else {}
                with lock:
                    result = handle(root, client, self.path, body)
                self.reply(200, {"ok": True, "result": result})
            except LabError as error:
                self.reply(200, {"ok": False, "error": {"code": error.code, "message": str(error), "exit_code": error.exit_code}})
            except Exception:
                self.reply(500, {"ok": False, "error": {"code": "E_BACKEND_UNAVAILABLE", "message": "teacher relay failed"}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def call(path, body, env=None):
    """From inside the agent: ask the launcher's relay."""
    env = os.environ if env is None else env
    request = urllib.request.Request(env["MLAB_TEACHER_URL"].rstrip("/") + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + env.get("MLAB_TEACHER_TOKEN", "")})
    try:
        try:
            response = urllib.request.urlopen(request, timeout=float(env.get("MLAB_TEACHER_TIMEOUT_S", "300")))
        except urllib.error.HTTPError as error:
            response = error  # the relay's structured error body
        with response:
            packet = json.load(response)
    except (OSError, urllib.error.URLError, ValueError):
        raise LabError("E_BACKEND_UNAVAILABLE", "The teacher relay is not answering; restart ./claude", 4) from None
    if not packet.get("ok"):
        error = packet.get("error") or {}
        raise LabError(error.get("code", "E_BACKEND_UNAVAILABLE"), error.get("message", "teacher failed"), error.get("exit_code", 4))
    return packet["result"]


def teacher(lab_root, path, body):
    """Run a teacher request: through the relay when the launcher provides one, else directly with MLAB_TEACHER_*."""
    if os.environ.get("MLAB_TEACHER_URL"):
        return call(path, body)
    from .client import TeacherClient, configuration
    config = configuration()
    if config is None:
        raise LabError("E_BACKEND_UNAVAILABLE", "No teacher configured: set MLAB_TEACHER_BASE_URL, MLAB_TEACHER_MODEL "
                       "(default deepseek-v4.1-flash) and MLAB_TEACHER_API_KEY, then start ./claude again", 4)
    return handle(lab_root, TeacherClient(config), path, body)
