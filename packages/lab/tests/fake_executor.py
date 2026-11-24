"""A stand-in GPU sandbox that speaks mlab.executor.v1, for tests and for sandbox integrators to copy."""
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

IDENTITY = {"model_revision": "Qwen/Qwen3-1.7B@0060bc56", "devices": ["AMD Radeon RX 9070 XT"],
            "runtime": {"name": "vllm", "version": "0.11.0"}, "weight_dtype": "bfloat16", "kv_dtype": "bfloat16",
            "activation_dtype": "bfloat16", "driver": "rocm-6.4.1"}


def observe(requests, ttft_ms=120.0, tpot_ms=12.0):
    rows = []
    for row in requests:
        ready = float(row.get("arrival_ms", 0))
        first = ready + ttft_ms
        tokens = int(row["output_tokens_target"])
        rows.append({"request_id": row["request_id"], "ready_ms": ready, "started_ms": ready, "first_token_ms": first,
                     "finished_ms": first + tpot_ms * tokens, "output_tokens": tokens,
                     "output_tokens_target": tokens, "status": "completed", "service_ms": ttft_ms + tpot_ms * tokens})
    return rows


class FakeExecutor:
    """`result(job)` builds the MeasurementRecord; override it to test failure handling."""

    def __init__(self, kind="measured", identity=None, drop_request=False, token=None):
        self.jobs, self.submitted = {}, []
        self.kind, self.identity, self.drop_request, self.token = kind, identity or IDENTITY, drop_request, token
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value):
                raw = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def authorized(self):
                return fake.token is None or self.headers.get("Authorization") == "Bearer " + fake.token

            def do_POST(self):
                if not self.authorized():
                    return self.reply(401, {"error": "token"})
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/v1/jobs":
                    assert body["schema"] == "mlab.executor.v1"
                    job = uuid.uuid4().hex[:8]
                    fake.submitted.append(body)
                    fake.jobs[job] = {"job_id": job, "status": "succeeded", "result": fake.result(body)}
                    return self.reply(200, {"job_id": job})
                job = self.path.split("/")[3]
                fake.jobs[job]["status"] = "cancelled"
                self.reply(200, {"status": "cancelled"})

            def do_GET(self):
                if not self.authorized():
                    return self.reply(401, {"error": "token"})
                self.reply(200, fake.jobs[self.path.rsplit("/", 1)[1]])

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def result(self, body):
        rows = observe(body["requests"])
        if self.drop_request:
            rows = rows[1:]
        return {"observations": rows, "vram_peak_gib": 9.5, "kv_peak_gib": 3.0, "identity": self.identity,
                "source": {"kind": self.kind, "reference": "fake-sandbox/run-" + body["client_job_id"]},
                "step_records": [{"phase": "decode", "batch": 4, "ms": 11.8}], "logs": ["fake sandbox"]}

    def close(self):
        self.server.shutdown()
        self.server.server_close()
