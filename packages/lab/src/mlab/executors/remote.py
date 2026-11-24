"""A GPU sandbox reached over the `mlab.executor.v1` job API (see base.py)."""
import json
import os
import time
from collections import Counter
import urllib.error
import urllib.parse
import urllib.request

from pydantic import ValidationError

from ..core.io import LabError, require
from ..gateway import plain_http_allowed
from .base import SCHEMA, MeasurementRecord

TERMINAL = {"succeeded", "failed", "cancelled"}
POLL_RETRIES = 4  # consecutive failed polls tolerated (5xx, timeouts, network blips) before giving up


class HardwareTimeout(LabError):
    """The job did not finish before the run deadline; the worker records the run as timed_out."""


class RemoteExecutor:
    name = "remote"

    def __init__(self, base_url, token=None, poll_s=2.0, timeout_s=3600):
        parsed = urllib.parse.urlsplit(base_url)
        require(parsed.scheme == "https" or (parsed.scheme == "http" and plain_http_allowed(parsed.hostname)),
                "E_SCHEMA", "The hardware service must use HTTPS (plain HTTP only for local or private hosts)", 2)
        self.base = base_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("MLAB_HARDWARE_TOKEN", "")
        self.poll_s, self.timeout_s = poll_s, timeout_s

    def _call(self, method, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(self.base + path, json.dumps(body).encode() if body is not None else None, headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise LabError("E_BACKEND_UNAVAILABLE", f"Hardware service HTTP {error.code} on {path}", 4) from None
        except (OSError, ValueError):
            raise LabError("E_BACKEND_UNAVAILABLE", "Hardware service unreachable or returned invalid JSON", 4) from None

    def submit(self, run, spec, requests):
        reply = self._call("POST", "/v1/jobs", {"schema": SCHEMA, "client_job_id": run["id"] + ":" + run["spec_hash"],
                                                "task": spec["task"], "candidate_id": spec["candidate_id"],
                                                "spec": spec, "requests": requests})
        require(isinstance(reply.get("job_id"), str) and reply["job_id"], "E_BACKEND_UNAVAILABLE", "Hardware service returned no job id", 4)
        return reply["job_id"]

    def cancel(self, job_id):
        """Best effort: a job the lab stops waiting for must not keep a GPU busy."""
        try:
            self._call("POST", f"/v1/jobs/{urllib.parse.quote(job_id)}/cancel", {})
        except LabError:
            pass

    def wait(self, job_id, cancelled=lambda: False):
        deadline = time.monotonic() + self.timeout_s
        failures = 0
        while time.monotonic() < deadline:
            if cancelled():
                raise LabError("E_NEEDS_APPROVAL", "Measurement cancelled", 3)
            try:
                state = self._call("GET", f"/v1/jobs/{urllib.parse.quote(job_id)}")
                failures = 0
            except LabError:
                failures += 1
                if failures > POLL_RETRIES:
                    raise
                time.sleep(min(self.poll_s * 2 ** failures, 30))
                continue
            if state.get("status") in TERMINAL:
                return state
            time.sleep(self.poll_s)
        raise HardwareTimeout("E_BACKEND_UNAVAILABLE", f"Hardware job did not finish within {self.timeout_s:g} s", 4)

    def execute(self, root, run, spec, requests, cancelled=lambda: False):
        job = self.submit(run, spec, requests)
        try:
            state = self.wait(job, cancelled)
        except BaseException:
            self.cancel(job)
            raise
        if state["status"] != "succeeded":
            raise LabError("E_BACKEND_UNAVAILABLE", f"Hardware job {job} {state['status']}: {str(state.get('error', ''))[:300]}", 4)
        try:
            record = MeasurementRecord.model_validate(state.get("result"))
        except ValidationError as error:
            raise LabError("E_MOCK_INVALID", "Hardware result violates mlab.executor.v1: " + str(error).splitlines()[0], 4) from None
        expected = Counter(r["request_id"] for r in requests)
        require(Counter(o.get("request_id") for o in record.observations) == expected, "E_MOCK_INVALID",
                "Hardware result must report every request exactly once", 4)
        return job, record
