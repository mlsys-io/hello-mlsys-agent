"""Stateless job entrypoints (schema `mlab.jobs.v1`): JSON in, JSON out, nothing kept between calls.

Any job runner can call them, so the same code serves a laptop, CI and a cluster:

    python -m mlab.jobs simulate < request.json > response.json        # e.g. inside `docker run`
    from mlab.jobs import run; run("simulate", payload_text)           # e.g. a FlowMesh/Lumilake python task

Jobs
  simulate   {"requests": [<mlab.simulator.v1 request>, ...]}  ->  {"results": [{"status", "response"}, ...]}
             Pass each request as its exact JSON text (a string) to keep its input_sha256; an object is accepted
             and serialized compactly. Each "response" is the exact JSON text POST /v1/simulate would return.
  sweep      {"task": "9.2" | "9.3"}  ->  the reference optimum (mlab.teacher.sweep) for the exercise profile
  calibrate  {"train": <dataset>, "holdout": <dataset>, "current_profile": <profile> | null, "version": "vN",
              "budgets": {...} | null, "allow_test_fixture": false}
             ->  {"decision": {...}, "profile": <fitted candidate Profile>}   (gate in scripts/calibration/loop.py)
  measure    reserved for GPU sandbox measurements (mlab.executor.v1); not implemented yet

Errors exit with status 2 and print {"schema", "ok": false, "error": {"code", "message"}}.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

SCHEMA = "mlab.jobs.v1"


class JobError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _repo_root():
    from .core.io import ROOT  # packages/lab
    return ROOT.parent.parent


def simulate(payload):
    try:
        from mlsys_sim.agent_service import handle_agent_request
    except ImportError:
        raise JobError("E_UNAVAILABLE", "The simulator package (mlsys_sim) is not installed in this environment") from None
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise JobError("E_SCHEMA", "simulate needs a nonempty 'requests' list")
    results = []
    for request in requests:
        if isinstance(request, str):
            raw = request.encode("utf-8")
        elif isinstance(request, dict):
            raw = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        else:
            raise JobError("E_SCHEMA", "each request is a JSON object or its exact JSON text")
        status, body = handle_agent_request(raw)
        results.append({"status": status, "response": json.dumps(body, ensure_ascii=False, allow_nan=False)})
    import mlsys_sim
    return {"results": results, "simulator_version": mlsys_sim.__version__}


def sweep(payload):
    from .backends.simulator_profiles import PROFILE
    task = payload.get("task")
    if task not in {"9.2", "9.3"}:
        raise JobError("E_SCHEMA", "sweep needs 'task': '9.2' or '9.3' (9.1 is a satisfy task)")
    if payload.get("profile", PROFILE) != PROFILE:
        raise JobError("E_SCHEMA", f"only the {PROFILE} exercise profile is supported")
    from .teacher.sweep import build
    return {"reference": build(task)}


def calibrate(payload):
    loop_path = _repo_root() / "scripts/calibration/loop.py"
    if not loop_path.exists():
        raise JobError("E_UNAVAILABLE", "calibrate needs the full hello-mlsys-agent checkout (scripts/calibration/loop.py)")
    import importlib.util
    import yaml
    spec = importlib.util.spec_from_file_location("hello_mlsys_calibration_loop", loop_path)
    loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loop)
    for key in ("train", "holdout", "version"):
        if not payload.get(key):
            raise JobError("E_SCHEMA", f"calibrate needs '{key}'")
    budgets = payload.get("budgets") or yaml.safe_load((_repo_root() / "calibration/budgets.yaml").read_text())
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        paths = {}
        for key in ("train", "holdout", "current_profile"):
            if payload.get(key) is not None:
                paths[key] = scratch / f"{key}.json"
                paths[key].write_text(json.dumps(payload[key]))
        fitted = scratch / "candidate.json"
        try:
            decision = loop.gate(paths["train"], paths["holdout"], paths.get("current_profile"), str(payload["version"]),
                                 budgets, bool(payload.get("allow_test_fixture")), keep=fitted)
        except SystemExit as error:  # the simulator refused the data; report it as a job error
            raise JobError("E_SIMULATOR", str(error)) from None
        return {"decision": decision, "profile": json.loads(fitted.read_text())}


def measure(payload):
    raise JobError("E_NOT_IMPLEMENTED", "GPU sandbox measurements arrive with the measurement harness (mlab.executor.v1)")


JOBS = {"simulate": simulate, "sweep": sweep, "calibrate": calibrate, "measure": measure}


def run(name, payload_text):
    """Run job `name` on a JSON payload text; returns the JSON result text. Raises JobError."""
    if name not in JOBS:
        raise JobError("E_SCHEMA", f"unknown job {name!r}; jobs: {', '.join(JOBS)}")
    try:
        payload = json.loads(payload_text) if payload_text.strip() else {}
    except ValueError as error:
        raise JobError("E_SCHEMA", f"payload is not JSON: {error}") from None
    if not isinstance(payload, dict):
        raise JobError("E_SCHEMA", "payload must be a JSON object")
    result = JOBS[name](payload)
    return json.dumps({"schema": SCHEMA, "ok": True, "job": name, **result}, ensure_ascii=False, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m mlab.jobs", description="Stateless mlab job entrypoints (mlab.jobs.v1)")
    parser.add_argument("job", choices=[*JOBS, "list"])
    parser.add_argument("--input", type=Path, help="payload file (default: stdin)")
    args = parser.parse_args(argv)
    if args.job == "list":
        print(json.dumps({"schema": SCHEMA, "jobs": sorted(JOBS)}))
        return 0
    payload = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
    try:
        sys.stdout.write(run(args.job, payload) + "\n")
        return 0
    except JobError as error:
        sys.stdout.write(json.dumps({"schema": SCHEMA, "ok": False, "job": args.job,
                                     "error": {"code": error.code, "message": str(error)}}) + "\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
