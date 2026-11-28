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
  templates  {"task": "9.1" | "9.2" | "9.3"}  ->  the exercise's objective, constraints, decision space, holdouts and
             template designs (baseline and examples) for the simulator profile
  evaluate   {"task", "candidates": [<lab Candidate>...], "seeds": [42], "workloads": ["development", <holdout>...]}
             ->  {"results": [{candidate_id, seed, workload, status, meets_constraints, metrics, constraints, objective}]}
             Each candidate is checked against the task first (rejected ones get "error"); constraints are the task's
             performance constraints, since the exercise profile does not evaluate model quality.
             Optional "resamples": [{"seed": int, "variant": "development" | "short_mix" | "long_mix"}, ...] (at most 8)
             also scores every design on fresh prompt mixes generated from the DEVELOPMENT reference trace only (never a
             holdout): arrival times stay, each request's prompt is redrawn from the task's empirical pool. Their rows
             carry workload "resample:<variant>:<seed>", resample_seed, resample_variant and trace_sha256; the job also
             returns "resamples" describing each trace. short_mix/long_mix (80% / 20% short prompts) are 9.2 only.
             With resamples, "workloads" may be []. Resamples count in the evaluation budget.
  measure    reserved for GPU sandbox measurements (mlab.executor.v1); not implemented yet

`templates`, `evaluate` and `sweep` take an optional "profile": the exercise profile, one per target GPU.
  protocol-v1.20  RX 9070 XT (the default; unchanged)        protocol-v1.21  RTX 5080 (16 GB, 960 GB/s, 112.6 dense BF16 TFLOPS)

Errors exit with status 2 and print {"schema", "ok": false, "error": {"code", "message"}}.
"""
import argparse
import contextlib
import json
import shutil
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
    task = payload.get("task")
    if task not in {"9.2", "9.3"}:
        raise JobError("E_SCHEMA", "sweep needs 'task': '9.2' or '9.3' (9.1 is a satisfy task)")
    profile = _profile(payload)
    from .teacher.sweep import build
    return {"reference": build(task, profile)}


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


MAX_EVALUATIONS = 48
MAX_RESAMPLES = 8
TASK_IDS = ("9.1", "9.2", "9.3")
RESAMPLE_VARIANTS = ("development", "short_mix", "long_mix")


def _profile(payload):
    """The exercise profile a job runs on: the default unless the payload names another published one."""
    from .backends.simulator_profiles import PROFILE, PROFILES
    profile = payload.get("profile", PROFILE)
    if profile not in PROFILES:
        raise JobError("E_SCHEMA", f"unknown profile {profile!r}; use one of {', '.join(sorted(PROFILES))}")
    return profile


@contextlib.contextmanager
def _exercise(task, profile=None):
    """A throwaway workspace for `task` on the in-tree simulator's exercise profile."""
    from .backends.simulator_profiles import PROFILE
    from .core.workspace import Lab, initialize
    if task not in TASK_IDS:
        raise JobError("E_SCHEMA", f"'task' must be one of {', '.join(TASK_IDS)}")
    scratch = Path(tempfile.mkdtemp(prefix="mlab-job-"))
    try:
        initialize(scratch / "ws", task, "mlsys", simulator_profile=profile or PROFILE)
        lab = Lab(scratch / "ws", readonly=True)
        try:
            yield lab
        finally:
            lab.close()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _flat_metrics(metrics):
    flat = {}
    for name, value in metrics.items():
        if type(value) in {int, float}:
            flat[name] = value
        elif isinstance(value, dict):
            for stat, inner in value.items():
                if type(inner) in {int, float}:
                    flat[f"{name}.{stat}"] = inner
    return flat


def templates(payload):
    from .core.io import load
    with _exercise(payload.get("task"), _profile(payload)) as lab:
        task = lab.task
        root = lab.root / "task"
        designs = [load(root / "baselines/baseline.yaml")]
        for path in sorted((root / "examples/candidates").glob("*.yaml")):
            design = load(path)
            if design["id"] not in {d["id"] for d in designs}:
                designs.append(design)
        profile = load(root / "profile.json")
        return {"task": {key: task[key] for key in ("id", "title", "objective", "constraints", "decision_space", "holdouts",
                                                    "workload_mode", "key_metrics", "budget", "infra_options")},
                "profile": {"id": profile["id"], "notice": profile["notice"]}, "templates": designs}


def _resample_requests(payload):
    """The validated "resamples" list: unique {"seed", "variant"} pairs, at most MAX_RESAMPLES."""
    resamples = payload.get("resamples", [])
    if not isinstance(resamples, list) or len(resamples) > MAX_RESAMPLES:
        raise JobError("E_SCHEMA", f"'resamples' must be a list of at most {MAX_RESAMPLES} {{seed, variant}} objects")
    seen = []
    for item in resamples:
        if not (isinstance(item, dict) and set(item) <= {"seed", "variant"} and type(item.get("seed")) is int
                and item["seed"] >= 0 and item.get("variant", "development") in RESAMPLE_VARIANTS):
            raise JobError("E_SCHEMA", "each resample is {\"seed\": <int >= 0>, \"variant\": "
                                       + " | ".join(RESAMPLE_VARIANTS) + "}")
        pair = (item["seed"], item.get("variant", "development"))
        if pair in seen:
            raise JobError("E_SCHEMA", f"resample {pair[1]}:{pair[0]} is listed twice")
        seen.append(pair)
    if payload.get("task") != "9.2" and any(variant != "development" for _, variant in seen):
        raise JobError("E_SCHEMA", "short_mix and long_mix resamples bias the 9.2 prompt mix; other tasks take 'development'")
    return seen


def _resample_traces(lab, resamples):
    """Write each resampled workload into the throwaway lab's task folder and return its frozen reference.

    Holdout hygiene: every resample is generated from the task's DEVELOPMENT reference trace
    (task["frozen"]["workload_requests"], hash-checked here), never from a holdout file, so the holdouts stay unseen by
    a design chosen on resamples. The holdouts are themselves draws of the same generator from the same reference
    (workspace.initialize uses seeds 1042 + i), so a resample that reproduces a holdout's bytes is refused.
    Generation is deterministic (mlab.core.workloads.generate with the given seed and variant), and the trace file is
    named by its sha256, so equal resamples produce byte-identical requests.
    """
    from .core.io import hashed, inside
    from .core.workloads import write_resample
    task_dir = lab.root / "task"
    development = lab.task["frozen"]["workload_requests"]
    path = inside(task_dir, development["path"])
    if hashed(path) != development["sha256"]:
        raise JobError("E_FROZEN_CHANGED", "the development reference trace does not match its frozen hash")
    base = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    holdouts = {reference["sha256"]: key.removeprefix("holdout_") for key, reference in lab.task["frozen"].items()
                if key.startswith("holdout_")}
    traces = []
    for seed, variant in resamples:
        trace, metadata = write_resample(task_dir / "resamples", lab.task["id"], base, seed, variant)
        if metadata["sha256"] in holdouts:
            raise JobError("E_SCHEMA", f"resample {variant}:{seed} reproduces the holdout {holdouts[metadata['sha256']]!r}; "
                                       "pick another seed so the holdouts stay unseen")
        traces.append({"workload": f"resample:{variant}:{seed}", "seed": seed, "variant": variant,
                       "reference": {"path": trace.relative_to(task_dir).as_posix(), "sha256": metadata["sha256"]},
                       "base_sha256": development["sha256"], "requests": metadata["requests"]})
    return traces


def evaluate(payload):
    from .backends.simulator_backend import request_bytes
    from .backends.simulator_http import SimulatorAPIError, SimulatorInProcessClient
    from .core.io import LabError
    from .core.metrics import get_metric, verdict
    candidates = payload.get("candidates")
    seeds = payload.get("seeds", [42])
    workloads = payload.get("workloads", ["development"])
    if not isinstance(candidates, list) or not candidates:
        raise JobError("E_SCHEMA", "evaluate needs a nonempty 'candidates' list")
    if not (isinstance(seeds, list) and seeds and all(type(seed) is int for seed in seeds)):
        raise JobError("E_SCHEMA", "'seeds' must be a nonempty list of integers")
    resamples = _resample_requests(payload)
    if not (isinstance(workloads, list) and (workloads or resamples) and all(isinstance(w, str) for w in workloads)):
        raise JobError("E_SCHEMA", "'workloads' must be a nonempty list of names")
    if len(candidates) * len(seeds) * (len(workloads) + len(resamples)) > MAX_EVALUATIONS:
        raise JobError("E_BUDGET", f"at most {MAX_EVALUATIONS} candidate x seed x (workload + resample) evaluations per job")
    client = SimulatorInProcessClient()
    results = []
    with _exercise(payload.get("task"), _profile(payload)) as lab:
        unknown = sorted(set(workloads) - {"development", *lab.task["holdouts"]})
        if unknown:
            raise JobError("E_SCHEMA", f"unknown workloads {unknown}; use development or {lab.task['holdouts']}")
        traces = _resample_traces(lab, resamples)
        objective = lab.task["objective"]
        settings = lab.config["simulator_service"]
        # Named workloads first, exactly as before; then each resample, run on its own frozen reference.
        runs = [(workload, None if workload == "development" else workload, None, {}) for workload in workloads]
        runs += [(trace["workload"], trace["workload"], trace["reference"],
                  {"resample_seed": trace["seed"], "resample_variant": trace["variant"],
                   "trace_sha256": trace["reference"]["sha256"]}) for trace in traces]
        for raw in candidates:
            identifier = raw.get("id") if isinstance(raw, dict) else None
            try:
                candidate = lab.validate_candidate(raw, seeds[0])
            except (LabError, ValueError) as error:
                results.append({"candidate_id": identifier, "status": "rejected",
                                "error": {"code": getattr(error, "code", "E_SCHEMA"), "message": str(error)[:400]}})
                continue
            for seed in seeds:
                for workload, holdout, reference, extra in runs:
                    spec = lab.make_spec(candidate, seed, 0, "holdout" if holdout else "development", holdout, reference)
                    row = {"candidate_id": candidate["id"], "seed": seed, "workload": workload, **extra}
                    try:
                        raw_request = request_bytes(lab.root, spec, f"{candidate['id']}:{seed}:{workload}", settings)
                        result = client.simulate(raw_request)["result"]
                    except SimulatorAPIError as error:
                        results.append({**row, "status": "rejected",
                                        "error": {"code": error.error.get("code", "E_SIMULATOR"), "message": str(error)[:400]}})
                        continue
                    judged, constraints = verdict(result["metrics"], lab.task, {"passes_floor": True},
                                                  result["status"], result["failure_class"])
                    value = get_metric(result["metrics"], objective["metric"]) if objective.get("metric") else None
                    results.append({**row, "status": result["status"], "failure_class": result["failure_class"],
                                    "meets_constraints": judged == "constraints_met",
                                    "metrics": _flat_metrics(result["metrics"]),
                                    "constraints": [{k: c.get(k) for k in ("metric", "op", "value", "observed", "met")}
                                                    for c in constraints if c.get("value") is not None],
                                    "objective": {**objective, "value": value}})
    import mlsys_sim
    described = [{key: trace[key] for key in ("workload", "seed", "variant", "base_sha256", "requests")}
                 | {"trace_sha256": trace["reference"]["sha256"]} for trace in traces]
    return {"results": results, **({"resamples": described} if described else {}),
            "simulator_version": mlsys_sim.__version__}


def measure(payload):
    raise JobError("E_NOT_IMPLEMENTED", "GPU sandbox measurements arrive with the measurement harness (mlab.executor.v1)")


JOBS = {"simulate": simulate, "templates": templates, "evaluate": evaluate, "sweep": sweep, "calibrate": calibrate,
        "measure": measure}


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
