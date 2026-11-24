"""The teacher's hidden reference optimum for a task under a simulator profile.

Sweeps a grid of configurations around the task's baseline through the in-tree simulator (the same request path
as student runs: make_spec -> request_bytes -> in-process simulate) and keeps the best one that meets every
performance constraint on the development workload. The result ships inside mlab
(src/mlab/teacher/reference/<task>-<profile>.json), never in a workspace, so the student's agent cannot read it.

    python -m mlab.teacher.sweep --task 9.2 [--task 9.3]   (or the `sweep` job in mlab.jobs)
"""
import argparse
import copy
import itertools
import json
import sys
import tempfile
from pathlib import Path

from mlab.backends.simulator_backend import request_bytes
from mlab.backends.simulator_http import SimulatorInProcessClient
from mlab.backends.simulator_profiles import PROFILE
from mlab.core.io import load
from mlab.core.metrics import get_metric, verdict
from mlab.core.store import now
from mlab.core.workspace import Lab, initialize

OUT = Path(__file__).resolve().parent / "reference"


def grid(task_id, baseline):
    """Candidate strategies around the baseline, within the exercise's decision space."""
    def variant(**changes):
        candidate = copy.deepcopy(baseline)
        serving = candidate["strategy"]["serving"]
        deployments = candidate["strategy"]["deployments"]
        for key, value in changes.items():
            if key == "model":
                deployments[0]["model"] = value
            elif key == "worker_replicas":
                deployments[1].update(replicas=value, devices=[f"gpu{i}" for i in range(1, value + 1)])
            else:
                serving[key] = value
        candidate["id"] = "ref_" + "_".join(f"{k}-{v}" for k, v in changes.items()).replace(".", "p").lower()
        candidate["rationale"] = "Reference sweep point"
        return candidate
    if task_id == "9.2":
        axes = {"model": ["llm-0.6b", "llm-1.7b", "llm-4b"], "max_concurrency": [1, 2, 4, 8, 12],
                "chunked_prefill_tokens": [None, 256, 1024, 4096], "continuous_batching": [True, False]}
    elif task_id == "9.3":
        axes = {"worker_replicas": [1, 2, 3], "max_concurrency": [1, 2, 4, 8, 16],
                "chunked_prefill_tokens": [None, 1024], "continuous_batching": [True, False]}
    else:
        raise SystemExit("9.1 is a satisfy task; it needs no reference optimum")
    for values in itertools.product(*axes.values()):
        yield variant(**dict(zip(axes, values)))


def build(task_id):
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch) / "ws"
        initialize(root, task_id, "mlsys", simulator_profile=PROFILE)
        lab = Lab(root, readonly=True)
        settings = lab.config["simulator_service"]
        client = SimulatorInProcessClient()
        objective = lab.task["objective"]
        performance_only = {"passes_floor": True}  # quality is not evaluated in the exercise profile
        rows = []
        for candidate in grid(task_id, load(root / "task/baselines/baseline.yaml")):
            spec = lab.make_spec(candidate, 42, 0, "development")
            try:
                result = client.simulate(request_bytes(root, spec, "reference", settings))["result"]
            except Exception as error:  # an infeasible point is data, not a failure of the sweep
                rows.append({"id": candidate["id"], "error": type(error).__name__})
                continue
            metrics = result["metrics"]
            judged, constraints = verdict(metrics, lab.task, performance_only, result["status"], result["failure_class"])
            rows.append({"id": candidate["id"], "feasible": judged == "constraints_met",
                         "value": get_metric(metrics, objective["metric"]), "strategy": candidate["strategy"],
                         "constraints": {c["metric"]: c["observed"] for c in constraints if c["value"] is not None}})
        lab.close()
    feasible = [r for r in rows if r.get("feasible") and type(r.get("value")) in {int, float}]
    feasible.sort(key=lambda r: r["value"], reverse=objective["direction"] == "max")
    import mlsys_sim
    return {"schema": "mlab.v1", "task": task_id, "profile": PROFILE, "simulator_version": mlsys_sim.__version__,
            "objective": objective, "generated_at": now(), "grid_size": len(rows), "feasible": len(feasible),
            "best": feasible[0] if feasible else None, "top": feasible[:5],
            "note": "Best of a coarse grid on the development workload with seed 42; a student may beat it off-grid, "
                    "so scores cap at 1.0 of this value."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", action="append", choices=["9.2", "9.3"], required=True)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for task_id in args.task:
        reference = build(task_id)
        path = OUT / f"{task_id}-{PROFILE}.json"
        path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")
        best = reference["best"]
        print(f"{task_id}: {reference['feasible']}/{reference['grid_size']} feasible; best "
              f"{reference['objective']['metric']}={best['value'] if best else None} ({best['id'] if best else '-'}) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
