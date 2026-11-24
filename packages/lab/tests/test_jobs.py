"""mlab.jobs: the stateless entrypoints a container or a FlowMesh python task runs."""
import copy
import hashlib
import json
import subprocess
import sys

import pytest

from mlab.backends.simulator_backend import request_bytes
from mlab.backends.simulator_http import SimulatorInProcessClient
from mlab.backends.simulator_profiles import PROFILE
from mlab.core.io import ROOT, load
from mlab.core.workspace import Lab, initialize
from mlab.jobs import JobError, run

REPO = ROOT.parent.parent
LOOP = REPO / "scripts/calibration/loop.py"


def requests_for(tmp_path, task="9.2"):
    root = tmp_path / "ws"
    initialize(root, task, "mlsys", simulator_profile=PROFILE)
    lab = Lab(root, readonly=True)
    try:
        baseline = load(root / "task/baselines/baseline.yaml")
        variant = copy.deepcopy(baseline)
        variant["id"] = "c4"
        variant["strategy"]["serving"]["max_concurrency"] = 4
        settings = lab.config["simulator_service"]
        return [request_bytes(root, lab.make_spec(c, 42, 0, "development"), c["id"], settings) for c in (baseline, variant)]
    finally:
        lab.close()


def test_simulate_is_byte_identical_to_the_in_process_client(tmp_path):
    raws = requests_for(tmp_path)
    result = json.loads(run("simulate", json.dumps({"requests": [raw.decode() for raw in raws]})))
    assert result["ok"] and result["schema"] == "mlab.jobs.v1" and len(result["results"]) == 2
    client = SimulatorInProcessClient()
    for raw, item in zip(raws, result["results"]):
        expected = client.simulate(raw)["response_bytes"]
        assert item["status"] == 200 and item["response"].encode("utf-8") == expected
        assert json.loads(item["response"])["input_sha256"] == "sha256:" + hashlib.sha256(raw).hexdigest()


def test_the_command_line_form_matches_and_reports_errors(tmp_path):
    raw = requests_for(tmp_path)[0]
    done = subprocess.run([sys.executable, "-m", "mlab.jobs", "simulate"], input=json.dumps({"requests": [raw.decode()]}),
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["results"][0]["response"].encode() == SimulatorInProcessClient().simulate(raw)["response_bytes"]
    bad = subprocess.run([sys.executable, "-m", "mlab.jobs", "simulate"], input="{}", capture_output=True, text=True, timeout=60)
    assert bad.returncode == 2 and json.loads(bad.stdout)["error"]["code"] == "E_SCHEMA"


def test_a_rejected_request_is_a_result_not_a_job_failure(tmp_path):
    raw = json.loads(requests_for(tmp_path)[0])
    raw["strategy"]["deployments"][0]["model"] = "llm-unknown"
    [item] = json.loads(run("simulate", json.dumps({"requests": [raw]})))["results"]
    assert item["status"] == 422 and json.loads(item["response"])["error"]["code"] == "UNKNOWN_MODEL_ID"


def test_sweep_reproduces_the_shipped_reference():
    reference = json.loads(run("sweep", json.dumps({"task": "9.2"})))["reference"]
    shipped = json.loads((ROOT / "src/mlab/teacher/reference/9.2-protocol-v1.20.json").read_text())
    assert reference["best"]["value"] == shipped["best"]["value"] and reference["feasible"] == shipped["feasible"]


def grid(noisy=False):
    fixture = json.loads((REPO / "packages/simulator/examples/v1/step_measurements.fixture.json").read_text())
    train = [(b, c) for b in (1, 2, 4, 8) for c in (64, 256, 1024)]
    holdout = [(1, 128), (2, 160), (4, 640), (8, 512)]
    def records(points, prefix):
        rows = []
        for b, c in points:
            ms = 1.0 + 0.2 * b + 0.002 * b * c
            if noisy:
                ms *= 0.3 + 1.4 * (int(hashlib.sha256(f"{b}-{c}".encode()).hexdigest()[:4], 16) / 0xFFFF)
            rows.append({"record_id": f"{prefix}-{b}-{c}", "group_id": f"{prefix}-{b}-{c}", "phase": "decode", "stage_index": 0,
                         "query_tokens": [1] * b, "context_lengths": [c] * b, "duration_ms": round(ms, 6)})
        return rows
    def dataset(points, name):
        return {"schema_version": 1, "dataset_id": name, "identity": fixture["identity"],
                "source": {"kind": "test_fixture", "reference": "synthetic grid"}, "records": records(points, name)}
    return dataset(train, "train"), dataset(holdout, "holdout")


@pytest.mark.skipif(not LOOP.exists(), reason="needs the full hello-mlsys-agent checkout")
def test_calibrate_fits_and_gates_statelessly():
    train, holdout = grid()
    first = json.loads(run("calibrate", json.dumps({"train": train, "holdout": holdout, "version": "v1",
                                                    "allow_test_fixture": True})))
    assert first["decision"]["promote"] and first["profile"]["profile_version"] == "v1"
    noisy_train, noisy_holdout = grid(noisy=True)
    second = json.loads(run("calibrate", json.dumps({"train": noisy_train, "holdout": noisy_holdout, "version": "v2",
                                                     "current_profile": first["profile"], "allow_test_fixture": True})))
    assert not second["decision"]["promote"] and second["decision"]["current_on_holdout"] is not None
    with pytest.raises(JobError, match="TEST_FIXTURE_NOT_ALLOWED") as refused:  # fixtures need the explicit flag
        run("calibrate", json.dumps({"train": train, "holdout": holdout, "version": "v1"}))
    assert refused.value.code == "E_SIMULATOR"


def test_unknown_and_reserved_jobs():
    with pytest.raises(JobError, match="unknown job"):
        run("deploy", "{}")
    with pytest.raises(JobError) as error:
        run("measure", "{}")
    assert error.value.code == "E_NOT_IMPLEMENTED"


@pytest.mark.parametrize("task", ["9.1", "9.2", "9.3"])
def test_templates_describe_the_exercise(task):
    out = json.loads(run("templates", json.dumps({"task": task})))
    ids = [t["id"] for t in out["templates"]]
    assert ids[0] == "baseline" and len(ids) == len(set(ids)) >= 2
    assert out["task"]["id"] == task and out["task"]["decision_space"] and out["profile"]["id"] == PROFILE


def test_evaluate_grades_designs_and_rejects_out_of_space_ones():
    base = json.loads(run("templates", json.dumps({"task": "9.2"})))["templates"][0]
    c4 = copy.deepcopy(base) | {"id": "c4"}
    c4["strategy"]["serving"]["max_concurrency"] = 4
    bad = copy.deepcopy(base) | {"id": "bad"}
    bad["strategy"]["serving"]["max_concurrency"] = 999
    out = json.loads(run("evaluate", json.dumps({"task": "9.2", "candidates": [base, c4, bad],
                                                 "workloads": ["development", "long_prompt_distribution"]})))
    rows = {(r["candidate_id"], r.get("workload")): r for r in out["results"]}
    assert rows[("baseline", "development")]["meets_constraints"] is True
    assert rows[("baseline", "development")]["objective"] == {"direction": "max", "metric": "goodput_tok_s", "value": 22.4}
    assert rows[("c4", "development")]["meets_constraints"] is False
    assert rows[("c4", "development")]["metrics"]["ttft_ms.p99"] > 2000
    assert rows[("bad", None)]["status"] == "rejected" and "outside decision space" in rows[("bad", None)]["error"]["message"]


def test_evaluate_matches_the_workspace_simulate_path(tmp_path):
    """The job's request bytes are the lab's own, so its numbers equal a workspace run of the same design."""
    raw = requests_for(tmp_path)[0]
    expected = SimulatorInProcessClient().simulate(raw)["result"]["metrics"]
    base = json.loads(run("templates", json.dumps({"task": "9.2"})))["templates"][0]
    [row] = json.loads(run("evaluate", json.dumps({"task": "9.2", "candidates": [base]})))["results"]
    assert row["metrics"]["ttft_ms.p99"] == expected["ttft_ms"]["p99"]


@pytest.mark.parametrize("payload, code", [
    ({"task": "9.9", "candidates": [{}]}, "E_SCHEMA"),
    ({"task": "9.2", "candidates": [{}], "workloads": ["nope"]}, "E_SCHEMA"),
    ({"task": "9.2", "candidates": [{}] * 25, "workloads": ["development", "short_prompt_distribution"]}, "E_BUDGET"),
])
def test_evaluate_refuses_bad_requests(payload, code):
    with pytest.raises(JobError) as error:
        run("evaluate", json.dumps(payload))
    assert error.value.code == code
