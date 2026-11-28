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
from mlab.core import workloads
from mlab.core.workloads import generate, write_resample
from mlab.core.workspace import Lab, initialize
from mlab.jobs import MAX_EVALUATIONS, MAX_RESAMPLES, JobError, run

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


# --- exercise profiles: one per target GPU -------------------------------------------------------------------

V120, V121 = "protocol-v1.20", "protocol-v1.21"


def evaluate_baseline(task, profile=None):
    payload = {"task": task} | ({"profile": profile} if profile else {})
    template = json.loads(run("templates", json.dumps(payload)))
    out = json.loads(run("evaluate", json.dumps(payload | {"candidates": [template["templates"][0]],
                                                           "seeds": [42], "workloads": ["development"]})))
    return template, out["results"][0]


def test_default_profile_is_unchanged_and_names_its_hardware():
    template, result = evaluate_baseline("9.2")
    assert template["profile"]["id"] == V120 and template["templates"][0]["infra"] == "rx9070xt_x1"
    assert result["metrics"]["goodput_tok_s"] == 22.4 and round(result["metrics"]["ttft_ms.p99"], 1) == 709.7
    explicit = evaluate_baseline("9.2", V120)[1]
    assert explicit["metrics"] == result["metrics"]  # naming the default changes nothing


def test_rtx5080_profile_uses_its_own_hardware_and_rates():
    template, result = evaluate_baseline("9.2", V121)
    assert template["profile"]["id"] == V121 and "RTX 5080" in template["profile"]["notice"]
    assert {t["infra"] for t in template["templates"]} == {"rtx5080_x1"}
    assert template["task"]["infra_options"] == ["rtx5080_x1"]
    amd = evaluate_baseline("9.2", V120)[1]["metrics"]
    # 960 GB/s vs 640: decode is bandwidth-bound and faster. 112.6 vs 195 TFLOPS: prefill is compute-bound and slower.
    assert result["metrics"]["decode_tok_s"] > amd["decode_tok_s"] * 1.3
    assert result["metrics"]["ttft_ms.p99"] > amd["ttft_ms.p99"] * 1.4


@pytest.mark.parametrize("task, infra", [("9.1", "rtx5080_x1"), ("9.3", "rtx5080_x4")])
def test_rtx5080_profile_covers_every_task(task, infra):
    template, result = evaluate_baseline(task, V121)
    assert {t["infra"] for t in template["templates"]} == {infra}
    assert result["status"] == "succeeded"


def test_unknown_profile_is_refused_before_any_work():
    for job in ("templates", "evaluate", "sweep"):
        with pytest.raises(JobError, match="unknown profile"):
            run(job, json.dumps({"task": "9.2", "profile": "protocol-v9", "candidates": [{}]}))


def test_a_v121_design_cannot_name_the_other_card():
    template = json.loads(run("templates", json.dumps({"task": "9.2", "profile": V121})))
    design = copy.deepcopy(template["templates"][0])
    design["infra"] = "rx9070xt_x1"
    out = json.loads(run("evaluate", json.dumps({"task": "9.2", "profile": V121, "candidates": [design],
                                                 "seeds": [42], "workloads": ["development"]})))
    assert out["results"][0]["status"] == "rejected"


def test_each_profile_ships_its_own_reference_optimum():
    for task in ("9.2", "9.3"):
        shipped = json.loads((ROOT / f"src/mlab/teacher/reference/{task}-{V121}.json").read_text())
        assert shipped["profile"] == V121 and shipped["best"]["feasible"] is True
        assert shipped["best"]["value"] != json.loads(
            (ROOT / f"src/mlab/teacher/reference/{task}-{V120}.json").read_text())["best"]["value"] or task == "9.2"


def test_sweep_reproduces_the_rtx5080_reference():
    reference = json.loads(run("sweep", json.dumps({"task": "9.3", "profile": V121})))["reference"]
    shipped = json.loads((ROOT / f"src/mlab/teacher/reference/9.3-{V121}.json").read_text())
    assert reference["best"]["value"] == shipped["best"]["value"] and reference["profile"] == V121


# --- resampled workloads: fresh prompt mixes from the development reference ------------------------------------

RESAMPLES = [{"seed": 7, "variant": "short_mix"}, {"seed": 7, "variant": "long_mix"}, {"seed": 11, "variant": "development"}]
LEGACY_ROW_KEYS = {"candidate_id", "seed", "workload", "status", "failure_class", "meets_constraints", "metrics",
                   "constraints", "objective"}


def development_rows(tmp_path, task="9.2", profile=V121):
    """The development reference rows and the frozen references of a fresh workspace, as the job sees them."""
    root = tmp_path / "ws"
    initialize(root, task, "mlsys", simulator_profile=profile)
    lab = Lab(root, readonly=True)
    try:
        frozen = lab.task["frozen"]
        path = root / "task" / frozen["workload_requests"]["path"]
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()], frozen
    finally:
        lab.close()


def short_share(rows):
    return sum(row["input_tokens"] <= 1024 for row in rows) / len(rows)


def baseline_design(task="9.2", profile=V121):
    return json.loads(run("templates", json.dumps({"task": task, "profile": profile})))["templates"][0]


def test_a_resample_is_deterministic_and_content_addressed(tmp_path):
    base, _ = development_rows(tmp_path)
    first, meta = write_resample(tmp_path / "a", "9.2", base, 7, "short_mix")
    again, _ = write_resample(tmp_path / "b", "9.2", base, 7, "short_mix")
    assert first.read_bytes() == again.read_bytes() and first.name == again.name
    assert first.name == meta["sha256"].removeprefix("sha256:") + ".requests.jsonl"
    assert meta["sha256"] == "sha256:" + hashlib.sha256(first.read_bytes()).hexdigest()
    other, _ = write_resample(tmp_path / "c", "9.2", base, 8, "short_mix")
    assert other.name != first.name  # a different seed is a different mix
    # Through the job: the same payload twice gives the same traces and the same numbers.
    payload = json.dumps({"task": "9.2", "profile": V121, "candidates": [baseline_design()], "workloads": [],
                          "resamples": RESAMPLES})
    one, two = json.loads(run("evaluate", payload)), json.loads(run("evaluate", payload))
    assert one["resamples"] == two["resamples"] and one["results"] == two["results"]


def test_a_resample_keeps_arrivals_and_ids_and_redraws_prompts_from_the_pool(tmp_path):
    base, _ = development_rows(tmp_path)
    pool = {(r["input_tokens"], r["output_tokens_target"], r["category"], r["prefix_group"]) for r in base}
    for seed, variant in [(7, "short_mix"), (7, "long_mix"), (11, "development")]:
        rows = generate("9.2", base, seed, variant)
        assert [(r["request_id"], r["arrival_ms"]) for r in rows] == [(r["request_id"], r["arrival_ms"]) for r in base]
        assert all((r["input_tokens"], r["output_tokens_target"], r["category"], r["prefix_group"]) in pool for r in rows)
    assert generate("9.2", base, 11, "development") != base  # the prompt mix does change


def test_the_variants_shift_the_short_long_share(tmp_path):
    base, _ = development_rows(tmp_path)
    def mean_share(variant):
        return sum(short_share(generate("9.2", base, seed, variant)) for seed in range(200)) / 200
    development, short, long = mean_share("development"), mean_share("short_mix"), mean_share("long_mix")
    assert abs(development - short_share(base)) < 0.05  # unbiased: the reference's own 9/12 short
    assert short > development and short > 0.75
    assert long < 0.3 < development


def test_resample_rows_are_shaped_like_named_rows_plus_their_provenance(tmp_path):
    _, frozen = development_rows(tmp_path)
    out = json.loads(run("evaluate", json.dumps({"task": "9.2", "profile": V121, "candidates": [baseline_design()],
                                                 "workloads": ["development"], "resamples": RESAMPLES})))
    rows = out["results"]
    assert [r["workload"] for r in rows] == ["development", "resample:short_mix:7", "resample:long_mix:7",
                                             "resample:development:11"]
    assert set(rows[0]) == LEGACY_ROW_KEYS
    described = {d["workload"]: d for d in out["resamples"]}
    for row, asked in zip(rows[1:], RESAMPLES):
        assert set(row) == LEGACY_ROW_KEYS | {"resample_seed", "resample_variant", "trace_sha256"}
        assert (row["resample_seed"], row["resample_variant"]) == (asked["seed"], asked["variant"])
        assert row["trace_sha256"] == described[row["workload"]]["trace_sha256"]
        assert row["status"] == "succeeded" and "ttft_ms.p99" in row["metrics"] and "goodput_tok_s" in row["metrics"]
        assert row["objective"]["metric"] == "goodput_tok_s" and row["objective"]["value"] == row["metrics"]["goodput_tok_s"]
    # Holdout hygiene: every trace was drawn from the development reference and none is a holdout.
    holdouts = {reference["sha256"] for key, reference in frozen.items() if key.startswith("holdout_")}
    assert {d["base_sha256"] for d in out["resamples"]} == {frozen["workload_requests"]["sha256"]}
    assert not holdouts & {d["trace_sha256"] for d in out["resamples"]}
    assert len({d["trace_sha256"] for d in out["resamples"]}) == 3
    # The prompt mix moves the numbers: long prompts cost TTFT, so long_mix is no faster than short_mix.
    by = {r["workload"]: r["metrics"] for r in rows}
    assert by["resample:long_mix:7"]["ttft_ms.p99"] > by["resample:short_mix:7"]["ttft_ms.p99"]


def test_a_resample_is_generated_from_the_development_reference_only(tmp_path, monkeypatch):
    base, _ = development_rows(tmp_path)
    seen = []
    original = workloads.write_resample
    monkeypatch.setattr(workloads, "write_resample", lambda d, t, rows, s, v: seen.append(rows) or original(d, t, rows, s, v))
    run("evaluate", json.dumps({"task": "9.2", "profile": V121, "candidates": [baseline_design()], "workloads": [],
                                "resamples": RESAMPLES[:1]}))
    assert seen == [base]


def test_a_resample_that_reproduces_a_holdout_is_refused():
    # workspace.initialize draws short_prompt_distribution as generate(base, 1042, "short_mix").
    with pytest.raises(JobError, match="reproduces the holdout 'short_prompt_distribution'") as error:
        run("evaluate", json.dumps({"task": "9.2", "profile": V121, "candidates": [baseline_design()],
                                    "resamples": [{"seed": 1042, "variant": "short_mix"}]}))
    assert error.value.code == "E_SCHEMA"


def test_the_budget_counts_resamples():
    nine = [{"seed": s, "variant": "development"} for s in range(MAX_RESAMPLES)]
    # 6 designs x 1 seed x (1 workload + 8 resamples) = 54 > 48, although 6 x 1 x 1 alone is well inside the budget.
    assert 6 * (1 + MAX_RESAMPLES) > MAX_EVALUATIONS >= 6
    with pytest.raises(JobError) as error:
        run("evaluate", json.dumps({"task": "9.2", "candidates": [{}] * 6, "resamples": nine}))
    assert error.value.code == "E_BUDGET"
    with pytest.raises(JobError) as error:  # and the seeds multiply them too: 3 x 2 x (1 + 8) = 54
        run("evaluate", json.dumps({"task": "9.2", "candidates": [{}] * 3, "seeds": [1, 2], "resamples": nine}))
    assert error.value.code == "E_BUDGET"


@pytest.mark.parametrize("payload", [
    {"task": "9.2", "resamples": [{"seed": s} for s in range(MAX_RESAMPLES + 1)]},
    {"task": "9.2", "resamples": [{"seed": 7, "variant": "short_mix"}] * 2},
    {"task": "9.2", "resamples": [{"seed": 7, "variant": "tiny_mix"}]},
    {"task": "9.2", "resamples": [{"seed": -1}]},
    {"task": "9.2", "resamples": [{"seed": True}]},
    {"task": "9.2", "resamples": [{"seed": 7, "extra": 1}]},
    {"task": "9.2", "resamples": {"seed": 7}},
    {"task": "9.1", "resamples": [{"seed": 7, "variant": "short_mix"}]},
    {"task": "9.2", "workloads": []},
])
def test_bad_resamples_are_refused(payload):
    with pytest.raises(JobError) as error:
        run("evaluate", json.dumps({"candidates": [{}], **payload}))
    assert error.value.code == "E_SCHEMA"


def test_without_resamples_evaluate_is_unchanged():
    design = baseline_design()
    plain = {"task": "9.2", "profile": V121, "candidates": [design], "workloads": ["development", "short_prompt_distribution"]}
    out = json.loads(run("evaluate", json.dumps(plain)))
    assert "resamples" not in out and all(set(r) == LEGACY_ROW_KEYS for r in out["results"])
    assert json.loads(run("evaluate", json.dumps(plain | {"resamples": []}))) == out
    # Adding resamples leaves the named rows byte-for-byte as they were.
    mixed = json.loads(run("evaluate", json.dumps(plain | {"resamples": RESAMPLES[:1]})))
    assert mixed["results"][:2] == out["results"]


@pytest.mark.parametrize("task", ["9.1", "9.3"])
def test_development_resamples_cover_the_other_tasks(task):
    out = json.loads(run("evaluate", json.dumps({"task": task, "profile": V121, "candidates": [baseline_design(task)],
                                                 "workloads": [], "resamples": [{"seed": 5, "variant": "development"}]})))
    [row] = out["results"]
    assert row["workload"] == "resample:development:5" and row["status"] == "succeeded"
