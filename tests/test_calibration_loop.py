"""The simulator improvement loop: sandbox runs -> dataset -> frozen holdout -> gated, versioned Profile."""
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/calibration"))
sys.path.insert(0, str(REPO / "packages/lab/tests"))
from loop import Loop  # noqa: E402

FIXTURE = json.loads((REPO / "packages/simulator/examples/v1/step_measurements.fixture.json").read_text())


# Held-out shapes sit between the training grid's context values, so a fitted Profile interpolates them from
# measured corners; a held-out shape exactly on grid values would need itself as a corner.
TRAIN = [(b, c) for b in (1, 2, 4, 8) for c in (64, 256, 1024)]
HOLDOUT = [(1, 128), (2, 160), (4, 640), (8, 512)]


def grid(noisy=False, scale=1.0, prefix="d"):
    """Decode steps. Smooth timings are bilinear in (batch, context), so interpolation recovers held-out shapes."""
    records = []
    for b, c in TRAIN + HOLDOUT:
        ms = scale * (1.0 + 0.2 * b + 0.002 * b * c)
        if noisy:
            ms *= 0.3 + 1.4 * (int(hashlib.sha256(f"{b}-{c}".encode()).hexdigest()[:4], 16) / 0xFFFF)
        group = ("holdout-" if (b, c) in HOLDOUT else "") + f"{prefix}-{b}-{c}"
        records.append({"record_id": f"{prefix}-{b}-{c}", "group_id": group, "phase": "decode",
                        "stage_index": 0, "query_tokens": [1] * b, "context_lengths": [c] * b, "duration_ms": round(ms, 6)})
    return {"schema_version": 1, "dataset_id": f"grid-{prefix}", "identity": copy.deepcopy(FIXTURE["identity"]),
            "source": {"kind": "test_fixture", "reference": "synthetic grid for the loop tests"}, "records": records}


@pytest.fixture
def loop(tmp_path):
    root = tmp_path / "calibration"
    root.mkdir()
    shutil.copyfile(REPO / "calibration/budgets.yaml", root / "budgets.yaml")
    return Loop(root, allow_fixture=True)


def write(tmp_path, name, dataset):
    path = tmp_path / name
    path.write_text(json.dumps(dataset))
    return path


def test_a_better_profile_is_promoted_and_a_worse_one_refused(loop, tmp_path):
    first = loop.ingest(write(tmp_path, "smooth.json", grid()))
    assert first["holdout_records"] == 4 and len(first["split"]["declared_groups"]) == 4
    decision = loop.promote(first["sha256"][:12])
    assert decision["promote"] and decision["version"] == "v1" and decision["current"] is None
    assert decision["candidate"]["mape_percent"] < 1 and decision["candidate"]["coverage"] >= 0.5
    key = first["identity_key"]
    assert (loop.root / "profiles" / key / "CURRENT").read_text().strip() == "v1"

    noisy = loop.ingest(write(tmp_path, "noisy.json", grid(noisy=True, prefix="n")))
    refused = loop.promote(noisy["sha256"][:12])
    assert not refused["promote"] and refused["current"] == "v1" and refused["current_on_holdout"] is not None
    assert any("MAPE" in reason for reason in refused["reasons"])
    assert (loop.root / "profiles" / key / "CURRENT").read_text().strip() == "v1"
    assert not (loop.root / "profiles" / key / "v2.json").exists()

    again = loop.ingest(write(tmp_path, "smooth2.json", grid(prefix="s")))
    accepted = loop.promote(again["sha256"][:12])
    assert accepted["promote"] and accepted["version"] == "v2" and accepted["current"] == "v1"
    events = [e["event"] for e in loop.ledger()]
    assert events == ["ingest", "promote", "ingest", "refuse", "ingest", "promote"]
    assert loop.verify() == {"ok": True, "problems": [], "ingested": 3, "promoted": 2}


def test_the_holdout_split_is_frozen_at_ingest(loop, tmp_path):
    entry = loop.ingest(write(tmp_path, "smooth.json", grid()))
    _, holdout = loop.files(entry)
    data = json.loads(holdout.read_text())
    data["records"] = data["records"][:-1]
    holdout.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="frozen"):
        loop.evaluate(entry["sha256"][:12])
    assert not loop.verify()["ok"]


def test_the_same_dataset_is_not_ingested_twice(loop, tmp_path):
    path = write(tmp_path, "smooth.json", grid())
    loop.ingest(path)
    with pytest.raises(SystemExit, match="already"):
        loop.ingest(path)


def test_sandbox_measurements_export_into_a_dataset(loop, tmp_path):
    """mlab measure on a sandbox -> measurement.json with step records -> a simulator dataset that ingests."""
    from mlab.cli import demo
    from fake_executor import FakeExecutor

    class Sandbox(FakeExecutor):
        def result(self, body):
            record = super().result(body)
            record["identity"] = {**record["identity"], "simulator_identity": FIXTURE["identity"]}
            record["step_records"] = grid()["records"]
            return record

    sandbox = Sandbox(kind="test_fixture")
    try:
        demo(tmp_path / "ws", "9.2", "fixture", hardware_url=sandbox.url)
    finally:
        sandbox.close()
    exported = loop.export([tmp_path / "ws"], tmp_path / "export")
    [path] = exported["datasets"]
    dataset = json.loads(Path(path).read_text())
    assert dataset["identity"] == FIXTURE["identity"] and dataset["source"]["kind"] == "test_fixture"
    assert len(dataset["records"]) == 16 and all(r["record_id"].split("/")[1].startswith("hw_0001:") for r in dataset["records"])
    entry = loop.ingest(path)
    assert entry["records"] == 16 and entry["holdout_records"] == 4  # the sandbox's declared holdout survives export
    strict = Loop(loop.root, allow_fixture=False)
    assert strict.export([tmp_path / "ws"], tmp_path / "strict") == {"datasets": []}  # fixtures need the explicit flag


def test_two_workspaces_export_without_colliding_ids(loop, tmp_path):
    from mlab.cli import demo
    from fake_executor import FakeExecutor

    class Sandbox(FakeExecutor):
        def result(self, body):
            record = super().result(body)
            record["identity"] = {**record["identity"], "simulator_identity": FIXTURE["identity"]}
            record["step_records"] = grid()["records"]
            return record

    sandbox = Sandbox(kind="test_fixture")
    try:
        for name in ("a", "b"):
            demo(tmp_path / name, "9.2", "fixture", hardware_url=sandbox.url)
    finally:
        sandbox.close()
    [path] = loop.export([tmp_path / "a", tmp_path / "b"], tmp_path / "export")["datasets"]
    records = json.loads(Path(path).read_text())["records"]
    assert len(records) == 32 and len({r["record_id"] for r in records}) == 32
    assert loop.ingest(path)["records"] == 32
