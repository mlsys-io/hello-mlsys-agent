"""The GPU-sandbox seam: mlab.executor.v1 behind `mlab measure`, with no other lab changes."""
import json

import pytest

from mlab.cli import demo
from mlab.core.io import LabError, load
from mlab.core.workspace import Lab, initialize
from mlab.executors.base import MeasurementRecord
from fake_executor import IDENTITY, FakeExecutor


def hardware_runs(root):
    lab = Lab(root, readonly=True)
    try:
        return [(run, lab.result(run["id"])[0]) for run in lab.store.all("run") if run.get("role") == "hardware"]
    finally:
        lab.close()


def test_a_sandbox_measurement_is_measured_evidence_with_its_raw_record(tmp_path):
    sandbox = FakeExecutor()
    try:
        summary = demo(tmp_path / "ws", "9.2", "fixture", hardware_url=sandbox.url)
    finally:
        sandbox.close()
    [(run, result)] = hardware_runs(tmp_path / "ws")
    assert run["backend"] == "remote" and result["evidence_level"] == "measured" and result["status"] == "succeeded"
    assert result["metrics"]["ttft_ms"]["p50"] == pytest.approx(120.0)
    record = json.loads((tmp_path / "ws" / run["folder"] / "measurement.json").read_text())
    assert record["identity"] == IDENTITY and record["step_records"]
    manifest = load(tmp_path / "ws" / run["folder"] / "manifest.json")
    assert manifest["provenance"]["source_kind"] == "measured" and "measurement.json" in manifest["sha256"]
    assert summary["alignment"]  # the slow flywheel still sets it against the frozen simulator prediction
    sent = sandbox.submitted[0]
    assert sent["spec"]["candidate_id"] == run["candidate"] and len(sent["requests"]) == result["requests"]["total"]


def test_a_test_fixture_sandbox_stays_mock(tmp_path):
    sandbox = FakeExecutor(kind="test_fixture")
    try:
        demo(tmp_path / "ws", "9.2", "fixture", hardware_url=sandbox.url)
    finally:
        sandbox.close()
    [(_, result)] = hardware_runs(tmp_path / "ws")
    assert result["evidence_level"] == "mock"


@pytest.mark.parametrize("sandbox", [{"identity": {**IDENTITY, "model_revision": "unknown"}}, {"drop_request": True}])
def test_an_incomplete_sandbox_result_fails_the_run_instead_of_counting(tmp_path, sandbox):
    service = FakeExecutor(**sandbox)
    try:
        with pytest.raises(LabError):  # the scripted demo stops when its measurement fails
            demo(tmp_path / "ws", "9.2", "fixture", hardware_url=service.url)
    finally:
        service.close()
    [(run, result)] = hardware_runs(tmp_path / "ws")
    assert run["status"] == "failed" and result["failure_class"] == "backend_error"


def test_remote_hardware_needs_its_url(tmp_path):
    with pytest.raises(LabError, match="hardware-url"):
        initialize(tmp_path / "ws", "9.2", "fixture", hardware="remote")
    initialize(tmp_path / "ok", "9.2", "fixture", hardware="remote", hardware_url="http://127.0.0.1:9/")
    assert load(tmp_path / "ok/.mlab/config.json")["hardware_service"] == {"base_url": "http://127.0.0.1:9"}


def test_measurement_record_contract():
    record = MeasurementRecord.model_validate({"observations": [{"request_id": "a"}], "identity": IDENTITY,
                                               "source": {"kind": "measured", "reference": "r"}})
    assert record.evidence_level == "measured"
    with pytest.raises(ValueError):
        MeasurementRecord.model_validate({"observations": [{"request_id": "a"}], "identity": {**IDENTITY, "devices": ["toy"]},
                                          "source": {"kind": "measured", "reference": "r"}})
