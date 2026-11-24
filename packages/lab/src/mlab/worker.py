import argparse
import time
from pathlib import Path
from .core.io import LabError, TERMINAL, load
from .core.store import Store, now
from .backends.local import STANDIN, finalize, fixture, load_requests, memory, observations, standin
from .backends.llm import estimate

# Local backends that wait a configured time first, so long jobs and queues can be rehearsed.
DELAYS = {"fixture": ("fixture_delay_s", "fixture_failure"), "standin": ("hardware_delay_s", "hardware_failure")}


def work(root, run_id):
    store = Store(root)
    store.begin()
    run = store.get("run", run_id)
    if run["status"] != "queued":
        store.rollback()
        store.close()
        return
    run.update(status="preparing", started_at=now(), progress={"phase": "loading_inputs"})
    store.put("run", run_id, run)
    store.commit()
    config = load(root / ".mlab/config.json")
    backend = run["backend"]
    try:
        spec = load(root / run["folder"] / "spec.json")
        if backend == "mlsys":
            from .backends.simulator_backend import execute
            execute(root, run, spec, config["simulator_service"])
            return
        if backend == "remote":
            measure_remotely(root, run_id, run, spec, config, store)
            return
        requests = load_requests(root, spec)
        catalog = load(root / "task/catalog.json")
        infra = spec["infra"]
        hardware = catalog["hardware"][infra["nodes"][0]["devices"][0]["hardware"]]
        overhead = 1.5 + (STANDIN["memory_overhead_gib"] if backend == "standin" else 0)
        peak, oom = memory(spec, hardware, catalog["models"], overhead)
        store.begin()
        current = store.get("run", run_id)
        if current["status"] == "cancelled":
            store.rollback()
            return
        phase = {"llm": "model_api", "fixture": "fixture", "standin": "hardware_queue"}[backend]
        current.update(status="running", progress={"phase": phase})
        store.put("run", run_id, current)
        store.commit()
        if backend in DELAYS:
            delay_key, failure_key = DELAYS[backend]
            until = time.monotonic() + config.get(delay_key, 0)
            started = time.monotonic()
            while time.monotonic() < until:
                if store.get("run", run_id)["status"] in {"cancelled", "timed_out", "failed"}:
                    return
                if time.monotonic() - started > config["run_timeout_s"]:
                    finalize(root, run_id, "timed_out", "timeout", [], {}, ["Wall-clock deadline exceeded."], {})
                    return
                time.sleep(.1)
            fault = config.get(failure_key)
            if fault:
                finalize(root, run_id, "failed", fault, [], {}, ["Injected failure: " + fault], {})
                return
        if oom:
            finalize(root, run_id, "failed", "oom", [], {"vram_peak_gib": peak}, ["Deterministic weight-memory lower bound exceeds usable device capacity."], {})
            return
        if backend == "fixture":
            data, provenance = fixture(spec, requests, peak), {"source": "deterministic synthetic fixture"}
            # Evaluation-only fault injection; never a student strategy parameter.
            rejected = config.get("fixture_reject_category")
            if rejected:
                for request, estimate_row in zip(requests, data["requests"]):
                    if request.get("category") == rejected:
                        estimate_row["status"] = "rejected"
        elif backend == "standin":
            data, observed = standin(spec, requests, peak, hardware["name"])
            provenance = {"source": "hardware stand-in with designed discrepancies", **observed}
        else:
            data, provenance = estimate(spec, requests, root / ".mlab/cache", provider=config.get("provider"))
        logs = ["Hardware stand-in: invented numbers, not a real GPU." if backend == "standin" else
                "Mock observations only. Shared metrics computed from requests.jsonl.", "Model secrets are never recorded."]
        if backend == "fixture" and config.get("fixture_log_bytes"):
            logs.append("X" * min(1_000_000, max(0, int(config["fixture_log_bytes"]))))
        rows = observations(data, requests, spec, schedule_resources=backend in {"fixture", "standin"})
        store.begin()
        current = store.get("run", run_id)
        if current["status"] in TERMINAL:
            store.rollback()
            return
        current.update(status="collecting", progress={"phase": "writing_artifacts"})
        store.put("run", run_id, current)
        store.event("run_phase", {"run_id": run_id, "phase": "collecting"})
        store.commit()
        finalize(root, run_id, "succeeded", None, rows,
                 {"vram_peak_gib": data["vram_peak_gib"], "kv_peak_gib": data["kv_peak_gib"]},
                 logs, provenance)
    except LabError as error:
        finalize(root, run_id, "failed", "backend_error", [], {}, [error.code + ": " + str(error)], {"error_code": error.code})
    except Exception as error:
        finalize(root, run_id, "failed", "backend_error", [], {}, ["E_MOCK_INVALID: worker could not produce valid artifacts: " + type(error).__name__], {})
    finally:
        store.close()


def measure_remotely(root, run_id, run, spec, config, store):
    """A measurement on a GPU sandbox over mlab.executor.v1; the raw record is kept as measurement.json."""
    from .contracts import RequestObservation
    from .executors.remote import RemoteExecutor
    requests = load_requests(root, spec)
    store.begin()
    current = store.get("run", run_id)
    if current["status"] in TERMINAL:
        store.rollback()
        return
    current.update(status="running", progress={"phase": "hardware_queue"})
    store.put("run", run_id, current)
    store.commit()
    service = config["hardware_service"]
    executor = RemoteExecutor(service["base_url"], poll_s=service.get("poll_s", 2.0), timeout_s=config["run_timeout_s"])
    stopped = lambda: store.get("run", run_id)["status"] in {"cancelled", "timed_out", "failed"}
    job, record = executor.execute(root, run, spec, requests, cancelled=stopped)
    inputs = {row["request_id"]: row for row in requests}
    rows = [RequestObservation(**{**observation, "depends_on": inputs[observation["request_id"]].get("depends_on", []),
                                  "gap_after_deps_ms": inputs[observation["request_id"]].get("gap_after_deps_ms", 0)}).model_dump()
            for observation in record.observations]
    provenance = {"source": "remote hardware executor", "endpoint": service["base_url"], "job_id": job,
                  "source_kind": record.source.kind, "source_reference": record.source.reference,
                  "identity": record.identity.model_dump()}
    finalize(root, run_id, "succeeded", None, rows, {"vram_peak_gib": record.vram_peak_gib, "kv_peak_gib": record.kv_peak_gib},
             [*record.logs, f"Measured by {service['base_url']} job {job} ({record.source.kind})."], provenance,
             wire={"measurement.json": record.model_dump_json().encode()}, evidence=record.evidence_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run", required=True)
    arguments = parser.parse_args()
    work(arguments.workspace.resolve(), arguments.run)
