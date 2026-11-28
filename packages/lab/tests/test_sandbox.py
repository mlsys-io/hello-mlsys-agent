"""The vLLM sandbox harness: the design->engine mapping, its refusals, and the record it returns.

No GPU is needed. `measure()` itself (vLLM + a card) is exercised by hand on a real GPU; everything else is here."""
import asyncio
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mlab.executors.base import MeasurementRecord
from mlab.sandbox import vllm_harness as harness

BASELINE = {
    "deployments": [{"model": "llm-1.7b", "weight_precision": "BF16", "kv_precision": "BF16", "context_tokens": 16384,
                     "devices": ["gpu0"], "replicas": 1, "parallel": {"kind": "none", "degree": 1}}],
    "serving": {"max_concurrency": 32, "gpu_memory_utilization": 0.8, "kv_fraction": None, "kv_budget_gib": None,
                "chunked_prefill_tokens": None, "continuous_batching": True},
    "cache": {}, "scheduling": {}, "speculative": False}


def design(**serving):
    strategy = copy.deepcopy(BASELINE)
    strategy["serving"].update(serving)
    return {"strategy": strategy}


def flag(args, name):
    return args[args.index(name) + 1]


def test_baseline_maps_to_engine_arguments():
    engine = harness.plan(design())
    assert engine["model"] == "Qwen/Qwen3-1.7B" and engine["context"] == 16384
    args = engine["args"]
    assert flag(args, "--max-model-len") == "16384" and flag(args, "--max-num-seqs") == "32"
    assert flag(args, "--gpu-memory-utilization") == "0.8" and flag(args, "--dtype") == "bfloat16"
    assert "--no-enable-prefix-caching" in args and "--enable-chunked-prefill" not in args
    assert flag(args, "--max-num-batched-tokens") == "16384"  # a whole-context budget is, in effect, unchunked prefill


def test_chunked_prefill_and_kv_budget_and_prefix_cache():
    engine = harness.plan(design(chunked_prefill_tokens=512, kv_budget_gib=2.0))
    args = engine["args"]
    assert "--enable-chunked-prefill" in args and flag(args, "--max-num-batched-tokens") == "512"
    assert flag(args, "--kv-cache-memory-bytes") == str(2 * 1024 ** 3)
    cached = design()
    cached["strategy"]["cache"] = {"prefix_cache": True}
    assert "--enable-prefix-caching" in harness.plan(cached)["args"]


def test_every_model_resolves_to_a_pinned_revision():
    for alias in ("llm-0.6b", "llm-1.7b", "llm-4b", "qwen3-1.7b"):
        spec = design()
        spec["strategy"]["deployments"][0]["model"] = alias
        engine = harness.plan(spec)
        assert len(engine["revision"]) == 40 and engine["revision"] == harness.REVISIONS[engine["model"]]


@pytest.mark.parametrize("change, words", [
    (lambda s: s["strategy"]["serving"].update(continuous_batching=False, max_concurrency=4), "batches continuously"),
    (lambda s: s["strategy"]["serving"].update(kv_fraction=0.5), "KV-cache fraction"),
    (lambda s: s["strategy"].update(speculative=True), "speculative"),
    (lambda s: s["strategy"]["deployments"].append(copy.deepcopy(s["strategy"]["deployments"][0])), "one deployment"),
    (lambda s: s["strategy"]["deployments"][0].update(replicas=2), "replicas"),
    (lambda s: s["strategy"]["deployments"][0]["parallel"].update(degree=2), "parallelism"),
    (lambda s: s["strategy"]["deployments"][0].update(weight_precision="FP8"), "BF16"),
    (lambda s: s["strategy"]["deployments"][0].update(model="llm-32b"), "no model is published"),
])
def test_a_design_vllm_cannot_reproduce_is_refused_not_approximated(change, words):
    spec = design()
    change(spec)
    with pytest.raises(harness.Unsupported, match=words):
        harness.plan(spec)


def test_non_continuous_batching_is_fine_for_one_request_at_a_time():
    harness.plan(design(continuous_batching=False, max_concurrency=1))  # the 9.1 designs


def test_prompts_are_exact_length_valid_ids_and_stable():
    row = {"request_id": "r003", "input_tokens": 6144}
    ids = harness.prompt_ids(row)
    assert len(ids) == 6144 and all(harness.VOCAB_LOW <= i < harness.VOCAB_HIGH for i in ids)
    assert ids == harness.prompt_ids(row)  # not Python's per-process hash()
    assert ids != harness.prompt_ids({"request_id": "r004", "input_tokens": 6144})


def observation(request_id="r001", ready=0.0, first=80.0, finished=400.0, tokens=64, status="completed"):
    return {"request_id": request_id, "ready_ms": ready, "started_ms": ready, "first_token_ms": first,
            "finished_ms": finished, "output_tokens": tokens, "output_tokens_target": 64, "status": status,
            "service_ms": None if finished is None else finished - ready}


LOG = "INFO ... Available KV cache memory: 4.00 GiB\nINFO ... Application startup complete."


def record(**override):
    base = dict(observations=[observation()], vram_peak_mib=13000, kv_usage_peak=0.25, log_text=LOG,
                engine=harness.plan(design()), revision="70d244cc", gpu=("NVIDIA GeForce RTX 5080", "GPU-abc", "595.99.02"),
                versions={"vllm": "0.34.0", "torch": "2.14.0+cu130"}, ready_s=430, label="t")
    return harness.build_record(**(base | override))


def test_the_record_validates_against_the_labs_own_contract_and_counts_as_measured():
    parsed = MeasurementRecord.model_validate(record())
    assert parsed.evidence_level == "measured"
    assert parsed.vram_peak_gib == round(13000 / 1024, 3) and parsed.kv_peak_gib == 1.0  # 4.00 GiB reserved x 25% used
    assert parsed.identity.devices == ["NVIDIA GeForce RTX 5080"] and parsed.identity.runtime["name"] == "vllm"
    assert parsed.identity.model_revision == "Qwen/Qwen3-1.7B@70d244cc"


def test_an_explicit_kv_budget_is_the_reserved_size_whatever_the_log_says():
    engine = harness.plan(design(kv_budget_gib=2.0))
    parsed = MeasurementRecord.model_validate(record(engine=engine, log_text="nothing useful", kv_usage_peak=0.5))
    assert parsed.kv_peak_gib == 1.0  # 2 GiB budget x 50% peak usage


def test_kv_log_lines_are_kept_in_the_record_for_diagnosis():
    noisy = "INFO a\nINFO GPU KV cache size: 123,456 tokens\nINFO b\nINFO Available KV cache memory: 4.00 GiB"
    logs = record(log_text=noisy)["logs"]
    assert any("GPU KV cache size: 123,456 tokens" in line for line in logs)


def test_kv_peak_is_absent_not_invented_when_the_engine_did_not_report_it():
    assert record(log_text="no memory line here")["kv_peak_gib"] is None
    assert record(kv_usage_peak=0.0)["kv_peak_gib"] is None


def test_failed_and_timed_out_requests_stay_visible_in_the_record():
    failed = observation("r002", first=None, finished=None, tokens=0, status="timed_out")
    parsed = MeasurementRecord.model_validate(record(observations=[observation(), failed]))
    assert [o["status"] for o in parsed.observations] == ["completed", "timed_out"]


# --- replay timing against a fake vLLM streaming endpoint (needs aiohttp, which the vLLM image ships) -------------

class FakeVLLM(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        import time
        time.sleep(0.05)  # "prefill"
        for _ in range(body["max_tokens"]):
            self.wfile.write(b'data: {"choices":[{"text":" a"}]}\n\n')
            self.wfile.flush()
            time.sleep(0.005)
        usage = {"usage": {"completion_tokens": body["max_tokens"]}}
        self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\ndata: [DONE]\n\n")


def test_replay_records_arrival_first_token_and_finish_times():
    pytest.importorskip("aiohttp")
    import time
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rows = [{"request_id": "r001", "arrival_ms": 0, "input_tokens": 8, "output_tokens_target": 6},
                {"request_id": "r002", "arrival_ms": 300, "input_tokens": 8, "output_tokens_target": 6}]
        t0 = time.monotonic() + 0.2
        out = asyncio.run(harness.replay(server.server_port, "m", rows, t0, t0 + 30))
    finally:
        server.shutdown()
    assert [o["status"] for o in out] == ["completed", "completed"] and all(o["output_tokens"] == 6 for o in out)
    first, second = out
    assert first["ready_ms"] == 0 and second["ready_ms"] == 300
    assert 40 <= first["first_token_ms"] - first["ready_ms"] < 400  # ~50 ms "prefill"
    assert second["finished_ms"] > second["first_token_ms"] >= second["ready_ms"]
    MeasurementRecord.model_validate(record(observations=out))  # the replay's rows satisfy the contract too


def test_the_cli_refuses_with_a_code_before_touching_a_gpu(monkeypatch, capsys):
    import io
    job = {"spec": design(kv_fraction=0.5), "requests": []}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(job)))
    assert harness.main() == 2
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and out["error"]["code"] == "E_UNSUPPORTED" and "KV-cache fraction" in out["error"]["message"]


# --- batches: measure_many with the download, server, sampler and replay layers faked ----------------------------

GPU_NAME = ("NVIDIA GeForce RTX 5080", "GPU-abc", "595.99.02")


def rows(*ids, tokens=8):
    return [{"request_id": i, "arrival_ms": 0, "input_tokens": tokens, "output_tokens_target": 4} for i in ids]


def spec(model="llm-1.7b", prefix_cache=False, **serving):
    out = design(**serving)
    out["strategy"]["deployments"][0]["model"] = model
    out["strategy"]["cache"] = {"prefix_cache": prefix_cache}
    return out


class FakeGPU:
    """The layers under measure_many, on a fake clock. Each replay reports the VRAM/KV peak `peaks` gives it."""

    def __init__(self, monkeypatch, tmp_path, startup_s=100.0, replay_s=50.0, download_s=30.0, nvml=None):
        self.now, self.startup_s, self.replay_s, self.download_s = 0.0, startup_s, replay_s, download_s
        self.nvml, self.meters, self.smi_power = nvml, [], ["[N/A]"]  # no NVML, no power reading: by default
        self.downloads, self.servers, self.replays, self.kvs, self.resets = [], [], [], [], []
        self.fail_startup, self.reset_ok, self.peaks, self.timed_out = set(), True, {}, False
        fake = self

        class Clock:
            @staticmethod
            def monotonic():
                return fake.now

            @staticmethod
            def sleep(seconds):
                fake.now += seconds

        class Server:
            def __init__(self, path, engine, env=None, log_path=None):
                self.path, self.engine, self.env, self.port, self.stopped = path, engine, env or {}, 9000, False
                self.process, self.memory = None, harness.MemorySampler()  # never started: the fake sets its peak
                fake.servers.append(self)

            def log_text(self):
                return LOG

            def stop(self):
                self.stopped = True

        class KV:
            def __init__(self, port):
                self.peak, self.stop = 0.0, threading.Event()
                fake.kvs.append(self)

            def start(self):
                pass

        def wait_ready(port, process, deadline):
            server = fake.servers[-1]
            fake.now += fake.startup_s
            server.memory.peak_mib = 8000.0  # the startup profiling peak
            if len(fake.servers) in fake.fail_startup:
                raise RuntimeError("vLLM exited with status 1 during startup")

        async def replay(port, model, requests, t0, deadline):
            server = fake.servers[-1]
            fake.replays.append((len(fake.servers), [r["request_id"] for r in requests], deadline))
            vram, kv = fake.peaks.get(requests[0]["request_id"], (9000.0, 0.25))
            server.memory.peak_mib, fake.kvs[-1].peak = vram, kv
            end = max(fake.now + fake.replay_s, t0)
            if fake.nvml is not None:  # idle until the first release, busy until the last finish
                fake.nvml.run_until(t0)
                fake.now, fake.nvml.busy = t0, True
                fake.sample()  # what the meter's sampling thread reads as the replay starts...
                fake.nvml.run_until(end)
                fake.now = end
                fake.sample()  # ... and as it ends
                fake.nvml.busy = False
            fake.now = end
            if fake.timed_out:
                fake.now = max(fake.now, deadline)  # every request ran into the deadline
            status = "timed_out" if fake.timed_out else "completed"
            return [observation(r["request_id"], status=status) for r in requests]

        def download(model, revision):
            fake.downloads.append((model, revision))
            fake.now += fake.download_s
            return "/tmp/hf/" + model

        def reset(port):
            fake.resets.append(port)
            return fake.reset_ok

        def smi(*fields):
            if fields == ("power.draw",):
                return list(fake.smi_power)
            return ["0"] if fields == ("memory.used",) else list(GPU_NAME)

        class Meter(harness.EnergyMeter):
            def __init__(self, uuid=None, period_s=harness.POWER_PERIOD_S, nvml=None):
                super().__init__(uuid, period_s, nvml)
                fake.meters.append(self)

            def start(self):  # no sampling thread on a fake clock: the fake replay samples at t0 instead
                pass

        self.versions_calls = 0
        if nvml is not None:
            nvml.clock = self

        def versions():
            self.versions_calls += 1
            return {"vllm": "0.34.0", "torch": "2.14.0+cu130"}

        for name, value in {"time": Clock, "Server": Server, "KVSampler": KV, "wait_ready": wait_ready,
                            "replay": replay, "download_weights": download, "reset_prefix_cache": reset, "smi": smi,
                            "runtime_versions": versions, "EnergyMeter": Meter,
                            "nvml_module": lambda: fake.nvml}.items():
            monkeypatch.setattr(harness, name, value)
        monkeypatch.setenv("MLAB_HF_RO_CACHE", str(tmp_path / "ro-cache"))  # empty: every model is a miss


    def sample(self):
        for meter in self.meters:
            meter.mark()


class FakeNVML:
    """The NVML calls the energy meter makes, on FakeGPU's clock: `idle_w` while nothing runs, `busy_w` during a
    replay. `counter` / `power` False make that call raise NVML_ERROR_NOT_SUPPORTED, as on a card without it."""

    class NVMLError(Exception):
        pass

    def __init__(self, idle_w=30.0, busy_w=180.0, counter=True, power=True, by_uuid=True):
        self.idle_w, self.busy_w, self.counter, self.power, self.by_uuid = idle_w, busy_w, counter, power, by_uuid
        self.clock, self.t, self.mj, self.busy, self.handles = None, 0.0, 7.5e9, False, []

    def run_until(self, t):
        if t > self.t:
            self.mj += (t - self.t) * (self.busy_w if self.busy else self.idle_w) * 1000
            self.t = t

    def nvmlInit(self):
        pass

    def nvmlDeviceGetHandleByUUID(self, uuid):
        if not self.by_uuid:
            raise self.NVMLError("Not Found")
        self.handles.append(uuid)
        return "handle:" + uuid

    def nvmlDeviceGetHandleByIndex(self, index):
        self.handles.append(index)
        return f"handle:{index}"

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        if not self.counter:
            raise self.NVMLError("Not Supported")
        self.run_until(self.clock.now)
        return int(self.mj)  # mJ

    def nvmlDeviceGetPowerUsage(self, handle):
        if not self.power:
            raise self.NVMLError("Not Supported")
        self.run_until(self.clock.now)
        return int((self.busy_w if self.busy else self.idle_w) * 1000)  # mW


@pytest.fixture
def gpu(monkeypatch, tmp_path):
    return FakeGPU(monkeypatch, tmp_path)


def by_id(out):
    return {result["id"]: result for result in out["results"]}


def test_a_batch_downloads_each_model_once_and_starts_one_server_per_distinct_configuration(gpu):
    designs = [{"id": "a", "spec": spec()}, {"id": "b", "spec": spec("llm-4b")},
               {"id": "c", "spec": spec(max_concurrency=8)}, {"id": "d", "spec": spec(max_concurrency=8)},
               {"id": "e", "spec": spec("llm-4b", max_concurrency=4)}]
    out = harness.measure_many(designs, rows("r001", "r002"))
    assert [r["id"] for r in out["results"]] == list("abcde") and {r["status"] for r in out["results"]} == {"measured"}
    assert gpu.downloads == [("Qwen/Qwen3-1.7B", harness.REVISIONS["Qwen/Qwen3-1.7B"]),
                             ("Qwen/Qwen3-4B", harness.REVISIONS["Qwen/Qwen3-4B"])]
    # grouped by model: both 1.7B configurations run before the 4B ones, whatever the input order
    assert [s.engine["model"].split("-")[-1] for s in gpu.servers] == ["1.7B", "1.7B", "4B", "4B"]
    assert all(s.stopped for s in gpu.servers) and len(gpu.replays) == 4
    summary = out["summary"]
    assert (summary["downloads"], summary["servers"], summary["replays"], summary["measured"]) == (2, 4, 4, 5)
    for result in out["results"]:
        MeasurementRecord.model_validate(result["record"])  # the batch record is still the lab's contract


def test_designs_with_identical_engine_arguments_and_workload_share_one_replay(gpu):
    out = harness.measure_many([{"id": "x", "spec": spec()}, {"id": "y", "spec": spec()}], rows("r001"), label="c7")
    x, y = by_id(out)["x"], by_id(out)["y"]
    assert len(gpu.servers) == 1 and len(gpu.replays) == 1
    assert x["shares_replay_with"] == ["y"] and y["shares_replay_with"] == ["x"]
    assert x["record"]["observations"] == y["record"]["observations"]
    assert x["record"]["observations"] is not y["record"]["observations"]
    assert x["record"]["source"]["reference"].endswith("/c7-x") and y["record"]["source"]["reference"].endswith("/c7-y")


def test_one_server_replays_each_workload_with_fresh_peaks(gpu):
    gpu.peaks = {"d1": (15000.0, 0.9), "h1": (7000.0, 0.1)}
    designs = [{"id": "dev", "spec": spec(), "requests": rows("d1")},
               {"id": "holdout", "spec": spec(), "requests": rows("h1")}]
    out = harness.measure_many(designs)
    dev, holdout = by_id(out)["dev"]["record"], by_id(out)["holdout"]["record"]
    assert len(gpu.servers) == 1 and [r[1] for r in gpu.replays] == [["d1"], ["h1"]]
    assert len(gpu.kvs) == 2  # a KV sampler per workload
    assert dev["vram_peak_gib"] == round(15000 / 1024, 3) and dev["kv_peak_gib"] == round(4.0 * 0.9, 3)
    # the second workload does not inherit the first one's peaks; the server's startup peak is common to both
    assert holdout["vram_peak_gib"] == round(8000 / 1024, 3) and holdout["kv_peak_gib"] == round(4.0 * 0.1, 3)
    assert gpu.resets == []  # no prefix cache, nothing to empty
    assert dev["timings"]["server"] == holdout["timings"]["server"] == 1


def test_a_prefix_cache_is_emptied_between_workloads_or_the_server_restarted(gpu):
    designs = [{"id": "dev", "spec": spec(prefix_cache=True), "requests": rows("d1")},
               {"id": "holdout", "spec": spec(prefix_cache=True), "requests": rows("h1")}]
    harness.measure_many(designs)
    assert len(gpu.resets) == 1 and len(gpu.servers) == 1
    assert gpu.servers[0].env.get("VLLM_SERVER_DEV_MODE") == "1"
    gpu.reset_ok, gpu.resets, gpu.servers = False, [], []
    out = harness.measure_many(designs)
    assert len(gpu.servers) == 2 and gpu.servers[0].stopped and out["summary"]["measured"] == 2


def test_distinct_arguments_restart_the_server_each_time(gpu):
    designs = [{"id": f"d{i}", "spec": spec(max_concurrency=c)} for i, c in enumerate((1, 8, 1))]
    harness.measure_many(designs, rows("r001"))
    assert len(gpu.servers) == 2 and len(gpu.replays) == 2  # 1 and 1 are the same configuration
    assert [flag(s.engine["args"], "--max-num-seqs") for s in gpu.servers] == ["1", "8"]


def test_phase_timings_are_in_each_record_and_the_summary(gpu):
    out = harness.measure_many([{"id": "a", "spec": spec()}, {"id": "b", "spec": spec(max_concurrency=8)}],
                               rows("r001"))
    a = by_id(out)["a"]["record"]
    assert a["weights_from"] == "download"
    assert a["timings"] == {"download_s": 30.0, "startup_s": 100.0, "replay_s": 50.0, "server": 1,
                            "batch_total_s": out["summary"]["total_s"]}
    summary = out["summary"]
    assert (summary["download_s"], summary["startup_s"], summary["replay_s"]) == (30.0, 200.0, 100.0)
    assert summary["total_s"] == 30 + 2 * (100 + 50) and summary["deadline_s"] == harness.DEADLINE_S


def test_the_deadline_stops_starting_servers_that_cannot_finish(gpu):
    gpu.startup_s, gpu.replay_s = 1000.0, 200.0
    designs = [{"id": str(c), "spec": spec(max_concurrency=c)} for c in (1, 2, 4, 8)]
    out = harness.measure_many(designs, rows("r001"), deadline_s=3300)
    # 30 s download + 1200 s per server: the third would end at ~3633 s, past 3300
    assert [r["status"] for r in out["results"]] == ["measured", "measured", "not_measured", "not_measured"]
    assert by_id(out)["4"] == {"id": "4", "status": "not_measured", "reason": "out of time in this GPU session"}
    assert len(gpu.servers) == 2 and out["summary"]["not_measured"] == 2


def test_a_replay_cut_by_the_session_deadline_is_not_measured_not_a_slow_design(gpu):
    gpu.timed_out = True
    out = harness.measure_many([{"id": "a", "spec": spec()}], rows("r001"), deadline_s=200)
    assert gpu.replays[0][2] == 200  # the replay deadline is capped by the session's
    assert out["results"][0]["status"] == "not_measured"


def test_a_failed_startup_fails_its_designs_and_the_batch_goes_on(gpu):
    gpu.fail_startup = {1}
    out = harness.measure_many([{"id": "a", "spec": spec()}, {"id": "b", "spec": spec(max_concurrency=8)}],
                               rows("r001"))
    a, b = by_id(out)["a"], by_id(out)["b"]
    assert a["status"] == "failed" and a["error"]["code"] == "E_SANDBOX" and "during startup" in a["error"]["message"]
    assert b["status"] == "measured" and gpu.servers[0].stopped


def test_every_design_is_checked_before_the_gpu_stack_is_imported(gpu, monkeypatch):
    planned = []
    real_plan = harness.plan
    monkeypatch.setattr(harness, "plan", lambda s: planned.append(s) or real_plan(s))

    def versions():
        assert len(planned) == 3  # all three designs, including the refused one after a good one
        return {"vllm": "0.34.0", "torch": "2.14.0+cu130"}

    monkeypatch.setattr(harness, "runtime_versions", versions)
    out = harness.measure_many([{"id": "a", "spec": spec()}, {"id": "bad", "spec": design(kv_fraction=0.5)},
                                {"id": "c", "spec": spec(max_concurrency=8)}], rows("r001"))
    bad = by_id(out)["bad"]
    assert bad["status"] == "unsupported" and bad["error"]["code"] == "E_UNSUPPORTED"
    assert "KV-cache fraction" in bad["error"]["message"] and out["summary"]["unsupported"] == 1


def test_an_all_refused_batch_never_imports_torch_or_vllm(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` would raise ImportError
    monkeypatch.setitem(sys.modules, "vllm", None)
    out = harness.measure_many([{"id": "a", "spec": design(kv_fraction=0.5)},
                                {"id": "b", "spec": spec("llm-32b")}], [])
    assert [r["status"] for r in out["results"]] == ["unsupported", "unsupported"]


@pytest.mark.parametrize("designs", [[], [{"id": "a"}], [{"spec": {}}], [{"id": "a", "spec": {}}] * 2])
def test_a_malformed_batch_is_refused_before_anything_runs(designs, monkeypatch):
    monkeypatch.setattr(harness, "runtime_versions", lambda: pytest.fail("imported the GPU stack"))
    with pytest.raises(ValueError):
        harness.measure_many(designs, [])


# --- the read-only weight cache ------------------------------------------------------------------------------------

def hub_snapshot(root, model, revision, files):
    """A standard hub cache entry: snapshot files are symlinks into blobs/."""
    folder = root / ("models--" + model.replace("/", "--"))
    blobs, snapshot = folder / "blobs", folder / "snapshots" / revision
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        blob = blobs / ("blob-" + name)
        if content is not None:
            blob.write_text(content)
        (snapshot / name).symlink_to(blob)  # content None: a dangling link, as an interrupted download leaves
    return snapshot


SHARDED = {"config.json": "{}", "tokenizer.json": "{}", "tokenizer_config.json": "{}",
           "model.safetensors.index.json": json.dumps({"weight_map": {"a": "model-1.safetensors",
                                                                      "b": "model-2.safetensors"}}),
           "model-1.safetensors": "w1", "model-2.safetensors": "w2"}
MODEL, REVISION = "Qwen/Qwen3-1.7B", harness.REVISIONS["Qwen/Qwen3-1.7B"]


def test_a_complete_snapshot_is_found_in_a_hub_cache_or_an_hf_home(tmp_path):
    snapshot = hub_snapshot(tmp_path / "hub", MODEL, REVISION, SHARDED)
    assert harness.cached_snapshot(MODEL, REVISION, root=str(tmp_path / "hub")) == str(snapshot)
    assert harness.cached_snapshot(MODEL, REVISION, root=str(tmp_path)) == str(snapshot)  # HF_HOME/hub/...
    assert harness.cached_snapshot(MODEL, "0" * 40, root=str(tmp_path)) is None  # another revision
    assert harness.cached_snapshot(MODEL, REVISION, root="") is None  # disabled
    single = {k: v for k, v in SHARDED.items() if not k.startswith("model")} | {"model.safetensors": "w"}
    hub_snapshot(tmp_path / "other", "Qwen/Qwen3-0.6B", "rev", single)
    assert harness.cached_snapshot("Qwen/Qwen3-0.6B", "rev", root=str(tmp_path / "other"))


@pytest.mark.parametrize("broken", [
    lambda f: f.pop("model-2.safetensors"),               # a shard the index names is missing
    lambda f: f.update({"model-2.safetensors": None}),     # a dangling symlink (interrupted download)
    lambda f: f.pop("tokenizer.json"),
    lambda f: f.update({"model.safetensors.index.json": "not json"}),
    lambda f: [f.pop(k) for k in ("model.safetensors.index.json", "model-1.safetensors", "model-2.safetensors")],
])
def test_an_incomplete_snapshot_is_a_miss(tmp_path, broken):
    files = dict(SHARDED)
    broken(files)
    hub_snapshot(tmp_path, MODEL, REVISION, files)
    assert harness.cached_snapshot(MODEL, REVISION, root=str(tmp_path)) is None


def test_a_cache_hit_serves_offline_without_downloading(gpu, tmp_path, monkeypatch):
    snapshot = hub_snapshot(tmp_path / "ro", MODEL, REVISION, SHARDED)
    monkeypatch.setenv("MLAB_HF_RO_CACHE", str(tmp_path / "ro"))
    out = harness.measure_many([{"id": "a", "spec": spec()}, {"id": "b", "spec": spec("llm-4b")}], rows("r001"))
    a, b = by_id(out)["a"]["record"], by_id(out)["b"]["record"]
    assert a["weights_from"] == "cache" and b["weights_from"] == "download"
    assert gpu.downloads == [("Qwen/Qwen3-4B", harness.REVISIONS["Qwen/Qwen3-4B"])]
    assert gpu.servers[0].path == str(snapshot) and gpu.servers[0].env.get("HF_HUB_OFFLINE") == "1"
    assert "HF_HUB_OFFLINE" not in gpu.servers[1].env
    assert (out["summary"]["cache_hits"], out["summary"]["downloads"]) == (1, 1)


# --- measure() is one design through measure_many --------------------------------------------------------------------

def test_measure_still_returns_one_record(gpu):
    out = harness.measure({"spec": spec(), "requests": rows("r001", "r002"), "label": "cand-dev"})
    parsed = MeasurementRecord.model_validate(out)
    assert parsed.source.reference.endswith("/cand-dev") and [o["request_id"] for o in out["observations"]] == ["r001", "r002"]
    assert parsed.weights_from == "download" and len(gpu.servers) == 1 and gpu.servers[0].stopped


def test_measure_still_raises_unsupported_and_runtime_errors(gpu):
    with pytest.raises(harness.Unsupported, match="speculative"):
        harness.measure({"spec": {"strategy": {**design()["strategy"], "speculative": True}}, "requests": rows("r1")})
    gpu.fail_startup = {1}
    with pytest.raises(RuntimeError, match="during startup"):
        harness.measure({"spec": spec(), "requests": rows("r001")})


def test_a_patched_replay_is_what_measure_uses(gpu, monkeypatch):
    """The sandbox adapter replaces `harness.replay` for sequential workloads; the batch path must honour it."""
    seen = []

    async def sequential(port, model, requests, t0, deadline):
        seen.append([r["request_id"] for r in requests])
        return [observation(r["request_id"]) for r in requests]

    monkeypatch.setattr(harness, "replay", sequential)
    harness.measure_many([{"id": "a", "spec": spec()}], rows("r001", "r002"))
    assert seen == [["r001", "r002"]]


def test_the_cli_runs_a_batch(gpu, monkeypatch, capsys):
    import io
    job = {"designs": [{"id": "a", "spec": spec()}, {"id": "bad", "spec": design(kv_fraction=0.5)}],
           "requests": rows("r001"), "label": "c1", "deadline_s": 3000}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(job)))
    assert harness.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and [r["status"] for r in out["results"]] == ["measured", "unsupported"]
    assert out["summary"]["deadline_s"] == 3000


# --- traces and resamples: the bytes the simulator scored are the bytes replayed --------------------------------------

LONG_MIX_2 = {"seed": 2, "variant": "long_mix"}  # the simulated worst case of the 9.2 pick (0.6B, c16)
RTX5080_PROFILE = "protocol-v1.21"  # the card the sandbox measures


def lab_traces(task="9.2", resamples=(LONG_MIX_2,)):
    """What the sandbox adapter's CPU prepare stage sends: the development trace and each resample as the exact text of
    the file the simulator reads, with its frozen sha256. The resamples come from the code `evaluate` itself uses."""
    from mlab.core.io import inside
    from mlab.jobs import _exercise, _resample_requests, _resample_traces
    with _exercise(task, RTX5080_PROFILE) as lab:
        def text(reference):
            return inside(lab.root / "task", reference["path"]).read_bytes().decode("utf-8")
        development = lab.task["frozen"]["workload_requests"]
        out = {"development": {"text": text(development), "sha256": development["sha256"]}}
        for trace in _resample_traces(lab, _resample_requests({"task": task, "resamples": list(resamples)})):
            out[trace["workload"]] = {"text": text(trace["reference"]), "sha256": trace["reference"]["sha256"],
                                      "resample": {"seed": trace["seed"], "variant": trace["variant"]}}
    return out


def simulated_sha256(task="9.2", resamples=(LONG_MIX_2,)):
    """The trace_sha256 the simulator's `evaluate` reports for each resample, for the 0.6B c16 pick."""
    from mlab.jobs import run
    pick = json.loads(run("templates", json.dumps({"task": task, "profile": RTX5080_PROFILE})))["templates"][0]
    pick = {**pick, "id": "pick"}
    pick["strategy"]["deployments"][0]["model"] = "llm-0.6b"
    pick["strategy"]["serving"]["max_concurrency"] = 16
    out = json.loads(run("evaluate", json.dumps({"task": task, "profile": RTX5080_PROFILE, "candidates": [pick],
                                                 "seeds": [42], "workloads": [], "resamples": list(resamples)})))
    assert all(row["status"] == "succeeded" for row in out["results"]), out["results"]
    return {row["workload"]: row["trace_sha256"] for row in out["results"]}


def test_a_resample_is_replayed_byte_for_byte_and_reports_the_simulators_trace_sha256(gpu, monkeypatch):
    """The 9.2 worst case (0.6B, c16, long_mix:2) can only be confirmed on the card if the card replays the very
    trace the simulator scored: same generator, same bytes, same sha256 on both sides."""
    traces = lab_traces()
    replayed = []
    fake_replay = harness.replay

    async def spy(port, model, requests, t0, deadline):
        replayed.append(copy.deepcopy(requests))
        return await fake_replay(port, model, requests, t0, deadline)

    monkeypatch.setattr(harness, "replay", spy)
    c16 = spec("llm-0.6b", max_concurrency=16)
    out = harness.measure_many([{"id": name, "spec": c16, "trace": trace} for name, trace in traces.items()])
    dev, resample = by_id(out)["development"]["record"], by_id(out)["resample:long_mix:2"]["record"]
    assert resample["trace_sha256"] == simulated_sha256()["resample:long_mix:2"]
    assert dev["trace_sha256"] == traces["development"]["sha256"]  # the frozen development trace's own hash
    assert (resample["resample_seed"], resample["resample_variant"]) == (2, "long_mix")
    assert "resample_seed" not in dev
    # one server, two replays, each of exactly the rows in its text, parsed as the simulator backend parses them
    assert len(gpu.servers) == 1 and len(replayed) == 2
    for name, rows_sent in zip(traces, replayed):
        assert rows_sent == [json.loads(line) for line in traces[name]["text"].splitlines() if line.strip()]
    assert [r["arrival_ms"] for r in replayed[0]] == [r["arrival_ms"] for r in replayed[1]]  # arrivals kept
    assert [r["input_tokens"] for r in replayed[0]] != [r["input_tokens"] for r in replayed[1]]  # prompts redrawn
    for record in (dev, resample):
        MeasurementRecord.model_validate(record)
    assert out["summary"]["resamples"] == 1 and out["summary"]["measured"] == 2


def test_a_trace_whose_text_does_not_match_its_sha256_is_refused_before_anything_runs(monkeypatch):
    monkeypatch.setattr(harness, "runtime_versions", lambda: pytest.fail("imported the GPU stack"))
    trace = dict(lab_traces()["resample:long_mix:2"])
    trace["text"] = trace["text"].replace('"input_tokens": ', '"input_tokens": 1', 1)
    with pytest.raises(ValueError, match="hashes to sha256:"):
        harness.measure_many([{"id": "r", "spec": spec(), "trace": trace}])


@pytest.mark.parametrize("trace, words", [
    ({"text": "{}\n"}, "a trace is"),
    ({"text": "{}\n", "sha256": "abc"}, "a trace is"),
    ({"text": "", "sha256": "sha256:" + "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
     "at least one request"),
])
def test_a_malformed_trace_is_refused(trace, words):
    with pytest.raises(ValueError, match=words):
        harness.measure_many([{"id": "r", "spec": spec(), "trace": trace}])


def test_a_design_names_requests_or_a_trace_not_both():
    trace = lab_traces(resamples=())["development"]
    with pytest.raises(ValueError, match="both 'requests' and a 'trace'"):
        harness.measure_many([{"id": "r", "spec": spec(), "trace": trace, "requests": rows("r001")}])


def test_at_most_two_resamples_per_gpu_session(gpu, monkeypatch):
    three = lab_traces(resamples=(LONG_MIX_2, {"seed": 7, "variant": "long_mix"}, {"seed": 7, "variant": "short_mix"}))
    designs = [{"id": name, "spec": spec(), "trace": trace} for name, trace in three.items()]
    with pytest.raises(ValueError, match="at most 2 resampled workloads per GPU session"):
        harness.measure_many(designs)
    assert gpu.servers == [] and gpu.versions_calls == 0
    # the same resample for two designs is one resampled workload
    one = three["resample:long_mix:2"]
    out = harness.measure_many([{"id": "a", "spec": spec(), "trace": one}, {"id": "b", "spec": spec(max_concurrency=8),
                                                                          "trace": one}, *designs[:2]])
    assert out["summary"]["resamples"] == 1 and out["summary"]["measured"] == 4


def test_measure_takes_a_trace_too(gpu):
    trace = lab_traces()["resample:long_mix:2"]
    record = harness.measure({"spec": spec(), "trace": trace, "label": "c16-long_mix-2"})
    assert record["trace_sha256"] == trace["sha256"] and record["resample_variant"] == "long_mix"


# --- energy: NVML's counter, else its power reading, else nvidia-smi, else nothing --------------------------------

ENERGY_KEYS = ("energy_j", "energy_window_s", "energy_j_per_req", "power_w_avg", "power_w_idle", "energy_j_per_req_net")


def measured_energy(monkeypatch, tmp_path, nvml=None, smi_power=None):
    fake = FakeGPU(monkeypatch, tmp_path, nvml=nvml)
    if smi_power is not None:
        fake.smi_power = smi_power
    out = harness.measure_many([{"id": "a", "spec": spec()}], rows("r001", "r002"))
    result = by_id(out)["a"]
    assert result["status"] == "measured"
    MeasurementRecord.model_validate(result["record"])  # the energy fields are the lab's contract too
    return fake, result["record"], out["summary"]


def test_energy_comes_from_the_nvml_counter_over_the_replay_window(monkeypatch, tmp_path):
    fake, rec, summary = measured_energy(monkeypatch, tmp_path, FakeNVML(idle_w=30.0, busy_w=180.0))
    assert rec["energy_source"] == summary["energy_source"] == "nvml_counter"
    # replay: t0 = 1 s after the clock starts, the last request finishes 50 s after it: a 49 s window, all busy
    assert rec["energy_window_s"] == pytest.approx(49.0)
    assert rec["energy_j"] == pytest.approx(180.0 * 49) and rec["power_w_avg"] == pytest.approx(180.0)
    assert rec["energy_j_per_req"] == pytest.approx(180.0 * 49 / 2)  # two requests replayed
    assert rec["power_w_idle"] == pytest.approx(30.0)  # the idle baseline, sampled with the server loaded
    assert rec["energy_j_per_req_net"] == pytest.approx((180.0 - 30.0) * 49 / 2)
    assert fake.nvml.handles == ["GPU-abc"]  # the card nvidia-smi named, by uuid
    assert rec["timings"]["replay_s"] == 50.0  # the idle baseline is not part of the replay's time


def test_the_idle_baseline_is_sampled_before_every_replay(monkeypatch, tmp_path):
    fake = FakeGPU(monkeypatch, tmp_path, nvml=FakeNVML(idle_w=25.0))
    start = fake.now
    out = harness.measure_many([{"id": "a", "spec": spec()}, {"id": "b", "spec": spec(max_concurrency=8)}],
                               rows("r001"))
    assert [r["record"]["power_w_idle"] for r in out["results"]] == [pytest.approx(25.0)] * 2
    # two servers: each 30 s download (once) + 100 s startup + 5 s idle + 50 s replay
    assert fake.now - start == pytest.approx(30 + 2 * (100 + harness.IDLE_S + 50))


def test_a_card_without_the_counter_integrates_nvml_power(monkeypatch, tmp_path):
    _, rec, _ = measured_energy(monkeypatch, tmp_path, FakeNVML(idle_w=30.0, busy_w=180.0, counter=False))
    assert rec["energy_source"] == "nvml_power"
    assert rec["energy_j"] == pytest.approx(180.0 * 49) and rec["power_w_idle"] == pytest.approx(30.0)


def test_an_unknown_uuid_falls_back_to_the_first_visible_gpu(monkeypatch, tmp_path):
    fake, rec, _ = measured_energy(monkeypatch, tmp_path, FakeNVML(by_uuid=False))
    assert fake.nvml.handles == [0] and rec["energy_source"] == "nvml_counter"


def test_without_nvml_nvidia_smi_power_is_the_last_resort(monkeypatch, tmp_path):
    _, rec, _ = measured_energy(monkeypatch, tmp_path, None, smi_power=["42.5"])
    assert rec["energy_source"] == "nvidia_smi"
    assert rec["power_w_avg"] == pytest.approx(42.5) and rec["power_w_idle"] == pytest.approx(42.5)


def test_no_power_reading_at_all_is_unavailable_and_costs_no_idle_time(monkeypatch, tmp_path):
    fake, rec, summary = measured_energy(monkeypatch, tmp_path, FakeNVML(counter=False, power=False))
    assert rec["energy_source"] == summary["energy_source"] == "unavailable"
    assert not any(key in rec for key in ENERGY_KEYS)  # missing is absent, never 0
    assert fake.now == pytest.approx(30 + 100 + 50)  # no idle baseline taken: nothing to read it with
    parsed = MeasurementRecord.model_validate(rec)
    assert parsed.energy_j is None and parsed.energy_j_per_req is None


def test_energy_fields_report_only_what_was_measured():
    assert harness.energy_fields("nvml_counter", None, 10.0, 3) == {"energy_source": "unavailable"}
    assert harness.energy_fields("nvml_counter", 100.0, 0.0, 3) == {"energy_source": "unavailable"}
    assert harness.energy_fields("unavailable", 100.0, 10.0, 3) == {"energy_source": "unavailable"}
    no_idle = harness.energy_fields("nvml_power", 100.0, 10.0, 4)
    assert no_idle == {"energy_source": "nvml_power", "energy_j": 100.0, "energy_window_s": 10.0,
                       "energy_j_per_req": 25.0, "power_w_avg": 10.0}
    assert harness.energy_fields("nvml_power", 100.0, 10.0, 4, idle_w=8.0)["energy_j_per_req_net"] == 5.0


class Readings:
    """An EnergyMeter whose samples a test writes by hand."""

    def __init__(self, source, points):
        self.meter = harness.EnergyMeter.__new__(harness.EnergyMeter)
        self.meter.source, self.meter.samples, self.meter.lock = source, list(points), threading.Lock()


def test_the_counter_is_interpolated_at_the_window_edges():
    meter = Readings("nvml_counter", [(0.0, 1000.0), (1.0, 1100.0), (2.0, 1300.0), (3.0, 1310.0)]).meter
    assert meter.energy_j(0.5, 2.5) == pytest.approx(1305.0 - 1050.0)
    assert meter.energy_j(1.0, 2.0) == pytest.approx(200.0)


def test_sampled_power_is_integrated_by_trapezoids():
    meter = Readings("nvml_power", [(0.0, 10.0), (1.0, 30.0), (2.0, 30.0)]).meter
    assert meter.energy_j(0.0, 2.0) == pytest.approx(20.0 + 30.0)
    assert meter.energy_j(0.5, 1.5) == pytest.approx(0.5 * (20 + 30) / 2 + 0.5 * 30)


def test_a_window_the_samples_do_not_cover_is_no_reading():
    meter = Readings("nvml_counter", [(1.0, 10.0), (2.0, 20.0)]).meter
    assert meter.energy_j(0.5, 2.0) is None and meter.energy_j(1.0, 2.5) is None and meter.energy_j(2.0, 1.0) is None
    backwards = Readings("nvml_counter", [(0.0, 50.0), (1.0, 10.0)]).meter  # the driver was reloaded
    assert backwards.energy_j(0.0, 1.0) is None


def test_an_nvml_that_fails_to_initialise_is_no_nvml(monkeypatch):
    class Broken(FakeNVML):
        def nvmlInit(self):
            raise self.NVMLError("Driver Not Loaded")
    monkeypatch.setattr(harness, "smi", lambda *fields: ["[N/A]"])
    assert harness.energy_reader("GPU-abc", Broken()) == ("unavailable", None)


def test_the_sampling_thread_reads_until_stopped():
    import time

    class Clock:
        @property
        def now(self):
            return time.monotonic()
    nvml = FakeNVML(idle_w=40.0)
    nvml.clock, nvml.t = Clock(), time.monotonic()
    meter = harness.EnergyMeter("GPU-abc", period_s=0.01, nvml=nvml)
    meter.start()
    start = meter.mark()
    time.sleep(0.15)
    end = meter.mark()
    meter.stop.set()
    meter.join(timeout=2)
    assert not meter.is_alive() and len(meter.samples) >= 5
    assert meter.energy_j(start, end) / (end - start) == pytest.approx(40.0, rel=0.05)
