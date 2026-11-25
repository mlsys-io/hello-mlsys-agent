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
