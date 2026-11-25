"""Measure one design on a real GPU with vLLM. Standalone: it runs inside a `vllm/vllm-openai` container, where `mlab`
is not installed, so it uses only the standard library plus what that image ships (aiohttp, torch, vllm, huggingface_hub).

    measure({"spec": <mlab RunSpec>, "requests": [<workload rows>], "label": str}) -> <mlab.executor.v1 MeasurementRecord>

The record is exactly what `mlab.executors.base.MeasurementRecord` validates: one observation per request, memory
peaks, a complete hardware/software identity and `source.kind == "measured"`.

HONESTY RULE. A design is measured as designed or not at all. vLLM cannot reproduce some settings the exercises allow
(non-continuous batching beyond one request, a KV fraction, speculative decoding, multi-GPU placement); those raise
`Unsupported` naming the setting, so a student is never shown a measurement of something other than their choice.
"""
import asyncio
import json
import re
import socket
import subprocess
import sys
import threading
import time
import zlib

MODELS = {"llm-0.6b": "Qwen/Qwen3-0.6B", "llm-1.7b": "Qwen/Qwen3-1.7B", "llm-4b": "Qwen/Qwen3-4B",
          "qwen3-0.6b": "Qwen/Qwen3-0.6B", "qwen3-1.7b": "Qwen/Qwen3-1.7B", "qwen3-4b": "Qwen/Qwen3-4B"}
# Model revisions are pinned (Hugging Face commit hashes): "Qwen/Qwen3-1.7B" moves, a measurement must not.
REVISIONS = {"Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
             "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
             "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c"}
STARTUP_TIMEOUT_S = 900
RUN_TIMEOUT_S = 600
VOCAB_LOW, VOCAB_HIGH = 1000, 100000  # token ids that exist in every Qwen3 tokenizer


class Unsupported(Exception):
    """The design uses a setting the sandbox cannot reproduce faithfully."""


def plan(spec):
    """The vLLM engine arguments for a design, or Unsupported naming what cannot be reproduced."""
    strategy = spec["strategy"]
    deployments = strategy["deployments"]
    serving = strategy.get("serving") or {}
    if len(deployments) != 1:
        raise Unsupported("the sandbox serves one deployment on one GPU; this design has %d" % len(deployments))
    deployment = deployments[0]
    if deployment.get("replicas", 1) != 1 or (deployment.get("parallel") or {}).get("degree", 1) != 1:
        raise Unsupported("the sandbox has one GPU per run: replicas and parallelism are not available")
    if deployment["weight_precision"] != "BF16" or deployment.get("kv_precision", "BF16") != "BF16":
        raise Unsupported("only BF16 weights and KV cache are available")
    model = MODELS.get(deployment["model"])
    if model is None:
        raise Unsupported("no model is published for %r" % deployment["model"])
    concurrency = int(serving.get("max_concurrency", 1))
    if not serving.get("continuous_batching", False) and concurrency > 1:
        raise Unsupported("vLLM always batches continuously; a design that turns it off with concurrency above 1 "
                          "cannot be measured")
    if serving.get("kv_fraction") is not None:
        raise Unsupported("a KV-cache fraction is not a vLLM setting; use the memory utilization or a KV budget")
    if strategy.get("speculative"):
        raise Unsupported("speculative decoding is not available in the sandbox")
    context = int(deployment["context_tokens"])
    chunk = serving.get("chunked_prefill_tokens")
    args = ["--dtype", "bfloat16", "--max-model-len", str(context),
            "--gpu-memory-utilization", str(serving.get("gpu_memory_utilization", 0.8)),
            "--max-num-seqs", str(concurrency), "--seed", "0",
            # a per-step token budget of the whole context is, in effect, unchunked prefill
            "--max-num-batched-tokens", str(int(chunk) if chunk else max(context, 2048)),
            "--enable-prefix-caching" if (strategy.get("cache") or {}).get("prefix_cache") else "--no-enable-prefix-caching"]
    if chunk:
        args.append("--enable-chunked-prefill")
    if serving.get("kv_budget_gib") is not None:
        args += ["--kv-cache-memory-bytes", str(int(float(serving["kv_budget_gib"]) * 1024 ** 3))]
    return {"model": model, "args": args, "context": context, "revision": REVISIONS[model]}


def smi(*fields):
    out = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    return [part.strip() for part in out[0].split(",")] if out else []


class MemorySampler(threading.Thread):
    """Peak GPU memory in use (MiB), sampled while the server runs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mib, self.stop = 0.0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                self.peak_mib = max(self.peak_mib, float(smi("memory.used")[0]))
            except (ValueError, IndexError, OSError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.5)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def prompt_ids(row, seed=7):
    """A deterministic prompt of exactly `input_tokens` token ids (the traces carry lengths, not text)."""
    state = (zlib.crc32(row["request_id"].encode()) & 0xFFFF) ^ seed  # stable across processes
    ids = []
    for _ in range(int(row["input_tokens"])):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        ids.append(VOCAB_LOW + state % (VOCAB_HIGH - VOCAB_LOW))
    return ids


async def replay(port, model, requests, t0, deadline):
    import aiohttp
    url = f"http://127.0.0.1:{port}/v1/completions"

    async def one(session, row):
        ready = float(row.get("arrival_ms", 0))
        await asyncio.sleep(max(0.0, t0 + ready / 1000 - time.monotonic()))
        target = int(row["output_tokens_target"])
        obs = {"request_id": row["request_id"], "ready_ms": ready, "started_ms": ready, "first_token_ms": None,
               "finished_ms": None, "output_tokens": 0, "output_tokens_target": target, "status": "failed",
               "service_ms": None}
        body = {"model": model, "prompt": prompt_ids(row), "max_tokens": target, "min_tokens": target,
                "temperature": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
        try:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=max(5.0, deadline - time.monotonic()))) as response:
                if response.status != 200:
                    obs["failure_code"] = "HTTP_%d" % response.status
                    return obs
                async for raw in response.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    event = json.loads(line[5:])
                    now = (time.monotonic() - t0) * 1000
                    if event.get("choices") and obs["first_token_ms"] is None and event["choices"][0].get("text") is not None:
                        obs["first_token_ms"] = max(now, ready)
                    if event.get("usage"):
                        obs["output_tokens"] = int(event["usage"]["completion_tokens"])
                    obs["finished_ms"] = max(now, ready)
            ok = obs["output_tokens"] == target and obs["first_token_ms"] is not None
            obs["status"] = "completed" if ok else "truncated"
            if ok:
                obs["service_ms"] = obs["finished_ms"] - obs["started_ms"]
        except asyncio.TimeoutError:
            obs["status"], obs["failure_code"] = "timed_out", "TIMEOUT"
        except aiohttp.ClientError as error:
            obs["failure_code"] = type(error).__name__
        return obs

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        rows = await asyncio.gather(*(one(session, row) for row in requests))
    return list(rows)


def wait_ready(port, process, deadline):
    import urllib.request
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited with status %s during startup" % process.returncode)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise RuntimeError("vLLM did not become ready in %d s" % STARTUP_TIMEOUT_S)


class KVSampler(threading.Thread):
    """Peak KV-cache usage (0..1) the engine reports while requests are in flight."""
    GAUGES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")  # the second is the pre-V1 name

    def __init__(self, port):
        super().__init__(daemon=True)
        self.port, self.peak, self.stop = port, 0.0, threading.Event()

    def run(self):
        import urllib.request
        while not self.stop.is_set():
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/metrics", timeout=2) as response:
                    for line in response.read().decode().splitlines():
                        if line.startswith(self.GAUGES):
                            self.peak = max(self.peak, float(line.split()[-1]))
            except (OSError, ValueError):
                pass
            self.stop.wait(0.2)


def kv_reserved_gib(log_text, engine_args=()):
    """KV memory the engine reserved: the explicit budget when the design set one, else what vLLM logged.
    None when neither is known: a missing number is reported as missing, never guessed."""
    args = list(engine_args)
    if "--kv-cache-memory-bytes" in args:
        return int(args[args.index("--kv-cache-memory-bytes") + 1]) / 1024 ** 3
    found = re.search(r"(?:Available KV cache memory|KV cache memory)[:=]?\s*([0-9.]+)\s*GiB", log_text)
    return float(found.group(1)) if found else None


def kv_lines(log_text, limit=6):
    """Every log line that mentions the KV cache, kept in the record so the format is never a guess."""
    return [line.strip()[:240] for line in log_text.splitlines() if "KV cache" in line][:limit]


def build_record(*, observations, vram_peak_mib, kv_usage_peak, log_text, engine, revision, gpu, versions, ready_s, label):
    """The mlab.executor.v1 MeasurementRecord for one run. Pure, so tests can validate it against the lab's contract.

    gpu = (name, uuid, driver); versions = {"vllm": ..., "torch": ...}."""
    reserved = kv_reserved_gib(log_text, engine["args"])
    name, uuid, driver = gpu
    return {
        "observations": observations,
        "vram_peak_gib": round(vram_peak_mib / 1024, 3),
        "kv_peak_gib": round(reserved * kv_usage_peak, 3) if reserved is not None and kv_usage_peak else None,
        "identity": {"model_revision": f"{engine['model']}@{revision}", "devices": [name],
                     "runtime": {"name": "vllm", "version": versions["vllm"]}, "weight_dtype": "bfloat16",
                     "kv_dtype": "bfloat16", "activation_dtype": "bfloat16", "driver": driver,
                     "torch": versions["torch"], "gpu_uuid": uuid, "engine_args": engine["args"],
                     "model_revision_full": engine["revision"]},
        "source": {"kind": "measured", "reference": f"vllm-{versions['vllm']}/{uuid}/{label}"},
        "step_records": [],
        "logs": ["server ready after %.0f s (download + startup)" % ready_s] + kv_lines(log_text)
                + log_text.strip().splitlines()[-12:],
    }


def measure(job):
    """Run one design on this GPU. Returns a MeasurementRecord dict; raises Unsupported or RuntimeError."""
    spec, requests = job["spec"], job["requests"]
    engine = plan(spec)  # refuse first: nothing below runs for a design the sandbox cannot reproduce
    import torch
    import vllm
    from huggingface_hub import snapshot_download
    started = time.monotonic()
    path = snapshot_download(engine["model"], revision=engine["revision"])
    port = free_port()
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", path,
               "--served-model-name", engine["model"], "--port", str(port), "--host", "127.0.0.1", *engine["args"]]
    memory, kv = MemorySampler(), KVSampler(port)
    memory.start()
    log_path = "/tmp/vllm-server.log"
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(port, process, time.monotonic() + STARTUP_TIMEOUT_S)
        ready_s = time.monotonic() - started
        kv.start()
        t0 = time.monotonic() + 1.0  # arrivals are measured from here
        observations = asyncio.run(replay(port, engine["model"], requests, t0, t0 + RUN_TIMEOUT_S))
        log_text = open(log_path).read()
    finally:
        memory.stop.set()
        kv.stop.set()
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    gpu = (smi("name", "uuid", "driver_version") + ["unknown"] * 3)[:3]
    return build_record(observations=observations, vram_peak_mib=memory.peak_mib, kv_usage_peak=kv.peak,
                        log_text=log_text, engine=engine, revision=engine["revision"][:8], gpu=tuple(gpu),
                        versions={"vllm": vllm.__version__, "torch": torch.__version__}, ready_s=ready_s,
                        label=job.get("label", "run"))


def main():
    """`docker run -i --gpus all <sandbox image>`: a job {"spec", "requests", "label"} on stdin, a record on stdout.
    Exit 2 with {"ok": false, "error": {...}} when the design cannot be measured (Unsupported) or the run failed."""
    try:
        record = measure(json.loads(sys.stdin.read()))
    except Unsupported as error:
        print(json.dumps({"ok": False, "error": {"code": "E_UNSUPPORTED", "message": str(error)}}))
        return 2
    except (RuntimeError, OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": {"code": "E_SANDBOX", "message": str(error)[:400]}}))
        return 2
    print(json.dumps({"ok": True, "record": record}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
