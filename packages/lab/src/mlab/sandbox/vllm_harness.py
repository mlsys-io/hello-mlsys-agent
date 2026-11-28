"""Measure designs on a real GPU with vLLM. Standalone: it runs inside a `vllm/vllm-openai` container, where `mlab`
is not installed, so it uses only the standard library plus what that image ships (aiohttp, torch, vllm, huggingface_hub).

    measure({"spec": <mlab RunSpec>, "requests": [<workload rows>], "label": str}) -> <mlab.executor.v1 MeasurementRecord>
    measure_many([{"id", "spec", "requests"?, "label"?}, ...], requests, deadline_s=3300) -> {"results", "summary"}

The record is exactly what `mlab.executors.base.MeasurementRecord` validates: one observation per request, memory
peaks, a complete hardware/software identity and `source.kind == "measured"`, plus where the weights came from and how
long each phase took.

BATCHING. `measure_many` downloads (or finds in a read-only cache) each model revision once, starts one vLLM server per
distinct engine configuration, and replays each distinct workload on it once; designs that share both get the same
measurement under their own ids. A GPU session has a time limit, so a design is only started when the time left fits
its estimated startup and replay; the rest come back `not_measured`, never half-measured.

HONESTY RULE. A design is measured as designed or not at all. vLLM cannot reproduce some settings the exercises allow
(non-continuous batching beyond one request, a KV fraction, speculative decoding, multi-GPU placement); those raise
`Unsupported` naming the setting, so a student is never shown a measurement of something other than their choice.
"""
import asyncio
import json
import math
import os
import re
import signal
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
DEADLINE_S = 3300  # a GPU session is killed at 3600 s; stop starting servers early enough to report what was measured
OUT_OF_TIME = "out of time in this GPU session"
RO_CACHE_ENV, RO_CACHE_DEFAULT = "MLAB_HF_RO_CACHE", "/cache/hf"  # a read-only Hugging Face hub cache, if mounted
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}  # served from the cache: no network at all
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
    """Peak GPU memory in use (MiB), sampled while the server runs. `take_peak` closes one window and opens the next,
    so two workloads served by the same server each get their own peak."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mib, self.stop, self.lock = 0.0, threading.Event(), threading.Lock()

    def run(self):
        while not self.stop.is_set():
            try:
                used = float(smi("memory.used")[0])
                with self.lock:
                    self.peak_mib = max(self.peak_mib, used)
            except (ValueError, IndexError, OSError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.5)

    def take_peak(self):
        with self.lock:
            peak, self.peak_mib = self.peak_mib, 0.0
        return peak


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


# --- weights -------------------------------------------------------------------------------------------------------

REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def snapshot_complete(path):
    """True when a Hugging Face snapshot directory holds the config, the tokenizer and every weight file. Snapshot
    entries are symlinks into blobs/, so a missing or dangling one (an interrupted download) counts as absent."""
    def present(name):
        file = os.path.join(path, name)
        return os.path.basename(name) == name and os.path.isfile(file) and os.path.getsize(file) > 0

    if not os.path.isdir(path) or not all(present(name) for name in REQUIRED_FILES):
        return False
    if present("model.safetensors.index.json"):
        try:
            with open(os.path.join(path, "model.safetensors.index.json")) as handle:
                shards = set(json.load(handle)["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False
        return bool(shards) and all(isinstance(shard, str) and present(shard) for shard in shards)
    return present("model.safetensors")


def cached_snapshot(model, revision, root=None):
    """The path of a complete snapshot of `model` at `revision` in the read-only cache ($MLAB_HF_RO_CACHE, default
    /cache/hf; empty disables it), laid out as a standard hub cache, or None. The root may be the hub cache itself or
    an HF_HOME that holds one under hub/."""
    root = os.environ.get(RO_CACHE_ENV, RO_CACHE_DEFAULT) if root is None else root
    if not root:
        return None
    folder = "models--" + model.replace("/", "--")
    for hub in (root, os.path.join(root, "hub")):
        snapshot = os.path.join(hub, folder, "snapshots", revision)
        if snapshot_complete(snapshot):
            return snapshot
    return None


def download_weights(model, revision):
    """Download (or revalidate) a snapshot into $HF_HOME, as a single measurement always did."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=revision)


# --- the server ----------------------------------------------------------------------------------------------------

def runtime_versions():
    """Import the GPU stack (only once every design has been checked) and name its versions."""
    import torch
    import vllm
    return {"vllm": vllm.__version__, "torch": torch.__version__}


class Server:
    """One vLLM OpenAI server for one engine configuration, with a GPU memory sampler around it."""

    def __init__(self, path, engine, env=None, log_path="/tmp/vllm-server.log"):
        self.port, self.log_path = free_port(), log_path
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", path,
                   "--served-model-name", engine["model"], "--port", str(self.port), "--host", "127.0.0.1",
                   *engine["args"]]
        self.memory = MemorySampler()
        self.memory.start()
        with open(log_path, "w") as log:  # its own session, so a kill also reaches vLLM's engine-core child
            self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                            env={**os.environ, **(env or {})})

    def log_text(self):
        with open(self.log_path, errors="replace") as handle:
            return handle.read()

    def stop(self):
        self.memory.stop.set()
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except OSError:
                self.process.kill()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass


def reset_prefix_cache(port):
    """Empty a running server's prefix cache between two workloads. vLLM serves this route only in dev mode
    (VLLM_SERVER_DEV_MODE=1); False means the caller must restart the server instead."""
    import urllib.request
    request = urllib.request.Request(f"http://127.0.0.1:{port}/reset_prefix_cache", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status == 200
    except OSError:
        return False


def wait_gpu_released(baseline_mib, timeout_s=60):
    """Wait until the previous server's memory is back (within 512 MiB of what was in use before the first server),
    so the next server's memory utilization is computed on the same free memory."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            if float(smi("memory.used")[0]) <= baseline_mib + 512:
                return True
        except (ValueError, IndexError, OSError, subprocess.SubprocessError):
            return False
        time.sleep(1)
    return False


# --- measuring -----------------------------------------------------------------------------------------------------

class OutOfTime(Exception):
    """The time left in this GPU session cannot fit the next step."""


def check_batch(designs, requests):
    """Shape errors that make the whole batch unusable (ValueError), raised before anything is imported or started."""
    if not isinstance(designs, list) or not designs:
        raise ValueError("measure_many needs a nonempty list of designs")
    seen = set()
    for design in designs:
        if not isinstance(design, dict) or not isinstance(design.get("id"), str) or not design["id"] \
                or not isinstance(design.get("spec"), dict):
            raise ValueError("each design needs a nonempty string 'id' and a 'spec' object")
        if design["id"] in seen:
            raise ValueError(f"design id {design['id']!r} appears twice")
        seen.add(design["id"])
        if not isinstance(design.get("requests", requests), list):
            raise ValueError(f"design {design['id']!r} has no 'requests' list (its own or the batch's)")


def measure_many(designs, requests=None, *, deadline_s=DEADLINE_S, label=None):
    """Measure a batch of designs in one GPU session.

    designs: [{"id": str, "spec": <RunSpec>, "requests": [...] (optional, else the batch's), "label": str (optional)}]
    Returns {"results": [one per design, in order], "summary": {...}}; each result is one of
      {"id", "status": "measured", "record": <MeasurementRecord>, "shares_replay_with": [ids]}
      {"id", "status": "unsupported", "error": {"code": "E_UNSUPPORTED", "message"}}
      {"id", "status": "failed", "error": {"code": "E_SANDBOX", "message"}}
      {"id", "status": "not_measured", "reason": "out of time in this GPU session"}
    Every design is checked (plan) before torch or vLLM is imported. Designs are grouped by model revision (weights
    fetched once per group), then by engine arguments (one server each, restarted between them), then by workload
    (replayed once; designs identical in all three share that measurement). Each replay has its own VRAM and KV peak
    windows, and a server with prefix caching has its cache emptied (or is restarted) between workloads.
    `deadline_s` (None: no limit) counts from the call; a server is started, and a workload replayed, only if the time
    left fits the longest startup and replay seen so far in this batch (plus a download when one is needed)."""
    started = time.monotonic()
    end = math.inf if deadline_s is None else started + float(deadline_s)
    check_batch(designs, requests)
    results, planned = {}, []
    for design in designs:  # refuse first: nothing below runs for a design the sandbox cannot reproduce
        try:
            planned.append((design, plan(design["spec"])))
        except Unsupported as error:
            results[design["id"]] = {"id": design["id"], "status": "unsupported",
                                     "error": {"code": "E_UNSUPPORTED", "message": str(error)}}
    # (model, revision) -> engine args -> workload -> [(design, engine, rows)], all in first-seen order
    groups: dict = {}
    for design, engine in planned:
        rows = design.get("requests", requests)
        workload = json.dumps(rows, sort_keys=True, default=str)
        by_args = groups.setdefault((engine["model"], engine["revision"]), {})
        by_args.setdefault(tuple(engine["args"]), {}).setdefault(workload, []).append((design, engine, rows))
    seen: dict[str, list[float]] = {"download": [], "startup": [], "replay": []}
    totals = {"download_s": 0.0, "startup_s": 0.0, "replay_s": 0.0}
    counts = {"downloads": 0, "cache_hits": 0, "servers": 0, "replays": 0}
    records: list[dict] = []

    def need(*phases):
        if time.monotonic() + sum(max(seen[phase], default=0.0) for phase in phases) > end:
            raise OutOfTime()

    def fail(entries, error):
        for design, _, _ in entries:
            results[design["id"]] = {"id": design["id"], "status": "failed",
                                     "error": {"code": "E_SANDBOX", "message": str(error)[:400]}}

    if planned:
        versions = runtime_versions()
        gpu = tuple((smi("name", "uuid", "driver_version") + ["unknown"] * 3)[:3])
        try:
            baseline_mib = float(smi("memory.used")[0])
        except (ValueError, IndexError, OSError, subprocess.SubprocessError):
            baseline_mib = 0.0
        try:
            for (model, revision), by_args in groups.items():
                snapshot = cached_snapshot(model, revision)
                need("startup", "replay", *(() if snapshot else ("download",)))
                clock = time.monotonic()
                if snapshot:
                    path, weights_from, env = snapshot, "cache", dict(OFFLINE_ENV)
                    counts["cache_hits"] += 1
                else:
                    try:
                        path, weights_from, env = download_weights(model, revision), "download", {}
                    except (RuntimeError, OSError, ValueError) as error:
                        fail([entry for workloads in by_args.values() for entries in workloads.values()
                              for entry in entries], f"downloading {model} failed: {error}")
                        continue
                    counts["downloads"] += 1
                download_s = time.monotonic() - clock
                totals["download_s"] += download_s
                if weights_from == "download":
                    seen["download"].append(download_s)
                for workloads in by_args.values():
                    serve(workloads, path, env, weights_from, download_s, need, fail, seen, totals, counts,
                          records, results, versions, gpu, baseline_mib, end, label)
        except OutOfTime:
            pass
    for design in designs:
        results.setdefault(design["id"], {"id": design["id"], "status": "not_measured", "reason": OUT_OF_TIME})
    total_s = round(time.monotonic() - started, 2)
    for record in records:
        record["timings"]["batch_total_s"] = total_s
    statuses = [results[design["id"]]["status"] for design in designs]
    summary = {"total_s": total_s, "deadline_s": deadline_s, **{k: round(v, 2) for k, v in totals.items()}, **counts,
               **{status: statuses.count(status) for status in ("measured", "not_measured", "unsupported", "failed")}}
    return {"results": [results[design["id"]] for design in designs], "summary": summary}


def serve(workloads, path, env, weights_from, download_s, need, fail, seen, totals, counts, records, results,
          versions, gpu, baseline_mib, end, label):
    """One engine configuration: start its server, replay each distinct workload, stop it. Raises OutOfTime."""
    engine = workloads[next(iter(workloads))][0][1]
    prefix_cache = "--enable-prefix-caching" in engine["args"]
    pending = list(workloads.values())
    server = None
    try:
        while pending:
            entries = pending[0]
            if server is not None:
                need("replay")
                if prefix_cache and not reset_prefix_cache(server.port):
                    server.stop()
                    server = None
            if server is None:
                need("startup", "replay")
                if counts["servers"]:
                    wait_gpu_released(baseline_mib)
                counts["servers"] += 1
                index, clock = counts["servers"], time.monotonic()
                # dev mode only for the prefix-cache reset route, and only when the server will need it
                extra = {"VLLM_SERVER_DEV_MODE": "1"} if prefix_cache and len(pending) > 1 else {}
                server = Server(path, engine, env={**env, **extra}, log_path=f"/tmp/vllm-server-{index}.log")
                limit = min(clock + STARTUP_TIMEOUT_S, end)
                try:
                    wait_ready(server.port, server.process, limit)
                except RuntimeError as error:
                    server.stop()
                    server = None
                    if limit == end and time.monotonic() >= end:
                        raise OutOfTime() from None
                    fail([entry for entries in pending for entry in entries], error)
                    return
                startup_s = time.monotonic() - clock
                seen["startup"].append(startup_s)
                totals["startup_s"] += startup_s
                startup_peak = server.memory.take_peak()
            server.memory.take_peak()  # the idle time since the last window belongs to no workload
            kv = KVSampler(server.port)
            kv.start()
            clock = time.monotonic()
            t0 = clock + 1.0  # arrivals are measured from here
            cap = min(t0 + RUN_TIMEOUT_S, end)
            rows = entries[0][2]
            try:
                observations = asyncio.run(replay(server.port, engine["model"], rows, t0, cap))
                log_text = server.log_text()
            except (RuntimeError, OSError, ValueError) as error:
                fail(entries, error)
                pending.pop(0)
                continue
            finally:
                kv.stop.set()
            replay_s = time.monotonic() - clock
            seen["replay"].append(replay_s)
            totals["replay_s"] += replay_s
            counts["replays"] += 1
            if cap == end and time.monotonic() >= end and any(o["status"] == "timed_out" for o in observations):
                raise OutOfTime()  # cut by the session, not by the design: not a measurement of it
            vram_peak = max(startup_peak, server.memory.take_peak())
            ids = [design["id"] for design, _, _ in entries]
            for design, _, _ in entries:
                name = design.get("label") or (f"{label}-{design['id']}" if label else design["id"])
                record = build_record(observations=json.loads(json.dumps(observations)), vram_peak_mib=vram_peak,
                                      kv_usage_peak=kv.peak, log_text=log_text, engine=engine,
                                      revision=engine["revision"][:8], gpu=gpu, versions=versions,
                                      ready_s=download_s + startup_s, label=name)
                record["weights_from"] = weights_from
                record["timings"] = {"download_s": round(download_s, 2), "startup_s": round(startup_s, 2),
                                     "replay_s": round(replay_s, 2), "server": index}
                records.append(record)
                results[design["id"]] = {"id": design["id"], "status": "measured", "record": record,
                                         "shares_replay_with": [other for other in ids if other != design["id"]]}
            pending.pop(0)
    finally:
        if server is not None:
            server.stop()


def measure(job):
    """Run one design on this GPU. Returns a MeasurementRecord dict; raises Unsupported or RuntimeError."""
    out = measure_many([{"id": "run", "spec": job["spec"], "label": job.get("label", "run")}], job["requests"],
                       deadline_s=None)
    result = out["results"][0]
    if result["status"] == "unsupported":
        raise Unsupported(result["error"]["message"])
    if result["status"] != "measured":
        raise RuntimeError((result.get("error") or {}).get("message") or result.get("reason") or "not measured")
    return result["record"]


def main():
    """`docker run -i --gpus all <sandbox image>`, a job on stdin and the answer on stdout:

      {"spec", "requests", "label"}                          -> {"ok": true, "record"}       one design
      {"designs": [{"id", "spec", ...}], "requests", "label", "deadline_s"}
                                                             -> {"ok": true, "results", "summary"}   a batch
    Exit 2 with {"ok": false, "error": {...}} when the design cannot be measured (Unsupported) or the run failed; in a
    batch each design carries its own status, so exit 2 only means the batch itself was malformed."""
    try:
        job = json.loads(sys.stdin.read())
        if isinstance(job, dict) and "designs" in job:
            out = measure_many(job["designs"], job.get("requests"), deadline_s=job.get("deadline_s", DEADLINE_S),
                               label=job.get("label"))
            print(json.dumps({"ok": True, **out}))
            return 0
        record = measure(job)
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
