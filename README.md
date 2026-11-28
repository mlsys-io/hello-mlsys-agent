# hello-mlsys-agent

Practise ML-systems design with your own coding agent. You frame a hypothesis about an LLM-serving system,
dispatch experiments, analyse what comes back and decide. A deterministic simulator answers every experiment,
and a teacher model coaches you and grades the finished task.

| Part | What it is |
| --- | --- |
| [`packages/simulator`](packages/simulator) | **mlsys-simulator**: a deterministic, CPU-only LLM-serving simulator (memory planning, KV preemption, continuous batching, chunked prefill, TP, analytical PP). Runs in-process. |
| [`packages/lab`](packages/lab) | **hello-mlsys-lab**: the `mlab` workflow CLI, the three design tasks (9.1 edge, 9.2 SLA, 9.3 DAG), the student workspace and a local course page. |
| Student agent | **Claude Code** with the workspace's instructions, permissions and hooks. The reference model is **Qwen 3.8 27B** on any OpenAI-compatible endpoint. |
| Teacher agent | `mlab ask` coaches and `mlab assess` grades against a rubric. The reference model is **DeepSeek V4.1 Flash**. |

## Quickstart

Linux, [uv](https://docs.astral.sh/uv/), and Node.js/npm (used for the pinned Claude Code).

```bash
git clone https://github.com/mlsys-io/hello-mlsys-agent && cd hello-mlsys-agent
scripts/setup.sh                 # Python 3.12 workspace + Claude Code pinned in agent/compatibility.json
source .venv/bin/activate

# The student agent: any OpenAI-compatible chat API with function calling (vLLM, SGLang, OpenRouter, ...)
export QWEN38_BASE_URL=http://127.0.0.1:8000/v1  QWEN38_MODEL=qwen3.8-27b  # QWEN38_API_KEY=... if needed
# The teacher (optional, but needed for mlab ask / assess)
export MLAB_TEACHER_BASE_URL=https://...  MLAB_TEACHER_MODEL=deepseek-v4.1-flash  MLAB_TEACHER_API_KEY=...

mlab doctor --live               # checks Claude Code and makes one real tool-call round trip to the model
mlab init my-lab --task 9.2      # the SLA task, on the in-tree simulator
cd my-lab && ./claude            # trust the workspace once, then state your hypothesis
```

No model handy? `mlab demo --simulator mlsys --out demo` runs a scripted student through a full task offline.

## How a session works

```
 you ──► Claude Code (Qwen 3.8 27B) ──► mlab CLI ──► mlsys-simulator (in-process)
              │   ▲                        │              hardware stand-in / GPU sandbox (later)
              │   └── loopback gateway ◄───┘ ./claude holds the model key and the teacher key
              └─────────────► mlab ask / mlab assess ──► teacher (DeepSeek V4.1 Flash)
```

- **Fast flywheel**: hypothesis, plan, review, simulate, analyse, decide. The simulator has no total run budget.
- **Slow flywheel**: you pick hardware points, mlab freezes the simulator's predictions first, then they run.
  Today that is a stand-in with made-up numbers; real GPU sandboxes plug in later.
- **You own the decisions.** Plan approvals, decisions and conclusions all need your confirmation. The agent
  drafts and runs, and the teacher coaches without handing over a configuration.
- **Evidence is labelled honestly.** Simulator results are uncalibrated predictions (`simulated`). Fixture and
  stand-in results are placeholders (`mock`). Nothing here claims real GPU performance.

**From sandbox to a better simulator.** `mlab init --hardware remote --hardware-url URL` measures on a GPU
sandbox that speaks the `mlab.executor.v1` job API. Its step timings flow through
[`calibration/`](calibration/README.md): export, ingest with a frozen holdout, then promote a new simulator
Profile only if it beats the current one on that holdout and stays within the error budgets.

Bring your own agent if you prefer: the workspace contract is `AGENTS.md` plus the `mlab` CLI. Other Claude Code
model profiles live in `packages/lab/agent/profiles/`. See the [lab README](packages/lab/README.md) for details.

## Tool image and jobs

Every computation is also a stateless job (`mlab.jobs.v1`, JSON in and out), so a laptop, CI or a cluster job
runner execute the same code:

| Job | Input | Output |
| --- | --- | --- |
| `simulate` | `{"requests": [<mlab.simulator.v1 request text>, ...]}` | each response exactly as `POST /v1/simulate` returns it |
| `templates` | `{"task": "9.2", "profile": "protocol-v1.21"}` | the exercise's objective, constraints, decision space, holdouts and template designs |
| `evaluate` | `{"task", "candidates": [...], "seeds", "workloads", "resamples"?}` | per design, seed and workload: metrics, constraint verdicts and the objective; out-of-space designs are rejected with the reason. `resamples` (up to 8 `{"seed", "variant"}`, variant `development`, or `short_mix`/`long_mix` on 9.2) adds fresh prompt mixes drawn from the development trace only, as workloads `resample:<variant>:<seed>` |
| `sweep` | `{"task": "9.2"}` | the reference optimum for the exercise profile |
| `calibrate` | train and holdout datasets, the current Profile, a version | the gate's decision and the fitted Profile |
| `measure` | reserved for GPU sandbox measurements | |

```bash
python -m mlab.jobs simulate < payload.json                                   # in the workspace
docker run --rm -i --user mlab --network none ghcr.io/mlsys-io/hello-mlsys-agent:<tag> simulate < payload.json
```

`profile` (optional on `templates`, `evaluate`, `sweep`) picks the target GPU: `protocol-v1.20` RX 9070 XT (default), `protocol-v1.21` RTX 5080.

Tags `vX.Y.Z` publish the image to GHCR (`.github/workflows/image.yml`); deployments pin its digest. A job
runner can also import `mlab.jobs.run(name, payload_text)` inside the image. Teacher prompts
(`packages/lab/src/mlab/teacher/prompts/`) and rubrics (`packages/lab/tasks/*/rubric.yaml`) are data files, so
other front ends can reuse them from a pinned tag.


## GPU sandbox (real measurements)

The simulator predicts; the sandbox measures. `Dockerfile.sandbox` is vLLM plus a small harness
(`packages/lab/src/mlab/sandbox/vllm_harness.py`) that serves **one design at a time on one GPU**, replays the workload with its
real arrival times, and returns an `mlab.executor.v1` record: per-request first-token and finish times, peak GPU and KV
memory, and the hardware and software identity (card, driver, vLLM and PyTorch versions, model revision, exact engine
arguments). `protocol-v1.21` simulates the RTX 5080, so a prediction and its measurement are for the same card.

```bash
docker run --rm -i --gpus all ghcr.io/mlsys-io/hello-mlsys-agent-sandbox:<tag> < job.json > record.json
# job.json: {"spec": {"strategy": <mlab Strategy>}, "requests": [<workload rows>], "label": "run-1"}
```

- **Measured as designed, or refused.** vLLM cannot reproduce some settings the exercises allow: non-continuous
  batching with concurrency above 1, a KV-cache fraction, speculative decoding, replicas or parallelism. These exit with
  `E_UNSUPPORTED` and the setting named, before any download or GPU use, so nobody reads a measurement of something
  other than their choice. Chunked prefill, the memory utilization, a KV budget and prefix caching are mapped to vLLM
  arguments. Exercise 9.3 (four GPUs) cannot be measured on one card.
- **Pinned.** vLLM is pinned by digest and the three Qwen3 models by Hugging Face commit; weights download on first use
  (network access needed). Expect minutes per run, not seconds, and a cold image pull on a new machine.
- **What a number means.** The record is a single deterministic-seeded run of one workload on one card. It is evidence
  about that run, not a calibration of the simulator.
- **Batches.** `{"designs": [{"id", "spec", "requests"?, "label"?}], "requests", "label", "deadline_s"}` on stdin (or
  `measure_many(designs, requests, deadline_s=3300)`) measures many designs in one GPU session: every design is checked
  before vLLM is imported, weights are fetched once per model revision, one server is started per distinct set of engine
  arguments (restarted between them), and each distinct workload is replayed once on it with fresh VRAM/KV peaks (a
  prefix cache is emptied, or the server restarted, between workloads). Designs identical in all of that share one
  measurement. Once the time left cannot fit the longest startup and replay seen so far, the remaining designs come back
  `{"status": "not_measured", "reason": "out of time in this GPU session"}`. The answer is
  `{"ok": true, "results": [...], "summary": {...}}`, one result per design with its own status.
- **Traces and resamples.** A design may send its workload as `"trace": {"text", "sha256", "resample"?}`: the exact
  JSONL text of the trace file and its `sha256:<hex>`. The sandbox hashes the text, refuses it unless it matches, and
  parses it the way the simulator parses that file, so a measurement and a simulation that report the same
  `trace_sha256` replayed the same bytes. A resampled workload (`evaluate`'s `resamples`, generated on CPU by the same
  `write_resample` code, never inside the sandbox) is a trace with `"resample": {"seed", "variant"}`; at most 2
  distinct ones per GPU session. The record of a trace carries `trace_sha256`, plus `resample_seed` and
  `resample_variant` for a resample.
- **Weight cache.** A complete snapshot in a read-only Hugging Face hub cache at `$MLAB_HF_RO_CACHE` (default
  `/cache/hf`, either the hub directory or an `HF_HOME` holding `hub/`; empty disables it) is served offline with no
  download; otherwise the weights download to `HF_HOME` as before. Each record says `weights_from: cache | download`.
- **Timings.** Each record carries `timings` (`download_s`, `startup_s`, `replay_s` of the phases that produced it,
  the `server` it ran on and `batch_total_s`); the batch `summary` has the same phases summed without double counting,
  plus counts of downloads, cache hits, servers, replays and of each result status.

The image is built and published by `.github/workflows/sandbox-image.yml` for `sandbox-vX.Y.Z` tags. The harness is
unit-tested without a GPU in `packages/lab/tests/test_sandbox.py`.

## Development

```bash
(cd packages/simulator && ../../.venv/bin/python -m unittest discover -s tests)
packages/lab/scripts/verify-cli.sh
```

After setup, rebuild the course page and run its browser tests with a checkout-local Chromium.

```bash
packages/lab/scripts/build-web.sh
cd packages/lab/frontend
npm run install:browser
npm run test:browser
```

CI (`.github/workflows/ci.yml`) runs the Python and browser tests, the scripted demos on the fixture and the simulator, the
calibration-loop tests and ledger check (`tests/`, `scripts/calibration/loop.py verify`), a standalone install
of the lab, and a mirror dry run. `live-eval.yml` smoke-tests the real models nightly when
their secrets are set.

### This repository is the source of truth

`packages/simulator` and `packages/lab` were imported with full history and are published as read-only mirrors:

| Folder | Mirror |
| --- | --- |
| `packages/simulator` | [eyuansu62/mlsys-simulator](https://github.com/eyuansu62/mlsys-simulator) |
| `packages/lab` | [taowangcheng/hello-mlsys-lab](https://github.com/taowangcheng/hello-mlsys-lab) |

On every push to `main`, `.github/workflows/mirror.yml` splits each folder with `git subtree` and pushes it
**fast-forward only**. If someone commits to a mirror directly, the push is refused and an issue is opened; run
`scripts/mirror/absorb-upstream.sh <folder>` and merge the PR it prepares. `scripts/mirror/check.sh` is the dry
run. Release tags `sim-vX.Y.Z` / `lab-vX.Y.Z` become `vX.Y.Z` on the mirrors. Targets are listed in
`mirror/targets.tsv`.

## License

MIT. See [LICENSE](LICENSE); each package carries its own copy.
