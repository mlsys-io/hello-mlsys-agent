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

Bring your own agent if you prefer: the workspace contract is `AGENTS.md` plus the `mlab` CLI. Other Claude Code
model profiles live in `packages/lab/agent/profiles/`. See the [lab README](packages/lab/README.md) for details.

## Development

```bash
(cd packages/simulator && ../../.venv/bin/python -m unittest discover -s tests)
(cd packages/lab && ../../.venv/bin/python -m pytest -q)
```

CI (`.github/workflows/ci.yml`) runs both suites, the scripted demos on the fixture and the simulator, a
standalone install of the lab, and a mirror dry run. `live-eval.yml` smoke-tests the real models nightly when
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
