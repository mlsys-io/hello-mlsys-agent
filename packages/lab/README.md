# hello-mlsys-lab

> **Read-only mirror.** This repository is published from [`packages/lab`](https://github.com/mlsys-io/hello-mlsys-agent/tree/main/packages/lab) in [mlsys-io/hello-mlsys-agent](https://github.com/mlsys-io/hello-mlsys-agent). Open issues and pull requests there; commits pushed here directly block the next sync until they are merged back there.

The learning workspace for weeks 6–8 of hello-mlsys, available as a local course page and in the terminal. Students learn through conversation with their own coding agent; the reference agent is Claude Code driving Qwen 3.8 27B: ask about the task, work through an idea, plan and run experiments, investigate earlier results, and compare explanations. Experiment editors, execution history and charts support that conversation, with direct controls available throughout.

The page keeps task and session navigation on the left, task information and evidence in the center, and conversation on the right. Configuration histories, measurement analysis, optional reflections and full reports stay in the workspace. A new conversation starts a clean session; earlier work remains in its original session, and only the task's hardware budget is shared.

Experiments can use simulation or hardware measurement. Simulation has no total run budget, but each batch has a size limit. Measurements share a task budget and currently use a stand-in until the AMD sandbox is connected. The fixture, LLM estimator and stand-in produce demonstration numbers; the simulator produces uncalibrated predictions.

The default `mlsys` backend runs the in-tree MLSys Simulator (`packages/simulator`, 1.24) in-process; `--simulator-url` uses its loopback HTTP service instead, and a standalone checkout without the simulator package falls back to that service. The `protocol-v1.20` session profile (applied by default with `--simulator mlsys`, and disclosed in `task/profile.json`) runs all three task workloads with supported Qwen3 BF16 configurations on RX9070XT; it does not replace the original course baselines. Simulation is uncalibrated, model quality is not evaluated, and sandbox measurements remain synthetic. Modeled memory or request failures remain visible and can be checked on the sandbox after student confirmation. See the [application integration and recordings](integrations/mlsys-simulator/APP_INTEGRATION.zh-CN.md).

| To read about | Go to |
| --- | --- |
| Implementation design and plan | [docs/plan.md](docs/plan.md) (Chinese copy: [docs/plan.zh-CN.md](docs/plan.zh-CN.md)) |
| Workbench design, build and browser checks | [Frontend guide](docs/web-frontend.md) |
| Course page API | [docs/web-api.md](docs/web-api.md) |
| Backend interface and artifact formats | [docs/contract-v0.1.md](docs/contract-v0.1.md) |
| External simulator API handoff: contract, examples and acceptance | [Simulator team start page](integrations/mlsys-simulator/README.md) |
| Scope of the v0.1 implementation (fast flywheel only, the student reports numeric predictions) | [Initial implementation notes](docs/implementation-v0.1.md) |
| Item-by-item check of v0.1 against the plan and the contract | [Conformance check record](docs/conformance-v0.1.md) |
| v0.1 architecture, student interaction, startup and recovery | [Initial agent architecture](docs/initial-agent-architecture.md) |
| Demo steps, inputs and expected results | [Demo steps](docs/demo.md) |

## Environment

On Linux with [uv](https://docs.astral.sh/uv/) and Node.js/npm, run these commands from the root of your checkout.
Setup selects Python 3.12 and installs the pinned Claude Code.

```bash
scripts/setup.sh
source .venv/bin/activate
mlab doctor
```

`scripts/setup.sh` installs the Python dependencies and the Claude Code version pinned in `agent/compatibility.json`
into the environment's `bin/`, without touching a global Claude Code.

### The student agent

The default profile `qwen3.8-27b` runs **Claude Code on Qwen 3.8 27B** through any OpenAI-compatible chat API
with function calling: a self-hosted vLLM or SGLang, OpenRouter or a campus service.

```bash
export QWEN38_BASE_URL=http://127.0.0.1:8000/v1   # ends in /v1
export QWEN38_MODEL=qwen3.8-27b                   # the name your server serves
export QWEN38_API_KEY=...                         # if the server needs one
mlab doctor --live                                # one real function-call round trip
```

`./claude` in a workspace runs `mlab.launch`. It holds the key, starts a loopback Anthropic-to-OpenAI gateway for
the session and starts Claude Code against it; Claude Code only sees the gateway URL and a local token. The
gateway sends Qwen's `enable_thinking` switch (`QWEN38_THINKING=false` turns reasoning off), raises short output
budgets to `QWEN38_MIN_TOKENS` (default 8192) so reasoning cannot swallow a reply, strips inline `<think>`
blocks, and reports a reply that ran out of budget instead of returning it empty. Plain `http://` is accepted
only for loopback and private-network hosts.

Other profiles in `agent/profiles/`: `qwen3.8-27b-anthropic` (a host that serves the Messages API directly),
`deepseek-flash`, `deepseek-v4-pro`, `qwen3.6-flash` and `claude-haiku` (the Claude Code account login). Pick one
with `mlab init --provider NAME`. Students may also bring another agent: the workspace contract is `AGENTS.md`
plus the `mlab` CLI.

### The teacher

The teacher is a second model that coaches and grades; the reference is **DeepSeek V4.1 Flash** on any
OpenAI-compatible endpoint:

```bash
export MLAB_TEACHER_BASE_URL=https://...   # OpenAI-compatible, ends in /v1
export MLAB_TEACHER_MODEL=deepseek-v4.1-flash   # the provider's id for it
export MLAB_TEACHER_API_KEY=...
```

- `mlab ask "QUESTION"` coaches: it explains mechanisms, cites the student's own runs and ends with a guiding
  question. It never names a value for a decision variable; a draft that does is rewritten once, then redacted.
  The agent may relay a question only when the student asks (`Bash(mlab ask*)` is an *ask* rule), questions are
  budgeted (`MLAB_TEACHER_QUESTIONS`, default 20) and logged to `.mlab/teacher.jsonl`. The teacher never sees
  holdout workloads, and sees holdout results only after a conclusion.
- `mlab assess` (also run by `mlab conclude` when a teacher is configured) grades against `task/rubric.yaml`
  and writes `assessments/NNN.json` and `.md`. Judged criteria (framing, analysis, dispatch, decision) come
  from the teacher with mandatory citations; citations to records that do not exist are dropped. Deterministic
  criteria come from mlab: the conclusion's evidence level and the objective against a reference optimum that
  ships inside mlab (`scripts/teacher/build_reference.py`), never in the workspace. The agent cannot edit
  `assessments/`. `mlab assess --export WS...` prints the latest assessment of each workspace for instructors.
- Under `./claude` the launcher holds `MLAB_TEACHER_*` and serves the teacher on loopback; the student's agent
  gets only `MLAB_TEACHER_URL` and a session token.

## Talking to the agent

```bash
mlab init --task 9.2 workspaces/my-sla
cd workspaces/my-sla && ./claude
```

The workspace's `.claude/settings.json` allows `mlab` commands, asks the student before approvals, decisions,
unlocks and conclusions, and denies edits to frozen evidence. The first launch asks you to trust the workspace.
`--simulator fixture` uses scripted mock numbers and `--simulator llm` has a model estimate them (labelled mock).
The slow flywheel defaults to the hardware stand-in (`--hardware standin`), which creates four kinds of gap by
fixed rules, so comparing simulation with measurement can be practised before real sandboxes are connected.

### GPU sandboxes

`mlab init --hardware remote --hardware-url https://sandbox.example/` sends each approved measurement run to a
GPU sandbox that speaks the `mlab.executor.v1` job API (`src/mlab/executors/base.py`): submit the run spec and
its requests, poll until a terminal state, and receive one observation per request, memory peaks, the full
hardware/software identity and a declared source. A run is `measured` evidence only when the sandbox declares a
measured source with an explicit identity; the raw record is kept as `measurement.json` for calibration.
`MLAB_HARDWARE_TOKEN` is sent as a bearer token. `tests/fake_executor.py` is a minimal reference service, and
`mlab demo --hardware-url URL` smoke-tests a real one end to end.

## Course page

```bash
mlab web workspaces/my-course
```

The first run creates the course directory and opens a page served on 127.0.0.1. For a remote machine, use an SSH or editor port forward. Each task opens with its overview and a persistent conversation beside it. Ask a conceptual question without preparing an experiment, request a simulation, or bring earlier evidence from this session into the discussion with **Add from history** or an `@` reference. Requested sandbox measurements show their configuration, workload and budget for confirmation before execution. Students can also edit and run experiments directly, inspect request timings, compare results and write their own observations.

The page drives Claude Code through the Agent SDK (`--agent sdk`, the default). In the workbench conversation the built-in tools are off and the page's lab tools (Explore, read-only Bash, paged Read) are served to Claude Code as in-process MCP tools, so confirmations are the same page cards. `--agent fake` is an offline preview with scripted requests. `mlab doctor --web` checks the web dependencies and local TCP access.

The React workbench is included as built static assets. Developers rebuild it with `scripts/build-web.sh`. See [Frontend design and verification](docs/web-frontend.md). Do not use the same workspace from the course page and the terminal at the same time.

## Offline checks

Run these commands from the lab directory (`packages/lab` in the merged repository).

```bash
mlab demo --out workspaces/my-first-demo                     # scripted student, fixture numbers
mlab demo --simulator mlsys --out workspaces/demo-simulated  # the same walk on the in-tree simulator
scripts/verify-cli.sh
```

Use `scripts/verify-cli.sh --with-claude` to include tests that run the pinned Claude Code against a local fake model.
`scripts/e2e/fresh-student.mjs` walks the course page as a new student with the real configured models (a question, a simulated experiment, a teacher question, a sandbox measurement), approving the cards in the browser; see its header for setup.
Passing offline tests does not establish the quality or availability of the real model.

The recording scripts under `frontend/scripts/record-*.mjs` and the galleries they build document earlier
Codex-based recordings; they have not been ported to the Claude Code page yet.

## Repository layout

```text
docs/            design, plan, contract and demo steps
src/mlab/        the mlab CLI, state machine, contracts, backends and course page (src/mlab/web)
frontend/        React workbench source, pinned dependencies and browser tests
scripts/         setup, launch and regression scripts
template/        student workspace template
tasks/           draft configurations of the three tasks; thresholds, workloads and quality tables are all draft
catalog/         hardware, models and infra, with a source for every number
examples/        paper examples; all numbers are made up
agent/           model profiles and framework version compatibility records
config/          API templates; local configurations with keys are ignored by Git
evals/           direction checks and entry points for evaluation scenarios
```
