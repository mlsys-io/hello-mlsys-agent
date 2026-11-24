# Start this task

Run `./claude` in this workspace; any terminal works. It starts Claude Code with this workspace's instructions
(AGENTS.md), permissions and hooks, driving the model chosen at `mlab init` (default: Qwen 3.8 27B). For an
OpenAI-compatible model, `./claude` also runs a small local gateway; the provider key stays in that process.
The first time, trust this workspace: check the directory and permissions, then choose Yes.
You can start with: "My initial hypothesis is ..., another explanation is ...; if ..., I am wrong. Please help me design a controlled experiment."
This task has two flywheels: the simulator has no run limit; hardware measurement has a budget, and you decide which points to measure. To find out why the simulator and the hardware disagree, ask the agent to analyse the data with you.
Plan and hardware-point approvals, decisions, unlocks and conclusions all need your confirmation in the same window. You can submit conclusions more than once; the latest one counts.
On the course web page (`mlab web`) you do the same things on the page: write your words in the forms in the middle column, or tell the agent in the right column; steps that need your confirmation appear as confirmation cards. Do not use one workspace in the web page and the terminal at the same time.
Simulator results (evidence level `simulated`) are uncalibrated predictions from the mlsys simulator, not GPU measurements. The fixture, the LLM estimator and the hardware stand-in produce invented placeholder numbers (`mock`).
