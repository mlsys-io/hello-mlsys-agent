"""What the teacher may see of a workspace. Holdout workloads are never included; holdout results only after the
student has submitted a conclusion, and then only as verdicts."""
import json

from ..core.io import load
from ..core.student import LOG as STUDENT_LOG

TEXTS = ("hypothesis.md", "review.md", "analysis.md", "decision.md")
LIMIT = 6000  # characters per student text


def clip(text, limit=LIMIT):
    return text if len(text) <= limit else text[:limit] + f"\n[... {len(text) - limit} more characters]"


def task_view(lab):
    task = lab.task
    return {"id": task["id"], "title": task["title"], "objective": task["objective"],
            "constraints": [{k: c.get(k) for k in ("metric", "op", "value", "status")} for c in task["constraints"]],
            "decision_space": task["decision_space"], "budget": task["budget"], "workload_mode": task["workload_mode"]}


def run_view(lab, run, include_holdout_metrics):
    view = {k: run.get(k) for k in ("id", "role", "backend", "split", "status", "verdict", "workflow", "candidate", "measurement")}
    if run["split"] == "holdout":
        view["holdout"] = run.get("holdout")
        if not include_holdout_metrics:
            return view
    try:
        result, _ = lab.result(run["id"])
    except Exception:  # an unfinished run has no result yet
        return view
    key = set(lab.task.get("key_metrics", [])) | {lab.task["objective"]["metric"]}
    view["evidence_level"] = result.get("evidence_level")
    view["metrics"] = {name: value for name, value in result["metrics"].items()
                       if name.split(".")[0] in {k.split(".")[0] for k in key}}
    view["constraints"] = [{k: c.get(k) for k in ("metric", "op", "value", "observed", "met")} for c in result["constraints"]]
    return view


def student_words(root, limit=40):
    path = root / STUDENT_LOG
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [clip(str(row.get("prompt") or row.get("text") or ""), 1500) for row in rows[-limit:]]


def workspace_view(lab, purpose):
    """purpose 'coach': no holdout results at all. purpose 'assess': holdout verdicts after a conclusion."""
    concluded = lab.latest_conclusion() is not None
    include_holdout = purpose == "assess" and concluded
    runs = [r for r in lab.store.all("run") if include_holdout or r["split"] != "holdout"]
    rounds = []
    for workflow in lab.store.all("workflow"):
        folder = lab.folder(workflow)
        texts = {name: clip((folder / name).read_text(encoding="utf-8")) for name in TEXTS if (folder / name).exists()}
        plan = load(folder / "plan.yaml") if (folder / "plan.yaml").exists() else None
        rounds.append({"id": workflow["id"], "status": workflow["status"], "selected": workflow.get("selected_candidate"),
                       "plan": plan, **texts})
    measurements = [{k: m.get(k) for k in ("id", "status", "runs")} for m in lab.store.all("measurement")]
    view = {"task": task_view(lab), "rounds": rounds, "measurements": measurements,
            "runs": [run_view(lab, r, include_holdout) for r in runs], "student_messages": student_words(lab.root)}
    if purpose == "assess" and concluded:
        view["conclusion"] = {k: lab.latest_conclusion().get(k) for k in ("id", "conclusion", "reason", "selection", "evidence")}
    return view
