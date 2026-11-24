"""`mlab assess`: the rubric assessment of a concluded task.

Deterministic criteria (evidence level, objective against a reference optimum) are computed by mlab from run
artifacts. Judged criteria (framing, analysis, dispatch, decision) come from the teacher model, which must cite
evidence ids; citations that do not exist in the workspace are dropped and the drop is recorded.
"""
import json
import statistics
from importlib import resources

from ..core.io import ROOT, LabError, atomic, hashed, load, require
from ..core.metrics import get_metric
from ..core.store import now
from .coach import history as teacher_history
from . import prompt
from .context import workspace_view

SYSTEM = prompt("assess")  # teacher/prompts/assess.md: data, reusable by other front ends


def rubric_for(lab):
    path = lab.root / "task/rubric.yaml"
    require(path.exists(), "E_SCHEMA", "This task has no rubric (task/rubric.yaml)", 2)
    rubric = load(path)
    require(abs(sum(c["weight"] for c in rubric["criteria"]) - 100) < 1e-9, "E_SCHEMA", "Rubric weights must sum to 100", 2)
    return rubric


def reference_for(lab):
    """The hidden reference optimum for this task and simulator profile, shipped with mlab (never in the workspace)."""
    profile = lab.config.get("simulator_profile") or "none"
    name = f"{lab.task['id']}-{profile}.json"
    try:
        reference = json.loads(resources.files("mlab.teacher").joinpath("reference", name).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    best = reference.get("best") or {}
    return reference if type(best.get("value")) in {int, float} else None


def objective_of(lab, runs):
    metric = lab.task["objective"]["metric"]
    values = []
    for run in runs:
        result, _ = lab.result(run["id"])
        value = get_metric(result["metrics"], metric)
        if type(value) in {int, float}:
            values.append(value)
    return statistics.median(values) if values else None


def meets_performance(lab, run):
    """Every set performance constraint met. Exercise profiles do not evaluate model quality, so a run's verdict
    stays not_evaluated even when it meets them all; grading uses the performance constraints."""
    if run.get("verdict") == "constraints_met":
        return True
    if run["status"] != "succeeded":
        return False
    result, _ = lab.result(run["id"])
    judged = [c for c in result["constraints"] if c.get("value") is not None and c.get("status") != "tbd"]
    return bool(judged) and all(c.get("met") is True for c in judged) and result["quality"].get("passes_floor") is not False


def validated_on_performance(lab, selection, runs):
    """Development, repeat and every holdout of the selected configuration meet the performance constraints."""
    mine = [r for r in runs if r.get("config_hash") == selection["config_hash"] and r.get("role", "simulator") == "simulator"]
    ok = [r for r in mine if meets_performance(lab, r)]
    return ({"development", "repeat", "holdout"} <= {r["split"] for r in ok}
            and set(lab.task["holdouts"]) <= {r.get("holdout") for r in ok if r["split"] == "holdout"})


def deterministic(lab, rubric):
    conclusion = lab.latest_conclusion()
    selection = conclusion["selection"]
    runs = lab.store.all("run")
    simulated = [r for r in runs if r.get("role", "simulator") == "simulator" and r["status"] == "succeeded"]
    selected_dev = [r for r in simulated if r.get("config_hash") == selection["config_hash"] and r["split"] == "development"]
    feasible = [r for r in simulated if r["split"] == "development" and meets_performance(lab, r)]
    direction = lab.task["objective"]["direction"]
    by_config = {}
    for run in feasible:
        by_config.setdefault(run.get("config_hash"), []).append(run)
    student_values = [v for v in (objective_of(lab, group) for group in by_config.values()) if v is not None]
    student_best = None if direction == "satisfy" or not student_values else (max if direction == "max" else min)(student_values)
    selected_feasible = any(meets_performance(lab, r) for r in selected_dev)
    selected_value = objective_of(lab, selected_dev)
    reference = reference_for(lab)
    best = reference["best"]["value"] if reference else student_best
    ratio = None
    if direction == "satisfy":
        ratio = 1.0 if selected_feasible else None
    elif selected_feasible and selected_value is not None and best:
        ratio = selected_value / best if direction == "max" else best / selected_value
        ratio = max(0.0, min(ratio, 1.0))
    hardware = [r for r in runs if r.get("role") == "hardware" and r.get("billable", True)]
    performance_validated = validated_on_performance(lab, selection, runs)
    facts = {"conclusion": conclusion["conclusion"], "conclusion_reason": conclusion["reason"],
             "validated_on_performance": performance_validated,
             "selection": f"{selection['workflow']}/{selection['candidate']}",
             "objective": {"metric": lab.task["objective"]["metric"], "direction": direction,
                           "selected_value": selected_value, "selected_meets_constraints": selected_feasible,
                           "student_best_feasible": student_best,
                           "reference_best": reference["best"]["value"] if reference else None,
                           "basis": "reference optimum" if reference else "best feasible configuration the student simulated",
                           "ratio": ratio},
             "runs": {"simulator": len([r for r in runs if r.get("role", "simulator") == "simulator"]),
                      "hardware_used": len(hardware), "hardware_budget": lab.task["budget"]["hardware"]["runs_total"]},
             "rounds": len(lab.store.all("workflow")),
             "teacher_questions": sum(1 for e in teacher_history(lab.root) if e.get("kind") == "ask")}
    scores = {}
    for criterion in rubric["criteria"]:
        if criterion["kind"] != "deterministic":
            continue
        if criterion["id"] == "evidence_level":
            score = criterion["scores"].get(conclusion["conclusion"], 0)
            why = f"Conclusion {conclusion['conclusion']}: {conclusion['reason']}"
            hardware_conflict = conclusion["conclusion"] == "unresolved" and "Hardware results conflict" in conclusion["reason"]
            if performance_validated and not hardware_conflict and score < criterion.get("performance_validated", 0):
                score = criterion["performance_validated"]
                why += (" Development, repeat and every holdout meet the performance constraints; model quality is"
                        " not evaluated in this profile.")
        elif criterion["id"] == "objective":
            score = round(4 * ratio, 2) if ratio is not None else 0
            if not selected_feasible:
                why = "The selected configuration does not meet the performance constraints in development runs."
            elif direction == "satisfy":
                why = "The selected configuration meets every performance constraint in development runs."
            elif type(selected_value) in {int, float} and type(best) in {int, float}:
                why = f"{facts['objective']['metric']} {selected_value:.4g} vs {facts['objective']['basis']} {best:.4g}."
            else:
                why = f"No numeric {facts['objective']['metric']} to compare for the selection or the {facts['objective']['basis']}."
        else:
            raise LabError("E_SCHEMA", f"Unknown deterministic criterion {criterion['id']}", 2)
        scores[criterion["id"]] = {"score": score, "rationale": why, "evidence": [r["id"] for r in selected_dev]}
    return facts, scores


def next_key(root):
    folder = root / "assessments"
    return f"{len(list(folder.glob('*.json'))) + 1:03}" if folder.exists() else "001"


def assess(lab, client):
    require(lab.latest_conclusion(), "E_NEEDS_APPROVAL", "Submit a conclusion first (mlab conclude); the assessment grades it", 3)
    rubric = rubric_for(lab)
    facts, scores = deterministic(lab, rubric)
    judged = [c for c in rubric["criteria"] if c["kind"] == "judged"]
    user = json.dumps({"rubric": judged, "facts": facts, "workspace": workspace_view(lab, "assess")}, ensure_ascii=False)
    reply = client.complete(SYSTEM, user, max_tokens=6000)
    known = {r["id"] for r in lab.store.all("run")} | {w["id"] for w in lab.store.all("workflow")} | {
        m["id"] for m in lab.store.all("measurement")}
    dropped = []
    for criterion in judged:
        item = (reply.get("criteria") or {}).get(criterion["id"]) or {}
        score = item.get("score")
        score = min(4, max(0, int(score))) if type(score) in {int, float} else 0
        cited = [e for e in item.get("evidence", []) if isinstance(e, str)]
        dropped += [e for e in cited if e not in known]
        scores[criterion["id"]] = {"score": score, "rationale": str(item.get("rationale", "Not graded by the teacher."))[:1200],
                                   "evidence": [e for e in cited if e in known]}
    total = round(sum(c["weight"] * scores[c["id"]]["score"] / 4 for c in rubric["criteria"]), 1)
    key = next_key(lab.root)
    record = {"schema": "mlab.v1", "id": key, "task": lab.task["id"], "at": now(), "teacher_model": client.model,
              "rubric_hash": hashed(rubric), "conclusion": lab.latest_conclusion()["id"], "total": total, "out_of": 100,
              "criteria": [{"id": c["id"], "title": c["title"], "kind": c["kind"], "weight": c["weight"], **scores[c["id"]]}
                           for c in rubric["criteria"]],
              "facts": facts, "strengths": [str(s) for s in reply.get("strengths", [])][:5],
              "suggestions": [str(s) for s in reply.get("suggestions", [])][:5], "dropped_citations": sorted(set(dropped))}
    atomic(lab.root / "assessments" / f"{key}.json", record)
    atomic(lab.root / "assessments" / f"{key}.md", render(record), text=True)
    return {"assessment": key, "total": total, "out_of": 100, "result_path": f"assessments/{key}.md",
            "criteria": {c["id"]: c["score"] for c in record["criteria"]}, "teacher_model": client.model}


def render(record):
    lines = [f"# Assessment {record['id']} — task {record['task']}", "",
             f"**{record['total']} / {record['out_of']}** · teacher `{record['teacher_model']}` · conclusion {record['conclusion']}", "",
             "| Criterion | Weight | Score (0-4) | Why |", "| --- | --- | --- | --- |"]
    for c in record["criteria"]:
        cited = f" ({', '.join(c['evidence'])})" if c["evidence"] else ""
        lines.append(f"| {c['title']} | {c['weight']} | {c['score']} | {c['rationale'].replace('|', '/')}{cited} |")
    for title, key in (("Strengths", "strengths"), ("Suggestions", "suggestions")):
        if record[key]:
            lines += ["", f"## {title}", ""] + [f"- {item}" for item in record[key]]
    if record["dropped_citations"]:
        lines += ["", "Citations the teacher made to records that do not exist were dropped: " + ", ".join(record["dropped_citations"])]
    return "\n".join(lines) + "\n"


def export(roots):
    """Instructor export: the latest assessment of each workspace, one JSON object per line."""
    rows = []
    for root in roots:
        found = sorted((root / "assessments").glob("*.json"))
        if found:
            rows.append({"workspace": str(root), **load(found[-1])})
    return rows


__all__ = ["assess", "export", "rubric_for", "ROOT"]
