"""`mlab ask`: Socratic coaching that explains and points at evidence, never hands over a configuration."""
import json
import os
import re

from ..core.io import LabError, require
from ..core.store import now
from .context import workspace_view

LOG = ".mlab/teacher.jsonl"
DEFAULT_QUESTIONS = 20

SYSTEM = """You are the teacher in an ML-systems design lab. A student, working with their own coding agent, \
designs LLM-serving configurations and tests them on a deterministic simulator (and later on real GPUs).
Your job is to coach, not to solve.
- Explain concepts the question needs: KV-cache memory, continuous batching, chunked prefill, TTFT vs TPOT,
  goodput under an SLA, tensor/pipeline parallelism vs replicas, roofline reasoning, queueing under load.
- Point at the student's own evidence by run or round id (sim_0003, r002) when it bears on the question.
- Never give a complete configuration, a specific value to choose for a decision variable, or the answer to the
  task. If asked for one, explain how the student could find it themselves instead.
- Distinguish what the evidence shows from hypotheses; uncalibrated simulation is not a GPU measurement.
- End with one guiding question. Keep the answer under 200 words, in the student's language.
Reply with a JSON object: {"answer": str, "concepts": [str], "evidence": [run or round ids you cited],
"follow_up_question": str}."""


def leaf_names(task):
    return sorted({path.split(".")[-1] for path in task["decision_space"]} - {"kind", "degree", "model"}, key=len, reverse=True)


def leaks(text, task):
    """Lines that assign a specific value to a decision variable, e.g. 'max_concurrency: 16'."""
    names = "|".join(map(re.escape, leaf_names(task)))
    if not names:
        return []
    pattern = re.compile(rf"\b({names})\b\s*(?:[:=]|to|of|at)\s*`?\d[\d.]*", re.IGNORECASE)
    return [match.group(0) for match in pattern.finditer(text)]


def redact(text, task):
    names = "|".join(map(re.escape, leaf_names(task)))
    return re.sub(rf"\b({names})\b\s*(?:[:=]|to|of|at)\s*`?\d[\d.]*", r"\1 = [your choice to test]", text, flags=re.IGNORECASE)


def history(root):
    path = root / LOG
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(root, entry):
    path = root / LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ask(lab, question, client, limit=None):
    question = (question or "").strip()
    require(question, "E_SCHEMA", "Ask a question: mlab ask \"...\"", 2)
    require(len(question) <= 4000, "E_SCHEMA", "Keep the question under 4000 characters", 2)
    limit = limit if limit is not None else int(os.environ.get("MLAB_TEACHER_QUESTIONS", DEFAULT_QUESTIONS))
    used = sum(1 for entry in history(lab.root) if entry.get("kind") == "ask")
    if used >= limit:
        raise LabError("E_BUDGET", f"The teacher's question budget for this task is used up ({used}/{limit}).", 3)
    earlier = [{"q": e["question"], "a": e["answer"]} for e in history(lab.root) if e.get("kind") == "ask"][-4:]
    user = json.dumps({"question": question, "earlier_exchanges": earlier,
                       "workspace": workspace_view(lab, "coach")}, ensure_ascii=False)
    reply = client.complete(SYSTEM, user)
    answer = str(reply.get("answer", "")).strip()
    found = leaks(answer, lab.task)
    if found:
        reply = client.complete(SYSTEM + "\nYour previous draft named specific values for decision variables ("
                                + ", ".join(found[:5]) + "). Rewrite it without any specific values.", user)
        answer = str(reply.get("answer", "")).strip()
        if leaks(answer, lab.task):
            answer = redact(answer, lab.task)
    require(answer, "E_BACKEND_UNAVAILABLE", "The teacher returned an empty answer", 4)
    known = {r["id"] for r in lab.store.all("run")} | {w["id"] for w in lab.store.all("workflow")}
    evidence = [e for e in reply.get("evidence", []) if isinstance(e, str) and e in known]
    entry = {"kind": "ask", "at": now(), "model": client.model, "question": question, "answer": answer,
             "concepts": [str(c) for c in reply.get("concepts", [])][:8], "evidence": evidence,
             "follow_up_question": str(reply.get("follow_up_question", ""))[:600], "redacted": bool(found)}
    append(lab.root, entry)
    return {"answer": answer, "follow_up_question": entry["follow_up_question"], "concepts": entry["concepts"],
            "evidence": evidence, "questions_used": used + 1, "questions_limit": limit, "teacher_model": client.model,
            "next_actions": [{"actor": "student", "do": "think", "say": "The teacher coaches; the decisions stay yours."}]}
