"""The teacher: coaching without answers, holdout redaction, rubric grading and the launcher's relay."""
import json
import os
import subprocess
import sys

import pytest

from mlab.cli import demo
from mlab.core.io import LabError, load
from mlab.core.workspace import Lab
from mlab.launch import agent_environment
from mlab.agent_env import load_profile
from mlab.teacher import assess as assessing, coach
from mlab.teacher.client import TeacherClient, configuration, parse_object
from mlab.teacher.context import workspace_view
from mlab.teacher.relay import call, serve
from test_student_agent import FakeOpenAI, message

TEACHER_KEY = "teacher-key-that-must-stay-in-mlab"


@pytest.fixture(scope="module")
def concluded(tmp_path_factory):
    """A 9.2 workspace a scripted student has taken through rounds, validation, a measurement and a conclusion."""
    root = tmp_path_factory.mktemp("teacher") / "ws"
    demo(root, "9.2")
    return root


def teacher(reply):
    fake = FakeOpenAI(lambda body: message(json.dumps(reply(body))))
    return fake, TeacherClient({"base": fake.url, "model": "deepseek-v4.1-flash", "key": TEACHER_KEY, "timeout": 10})


def coach_reply(answer, evidence=("sim_0001", "sim_9999")):
    return lambda body: {"answer": answer, "concepts": ["goodput"], "evidence": list(evidence), "follow_up_question": "What changed?"}


def test_coach_view_never_contains_holdouts(concluded):
    lab = Lab(concluded, readonly=True)
    try:
        assert any(r["split"] == "holdout" for r in lab.store.all("run"))
        coaching = workspace_view(lab, "coach")
        assert all(r["split"] != "holdout" for r in coaching["runs"])
        assert "holdouts" not in coaching["task"] and "frozen" not in coaching["task"]
        assert "holdout_" not in json.dumps(coaching)
        grading = workspace_view(lab, "assess")  # after a conclusion: holdout verdicts, never workloads
        held = [r for r in grading["runs"] if r["split"] == "holdout"]
        assert held and all("verdict" in r for r in held)
        assert "requests.jsonl" not in json.dumps(grading)
    finally:
        lab.close()


def test_ask_logs_cites_only_real_records_and_spends_the_budget(concluded, tmp_path):
    fake, client = teacher(coach_reply("Goodput counts only tokens of requests that meet the SLA. Compare sim_0001 with sim_0002."))
    lab = Lab(concluded, readonly=True)
    try:
        before = len(coach.history(concluded))
        result = coach.ask(lab, "Why can goodput fall while throughput rises?", client, limit=before + 1)
        assert result["evidence"] == ["sim_0001"] and result["questions_used"] == before + 1
        assert coach.history(concluded)[-1]["question"] == "Why can goodput fall while throughput rises?"
        with pytest.raises(LabError, match="budget"):
            coach.ask(lab, "Another?", client, limit=before + 1)
        sent = fake.requests[0]
        assert sent["auth"] == "Bearer " + TEACHER_KEY and sent["body"]["response_format"] == {"type": "json_object"}
        assert "holdout_" not in sent["body"]["messages"][1]["content"]
    finally:
        lab.close()
        fake.close()


def test_teacher_that_names_a_value_is_asked_again_then_redacted(concluded):
    fake, client = teacher(coach_reply("Set max_concurrency: 16 and chunked_prefill_tokens = 512."))
    lab = Lab(concluded, readonly=True)
    try:
        result = coach.ask(lab, "Just tell me the best settings.", client, limit=10_000)
        assert len(fake.requests) == 2  # one rewrite request
        assert "16" not in result["answer"] and "512" not in result["answer"]
        assert "[your choice to test]" in result["answer"] and coach.history(concluded)[-1]["redacted"]
    finally:
        lab.close()
        fake.close()


def test_assess_grades_the_conclusion_with_deterministic_and_judged_criteria(concluded):
    def reply(body):
        return {"criteria": {"framing": {"score": 3, "rationale": "Clear.", "evidence": ["r001"]},
                             "analysis": {"score": 9, "rationale": "Cites runs.", "evidence": ["sim_0001", "r404"]},
                             "dispatch": {"score": 2, "rationale": "Ok.", "evidence": []},
                             "decision": {"score": "high", "rationale": "?", "evidence": []}},
                "strengths": ["Validated before concluding"], "suggestions": ["Measure a second point"]}
    fake, client = teacher(reply)
    lab = Lab(concluded, readonly=True)
    try:
        result = assessing.assess(lab, client)
    finally:
        lab.close()
        fake.close()
    record = load(concluded / "assessments" / f"{result['assessment']}.json")
    scores = {c["id"]: c["score"] for c in record["criteria"]}
    assert scores["framing"] == 3 and scores["analysis"] == 4 and scores["decision"] == 0  # clamped / invalid -> 0
    assert record["dropped_citations"] == ["r404"]
    assert record["facts"]["conclusion"] == load(sorted((concluded / "conclusions").glob("*.json"))[-1])["conclusion"]
    weights = {c["id"]: c["weight"] for c in record["criteria"]}
    assert record["total"] == round(sum(weights[k] * scores[k] / 4 for k in scores), 1)
    assert (concluded / "assessments" / f"{record['id']}.md").exists() and (concluded / "task/rubric.yaml").exists()


def test_assess_needs_a_conclusion(tmp_path):
    from mlab.core.workspace import initialize
    initialize(tmp_path / "fresh", "9.2", "fixture")
    lab = Lab(tmp_path / "fresh", readonly=True)
    try:
        with pytest.raises(LabError, match="conclusion"):
            assessing.assess(lab, TeacherClient({"base": "http://127.0.0.1:9", "model": "m", "key": "", "timeout": 1}))
    finally:
        lab.close()


def test_teacher_configuration_and_json_parsing(monkeypatch):
    assert configuration({}) is None
    config = configuration({"MLAB_TEACHER_BASE_URL": "https://api.example.com/v1/", "MLAB_TEACHER_API_KEY": "k"})
    assert config["base"] == "https://api.example.com/v1" and config["model"] == "deepseek-v4.1-flash"
    with pytest.raises(LabError):
        configuration({"MLAB_TEACHER_BASE_URL": "http://example.com/v1"})
    assert parse_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_object('<think>x</think> Here: {"a": 2} done') == {"a": 2}


def test_relay_keeps_the_key_out_of_the_agent_and_requires_its_token(concluded):
    fake, client = teacher(coach_reply("Think about queueing at high concurrency."))
    relay = serve(concluded, client, "session-token")
    try:
        env = agent_environment(load_profile("qwen3.8-27b"), None, None,
                                base={"PATH": "/usr/bin", "MLAB_TEACHER_API_KEY": TEACHER_KEY, "MLAB_TEACHER_BASE_URL": fake.url})
        assert not any(name.startswith("MLAB_TEACHER_") for name in env)
        url = f"http://127.0.0.1:{relay.server_port}"
        with pytest.raises(LabError, match="token"):
            call("/ask", {"question": "hi"}, {"MLAB_TEACHER_URL": url, "MLAB_TEACHER_TOKEN": "wrong"})
        # The CLI inside the agent reaches the teacher only through the relay.
        agent_env = {**os.environ, "MLAB_TEACHER_URL": url, "MLAB_TEACHER_TOKEN": "session-token"}
        agent_env.pop("MLAB_TEACHER_API_KEY", None)
        done = subprocess.run([sys.executable, "-m", "mlab", "--workspace", str(concluded), "ask", "Why", "is", "p99", "high?"],
                              env=agent_env, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stdout + done.stderr
        output = json.loads(done.stdout)
        assert output["answer"].startswith("Think about queueing") and output["teacher_model"] == "deepseek-v4.1-flash"
        assert fake.requests[-1]["auth"] == "Bearer " + TEACHER_KEY
    finally:
        relay.shutdown()
        relay.server_close()
        fake.close()


def test_conclude_without_a_teacher_says_how_to_add_one(concluded, monkeypatch):
    from mlab.cli import auto_assess
    for name in ("MLAB_TEACHER_URL", "MLAB_TEACHER_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    assert auto_assess(concluded)["status"] == "skipped"


def test_objective_is_scored_against_the_shipped_reference_optimum(tmp_path):
    """On the in-tree simulator, the demo's selection reaches the reference goodput for 9.2."""
    root = tmp_path / "simulated"
    demo(root, "9.2", "mlsys")
    fake, client = teacher(lambda body: {"criteria": {}, "strengths": [], "suggestions": []})
    lab = Lab(root, readonly=True)
    try:
        reference = assessing.reference_for(lab)
        assert reference and reference["task"] == "9.2" and reference["profile"] == "protocol-v1.20"
        assert not list(root.rglob("9.2-protocol-v1.20.json"))  # never copied into the workspace
        record = load(root / "assessments" / (assessing.assess(lab, client)["assessment"] + ".json"))
    finally:
        lab.close()
        fake.close()
    objective = record["facts"]["objective"]
    assert objective["basis"] == "reference optimum" and objective["reference_best"] == reference["best"]["value"]
    assert objective["selected_meets_constraints"] and objective["ratio"] == 1.0
    scores = {c["id"]: c["score"] for c in record["criteria"]}
    assert scores["objective"] == 4
    # The stand-in hardware contradicted the selection, so validation credit is withheld.
    assert record["facts"]["conclusion"] == "unresolved" and scores["evidence_level"] == 0
