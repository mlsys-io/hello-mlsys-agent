"""The reference student agent: Claude Code on Qwen 3.8 27B through the loopback gateway, no Codex anywhere."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mlab.agent_env import DEFAULT_PROVIDER, endpoint, load_profile, substitute
from mlab.core.io import ROOT, LabError, load
from mlab.core.workspace import initialize
from mlab.gateway import Gateway, GatewayError, check_endpoint, from_openai, plain_http_allowed, to_openai
from mlab.launch import agent_environment

SECRET = "upstream-key-that-must-stay-in-mlab"


class FakeOpenAI:
    """A minimal OpenAI-compatible chat server that records the requests it receives."""

    def __init__(self, reply):
        self.requests, self.reply = [], reply
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                raw = json.dumps(fake.reply(body)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def message(content=None, tool_calls=None, finish="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {})},
                         "finish_reason": finish}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def test_no_codex_is_left_in_the_package():
    assert not [p for p in (ROOT / "src").rglob("*codex*.py")]
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "codex_client" not in text and "from .codex" not in text and "from ..codex" not in text, path
    assert not list((ROOT / "agent/profiles").glob("codex-*.json"))


def test_courses_and_workspaces_made_for_codex_still_open(capsys):
    assert load_profile("codex-luna")["id"] == DEFAULT_PROVIDER
    assert "removed with Codex" in capsys.readouterr().err
    from mlab.terminal import main
    assert main([]) == 2 and "./claude" in capsys.readouterr().err


def test_default_workspace_uses_claude_code_on_qwen_without_endpoint_variables(workspace_path, monkeypatch):
    for name in [n for n in os.environ if n.startswith("QWEN38_")]:
        monkeypatch.delenv(name)
    result = initialize(workspace_path, "9.2", "fixture")
    assert DEFAULT_PROVIDER == "qwen3.8-27b"
    assert load(workspace_path / ".mlab/config.json")["provider"] == "qwen3.8-27b"
    assert (workspace_path / ".claude/settings.json").is_file() and (workspace_path / "CLAUDE.md").is_file()
    launcher = (workspace_path / "claude").read_text()
    assert "-m mlab.launch --workspace" in launcher and "./claude" in result["next_actions"][0]["say"]
    settings = load(workspace_path / ".claude/settings.local.json")["env"]
    assert "ANTHROPIC_BASE_URL" not in settings  # the gateway URL is per session, set by the launcher
    assert settings["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"


def test_profile_placeholders_take_defaults_and_typed_literals(monkeypatch):
    monkeypatch.delenv("X_UNSET", raising=False)
    monkeypatch.setenv("X_SET", "false")
    assert substitute("${X_UNSET:-8192}") == 8192
    assert substitute("${X_SET}") is False
    assert substitute("a-${X_UNSET:-b}-c") == "a-b-c"
    with pytest.raises(LabError, match="X_UNSET"):
        substitute("${X_UNSET}")


def test_qwen_profile_resolves_an_openai_endpoint_and_optional_key(monkeypatch):
    monkeypatch.setenv("QWEN38_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.delenv("QWEN38_API_KEY", raising=False)
    monkeypatch.delenv("QWEN38_MODEL", raising=False)
    monkeypatch.delenv("QWEN38_THINKING", raising=False)
    resolved = endpoint(load_profile("qwen3.8-27b"))
    assert resolved["protocol"] == "openai" and resolved["model"] == "qwen3.8-27b" and resolved["key"] == ""
    assert resolved["min_tokens"] == 8192 and resolved["max_tokens"] == 32768
    assert resolved["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    monkeypatch.delenv("QWEN38_BASE_URL")
    with pytest.raises(LabError, match="QWEN38_BASE_URL"):
        endpoint(load_profile("qwen3.8-27b"))


def test_gateway_raises_short_budgets_and_strips_inline_reasoning():
    request = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 512}
    assert to_openai(request, "m", 32768, 8192)["max_tokens"] == 8192
    assert to_openai({**request, "max_tokens": 64000}, "m", 32768, 8192)["max_tokens"] == 32768
    reply = from_openai(message("<think>private chain</think>\nThe answer."), "m")
    assert reply["content"] == [{"type": "text", "text": "The answer."}]
    with pytest.raises(GatewayError, match="output budget"):
        from_openai(message("<think>never finished", finish="length"), "m")


def test_plain_http_is_limited_to_local_and_private_hosts(monkeypatch):
    monkeypatch.delenv("MLAB_ALLOW_HTTP", raising=False)
    assert plain_http_allowed("127.0.0.1") and plain_http_allowed("10.20.30.40") and plain_http_allowed("localhost")
    assert not plain_http_allowed("example.com") and not plain_http_allowed("8.8.8.8")
    with pytest.raises(GatewayError, match="HTTPS"):
        Gateway("http://example.com/v1", "m", "k", "t")
    monkeypatch.setenv("MLAB_ALLOW_HTTP", "1")
    Gateway("http://example.com/v1", "m", "k", "t")


def test_live_check_and_thinking_switch_reach_the_upstream():
    def reply(body):
        if isinstance(body.get("tool_choice"), dict):
            return message(None, [{"id": "call_1", "type": "function",
                                   "function": {"name": "check_status", "arguments": '{"experiment_id": "probe"}'}}])
        return message("ready")
    fake = FakeOpenAI(reply)
    try:
        gateway = Gateway(fake.url, "qwen3.8-27b", SECRET, "t", max_tokens=32768, min_tokens=8192,
                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        assert check_endpoint(gateway) == {"function_call": True, "tool_result_round_trip": True}
        assert all(r["body"]["chat_template_kwargs"] == {"enable_thinking": False} for r in fake.requests)
        assert all(r["body"]["max_tokens"] == 8192 and r["auth"] == "Bearer " + SECRET for r in fake.requests)
    finally:
        fake.close()


def test_agent_environment_strips_provider_and_teacher_secrets():
    base = {"PATH": "/usr/bin", "QWEN38_API_KEY": SECRET, "MLAB_TEACHER_API_KEY": "teacher", "DEEPSEEK_API_KEY": "d",
            "ANTHROPIC_API_KEY": "a", "CLAUDE_CODE_USE_BEDROCK": "1", "HOME": "/home/x"}
    env = agent_environment(load_profile("qwen3.8-27b"), "http://127.0.0.1:9/", "local-token", "qwen3.8-27b", base=base)
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9/" and env["ANTHROPIC_AUTH_TOKEN"] == "local-token"
    assert env["ANTHROPIC_MODEL"] == env["CLAUDE_CODE_SUBAGENT_MODEL"] == "qwen3.8-27b" and env["HOME"] == "/home/x"
    assert not {"QWEN38_API_KEY", "MLAB_TEACHER_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
                "CLAUDE_CODE_USE_BEDROCK"} & set(env)
    assert SECRET not in json.dumps(env)


def test_launcher_gives_claude_code_only_the_gateway_and_relays_its_requests(workspace_path, tmp_path, monkeypatch):
    """./claude end to end: a stand-in Claude Code records its environment and sends one Messages request."""
    fake = FakeOpenAI(lambda body: message("hello from qwen"))
    claude = tmp_path / "claude"
    claude.write_text(f"""#!{sys.executable}
import json, os, sys, urllib.request
out = {{"argv": sys.argv[1:], "env": dict(os.environ)}}
request = urllib.request.Request(os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages",
    json.dumps({{"model": "x", "max_tokens": 100, "messages": [{{"role": "user", "content": "hi"}}]}}).encode(),
    {{"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["ANTHROPIC_AUTH_TOKEN"]}})
out["reply"] = json.load(urllib.request.urlopen(request, timeout=10))
open({str(tmp_path / "record.json")!r}, "w").write(json.dumps(out))
""")
    claude.chmod(0o755)
    from mlab.core.io import pinned_claude_version
    version = tmp_path / "bin"
    version.mkdir()
    # require_pinned_claude() asks the binary for its version first.
    (version / "claude").write_text(f"""#!/bin/sh
if [ "$1" = "--version" ]; then echo "{pinned_claude_version()} (Claude Code)"; exit 0; fi
exec {claude} "$@"
""")
    (version / "claude").chmod(0o755)
    try:
        initialize(workspace_path, "9.2", "fixture")
        env = {**os.environ, "QWEN38_BASE_URL": fake.url, "QWEN38_API_KEY": SECRET, "QWEN38_MODEL": "qwen3.8-27b",
               "PATH": f"{version}:{os.environ['PATH']}"}
        result = subprocess.run([sys.executable, "-c", (
            "import sys, mlab.core.io as io; from pathlib import Path; io.BIN_DIR = Path(sys.argv[1]); "
            "from mlab.launch import main; sys.exit(main(sys.argv[2:]))"),
            str(version), "--workspace", str(workspace_path), "-p", "hello"],
            env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        record = json.loads((tmp_path / "record.json").read_text())
    finally:
        fake.close()
    assert record["argv"][:4] == ["--setting-sources", "project,local", "--tools", "Bash,Read,Edit,Write"]
    assert record["argv"][-2:] == ["-p", "hello"]
    assert record["env"]["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")
    assert SECRET not in json.dumps(record["env"])
    assert record["reply"]["content"] == [{"type": "text", "text": "hello from qwen"}]
    assert fake.requests[0]["auth"] == "Bearer " + SECRET and fake.requests[0]["body"]["model"] == "qwen3.8-27b"
    assert Path(record["env"]["PATH"].split(":")[0]).name == "bin"
