"""`./claude` in a workspace: start the student's Claude Code against the workspace's provider profile.

For an OpenAI-compatible profile (the default, Qwen 3.8 27B) this process runs the loopback Anthropic->OpenAI
gateway for the length of the session. Claude Code only ever sees the gateway's URL and a one-session local
token; the provider key stays in this process. A native Messages profile gets its key as ANTHROPIC_AUTH_TOKEN.
Claude Code runs with project and local settings only and the agent's four tools.

    python -m mlab.launch --workspace DIR [claude arguments...]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from .agent_env import FOREIGN_PREFIXES, PROVIDER_SWITCHES, endpoint, load_profile, open_endpoint
from .core.io import LabError, load, require_pinned_claude, tool_path

CLAUDE_FLAGS = ["--setting-sources", "project,local", "--tools", "Bash,Read,Edit,Write"]


MODEL_VARIABLES = ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                   "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")


def agent_environment(profile, url, token, model=None, base=None):
    """The process environment for the student's Claude Code: provider variables and keys removed, then the
    endpoint this launcher opened. Every model alias names the profile's model; the workspace's
    settings.local.json can still override individual aliases."""
    env = dict(os.environ if base is None else base)
    key_variable = profile.get("api_key_env")
    for name in list(env):
        if name.startswith(FOREIGN_PREFIXES) or name in PROVIDER_SWITCHES or name == key_variable:
            env.pop(name)
    env.update({"PATH": tool_path(), "PYTHONUTF8": "1", "DISABLE_UPDATES": "1"})
    if url:
        env.update({"ANTHROPIC_BASE_URL": url, "ANTHROPIC_AUTH_TOKEN": token})
    if model:
        env.update(dict.fromkeys(MODEL_VARIABLES, model))
    return env


def start_teacher(root, env):
    """With MLAB_TEACHER_* in this process, serve the teacher on loopback and give the agent only its URL and token."""
    import secrets
    from .teacher.client import TeacherClient, configuration
    from .teacher.relay import serve
    config = configuration()
    if config is None:
        return None
    token = secrets.token_urlsafe(24)
    relay = serve(root, TeacherClient(config), token)
    env.update({"MLAB_TEACHER_URL": f"http://127.0.0.1:{relay.server_port}", "MLAB_TEACHER_TOKEN": token})
    print(f"mlab: teacher {config['model']} available to `mlab ask` and `mlab assess`.", file=sys.stderr)
    return relay


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlab.launch", description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, required=True)
    args, claude_args = parser.parse_known_args(argv)
    root = args.workspace.resolve()
    try:
        config = load(root / ".mlab/config.json")
        profile = load_profile(config["provider"])
        resolved = endpoint(profile)
        claude = require_pinned_claude()
        server, url, token = open_endpoint(resolved)
    except LabError as error:
        print(f"mlab: {error}", file=sys.stderr)
        return error.exit_code
    env = agent_environment(profile, url, token, resolved.get("model"))
    relay = start_teacher(root, env)
    if server:
        print(f"mlab: {profile['id']} -> {resolved['model']} via the local gateway; the provider key stays in this process.",
              file=sys.stderr)
    try:
        process = subprocess.Popen([claude, *CLAUDE_FLAGS, *claude_args], cwd=root, env=env)
        # Ctrl+C goes to the whole foreground group: Claude Code handles it, and this process keeps the
        # gateway up until Claude Code itself exits.
        while True:
            try:
                return process.wait()
            except KeyboardInterrupt:
                continue
    finally:
        for running in (server, relay):
            if running:
                running.shutdown()
                running.server_close()


if __name__ == "__main__":
    sys.exit(main())
