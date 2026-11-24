"""Environment for the Claude Code agents mlab starts: the terminal classroom and the web page."""
import json
import os
import re
import shutil
from pathlib import Path
from .core.io import ROOT, LabError, inside, load, tool_path
from .gateway import GatewayError

# The reference student agent: Claude Code driving Qwen 3.8 27B through an OpenAI-compatible endpoint.
DEFAULT_PROVIDER = "qwen3.8-27b"
FOREIGN_PREFIXES = ("ANTHROPIC_", "MLAB_LLM_", "MLAB_TEACHER_", "QWEN", "DEEPSEEK_", "DASHSCOPE_")
PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
PROVIDER_SWITCHES = {"CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDECODE"}


def child_environment(model, url, token, key_env, config_dir, auth_mode="bearer"):
    """Terminal classroom: a full environment for `claude -p` against a gateway, a native endpoint or the login."""
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(FOREIGN_PREFIXES) or name == key_env or name in PROVIDER_SWITCHES:
            env.pop(name, None)
    env.update({"ANTHROPIC_BASE_URL": url, "ANTHROPIC_AUTH_TOKEN": token, "ANTHROPIC_MODEL": model,
        "MLAB_LLM_PROTOCOL": "claude" if auth_mode == "login" else "anthropic",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model, "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "MAX_THINKING_TOKENS": "0", "DISABLE_UPDATES": "1",
        "CLAUDE_CONFIG_DIR": str(config_dir),
        "PATH": tool_path(),
        "PYTHONUTF8": "1"})
    if not shutil.which("mlab", path=env["PATH"]):
        raise GatewayError("mlab is not installed for this Python. Run scripts/setup.sh and start with scripts/start-agent.sh.")
    if auth_mode == "api_key":
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        env["ANTHROPIC_API_KEY"] = token
    if auth_mode == "login":
        for name in [n for n in env if n.startswith("ANTHROPIC_")] + ["CLAUDE_CONFIG_DIR", "CLAUDE_CODE_SUBAGENT_MODEL"]:
            env.pop(name, None)
        env["ANTHROPIC_MODEL"] = model
    return env


# Profiles that existed before Codex was removed; courses and workspaces made with them keep working.
REMOVED_PROVIDERS = {"codex-luna": DEFAULT_PROVIDER, "codex-soc-qwen": DEFAULT_PROVIDER}


def load_profile(name):
    if name in REMOVED_PROVIDERS:
        import sys
        print(f"mlab: provider {name} was removed with Codex; using {REMOVED_PROVIDERS[name]} (Claude Code).", file=sys.stderr)
        name = REMOVED_PROVIDERS[name]
    path = inside(ROOT / "agent/profiles", f"{name}.json")
    if not path.is_file():
        raise LabError("E_SCHEMA", f"unknown provider profile {name}", 2)
    return load(path)


def provider_key(profile):
    """The API key of a provider profile, from the variable the profile names; None for the account login."""
    variable = profile.get("api_key_env")
    if not variable:
        return None
    key = os.environ.get(variable) or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LabError("E_BACKEND_UNAVAILABLE", f"set {variable} for the {profile['id']} profile; keys are read from the environment only", 4)
    return key


def sanitize_process_environment():
    """The web server's own environment: provider variables it inherited must not reach the agents."""
    for name in list(os.environ):
        if name.startswith(FOREIGN_PREFIXES) or name in PROVIDER_SWITCHES:
            os.environ.pop(name, None)


def web_environment(key=None, config_dir=None):
    """Variables a web task agent adds on top of the sanitized process environment.

    The model and endpoint come from the workspace's settings.local.json, written from the profile by `mlab init`.
    With a key, Claude Code also gets its own config directory, so the student's own Claude settings stay out.
    """
    env = {"DISABLE_UPDATES": "1", "PATH": tool_path(), "PYTHONUTF8": "1", "MLAB_PROMPT_SOURCE": "web",
           "MLAB_LLM_PROTOCOL": "anthropic" if key else "claude"}
    if key:
        env["ANTHROPIC_AUTH_TOKEN"] = key
        if config_dir:
            Path(config_dir).mkdir(parents=True, exist_ok=True)
            env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    return env


def substitute(value, what="this provider"):
    """Fill ${VAR} and ${VAR:-default} from the environment, recursively. A string that is one placeholder and
    resolves to a JSON literal (true, 32768) becomes that value, so profiles can carry typed settings."""
    if isinstance(value, dict):
        return {key: substitute(item, what) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, what) for item in value]
    if not isinstance(value, str):
        return value

    def one(match):
        found = os.environ.get(match.group(1)) or match.group(2)
        if found is None:
            raise LabError("E_SCHEMA", f"set {match.group(1)} before selecting {what}", 2)
        return found

    resolved = PLACEHOLDER.sub(one, value)
    if PLACEHOLDER.fullmatch(value):
        try:
            literal = json.loads(resolved)
        except ValueError:
            return resolved
        if isinstance(literal, (bool, int, float)):
            return literal
    return resolved


def endpoint(profile):
    """How the agent reaches its model: 'login' (Claude Code's own account), 'anthropic' (a native Messages API)
    or 'openai' (an OpenAI-compatible API behind the loopback gateway). Values are resolved; the key is included."""
    protocol = profile.get("protocol") or ("anthropic" if profile.get("api_key_env") else "login")
    if protocol == "login":
        return {"protocol": "login"}
    what = f"the {profile['id']} profile"
    env = substitute(profile.get("env", {}), what)
    resolved = {"protocol": protocol,
                "base_url": substitute(profile.get("base_url") or env.get("ANTHROPIC_BASE_URL", ""), what),
                "model": substitute(profile.get("model") or env.get("ANTHROPIC_MODEL", ""), what),
                "max_tokens": int(substitute(profile.get("max_tokens", 32768), what)),
                "min_tokens": int(substitute(profile.get("min_tokens", 0), what)),
                "timeout_s": float(substitute(profile.get("timeout_s", 600), what)),
                "extra_body": substitute(profile.get("extra_body", {}), what)}
    if not resolved["base_url"] or not resolved["model"]:
        raise LabError("E_SCHEMA", f"{what} needs a base URL and a model", 2)
    variable = profile.get("api_key_env")
    key = os.environ.get(variable, "") if variable else ""
    if not key and not profile.get("api_key_optional"):
        raise LabError("E_BACKEND_UNAVAILABLE", f"set {variable} for {what}; keys are read from the environment only", 4)
    resolved["key"] = key
    return resolved


def endpoint_target(resolved, gateway_token=None):
    """The Gateway (OpenAI-compatible) or AnthropicEndpoint (native) for a resolved non-login profile."""
    import secrets
    from .gateway import AnthropicEndpoint, Gateway
    options = {"max_tokens": resolved["max_tokens"], "timeout": resolved["timeout_s"]}
    if resolved["protocol"] == "openai":
        return Gateway(resolved["base_url"], resolved["model"], resolved["key"], gateway_token or secrets.token_urlsafe(24),
                       min_tokens=resolved["min_tokens"], extra_body=resolved["extra_body"], **options)
    return AnthropicEndpoint(resolved["base_url"], resolved["model"], resolved["key"], "", **options)


def open_endpoint(resolved, gateway_token=None):
    """Start what the agent connects to. Returns (server or None, base URL, token for Claude Code)."""
    from .gateway import connect_claude
    if resolved["protocol"] == "login":
        return None, None, None
    return connect_claude(endpoint_target(resolved, gateway_token))
