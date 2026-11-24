"""One OpenAI-compatible chat call that must return a JSON object."""
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from ..core.io import LabError, require
from ..gateway import THINK, plain_http_allowed

DEFAULT_MODEL = "deepseek-v4.1-flash"
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def configuration(env=None):
    """The teacher's endpoint from MLAB_TEACHER_*; None when no teacher is configured."""
    env = os.environ if env is None else env
    base = env.get("MLAB_TEACHER_BASE_URL", "").rstrip("/")
    if not base:
        return None
    parsed = urllib.parse.urlsplit(base)
    require(parsed.scheme == "https" or (parsed.scheme == "http" and plain_http_allowed(parsed.hostname)),
            "E_SCHEMA", "MLAB_TEACHER_BASE_URL must use HTTPS (plain HTTP only for local or private hosts)", 2)
    return {"base": base, "model": env.get("MLAB_TEACHER_MODEL") or DEFAULT_MODEL, "key": env.get("MLAB_TEACHER_API_KEY", ""),
            "timeout": float(env.get("MLAB_TEACHER_TIMEOUT_S", "180"))}


def parse_object(text):
    text = FENCE.sub("", THINK.sub("", text or "").strip())
    try:
        value = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


class TeacherClient:
    def __init__(self, config):
        self.config = config

    @property
    def model(self):
        return self.config["model"]

    def complete(self, system, user, max_tokens=4096):
        """The model's reply as a dict. One retry without response_format for servers that reject it."""
        body = {"model": self.config["model"], "temperature": 0.2, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}}
        for attempt in range(2):
            try:
                return parse_object(self._post(body))
            except urllib.error.HTTPError as error:
                if error.code == 400 and attempt == 0 and "response_format" in body:
                    body.pop("response_format")
                    continue
                raise LabError("E_BACKEND_UNAVAILABLE", f"Teacher model HTTP {error.code}; check MLAB_TEACHER_* settings", 4) from None
            except ValueError:
                if attempt == 0:
                    body["messages"].append({"role": "user", "content": "Reply with one valid JSON object only."})
                    continue
                raise LabError("E_BACKEND_UNAVAILABLE", "The teacher model did not return a JSON object", 4) from None
            except (OSError, urllib.error.URLError):
                raise LabError("E_BACKEND_UNAVAILABLE", "Teacher model unreachable; check MLAB_TEACHER_BASE_URL", 4) from None
            except (KeyError, IndexError, TypeError):
                raise LabError("E_BACKEND_UNAVAILABLE", "The teacher endpoint did not return a chat completion", 4) from None
        raise LabError("E_BACKEND_UNAVAILABLE", "The teacher model did not return a JSON object", 4)

    def _post(self, body):
        headers = {"Content-Type": "application/json"}
        if self.config["key"]:
            headers["Authorization"] = "Bearer " + self.config["key"]
        request = urllib.request.Request(self.config["base"] + "/chat/completions", json.dumps(body).encode(), headers)

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        with urllib.request.build_opener(NoRedirect).open(request, timeout=self.config["timeout"]) as response:
            data = json.load(response)
        return data["choices"][0]["message"].get("content") or ""
