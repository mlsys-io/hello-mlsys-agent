"""The teacher for the course page: one loopback relay per workspace, started on first use.

The server reads MLAB_TEACHER_* once at startup (before it removes provider variables from its own
environment), so `mlab ask` / `mlab assess` run by the page's agents only receive a relay URL and token.
"""
import secrets
import threading

_config = None
_relays = {}
_lock = threading.Lock()


def configure():
    """Capture the teacher configuration from the server's environment; call before sanitizing it."""
    global _config
    from ..teacher.client import configuration
    _config = configuration()
    return _config


def env_for(root):
    """MLAB_TEACHER_URL/TOKEN/TIMEOUT_S for commands run in this workspace; empty without a teacher."""
    if _config is None:
        return {}
    from ..teacher.client import TeacherClient
    from ..teacher.relay import serve
    key = str(root)
    with _lock:
        if key not in _relays:
            token = secrets.token_urlsafe(24)
            _relays[key] = (serve(root, TeacherClient(_config), token), token)
        server, token = _relays[key]
    return {"MLAB_TEACHER_URL": f"http://127.0.0.1:{server.server_port}", "MLAB_TEACHER_TOKEN": token,
            "MLAB_TEACHER_TIMEOUT_S": str(int(4 * _config["timeout"] + 60))}


def close_all():
    with _lock:
        for server, _ in _relays.values():
            server.shutdown()
            server.server_close()
        _relays.clear()
