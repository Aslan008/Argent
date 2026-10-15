"""Webhooks: an HTTP request starts a room.

    POST http://127.0.0.1:8731/hook/<name>
    X-Argent-Secret: <the secret shown once when the hook was made>
    {"task": "...", "any_field_the_room_declares": "value"}

What makes this safe to leave running:

* it listens on the loopback address only, and a request whose `Host` is not that address is refused
  (a page in the browser cannot reach it by rebinding a name); to reach it from outside the person must
  add the name to `trusted_hosts` on purpose, or put their own tunnel in front;
* every hook has its own secret; only a hash of it is stored; a wrong secret is answered the same as an
  unknown hook, and repeated wrong ones are slowed down;
* only the fields the room declares (`defaults`), or the hook lists, come through - a request cannot
  write `_steps` or any other internal field into the run;
* a body is small (64 KB), values are plain text or numbers and short, one run per hook at a time;
* the run is unattended (see headless.py): a step that asks first is not run, a question ends the run.

The run itself happens in a child process, so a tool's output never lands on the host's screen and a
hang costs the run, not the program.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from logger import get_logger

log = get_logger("rooms.webhooks")

DEFAULT_PORT = 8731
PORT_ENV = "ARGENT_WEBHOOK_PORT"
MAX_BODY_BYTES = 64 * 1024
DRAIN_LIMIT_BYTES = 1024 * 1024
MAX_VALUE_CHARS = 4000
MAX_WAIT_SECONDS = 120
RUN_TIMEOUT_SECONDS = 30 * 60
MAX_RUNS_AT_ONCE = 4
#: wrong secrets from one address per window before it is turned away for the rest of the window
MAX_BAD_SECRETS = 5
BAD_SECRET_WINDOW_SECONDS = 60

_HOOK_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")
_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_RESERVED_FIELDS = frozenset({"task", "rooms_catalog", "error", "error_node"})
_STORE = "webhooks.json"


def _argent_dir() -> Path:
    from project_paths import project_root_or_cwd
    return Path(project_root_or_cwd()) / ".argent"


def store_path() -> Path:
    return _argent_dir() / _STORE


def port() -> int:
    try:
        return int(os.environ.get(PORT_ENV) or DEFAULT_PORT)
    except ValueError:
        return DEFAULT_PORT


# ── the hooks on disk ─────────────────────────────────────────────────────────────

def _load() -> dict:
    path = store_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("hooks", {})
    data.setdefault("trusted_hosts", [])
    return data


def _save(data: dict) -> None:
    from atomic_io import atomic_write_text
    store_path().parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(store_path(), json.dumps(data, ensure_ascii=False, indent=2))


def _hash(secret: str, salt: str) -> str:
    return hashlib.sha256((salt + secret).encode("utf-8")).hexdigest()


def list_hooks(room: str = None) -> list:
    """The hooks, without their secrets: [{name, room, params, enabled, created, last_run, ...}]."""
    out = []
    for name, hook in sorted(_load()["hooks"].items()):
        if room is not None and hook.get("room") != room:
            continue
        out.append({"name": name, "room": hook.get("room", ""), "params": list(hook.get("params") or []),
                    "enabled": bool(hook.get("enabled", True)), "created": hook.get("created"),
                    "last_run": hook.get("last_run"), "last_status": hook.get("last_status"),
                    "last_summary": hook.get("last_summary", ""), "task": hook.get("task", "")})
    return out


def add_hook(room: str, name: str = None, params=None, task: str = "") -> tuple:
    """Create a hook for `room`. Returns (hook dict, secret): the secret is shown once and not stored."""
    name = (name or f"{room}").strip()
    if not _HOOK_NAME.match(name):
        raise ValueError("the hook name may hold letters, digits, - and _ (up to 48 characters)")
    bad = [p for p in (params or []) if not _FIELD.match(str(p)) or p in _RESERVED_FIELDS or str(p).startswith("_")]
    if bad:
        raise ValueError(f"not usable as parameter names: {bad}")
    data = _load()
    if name in data["hooks"]:
        raise ValueError(f"a hook named '{name}' already exists")
    secret = secrets.token_urlsafe(24)
    salt = secrets.token_hex(8)
    data["hooks"][name] = {"room": room, "salt": salt, "secret_hash": _hash(secret, salt),
                           "params": [str(p) for p in (params or [])], "enabled": True, "task": task,
                           "created": datetime.now().isoformat(timespec="seconds")}
    _save(data)
    return next(h for h in list_hooks() if h["name"] == name), secret


def remove_hook(name: str) -> bool:
    data = _load()
    if name not in data["hooks"]:
        return False
    del data["hooks"][name]
    _save(data)
    return True


def set_enabled(name: str, enabled: bool) -> bool:
    data = _load()
    if name not in data["hooks"]:
        return False
    data["hooks"][name]["enabled"] = bool(enabled)
    _save(data)
    return True


def new_secret(name: str):
    """Replace a hook's secret (the old one stops working). Returns the new one, or None."""
    data = _load()
    hook = data["hooks"].get(name)
    if hook is None:
        return None
    secret = secrets.token_urlsafe(24)
    hook["salt"] = secrets.token_hex(8)
    hook["secret_hash"] = _hash(secret, hook["salt"])
    _save(data)
    return secret


def _record(name: str, status: str, summary: str) -> None:
    data = _load()
    hook = data["hooks"].get(name)
    if hook is None:
        return
    hook.update(last_run=datetime.now().isoformat(timespec="seconds"), last_status=status,
                last_summary=(summary or "")[:500])
    _save(data)


def trusted_hosts() -> list:
    return list(_load().get("trusted_hosts") or [])


def has_enabled_hooks() -> bool:
    return any(h["enabled"] for h in list_hooks())


def url_for(name: str, host: str = "127.0.0.1") -> str:
    return f"http://{host}:{port()}/hook/{name}"


# ── what a request may carry into a run ────────────────────────────────────────────

def clean_params(body, allowed: set) -> tuple:
    """(task, params, problems) from a request body: only declared fields, only plain short values."""
    if body is None:
        return "", {}, []
    if not isinstance(body, dict):
        return "", {}, ["the body must be a JSON object"]
    task, params, problems = "", {}, []
    for key, value in body.items():
        if key == "task":
            if not isinstance(value, str) or len(value) > MAX_VALUE_CHARS:
                problems.append("task must be text of up to %d characters" % MAX_VALUE_CHARS)
            else:
                task = value
            continue
        if not _FIELD.match(str(key)) or key in _RESERVED_FIELDS or key.startswith("_"):
            problems.append(f"'{key}' is not a usable parameter name")
        elif key not in allowed:
            problems.append(f"'{key}' is not a parameter of this hook")
        elif isinstance(value, bool) or isinstance(value, (int, float)):
            params[key] = value
        elif isinstance(value, str) and len(value) <= MAX_VALUE_CHARS:
            params[key] = value
        else:
            problems.append(f"'{key}' must be a number, true/false or text of up to {MAX_VALUE_CHARS} characters")
    return task, params, problems


def allowed_fields(hook: dict, library=None) -> set:
    """The fields a request may set: the hook's own list, else what the room declares in `defaults`."""
    listed = set(hook.get("params") or [])
    if listed:
        return listed
    try:
        if library is None:
            from src.rooms.session import build_library
            library = build_library()
        room = library.get(hook.get("room"))
        return set((room.defaults or {}).keys()) if room is not None else set()
    except Exception:
        return set()


# ── the run, in a child process ──────────────────────────────────────────────────

def run_in_child(room: str, task: str, params: dict, timeout: float = RUN_TIMEOUT_SECONDS) -> dict:
    """Run the room as `python -m src.rooms.headless` and return its verdict (never raises)."""
    import tempfile
    from project_paths import project_root_or_cwd
    from src.rooms.headless import RESULT_MARKER
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("ARGENT_TUI", None)
    env["PYTHONPATH"] = os.pathsep.join([str(root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    handle, pfile = tempfile.mkstemp(prefix="argent-hook-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(params, fh, ensure_ascii=False)
        exe = Path(sys.executable)
        python = str(exe.with_name("python.exe")) if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists() else sys.executable
        argv = [python, "-m", "src.rooms.headless", room, "--params-file", pfile]
        if task:
            argv += ["--task", task]
        try:
            proc = subprocess.run(argv, cwd=str(project_root_or_cwd()), env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return {"status": "error", "summary": f"timed out after {int(timeout)} s and was stopped", "room": room}
        except OSError as exc:
            return {"status": "error", "summary": f"could not start the run: {exc}", "room": room}
        for line in reversed((proc.stdout or "").splitlines()):
            if line.startswith(RESULT_MARKER):
                try:
                    payload = json.loads(line[len(RESULT_MARKER):])
                    if isinstance(payload, dict):
                        return payload
                except ValueError:
                    break
        tail = " ".join((proc.stderr or proc.stdout or "").split())[-300:]
        return {"status": "error", "room": room,
                "summary": f"the run ended without a result (exit code {proc.returncode}): {tail}"}
    finally:
        try:
            os.unlink(pfile)
        except OSError:
            pass


# ── the server ───────────────────────────────────────────────────────────────────

class WebhookServer:
    """The listener. `runner(room, task, params) -> dict` is injectable so the HTTP side is testable
    without starting a room."""

    def __init__(self, host: str = "127.0.0.1", port_number: int = None, runner=None, library=None):
        self.host = host
        self.requested_port = port() if port_number is None else port_number
        self._runner = runner or run_in_child
        self._library = library
        self._httpd = None
        self._thread = None
        self._busy = set()                     # hook names with a run in flight
        self._active = 0
        self._lock = threading.Lock()
        self._bad = {}                         # client address -> [timestamps of wrong secrets]
        self._workers = []

    # lifecycle
    @property
    def port(self) -> int:
        return self._httpd.server_address[1] if self._httpd else self.requested_port

    def start(self) -> "WebhookServer":
        if self._httpd is not None:
            return self
        server = self
        handler = type("Handler", (_Handler,), {"owner": server})
        self._httpd = ThreadingHTTPServer((self.host, self.requested_port), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="argent-webhooks")
        self._thread.start()
        log.info("webhooks listening on %s:%s", self.host, self.port)
        return self

    def stop(self) -> None:
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        self._thread = None

    def join(self, timeout: float = None) -> None:
        """Wait for the runs started so far (for tests)."""
        with self._lock:
            workers = list(self._workers)
        for w in workers:
            w.join(timeout)

    # the request
    def _throttled(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = [t for t in self._bad.get(client, []) if now - t < BAD_SECRET_WINDOW_SECONDS]
            self._bad[client] = hits
            return len(hits) >= MAX_BAD_SECRETS

    def _note_bad_secret(self, client: str) -> None:
        with self._lock:
            self._bad.setdefault(client, []).append(time.monotonic())

    def handle(self, name: str, headers, body_bytes: bytes, query: dict, client: str) -> tuple:
        """Everything a POST does, as (status, payload) - separate from the socket so it is testable."""
        if self._throttled(client):
            return HTTPStatus.TOO_MANY_REQUESTS, {"ok": False, "error": "too many wrong secrets; wait a minute"}
        hook = _load()["hooks"].get(name)
        given = (headers.get("X-Argent-Secret") or "").strip()
        auth = (headers.get("Authorization") or "").strip()
        if not given and auth.lower().startswith("bearer "):
            given = auth[7:].strip()
        good = bool(hook) and hmac.compare_digest(_hash(given, hook.get("salt", "")), hook.get("secret_hash", ""))
        if not good:
            self._note_bad_secret(client)
            return HTTPStatus.NOT_FOUND, {"ok": False, "error": "no such hook"}      # unknown hook and wrong secret read alike
        if not hook.get("enabled", True):
            return HTTPStatus.CONFLICT, {"ok": False, "error": "this hook is switched off"}
        body = None
        if body_bytes.strip():
            try:
                body = json.loads(body_bytes.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "the body is not valid JSON"}
        task, params, problems = clean_params(body, allowed_fields(hook, self._library))
        if problems:
            return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "; ".join(problems)}
        task = task or hook.get("task", "")
        with self._lock:
            if name in self._busy:
                return HTTPStatus.TOO_MANY_REQUESTS, {"ok": False, "error": "this hook is already running", "retry_after": 5}
            if self._active >= MAX_RUNS_AT_ONCE:
                return HTTPStatus.TOO_MANY_REQUESTS, {"ok": False, "error": "too many runs at once", "retry_after": 5}
            self._busy.add(name)
            self._active += 1
        result_box = {}
        done = threading.Event()

        def work():
            try:
                result_box["result"] = self._runner(hook["room"], task, params)
            except Exception as exc:                    # the listener outlives its runs
                result_box["result"] = {"status": "error", "summary": f"the run failed: {exc}"}
            finally:
                res = result_box["result"]
                try:
                    _record(name, res.get("status", "error"), res.get("summary", ""))
                finally:
                    with self._lock:
                        self._busy.discard(name)
                        self._active -= 1
                    done.set()

        worker = threading.Thread(target=work, daemon=True, name=f"webhook-{name}")
        with self._lock:
            self._workers = [w for w in self._workers if w.is_alive()] + [worker]
        worker.start()
        try:
            wait = min(float((query.get("wait") or ["0"])[0]), MAX_WAIT_SECONDS)
        except ValueError:
            wait = 0.0
        if wait > 0 and done.wait(wait):
            res = result_box["result"]
            code = HTTPStatus.OK if res.get("status") in ("ok", "denied") else HTTPStatus.INTERNAL_SERVER_ERROR
            return code, {"ok": res.get("status") == "ok", "status": res.get("status"), "exit": res.get("exit", ""),
                          "needs_person": bool(res.get("needs_person")), "summary": res.get("summary", "")[:4000]}
        return HTTPStatus.ACCEPTED, {"ok": True, "status": "started", "room": hook["room"]}


class _Handler(BaseHTTPRequestHandler):
    owner: WebhookServer = None
    protocol_version = "HTTP/1.1"
    server_version = "ArgentRooms"

    def log_message(self, fmt, *args):            # the host's screen is not for access logs
        log.debug("webhook: " + fmt, *args)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1") or host in [h.lower() for h in trusted_hosts()]

    def _reply(self, status, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.close_connection = True               # an early refusal leaves the body unread: do not reuse the socket
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            self.send_header("Retry-After", "5")
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if not self._host_ok():
            return self._reply(HTTPStatus.FORBIDDEN, {"ok": False, "error": "unexpected Host"})
        parts = urlsplit(self.path)
        match = re.match(r"^/hook/([A-Za-z0-9][A-Za-z0-9_-]{0,47})/?$", parts.path)
        if not match:
            return self._reply(HTTPStatus.NOT_FOUND, {"ok": False, "error": "no such hook"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._reply(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "bad Content-Length"})
        if length > MAX_BODY_BYTES:
            # the sender is still writing: read (a bounded amount of) the body and drop it, or the
            # answer is lost in the reset the unread bytes cause
            left = min(length, DRAIN_LIMIT_BYTES)
            while left > 0:
                chunk = self.rfile.read(min(65536, left))
                if not chunk:
                    break
                left -= len(chunk)
            return self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": f"the body is over {MAX_BODY_BYTES} bytes"})
        body = self.rfile.read(length) if length else b""
        status, payload = self.owner.handle(match.group(1), self.headers, body, parse_qs(parts.query),
                                            self.client_address[0])
        self._reply(status, payload)

    def do_GET(self):
        self._reply(HTTPStatus.METHOD_NOT_ALLOWED, {"ok": False, "error": "use POST"})

    do_PUT = do_DELETE = do_PATCH = do_GET


# ── the host's side ──────────────────────────────────────────────────────────────

_instance = None
_instance_lock = threading.Lock()


def start_if_configured():
    """Start this process's listener when at least one hook is switched on (and not twice). Returns
    the server, or None. A busy port (another Argent window already listens) is not an error."""
    global _instance
    if os.environ.get("ARGENT_DISABLE_AUTOMATIONS") == "1" or not has_enabled_hooks():
        return None
    with _instance_lock:
        if _instance is not None:
            return _instance
        try:
            _instance = WebhookServer().start()
        except OSError as exc:
            log.info("webhooks not started here: %s", exc)
            return None
        import atexit
        atexit.register(_instance.stop)
        return _instance


def main(argv=None) -> int:
    """python -m src.rooms.webhooks serve [--port N]"""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "serve":
        print("usage: python -m src.rooms.webhooks serve [--port N]", file=sys.stderr)
        return 2
    chosen = None
    if "--port" in args:
        at = args.index("--port")
        chosen = int(args[at + 1])
    server = WebhookServer(port_number=chosen).start()
    print(f"Argent webhooks: http://{server.host}:{server.port}/hook/<name>  (Ctrl+C to stop)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
