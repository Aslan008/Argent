"""The visual room editor's local server.

A person edits a room by dragging boxes on a page; the page talks to this tiny HTTP
server, which owns everything that matters: the SAME validator that guards chat-built and
hand-written rooms decides what may be saved, and only the user's own rooms directory is
ever written. The page itself holds no authority.

Safety of a local server:
  * it binds 127.0.0.1 only, on a random port;
  * every request carries a random token (in the opened URL, then a header) — a web page
    in another tab cannot guess it;
  * the Host header must be this server (DNS-rebinding), and a browser Origin, when sent,
    must be this server too;
  * bodies are capped, rooms are data (never executed), and packaged rooms are read-only.

Run it with `/rooms editor`; it stops with `/rooms editor stop` or with the program.
"""

import json
import secrets
import threading
from typing import Any, get_args
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

from src.rooms import diagram
from src.rooms import models as _m
from src.rooms.models import DERIVED_VALUES, LEETH_MODES, NOTIFY_CHANNELS, NodeType, Room
from src.rooms.validator import KNOWN_ROLES, NOT_IN_ROOMS

MAX_BODY = 512 * 1024
#: How much of a refused (too large) body is read and discarded so the refusal reaches the client.
DRAIN_LIMIT = 8 * 1024 * 1024
NAME_OK = lambda s: bool(s) and s.replace("_", "").isalnum() and s[0].isalpha() and s == s.lower() and len(s) <= 41  # noqa: E731
_INDEX = Path(__file__).parent / "editor" / "index.html"
#: The only other files the page may ask for (path -> (file in the editor folder, content type)).
_STATIC = {
    "/guide": ("guide.html", "text/html; charset=utf-8"),
    "/examples": ("examples.json", "application/json; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/ext.js": ("ext.js", "text/javascript; charset=utf-8"),
    "/run.js": ("run.js", "text/javascript; charset=utf-8"),
    "/simple.js": ("simple.js", "text/javascript; charset=utf-8"),
    "/profiles.js": ("profiles.js", "text/javascript; charset=utf-8"),
    "/stack.js": ("stack.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
}


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


#: How long the page waits for something that may start a process (an MCP server's tool list).
CATALOG_TIMEOUT = 25.0
MAX_PARAMS = 60


def _with_timeout(fn, seconds: float):
    """(ok, value-or-error text): `fn()` in a thread, abandoned after `seconds` (the thread is a daemon)."""
    box = {}

    def work():
        try:
            box["v"] = fn()
        except Exception as e:
            box["e"] = str(e)[:300]

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        return False, f"no answer within {seconds:g} s"
    if "e" in box:
        return False, box["e"]
    return True, box.get("v")


class EditorServer:
    def __init__(self, library_factory: Callable, tools: Callable, user_dir, host: str = "127.0.0.1", port: int = 0,
                 integrations: Any = None, runner_factory: Callable = None, tool_catalog: Callable = None):
        self._library_factory = library_factory     # () -> RoomLibrary (fresh view of disk)
        self._tools = tools                         # () -> iterable of tool names
        self._integrations = integrations           # catalogs of MCP / skills / plugins / Leeth (lazy default)
        self._runner_factory = runner_factory       # (human, confirm) -> NodeRunner (production default)
        self._tool_catalog = tool_catalog           # () -> [tool info] (production default)
        self._tool_info_cache = None
        self._runs = None
        self.user_dir = Path(user_dir)
        self.token = secrets.token_urlsafe(18)
        self._httpd = ThreadingHTTPServer((host, port), self._handler_class())
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self.host = host
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle --------------------------------------------------------------------------------
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/?t={self.token}"

    def start(self) -> "EditorServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="rooms-editor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:
            pass

    # -- the API (pure: dict in, (status, dict) out — testable without a socket) ------------------
    @property
    def integrations(self):
        if self._integrations is None:
            from src.rooms.integrations import Integrations
            self._integrations = Integrations()
        return self._integrations

    @property
    def runs(self):
        if self._runs is None:
            from src.rooms.editor_runs import RunManager
            self._runs = RunManager(self._runner_factory or self._production_runner)
        return self._runs

    def _production_runner(self, human, confirm):
        from src.rooms.session import build_runner
        return build_runner(human, library=self._library_factory(), confirm=confirm)

    def _tool_infos(self) -> list:
        """Parameters, group and risk of every registered tool (computed once per server)."""
        if self._tool_info_cache is None:
            if self._tool_catalog is not None:
                self._tool_info_cache = list(self._tool_catalog())
            else:
                from src.rooms.editor_catalog import tool_catalog
                self._tool_info_cache = tool_catalog()
        return self._tool_info_cache

    def api_state(self) -> dict:
        lib = self._library_factory()
        rooms = []
        for name in lib.names():
            r = lib.rooms[name]
            rooms.append({"name": name, "source": lib.sources.get(name, "?"), "description": r.description,
                          "tags": r.tags, "quarantined": bool(lib.stats.get(name, {}).get("quarantined"))})
        return {
            "rooms": rooms, "load_errors": lib.load_errors,
            "tools": sorted(self._tools()), "roles": sorted(KNOWN_ROLES),
            "channels": list(NOTIFY_CHANNELS), "derived": list(DERIVED_VALUES),
            "node_types": [t for t in get_args(NodeType) if t != "spawn_room"],   # spawn_room is the triage room's alone
            "tool_info": self._tool_infos(), "tool_groups": self._tool_groups(),
            "not_in_rooms": sorted(NOT_IN_ROOMS),
            "leeth": {"available": self.integrations.leeth_available(), "modes": list(LEETH_MODES)},
            "vocab": {
                "loop_modes": list(_m.LOOP_MODES), "power_modes": list(_m.POWER_MODES), "hint_modes": list(_m.HINT_MODES),
                "forget_modes": list(_m.FORGET_MODES), "memory_modes": list(_m.MEMORY_MODES),
                "sampling": {k: list(v) for k, v in _m.SAMPLING_BOUNDS.items()},
                "reasoning_efforts": list(_m.REASONING_EFFORTS),
                "loop_default_max": _m.LOOP_DEFAULT_MAX, "loop_max": _m.LOOP_MAX,
                "power_min_delay": _m.POWER_MIN_DELAY, "power_max_delay": _m.POWER_MAX_DELAY,
                "extra_derived": list(_m.EXTRA_DERIVED),
                "parallel_max_branches": _m.PARALLEL_MAX_BRANCHES, "parallel_max_depth": _m.PARALLEL_MAX_DEPTH,
                "merge_modes": list(_m.MERGE_MODES), "wait_for_modes": list(_m.WAIT_FOR_MODES),
                "branch_fail_modes": list(_m.BRANCH_FAIL_MODES),
                "mcp_call_timeout": self._mcp_call_timeout(),
            },
        }

    @staticmethod
    def _mcp_call_timeout() -> float:
        """How long Argent waits for one MCP call (the page warns about slow web-AI questions with it)."""
        try:
            from config import get_mcp_call_timeout
            return float(get_mcp_call_timeout())
        except Exception:
            return 120.0

    @staticmethod
    def _tool_groups() -> dict:
        from src.rooms.editor_catalog import GROUPS
        return dict(GROUPS)

    # -- what to pick from: MCP servers, skills, plugins --------------------------------------------
    def api_mcp(self, server: str = ""):
        if not server:
            return HTTPStatus.OK, {"servers": self.integrations.mcp_servers()}
        ok, value = _with_timeout(lambda: self.integrations.mcp_tools(server), CATALOG_TIMEOUT)
        if not ok:
            return HTTPStatus.BAD_GATEWAY, {"error": str(value), "server": server}
        return HTTPStatus.OK, {"server": server, "tools": value}

    def api_skills(self, name: str = ""):
        if not name:
            return HTTPStatus.OK, {"skills": [{"name": k, "description": v}
                                              for k, v in sorted(self.integrations.skills().items())]}
        return HTTPStatus.OK, {"name": name, "headings": self.integrations.skill_headings(name)}

    def api_plugins(self):
        return HTTPStatus.OK, {"plugins": self.integrations.plugins()}

    # -- runs from the page ------------------------------------------------------------------------
    @staticmethod
    def _params(raw) -> dict:
        if not isinstance(raw, dict):
            return {}
        out = {}
        for k, v in list(raw.items())[:MAX_PARAMS]:
            if isinstance(k, str) and k.replace("_", "").replace(".", "").isalnum():
                out[k] = v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v, ensure_ascii=False)
        return out

    def api_run(self, body: dict):
        problems, room = self._check(body.get("room"))
        if problems or room is None:
            return HTTPStatus.UNPROCESSABLE_ENTITY, {"ok": False, "errors": problems}
        params = self._params(body.get("params"))
        run = self.runs.start(room, params)
        return HTTPStatus.OK, {"ok": True, "run": run.id}

    def api_run_node(self, body: dict):
        problems, room = self._check(body.get("room"))
        if problems or room is None:
            return HTTPStatus.UNPROCESSABLE_ENTITY, {"ok": False, "errors": problems}
        node = next((n for n in room.nodes if n.id == body.get("node")), None)
        if node is None:
            return HTTPStatus.NOT_FOUND, {"ok": False, "errors": [f"no step '{body.get('node')}'"]}
        if node.type in ("human_pause", "room", "spawn_room", "condition", "break", "continue", "parallel", "join"):
            return HTTPStatus.UNPROCESSABLE_ENTITY, {"ok": False, "errors": [
                f"a step of type '{node.type}' cannot be run alone: it only makes sense inside a run"]}
        from src.rooms.editor_catalog import acts_outside
        reads_only = node.type in ("skill", "set", "wait", "foreach", "loop", "hint", "forget", "assert") or (
            node.type == "notify" and node.channel == "console") or (
            node.type == "tool" and not acts_outside(node.tool)) or (
            node.type == "git" and (node.mode or "status") in ("status", "diff", "log")) or (
            node.type == "http" and (node.mode or "get").lower() in ("get", "head")) or (
            node.type == "mcp" and (node.mode or "call") == "list") or (
            node.type == "leeth" and (node.mode or "delegate") in ("filter", "classify", "scan"))
        run = self.runs.start(room, self._params(body.get("params")), node_id=node.id, ask_first=not reads_only)
        return HTTPStatus.OK, {"ok": True, "run": run.id, "asks": not reads_only}

    def api_run_status(self, run_id: str, since: int = 0):
        run = self.runs.get(run_id)
        if run is None:
            return HTTPStatus.NOT_FOUND, {"error": f"no run '{run_id}'"}
        return HTTPStatus.OK, run.snapshot(max(0, since))

    def api_runs(self):
        return HTTPStatus.OK, {"runs": self.runs.list()}

    def api_answer(self, body: dict):
        try:
            prompt_id = int(body.get("prompt"))
        except (TypeError, ValueError):
            return HTTPStatus.BAD_REQUEST, {"ok": False, "errors": ["prompt id missing"]}
        ok = self.runs.answer(str(body.get("run") or ""), prompt_id, body.get("answer"))
        return (HTTPStatus.OK if ok else HTTPStatus.CONFLICT), {"ok": ok}

    def api_stop(self, body: dict):
        return HTTPStatus.OK, {"ok": self.runs.stop(str(body.get("run") or ""))}

    def api_triggers(self, room: str):
        """The schedules and webhooks that start `room` (all rooms when empty)."""
        from src.rooms import triggers, webhooks
        return HTTPStatus.OK, {"ok": True, "triggers": triggers.list_triggers(room or None),
                               "webhook_port": webhooks.port(), "webhooks_serving": webhooks._instance is not None}

    def api_trigger(self, body: dict):
        """Add / remove / switch a trigger. A webhook's secret is in the answer once and nowhere else."""
        from src.rooms import triggers, webhooks
        action, room = str(body.get("action") or ""), str(body.get("room") or "")
        try:
            if action == "add_schedule":
                if self._library_factory().get(room) is None:
                    return HTTPStatus.NOT_FOUND, {"ok": False, "errors": [f"no room '{room}' (save it first)"]}
                made = triggers.add_schedule(room, str(body.get("schedule") or ""), str(body.get("task") or ""),
                                             self._params(body.get("params")))
                return HTTPStatus.OK, {"ok": True, "trigger": made}
            if action == "add_webhook":
                if self._library_factory().get(room) is None:
                    return HTTPStatus.NOT_FOUND, {"ok": False, "errors": [f"no room '{room}' (save it first)"]}
                fields = [str(p) for p in (body.get("params") or []) if str(p).strip()]
                made, secret = triggers.add_webhook(room, str(body.get("name") or "") or None, fields, str(body.get("task") or ""))
                started = webhooks.start_if_configured() is not None or webhooks._instance is not None
                return HTTPStatus.OK, {"ok": True, "trigger": made, "secret": secret, "serving": started,
                                       "note": "The secret is shown only now: keep it where the sender can read it."}
            kind, name = str(body.get("kind") or ""), str(body.get("name") or "")
            if action == "remove":
                return HTTPStatus.OK, {"ok": triggers.remove(kind, name)}
            if action in ("enable", "disable"):
                return HTTPStatus.OK, {"ok": triggers.set_enabled(kind, name, action == "enable")}
            if action == "new_secret" and kind == "webhook":
                secret = webhooks.new_secret(name)
                return HTTPStatus.OK, {"ok": secret is not None, "secret": secret}
        except ValueError as exc:
            return HTTPStatus.UNPROCESSABLE_ENTITY, {"ok": False, "errors": [str(exc)]}
        return HTTPStatus.BAD_REQUEST, {"ok": False, "errors": [f"unknown action '{action}'"]}

    def api_room(self, name: str):
        lib = self._library_factory()
        room = lib.get(name)
        if room is None:
            return HTTPStatus.NOT_FOUND, {"error": f"no room '{name}'"}
        data = json.loads(room.model_dump_json(by_alias=True, exclude_none=True))
        return HTTPStatus.OK, {"room": data, "source": lib.sources.get(name, "?"),
                               "layout": self._read_layout(name, getattr(lib, "extra_dirs", ()))}

    def api_validate(self, body: dict):
        problems, room = self._check(body.get("room"))
        out = {"ok": not problems, "errors": problems}
        if room is not None:
            out["flow"] = diagram.render_flow(room)
            out["mermaid"] = diagram.render_mermaid(room)
            if not problems:
                from src.rooms.validator import room_warnings
                try:
                    out["warnings"] = room_warnings(room, self.integrations)
                except Exception:
                    out["warnings"] = []
        return HTTPStatus.OK, out

    def api_save(self, body: dict):
        data = body.get("room")
        problems, room = self._check(data)
        if problems or room is None:
            return HTTPStatus.UNPROCESSABLE_ENTITY, {"ok": False, "errors": problems}
        lib = self._library_factory()
        if lib.sources.get(room.room) == "starter":
            return HTTPStatus.CONFLICT, {"ok": False, "errors": [
                f"'{room.room}' is a packaged room and is read-only; save it under another name"]}
        if lib.get(room.room) is not None and not body.get("replace"):
            return HTTPStatus.CONFLICT, {"ok": False, "exists": True,
                                         "errors": [f"room '{room.room}' already exists"]}
        lib.user_dir = self.user_dir
        errors = lib.add_room(room, self._tools())
        if errors:
            return HTTPStatus.UNPROCESSABLE_ENTITY, {"ok": False, "errors": errors}
        layout = body.get("layout")
        if isinstance(layout, dict):
            self._write_layout(room.room, layout)
        return HTTPStatus.OK, {"ok": True, "saved": str(self.user_dir / f"{room.room}.json")}

    def api_delete(self, body: dict):
        name = str(body.get("name") or "")
        lib = self._library_factory()
        if lib.get(name) is None:
            return HTTPStatus.NOT_FOUND, {"ok": False, "errors": [f"no room '{name}'"]}
        if lib.sources.get(name) != "user":
            return HTTPStatus.CONFLICT, {"ok": False, "errors": ["only your own rooms can be deleted"]}
        users = lib.referenced_by(name)
        if users:
            return HTTPStatus.CONFLICT, {"ok": False, "errors": [f"room is called by: {', '.join(users)}"]}
        lib.user_dir = self.user_dir
        lib.remove_room(name)
        for folder in [self.user_dir, *getattr(lib, "extra_dirs", ())]:
            try:
                (Path(folder) / f"{name}.layout").unlink()
            except OSError:
                pass
        return HTTPStatus.OK, {"ok": True}

    # -- helpers ----------------------------------------------------------------------------------
    def _check(self, data) -> tuple:
        """(problems, Room|None) for a room object from the page."""
        if not isinstance(data, dict):
            return ["no room object"], None
        try:
            room = Room(**data)
        except Exception as e:
            return [f"schema: {str(e)[:600]}"], None
        if not NAME_OK(room.room):
            return [f"room name '{room.room}' must be snake_case (a-z, 0-9, _), starting with a letter"], room
        from src.rooms.validator import validate_room
        lib = self._library_factory()
        # spawn_room is the right of the starter triage room only (the same rule the library applies);
        # a saved room can never have it — saving under a starter name is refused with 409.
        spawns = room.room in getattr(lib, "_spawn_rooms", ()) and lib.sources.get(room.room) == "starter"
        problems = validate_room(room, set(self._tools()), allow_spawn=spawns)
        if not problems:
            problems = lib.reference_errors({**lib.rooms, room.room: room}).get(room.room, [])
        return problems, room

    def _layout_path(self, name: str) -> Path:
        return self.user_dir / f"{name}.layout"

    def _read_layout(self, name: str, other_dirs=()) -> dict:
        """The saved arrangement: next to the shared rooms first, then next to the room's own project folder
        (a room that has lived only in a project has its .layout there until the first save here)."""
        for folder in [self.user_dir, *other_dirs]:
            try:
                data = json.loads((Path(folder) / f"{name}.layout").read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                return data
        return {}

    def _write_layout(self, name: str, layout: dict) -> None:
        clean = {}
        for k, v in layout.items():
            try:
                clean[str(k)[:80]] = [float(v[0]), float(v[1])]
            except Exception:
                continue
        try:
            self.user_dir.mkdir(parents=True, exist_ok=True)
            self._layout_path(name).write_text(json.dumps(clean), encoding="utf-8")
        except OSError:
            pass

    # -- HTTP -------------------------------------------------------------------------------------
    def _handler_class(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "ArgentRooms"

            def log_message(self, fmt, *args):      # silence the default stderr access log
                return

            # request guards ---------------------------------------------------------------------
            def _host_ok(self) -> bool:
                host = (self.headers.get("Host") or "").lower()
                return host in (f"127.0.0.1:{server.port}", f"localhost:{server.port}")

            def _origin_ok(self) -> bool:
                origin = self.headers.get("Origin")
                if not origin:
                    return True
                return origin.lower() in (f"http://127.0.0.1:{server.port}", f"http://localhost:{server.port}")

            def _token_ok(self, query: dict) -> bool:
                given = self.headers.get("X-Rooms-Token") or (query.get("t") or [""])[0]
                return secrets.compare_digest(str(given), server.token)

            def _send(self, status, payload=None, raw=None, ctype="application/json; charset=utf-8", close=False):
                body = raw if raw is not None else _json_bytes(payload)
                self.send_response(int(status))
                if close:
                    self.send_header("Connection", "close")
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                 "style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:")
                self.end_headers()
                self.wfile.write(body)

            def _guard(self, query) -> bool:
                if not self._host_ok() or not self._origin_ok():
                    self._send(HTTPStatus.FORBIDDEN, {"error": "bad host or origin"})
                    return False
                if not self._token_ok(query):
                    self._send(HTTPStatus.UNAUTHORIZED, {"error": "bad token"})
                    return False
                return True

            # routes -----------------------------------------------------------------------------
            def do_GET(self):
                url = urlparse(self.path)
                query = parse_qs(url.query)
                if not self._guard(query):
                    return
                try:
                    if url.path == "/":
                        html = _INDEX.read_text(encoding="utf-8") if _INDEX.exists() else "<h1>editor missing</h1>"
                        # a <script>/<link> request cannot carry the header, so the page's own tags carry the token
                        html = html.replace("%%TOKEN%%", server.token)
                        return self._send(HTTPStatus.OK, raw=html.encode("utf-8"), ctype="text/html; charset=utf-8")
                    if url.path in _STATIC:          # the in-page guide and its ready-made examples
                        name, ctype = _STATIC[url.path]
                        path = _INDEX.parent / name
                        if not path.exists():
                            return self._send(HTTPStatus.NOT_FOUND, {"error": f"{name} is missing"})
                        return self._send(HTTPStatus.OK, raw=path.read_bytes(), ctype=ctype)
                    if url.path == "/api/state":
                        return self._send(HTTPStatus.OK, server.api_state())
                    if url.path == "/api/room":
                        status, payload = server.api_room((query.get("name") or [""])[0])
                        return self._send(status, payload)
                    if url.path == "/api/mcp":
                        status, payload = server.api_mcp((query.get("server") or [""])[0])
                        return self._send(status, payload)
                    if url.path == "/api/skills":
                        status, payload = server.api_skills((query.get("name") or [""])[0])
                        return self._send(status, payload)
                    if url.path == "/api/plugins":
                        status, payload = server.api_plugins()
                        return self._send(status, payload)
                    if url.path == "/api/runs":
                        status, payload = server.api_runs()
                        return self._send(status, payload)
                    if url.path == "/api/triggers":
                        status, payload = server.api_triggers((query.get("room") or [""])[0])
                        return self._send(status, payload)
                    if url.path == "/api/run":
                        try:
                            since = int((query.get("since") or ["0"])[0])
                        except ValueError:
                            since = 0
                        status, payload = server.api_run_status((query.get("id") or [""])[0], since)
                        return self._send(status, payload)
                    return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                except Exception as e:          # never leak a traceback to the page
                    return self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)[:200]})

            def do_POST(self):
                url = urlparse(self.path)
                query = parse_qs(url.query)
                if not self._guard(query):
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                if length < 0 or length > MAX_BODY:
                    # Say no and close. The client is still sending: unread bytes make the OS answer
                    # its next write with a reset, and the "no" is lost. So the body is read and
                    # thrown away first (bounded), then the answer goes out and the connection ends.
                    self.close_connection = True
                    left = min(max(length, 0), DRAIN_LIMIT)
                    while left > 0:
                        chunk = self.rfile.read(min(65536, left))
                        if not chunk:
                            break
                        left -= len(chunk)
                    return self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"},
                                      close=True)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("object expected")
                except Exception:
                    return self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
                handlers = {"/api/validate": server.api_validate, "/api/save": server.api_save,
                            "/api/delete": server.api_delete, "/api/run": server.api_run,
                            "/api/run_node": server.api_run_node, "/api/answer": server.api_answer,
                            "/api/stop": server.api_stop, "/api/trigger": server.api_trigger}
                fn = handlers.get(url.path)
                if fn is None:
                    return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                try:
                    status, payload = fn(body)
                    return self._send(status, payload)
                except Exception as e:
                    return self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)[:200]})

        return Handler


_running: Optional[EditorServer] = None


def running() -> Optional[EditorServer]:
    return _running


def start_editor(library_factory=None, tools=None, user_dir=None) -> EditorServer:
    """Start (or return) the process's editor server."""
    global _running
    if _running is not None:
        return _running
    from src.rooms.session import (
        available_tool_names, build_library, project_rooms_dir, stats_path, user_rooms_dir,
    )
    udir = Path(user_dir).resolve() if user_dir else user_rooms_dir().resolve()
    local = project_rooms_dir().resolve()      # pinned now too: the project the editor was opened in
    stats = stats_path().resolve()           # pinned now: a later /cd must not move what the editor edits
    _running = EditorServer(
        library_factory or (lambda: build_library(user_dir=udir, stats_file=stats, extra_dirs=[local])),
        tools or available_tool_names,
        udir,
    ).start()
    return _running


def stop_editor() -> bool:
    global _running
    if _running is None:
        return False
    _running.stop()
    _running = None
    return True
