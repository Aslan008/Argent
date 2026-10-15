"""The visual room editor's server: the guards a local web server needs (token, Host, Origin, body
cap) and the API that is the only thing allowed to write rooms (validator gate, packaged rooms
read-only, references respected, layout side-car kept apart from the rooms)."""

import http.client
import json

import pytest

from src.rooms import editor_server
from src.rooms.editor_server import EditorServer
from src.rooms.library import RoomLibrary, default_starter_dir

TOOLS = {"run_command", "search_web", "read_file", "list_directory", "grep_search", "search_files",
         "get_file_outline", "analyze_project", "read_git_diff", "read_webpage", "browser_open",
         "create_artifact", "write_file", "replace_in_file", "memory", "todo_write"}


def _room(name="my_room", **extra):
    data = {
        "room": name, "description": "Run a command and report",
        "budget": {"max_iterations": 5, "max_tokens": 10000},
        "nodes": [{"id": "check", "type": "tool", "tool": "run_command", "args": {"command": "echo hi"},
                   "sets": {"ok": "ok"}}],
        "edges": [{"from": "check", "to": "EXIT:done", "if": "state.ok"},
                  {"from": "check", "to": "EXIT:failed", "else": True}],
    }
    data.update(extra)
    return data


@pytest.fixture
def server(tmp_path):
    udir = tmp_path / "rooms"
    stats = tmp_path / "stats.json"

    def factory():
        lib = RoomLibrary(user_dir=udir, stats_path=stats)
        lib.load_dir(default_starter_dir(), TOOLS, origin="starter")
        if udir.exists():
            lib.load_dir(udir, TOOLS, origin="user")
        return lib

    srv = EditorServer(factory, lambda: TOOLS, udir).start()
    yield srv
    srv.stop()


def _req(srv, method, path, body=None, headers=None, host=None, token=True):
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    h = {"Host": host or f"127.0.0.1:{srv.port}"}
    if token:
        h["X-Rooms-Token"] = srv.token
    h.update(headers or {})
    payload = None
    if body is not None:
        payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
        h.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=payload, headers=h)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(raw)
    except ValueError:
        return resp.status, raw


# -- guards ------------------------------------------------------------------------------------------
def test_binds_loopback_only(server):
    assert server.host == "127.0.0.1"
    assert server.url.startswith("http://127.0.0.1:")
    assert "?t=" in server.url


def test_no_token_is_unauthorized(server):
    status, _ = _req(server, "GET", "/api/state", token=False)
    assert status == 401


def test_wrong_token_is_unauthorized(server):
    status, _ = _req(server, "GET", "/api/state", token=False, headers={"X-Rooms-Token": "nope"})
    assert status == 401


def test_token_in_query_opens_the_page(server):
    status, body = _req(server, "GET", f"/?t={server.token}", token=False)
    assert status == 200
    assert b"<svg" in body and b"%%TOKEN%%" not in body
    # <script>/<link> cannot send the header, so the served page carries the token in their URLs
    assert f"/app.js?t={server.token}".encode() in body and f"/app.css?t={server.token}".encode() in body
    status, js = _req(server, "GET", f"/app.js?t={server.token}", token=False)
    assert status == 200 and b"/api/validate" in js
    assert _req(server, "GET", "/app.js", token=False)[0] == 401


def test_foreign_host_header_is_refused(server):
    status, _ = _req(server, "GET", "/api/state", host="evil.example:80")
    assert status == 403


def test_foreign_origin_is_refused(server):
    status, _ = _req(server, "POST", "/api/validate", {"room": _room()},
                     headers={"Origin": "http://evil.example"})
    assert status == 403


def test_own_origin_is_accepted(server):
    status, _ = _req(server, "POST", "/api/validate", {"room": _room()},
                     headers={"Origin": f"http://127.0.0.1:{server.port}"})
    assert status == 200


def test_body_cap(server):
    big = json.dumps({"room": _room(description="x" * (editor_server.MAX_BODY + 10))})
    status, _ = _req(server, "POST", "/api/validate", big)
    assert status == 413


def test_invalid_json_body(server):
    status, _ = _req(server, "POST", "/api/validate", "{not json")
    assert status == 400


def test_unknown_route(server):
    assert _req(server, "GET", "/api/nope")[0] == 404
    assert _req(server, "POST", "/api/nope", {})[0] == 404


def test_responses_carry_a_csp(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("GET", "/api/state", headers={"Host": f"127.0.0.1:{server.port}", "X-Rooms-Token": server.token})
    resp = conn.getresponse()
    resp.read()
    assert "default-src 'self'" in (resp.getheader("Content-Security-Policy") or "")
    conn.close()


# -- state / room ------------------------------------------------------------------------------------
def test_state_lists_starter_rooms_and_vocabulary(server):
    status, data = _req(server, "GET", "/api/state")
    assert status == 200
    names = {r["name"]: r["source"] for r in data["rooms"]}
    assert names["debug_loop"] == "starter"
    assert set(data["node_types"]) == {"tool", "agent", "condition", "human_pause", "set", "notify", "room",
                                         "mcp", "skill", "plugin", "leeth", "foreach", "wait", "loop", "break",
                                         "continue", "power", "hint", "forget", "assert", "parallel", "join"}
    assert "run_command" in data["tools"] and "Coder" in data["roles"]
    assert "console" in data["channels"]


def test_room_returns_graph_and_empty_layout(server):
    status, data = _req(server, "GET", "/api/room?name=debug_loop")
    assert status == 200
    assert data["source"] == "starter"
    assert data["room"]["room"] == "debug_loop" and data["room"]["nodes"]
    assert data["layout"] == {}


def test_missing_room_is_404(server):
    assert _req(server, "GET", "/api/room?name=nope")[0] == 404


# -- validate ----------------------------------------------------------------------------------------
def test_validate_ok_returns_diagrams(server):
    status, data = _req(server, "POST", "/api/validate", {"room": _room()})
    assert status == 200 and data["ok"] is True and data["errors"] == []
    assert data["mermaid"].startswith("flowchart TD")
    assert "check" in data["flow"]


def test_validate_reports_graph_problems(server):
    bad = _room(edges=[{"from": "check", "to": "ghost"}])
    status, data = _req(server, "POST", "/api/validate", {"room": bad})
    assert status == 200 and data["ok"] is False and data["errors"]


def test_validate_rejects_unknown_tool(server):
    bad = _room()
    bad["nodes"][0]["tool"] = "rm_rf_everything"
    _, data = _req(server, "POST", "/api/validate", {"room": bad})
    assert data["ok"] is False


def test_validate_rejects_schema_garbage_and_bad_names(server):
    assert _req(server, "POST", "/api/validate", {"room": "text"})[1]["ok"] is False
    assert _req(server, "POST", "/api/validate", {"room": {"nodes": 5}})[1]["ok"] is False
    _, data = _req(server, "POST", "/api/validate", {"room": _room(name="Bad Name!")})
    assert data["ok"] is False and any("snake_case" in e for e in data["errors"])
    # a path-like name must never reach the filesystem
    _, data = _req(server, "POST", "/api/validate", {"room": _room(name="../evil")})
    assert data["ok"] is False


def test_validate_refuses_spawn_room_nodes(server):
    bad = _room()
    bad["nodes"].append({"id": "sp", "type": "spawn_room"})
    _, data = _req(server, "POST", "/api/validate", {"room": bad})
    assert data["ok"] is False


# -- save / delete -----------------------------------------------------------------------------------
def test_save_writes_room_and_layout_sidecar(server):
    layout = {"check": [10, 20], "EXIT:done": [10, 120], "junk": "x"}
    status, data = _req(server, "POST", "/api/save", {"room": _room(), "layout": layout})
    assert status == 200 and data["ok"] is True
    assert (server.user_dir / "my_room.json").exists()
    side = json.loads((server.user_dir / "my_room.layout").read_text(encoding="utf-8"))
    assert side == {"check": [10.0, 20.0], "EXIT:done": [10.0, 120.0]}   # malformed entries dropped
    # the side-car must not be mistaken for a room by the loader
    _, state = _req(server, "GET", "/api/state")
    assert state["load_errors"] == {}
    _, room = _req(server, "GET", "/api/room?name=my_room")
    assert room["source"] == "user" and room["layout"]["check"] == [10.0, 20.0]


def test_save_refuses_invalid_room_and_writes_nothing(server):
    status, data = _req(server, "POST", "/api/save", {"room": _room(edges=[{"from": "check", "to": "ghost"}])})
    assert status == 422 and data["ok"] is False and data["errors"]
    assert not (server.user_dir / "my_room.json").exists()


def test_save_never_overwrites_a_packaged_room(server):
    _, data = _req(server, "GET", "/api/room?name=debug_loop")
    status, out = _req(server, "POST", "/api/save", {"room": data["room"], "replace": True})
    assert status == 409 and out["ok"] is False
    assert not (server.user_dir / "debug_loop.json").exists()


def test_save_existing_needs_replace(server):
    assert _req(server, "POST", "/api/save", {"room": _room()})[0] == 200
    status, out = _req(server, "POST", "/api/save", {"room": _room(description="changed")})
    assert status == 409 and out.get("exists") is True
    status, _ = _req(server, "POST", "/api/save", {"room": _room(description="changed"), "replace": True})
    assert status == 200
    _, got = _req(server, "GET", "/api/room?name=my_room")
    assert got["room"]["description"] == "changed"


def test_saved_room_references_are_checked(server):
    caller = _room(name="caller", nodes=[{"id": "go", "type": "room", "room": "missing_one"}],
                   edges=[{"from": "go", "to": "EXIT:done"}])
    status, out = _req(server, "POST", "/api/save", {"room": caller})
    assert status == 422 and out["errors"]
    assert not (server.user_dir / "caller.json").exists()


def test_delete_own_room_removes_file_and_layout(server):
    _req(server, "POST", "/api/save", {"room": _room(), "layout": {"check": [1, 2]}})
    status, out = _req(server, "POST", "/api/delete", {"name": "my_room"})
    assert status == 200 and out["ok"] is True
    assert not (server.user_dir / "my_room.json").exists()
    assert not (server.user_dir / "my_room.layout").exists()


def test_delete_refuses_packaged_missing_and_referenced(server):
    assert _req(server, "POST", "/api/delete", {"name": "debug_loop"})[0] == 409
    assert _req(server, "POST", "/api/delete", {"name": "nope"})[0] == 404
    _req(server, "POST", "/api/save", {"room": _room(name="inner_room")})
    caller = _room(name="outer_room", nodes=[{"id": "go", "type": "room", "room": "inner_room"}],
                   edges=[{"from": "go", "to": "EXIT:done"}])
    assert _req(server, "POST", "/api/save", {"room": caller})[0] == 200
    status, out = _req(server, "POST", "/api/delete", {"name": "inner_room"})
    assert status == 409 and "outer_room" in " ".join(out["errors"])
    assert (server.user_dir / "inner_room.json").exists()


# -- lifecycle / command ----------------------------------------------------------------------------
def test_start_and_stop_editor_module_level(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    try:
        srv = editor_server.start_editor(user_dir=tmp_path / "rooms")
        assert editor_server.running() is srv
        assert editor_server.start_editor() is srv        # one server per process
        assert srv.user_dir == (tmp_path / "rooms").resolve()
    finally:
        assert editor_server.stop_editor() is True
    assert editor_server.running() is None
    assert editor_server.stop_editor() is False


def test_the_page_ships_in_the_package():
    assert editor_server._INDEX.exists()
    folder = editor_server._INDEX.parent
    html = "".join((folder / f).read_text(encoding="utf-8") for f in ("index.html", "app.js", "app.css", "guide.html"))
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")   # no external resources
    assert "https://" not in html


def test_rooms_editor_command(tmp_path, monkeypatch):
    from src.rooms import cli
    monkeypatch.chdir(tmp_path)
    opened = []
    monkeypatch.setattr(cli, "_open_in_browser", lambda url: opened.append(url) or True)
    out, err = [], []
    io = cli.Io(out=out.append, err=err.append, confirm=lambda *_: False, choose=lambda *_: None,
                human=lambda *_: "", confirm_step=lambda *_: True)
    try:
        cli.handle("editor", io)
        assert opened and opened[0].startswith("http://127.0.0.1:")
        assert any(opened[0] in line for line in out)
        cli.handle("editor status", io)
        assert any(opened[0] in line for line in out[1:])
    finally:
        cli.handle("editor stop", io)
    assert editor_server.running() is None
    assert not err


def test_free_text_starting_with_editor_is_a_task(monkeypatch):
    from src.rooms import cli
    assert cli._is_command("editor", "") and cli._is_command("editor", "stop")
    assert not cli._is_command("editor", "config for the project")


# -- the in-editor guide ----------------------------------------------------------------------------
_EDITOR_DIR = editor_server._INDEX.parent


def _guide_text() -> str:
    return (_EDITOR_DIR / "guide.html").read_text(encoding="utf-8")


def _examples() -> list:
    return json.loads((_EDITOR_DIR / "examples.json").read_text(encoding="utf-8"))


def test_guide_and_examples_are_served_behind_the_token(server):
    status, body = _req(server, "GET", "/guide")
    assert status == 200 and b'id="g-start"' in body
    status, data = _req(server, "GET", "/examples")
    assert status == 200 and isinstance(data, list) and data
    assert _req(server, "GET", "/guide", token=False)[0] == 401
    assert _req(server, "GET", "/examples", token=False)[0] == 401
    # only the two known files: no path traversal through the static routes
    assert _req(server, "GET", "/guide/../index.html")[0] == 404


def test_every_example_is_a_valid_room(server):
    names = set()
    for ex in _examples():
        assert {"name", "title", "summary", "room"} <= set(ex)
        assert ex["name"] == ex["room"]["room"], "the card and the room share a name"
        assert ex["name"] not in names
        names.add(ex["name"])
        status, out = _req(server, "POST", "/api/validate", {"room": ex["room"]})
        assert status == 200 and out["ok"], f"{ex['name']}: {out['errors']}"
        assert editor_server.NAME_OK(ex["name"])


def test_examples_do_not_collide_with_packaged_rooms():
    starter = {p.stem for p in default_starter_dir().glob("*.json")}
    assert not starter & {ex["name"] for ex in _examples()}


def test_guide_links_point_at_real_things():
    import re
    guide, page = _guide_text(), editor_server._INDEX.read_text(encoding="utf-8")
    script = (_EDITOR_DIR / "app.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([^"]+)"', guide)) | set(re.findall(r'id="([^"]+)"', page))
    for target in re.findall(r'data-goto="([^"]+)"', guide):
        assert target in ids, f"data-goto={target}"
    for sel in re.findall(r'data-flash="#([^"]+)"', guide):
        assert sel in ids, f"data-flash=#{sel}: no such control on the page"
    names = {ex["name"] for ex in _examples()}
    for name in re.findall(r'data-example="([^"]+)"', guide):
        assert name in names, f"data-example={name}"
    # the page's own openGuide(...) calls land on real sections
    for sec in re.findall(r'openGuide\("(g-[a-z]+)"\)', script) + re.findall(r'sec: "(g-[a-z]+)"', script):
        assert sec in ids, f"page links to missing guide section {sec}"
    # every section has a title for the navigation
    for m in re.finditer(r'<section class="g-sec" id="([^"]+)"([^>]*)>', guide):
        assert 'data-title="' in m.group(2), m.group(1)


def test_guide_keeps_up_with_the_engine(server):
    """When a node type, a derived value, a channel or a role is added, the guide must say so."""
    guide = _guide_text()
    state = server.api_state()
    for t in state["node_types"]:
        assert f"({t})" in guide, f"node type {t} is not described in the guide"
    for v in state["derived"]:
        assert f"<code>{v}</code>" in guide, f"derived value {v}"
    for c in state["channels"]:
        assert f"<code>{c}</code>" in guide, f"notify channel {c}"
    for r in state["roles"]:
        assert f"<code>{r}</code>" in guide, f"role {r}"


def test_page_wires_the_guide_and_the_legend():
    page = editor_server._INDEX.read_text(encoding="utf-8")
    script = (_EDITOR_DIR / "app.js").read_text(encoding="utf-8")
    for needle in ('id="btnGuide"', 'id="guide"', 'id="palette"', 'id="btnUndo"', 'id="tabWords"'):
        assert needle in page, needle
    for needle in ('fetch("/guide"', 'api("/examples")', '"F1"', "function undo", "function condBuilder", "function kvEditor"):
        assert needle in script, needle


def test_editor_knows_every_node_type_the_engine_has():
    """A node type without a TYPES entry, an icon and colours broke `draw()` (the triage room lost its
    exits and its `propose` block): every type from the data model must be drawable."""
    import re
    from typing import get_args
    from src.rooms.models import NodeType

    script = (_EDITOR_DIR / "app.js").read_text(encoding="utf-8")
    sprite = editor_server._INDEX.read_text(encoding="utf-8")
    css = (_EDITOR_DIR / "app.css").read_text(encoding="utf-8")
    types_block = script.split("const TYPES = {", 1)[1].split("\n};", 1)[0]
    for t in get_args(NodeType):
        assert re.search(rf"^\s+{t}: \{{icon: \"i-{t}\"", types_block, re.M), f"TYPES has no entry for {t}"
        assert f'id="i-{t}"' in sprite, f"no icon i-{t} in the sprite"
        assert f"--c-{t}:" in css and f"--k-{t}:" in css, f"no colours for {t}"



def test_starter_triage_validates_but_a_copy_may_not_spawn(server):
    """The packaged triage holds spawn_room legitimately; the editor must not flag it. Under any other name
    (a copy) the block is still refused, and the starter name itself stays read-only."""
    _, got = _req(server, "GET", "/api/room?name=triage")
    triage = got["room"]
    _, data = _req(server, "POST", "/api/validate", {"room": triage})
    assert data["ok"] is True, data["errors"]
    copy = {**triage, "room": "triage_copy"}
    _, data = _req(server, "POST", "/api/validate", {"room": copy})
    assert data["ok"] is False
    status, _ = _req(server, "POST", "/api/save", {"room": triage})
    assert status == 409
