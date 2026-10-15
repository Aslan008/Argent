"""The editor server's new surface: what to pick from (tools with parameters, MCP servers and their
tools, skills and their headings, plugins, Leeth), warnings on validate, and runs from the page
(whole room, one step, questions answered in the browser, stop)."""

import http.client
import json
import time

import pytest

from src.rooms import editor_server
from src.rooms.editor_catalog import group_of, tool_catalog
from src.rooms.editor_server import EditorServer
from src.rooms.library import RoomLibrary, default_starter_dir
from src.rooms.models import Node
from src.rooms.runner import NodeRunner

TOOLS = {"run_command", "read_file", "list_directory", "search_web", "browser_open", "write_file"}


class FakeCatalog:
    def __init__(self):
        self.fail_tools = False

    def mcp_servers(self):
        return {"unity": {"type": "stdio", "running": True}}

    def mcp_tools(self, server):
        if self.fail_tools or server != "unity":
            raise RuntimeError("server is not configured")
        return [{"name": "find_assets", "description": "d",
                 "params": [{"name": "query", "type": "string", "required": True, "description": "",
                             "enum": None, "default": None}]}]

    def skills(self):
        return {"docs": "How to write docs", "review": "Review checklist"}

    def skill_headings(self, name):
        return [{"level": 1, "title": "Docs"}, {"level": 2, "title": "Setup"}] if name == "docs" else []

    def plugins(self):
        return {"my_plugin": {"commands": ["hello"], "hooks": ["on_start"]}}

    def leeth_available(self):
        return True

    # what the runner would call
    def mcp(self, node, state):
        return "found: asset_1\nasset_2"

    def skill(self, node, state):
        return "# Docs\nbody"

    def plugin(self, node, state):
        return "hello!"

    def leeth(self, node, state):
        return "{}"


def _room(name="my_room", **extra):
    data = {
        "room": name, "description": "Find things",
        "budget": {"max_iterations": 20, "max_tokens": 10000},
        "nodes": [{"id": "find", "type": "mcp", "server": "unity", "tool": "find_assets",
                   "args": {"query": "tree"}, "sets": {"assets": "lines"}},
                  {"id": "show", "type": "notify", "message": "got {{assets}}"}],
        "edges": [{"from": "find", "to": "show"}, {"from": "show", "to": "EXIT:done"}],
    }
    data.update(extra)
    return data


def _tool_catalog():
    return [{"name": "run_command", "description": "Run", "group": "shell", "mutating": False,
             "params": [{"name": "command", "type": "string", "required": True, "description": ""}],
             "not_in_rooms": False}]


@pytest.fixture
def make_server(tmp_path):
    made = []

    def make(**kw):
        udir = tmp_path / "rooms"

        def factory():
            lib = RoomLibrary(user_dir=udir, stats_path=tmp_path / "stats.json")
            lib.load_dir(default_starter_dir(), TOOLS, origin="starter")
            return lib

        fake = kw.pop("integrations", None) or FakeCatalog()

        def runner_factory(human, confirm):
            return NodeRunner(integrations=fake, human_prompt=human, confirm=confirm,
                              tool_executor=kw.pop("tool_executor", None) or (lambda n, s: "tool ran"))

        srv = EditorServer(factory, lambda: TOOLS, udir, integrations=fake, runner_factory=runner_factory,
                           tool_catalog=_tool_catalog, **kw).start()
        made.append(srv)
        return srv

    yield make
    for s in made:
        s.stop()


def _req(srv, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    h = {"Host": f"127.0.0.1:{srv.port}", "X-Rooms-Token": srv.token}
    payload = None
    if body is not None:
        payload = json.dumps(body)
        h["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=h)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, json.loads(raw)


def _wait(srv, run_id, states=("done", "error", "stopped"), timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        status, snap = _req(srv, "GET", f"/api/run?id={run_id}")
        assert status == 200
        if snap["status"] in states:
            return snap
        time.sleep(0.03)
    raise AssertionError(f"run did not reach {states}: {snap['status']}")


# ── catalogs ─────────────────────────────────────────────────────────────────────

def test_state_has_every_node_type_and_tool_info(make_server):
    srv = make_server()
    status, state = _req(srv, "GET", "/api/state")
    assert status == 200
    for t in ("mcp", "skill", "plugin", "leeth", "foreach", "wait"):
        assert t in state["node_types"]
    assert state["tool_info"][0]["params"][0]["name"] == "command"
    assert "shell" in state["tool_groups"] and "rooms" in state["not_in_rooms"]
    assert state["leeth"]["available"] is True and "filter" in state["leeth"]["modes"]


def test_mcp_servers_and_tools(make_server):
    srv = make_server()
    assert _req(srv, "GET", "/api/mcp")[1]["servers"]["unity"]["running"] is True
    status, body = _req(srv, "GET", "/api/mcp?server=unity")
    assert status == 200 and body["tools"][0]["name"] == "find_assets"


def test_mcp_tools_failure_is_a_clean_error(make_server):
    fake = FakeCatalog()
    fake.fail_tools = True
    srv = make_server(integrations=fake)
    status, body = _req(srv, "GET", "/api/mcp?server=unity")
    assert status == 502 and "not configured" in body["error"]


def test_skills_headings_and_plugins(make_server):
    srv = make_server()
    skills = _req(srv, "GET", "/api/skills")[1]["skills"]
    assert [s["name"] for s in skills] == ["docs", "review"]
    assert _req(srv, "GET", "/api/skills?name=docs")[1]["headings"][1]["title"] == "Setup"
    assert _req(srv, "GET", "/api/plugins")[1]["plugins"]["my_plugin"]["commands"] == ["hello"]


def test_validate_returns_warnings_for_unknown_names(make_server):
    srv = make_server()
    room = _room(nodes=[{"id": "a", "type": "mcp", "server": "ghost", "tool": "t"},
                        {"id": "b", "type": "skill", "skill": "nothing"}],
                 edges=[{"from": "a", "to": "b"}, {"from": "b", "to": "EXIT:done"}])
    status, body = _req(srv, "POST", "/api/validate", {"room": room})
    assert status == 200 and body["ok"] is True
    text = " ".join(body["warnings"])
    assert "ghost" in text and "nothing" in text


def test_oversized_body_is_refused_cleanly(make_server):
    srv = make_server()
    big = {"room": _room(description="x" * (editor_server.MAX_BODY + 10))}
    for _ in range(5):                       # a reset instead of an answer would show up here
        assert _req(srv, "POST", "/api/validate", big)[0] == 413


# ── runs ─────────────────────────────────────────────────────────────────────────

def test_run_a_room_and_read_its_steps(make_server):
    srv = make_server()
    status, body = _req(srv, "POST", "/api/run", {"room": _room(), "params": {"task": "go"}})
    assert status == 200 and body["ok"]
    snap = _wait(srv, body["run"])
    assert snap["status"] == "done" and snap["outcome"]["exit"] == "done"
    kinds = [e["type"] for e in snap["events"]]
    assert kinds.count("node_start") == 2 and kinds.count("node_end") == 2 and "room_end" in kinds
    result = next(e for e in snap["events"] if e["type"] == "node_result" and e["node"] == "find")
    assert "asset_1" in result["output"] and "assets" in result["changed"]
    assert snap["outcome"]["state"]["task"] == "go"
    assert any(r["id"] == body["run"] for r in _req(srv, "GET", "/api/runs")[1]["runs"])


def test_invalid_room_is_not_run(make_server):
    srv = make_server()
    room = _room(nodes=[{"id": "a", "type": "tool", "tool": "ghost_tool"}],
                 edges=[{"from": "a", "to": "EXIT:done"}])
    status, body = _req(srv, "POST", "/api/run", {"room": room})
    assert status == 422 and not body["ok"]


def test_since_returns_only_new_events(make_server):
    srv = make_server()
    run = _req(srv, "POST", "/api/run", {"room": _room()})[1]["run"]
    snap = _wait(srv, run)
    again = _req(srv, "GET", f"/api/run?id={run}&since={snap['next']}")[1]
    assert again["events"] == [] and again["next"] == snap["next"]


def test_human_pause_asks_the_page(make_server):
    srv = make_server()
    room = _room(
        nodes=[{"id": "ask", "type": "human_pause", "prompt": "Which topic?", "into": "topic"},
               {"id": "show", "type": "notify", "message": "topic is {{topic}}"}],
        edges=[{"from": "ask", "to": "show"}, {"from": "show", "to": "EXIT:done"}])
    run = _req(srv, "POST", "/api/run", {"room": room})[1]["run"]
    snap = _wait(srv, run, states=("waiting",))
    assert snap["prompt"]["kind"] == "ask" and snap["prompt"]["text"] == "Which topic?"
    status, body = _req(srv, "POST", "/api/answer", {"run": run, "prompt": snap["prompt"]["id"], "answer": "cats"})
    assert status == 200 and body["ok"]
    done = _wait(srv, run)
    assert done["status"] == "done" and done["outcome"]["state"]["topic"] == "cats"


def test_stale_answer_is_refused(make_server):
    srv = make_server()
    status, body = _req(srv, "POST", "/api/answer", {"run": "nope", "prompt": 1, "answer": "x"})
    assert status == 409 and not body["ok"]


def test_confirm_declined_means_no(make_server):
    srv = make_server()
    room = _room(nodes=[{"id": "find", "type": "mcp", "server": "unity", "tool": "find_assets",
                         "confirm": True, "sets": {"assets": "lines"}}],
                 edges=[{"from": "find", "to": "EXIT:done"}])
    run = _req(srv, "POST", "/api/run", {"room": room})[1]["run"]
    snap = _wait(srv, run, states=("waiting",))
    assert snap["prompt"]["kind"] == "confirm" and "unity.find_assets" in snap["prompt"]["text"]
    _req(srv, "POST", "/api/answer", {"run": run, "prompt": snap["prompt"]["id"], "answer": False})
    done = _wait(srv, run)
    assert done["outcome"]["state"]["find.declined"] == "True"
    assert done["outcome"]["state"]["assets"] == "[]"


def test_stop_ends_a_waiting_run(make_server):
    srv = make_server()
    room = _room(nodes=[{"id": "ask", "type": "human_pause", "prompt": "?"}],
                 edges=[{"from": "ask", "to": "EXIT:done"}])
    run = _req(srv, "POST", "/api/run", {"room": room})[1]["run"]
    _wait(srv, run, states=("waiting",))
    assert _req(srv, "POST", "/api/stop", {"run": run})[1]["ok"] is True
    assert _wait(srv, run)["status"] == "stopped"


def test_error_in_a_step_is_reported_not_raised(make_server):
    class Boom(FakeCatalog):
        def mcp(self, node, state):
            raise RuntimeError("server died")
    srv = make_server(integrations=Boom())
    run = _req(srv, "POST", "/api/run", {"room": _room()})[1]["run"]
    snap = _wait(srv, run)
    assert snap["status"] == "error" and "server died" in snap["error"]


# ── one step ─────────────────────────────────────────────────────────────────────

def test_run_one_reading_step_without_asking(make_server):
    srv = make_server()
    room = _room(nodes=[{"id": "s", "type": "skill", "skill": "docs"}],
                 edges=[{"from": "s", "to": "EXIT:done"}])
    status, body = _req(srv, "POST", "/api/run_node", {"room": room, "node": "s"})
    assert status == 200 and body["asks"] is False
    snap = _wait(srv, body["run"])
    assert snap["status"] == "done" and snap["kind"] == "node"
    assert snap["outcome"]["state"]["s.output"].startswith("# Docs")


def test_run_one_acting_step_asks_first(make_server):
    srv = make_server()
    status, body = _req(srv, "POST", "/api/run_node", {"room": _room(), "node": "find"})
    assert status == 200 and body["asks"] is True
    snap = _wait(srv, body["run"], states=("waiting",))
    assert snap["prompt"]["kind"] == "confirm"
    _req(srv, "POST", "/api/answer", {"run": body["run"], "prompt": snap["prompt"]["id"], "answer": True})
    done = _wait(srv, body["run"])
    assert done["status"] == "done" and "asset_1" in done["outcome"]["state"]["find.output"]


def test_declined_single_step_does_not_run(make_server):
    srv = make_server()
    body = _req(srv, "POST", "/api/run_node", {"room": _room(), "node": "find"})[1]
    snap = _wait(srv, body["run"], states=("waiting",))
    _req(srv, "POST", "/api/answer", {"run": body["run"], "prompt": snap["prompt"]["id"], "answer": False})
    done = _wait(srv, body["run"])
    assert done["outcome"]["exit"] == "declined" and "find.output" not in done["outcome"]["state"]


def test_only_in_a_run_steps_are_refused(make_server):
    srv = make_server()
    room = _room(nodes=[{"id": "ask", "type": "human_pause"}], edges=[{"from": "ask", "to": "EXIT:done"}])
    status, body = _req(srv, "POST", "/api/run_node", {"room": room, "node": "ask"})
    assert status == 422
    status, _ = _req(srv, "POST", "/api/run_node", {"room": _room(), "node": "missing"})
    assert status == 404


# ── the catalog of Argent's tools ────────────────────────────────────────────────

def test_tool_catalog_describes_every_tool():
    cat = tool_catalog()
    names = {e["name"] for e in cat}
    from tools.schemas import AVAILABLE_TOOLS
    assert names == set(AVAILABLE_TOOLS)
    by = {e["name"]: e for e in cat}
    assert by["leeth_map"]["group"] == "leeth"
    assert any(p["name"] == "items" and p["required"] for p in by["leeth_map"]["params"])
    assert by["write_file"]["mutating"] is True and by["read_file"]["mutating"] is False
    assert by["rooms"]["not_in_rooms"] is True
    assert group_of("browser_open") == "web" and group_of("something_new") == "other"
    assert all(e["group"] != "other" for e in cat), [e["name"] for e in cat if e["group"] == "other"]


def test_state_has_the_vocabulary_of_the_controls(make_server):
    srv = make_server()
    state = _req(srv, "GET", "/api/state")[1]
    v = state["vocab"]
    assert "do_while" in v["loop_modes"] and "hibernate" in v["power_modes"] and v["power_min_delay"] == 10
    assert v["sampling"]["top_p"] == [0.0, 1.0] and "xhigh" in v["reasoning_efforts"]
    for t in ("loop", "break", "continue", "power", "hint", "forget", "assert"):
        assert t in state["node_types"]
    assert "spawn_room" not in state["node_types"]


def test_saving_refuses_a_step_the_rules_forbid(make_server):
    srv = make_server()
    room = _room(rules={"read_only": True},
                 nodes=[{"id": "w", "type": "tool", "tool": "write_file", "args": {"path": "a", "content": "b"}}],
                 edges=[{"from": "w", "to": "EXIT:done"}])
    body = _req(srv, "POST", "/api/validate", {"room": room})[1]
    assert body["ok"] is False and any("read-only" in e for e in body["errors"])


def test_a_loop_room_runs_from_the_page(make_server):
    srv = make_server()
    room = _room(
        nodes=[{"id": "loop", "type": "loop", "mode": "times", "stop": 3},
               {"id": "inc", "type": "set", "values": {"n": "= state.n + 1"}}],
        edges=[{"from": "loop", "if": "not state.loop_done", "to": "inc"}, {"from": "loop", "else": True, "to": "EXIT:done"},
               {"from": "inc", "to": "loop"}])
    run = _req(srv, "POST", "/api/run", {"room": room, "params": {"n": 0}})[1]["run"]
    snap = _wait(srv, run)
    assert snap["status"] == "done" and snap["outcome"]["state"]["n"] == "3"


def test_a_notification_step_asks_before_it_goes_out(make_server):
    srv = make_server()
    room = _room(nodes=[{"id": "n", "type": "notify", "message": "hi", "channel": "telegram"}],
                 edges=[{"from": "n", "to": "EXIT:done"}])
    body = _req(srv, "POST", "/api/run_node", {"room": room, "node": "n"})[1]
    assert body["asks"] is True
    _wait(srv, body["run"], states=("waiting",))
    _req(srv, "POST", "/api/stop", {"run": body["run"]})
    _wait(srv, body["run"])
