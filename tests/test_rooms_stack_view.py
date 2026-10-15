"""Rooms editor: the «Стек» view lays a graph out as a column of blocks.

The layout itself (`buildStack` in stack.js) is a pure function, so it is run under node when node is
installed (skipped otherwise); the page wiring is checked as text.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

EDITOR = Path(__file__).resolve().parent.parent / "src" / "rooms" / "editor"
NODE = shutil.which("node")

RUNNER = r"""
const {buildStack} = require(process.argv[1]);
const rooms = JSON.parse(require("fs").readFileSync(0, "utf8"));
const flat = (items, out) => {
  for (const it of items) {
    out.push(it.kind + (it.id ? ":" + it.id : "") + (it.name ? ":" + it.name : ""));
    if (it.head) out.push("head:" + it.head.id);
    if (it.body) flat(it.body, out);
    if (it.arms) for (const a of it.arms) { out.push("arm:" + a.kind); flat(a.items, out); }
  }
  return out;
};
const result = {};
for (const [name, room] of Object.entries(rooms)) {
  const t = buildStack(room);
  result[name] = {lines: flat(t.items, []), loose: t.loose};
}
console.log(JSON.stringify(result));
"""


def layout(rooms: dict) -> dict:
    done = subprocess.run([NODE, "-e", RUNNER, str(EDITOR / "stack.js")], input=json.dumps(rooms),
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def room(entry, nodes, edges):
    return {"entry": entry, "nodes": [{"id": i, "type": t} for i, t in nodes], "edges": edges}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
class TestLayout:
    def test_a_chain_is_a_column_ending_in_an_exit(self):
        r = layout({"r": room("a", [("a", "agent"), ("b", "tool")],
                              [{"from": "a", "to": "b"}, {"from": "b", "to": "EXIT:done"}])})["r"]
        assert r["lines"] == ["block:a", "block:b", "exit:done"] and r["loose"] == []

    def test_a_branch_that_meets_again_shows_the_meeting_step_once_below(self):
        r = layout({"r": room("c", [("c", "condition"), ("x", "tool"), ("y", "tool"), ("m", "notify")],
                              [{"from": "c", "if": "state.ok", "to": "x"}, {"from": "c", "else": True, "to": "y"},
                               {"from": "x", "to": "m"}, {"from": "y", "to": "m"}, {"from": "m", "to": "EXIT:done"}])})["r"]
        assert r["lines"] == ["block:c", "branch:c", "arm:if", "block:x", "arm:else", "block:y", "block:m", "exit:done"]

    def test_arms_that_end_apart_each_run_to_their_own_exit(self):
        r = layout({"r": room("c", [("c", "condition"), ("x", "tool")],
                              [{"from": "c", "if": "state.ok", "to": "x"}, {"from": "c", "else": True, "to": "EXIT:failed"},
                               {"from": "x", "to": "EXIT:done"}])})["r"]
        assert r["lines"] == ["block:c", "branch:c", "arm:if", "block:x", "exit:done", "arm:else", "exit:failed"]

    def test_a_cycle_becomes_a_repeat_with_the_way_out_below(self):
        r = layout({"r": room("w", [("w", "agent"), ("t", "tool")],
                              [{"from": "w", "to": "t"}, {"from": "t", "if": "state.ok", "to": "EXIT:done"},
                               {"from": "t", "else": True, "to": "w"}])})["r"]
        assert r["lines"][0] == "repeat:w" and "head:w" in r["lines"]
        assert r["lines"].count("repeat:t") == 0                  # one cycle is one «repeat», not one per step
        assert "block:t" in r["lines"] and "again:w" in r["lines"] and r["loose"] == []

    def test_parallel_arms_are_branches_and_the_join_follows(self):
        r = layout({"r": room("p", [("p", "parallel"), ("a", "tool"), ("b", "tool"), ("j", "join")],
                              [{"from": "p", "to": "a"}, {"from": "p", "to": "b"}, {"from": "a", "to": "j"},
                               {"from": "b", "to": "j"}, {"from": "j", "to": "EXIT:done"}])})["r"]
        rooms_lines = r["lines"]
        assert rooms_lines[:2] == ["block:p", "branch:p"] and rooms_lines[-2:] == ["block:j", "exit:done"]

    def test_a_step_nobody_reaches_is_listed_not_lost(self):
        r = layout({"r": room("a", [("a", "agent"), ("orphan", "tool")], [{"from": "a", "to": "EXIT:done"}])})["r"]
        assert r["loose"] == ["orphan"]

    def test_a_jump_back_to_a_shown_step_is_a_marker_not_a_copy(self):
        r = layout({"r": room("a", [("a", "agent"), ("b", "tool"), ("c", "tool")],
                              [{"from": "a", "if": "state.x", "to": "b"}, {"from": "a", "else": True, "to": "c"},
                               {"from": "b", "to": "EXIT:done"}, {"from": "c", "to": "b"}])})["r"]
        assert r["lines"].count("block:b") == 1

    def test_every_starter_room_and_example_is_laid_out_completely(self):
        from src.rooms import library
        rooms = {}
        for ex in json.loads((EDITOR / "examples.json").read_text(encoding="utf-8")):
            if isinstance(ex.get("room"), dict) and "nodes" in ex["room"]:
                rooms["ex_" + ex["name"]] = ex["room"]
        for p in (Path(library.__file__).parent / "starter").glob("*.json"):
            body = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(body, dict) and "nodes" in body:
                rooms["starter_" + p.stem] = body
        assert len(rooms) >= 10, "expected the starter rooms and the examples"
        for name, result in layout(rooms).items():
            shown = {line.split(":")[1] for line in result["lines"] if line.split(":")[0] in ("block", "head")}
            ids = {n["id"] for n in rooms[name]["nodes"]}
            lost = ids - shown - set(result["loose"])
            assert not lost, f"{name}: {sorted(lost)} are neither shown nor listed"


class TestPage:
    def test_the_tab_the_script_and_the_insertion_point_are_wired(self):
        html = (EDITOR / "index.html").read_text(encoding="utf-8")
        app = (EDITOR / "app.js").read_text(encoding="utf-8")
        assert 'id="tabStack"' in html and 'id="paneStack"' in html and "/stack.js?t=" in html
        assert html.index("/stack.js?t=") < html.index("/app.js?t=")
        assert 'view === "stack"' in app and "takePendingInsert()" in app
        from src.rooms import editor_server
        assert editor_server._STATIC["/stack.js"][0] == "stack.js"

    def test_the_view_keeps_nothing_of_its_own(self):
        js = (EDITOR / "stack.js").read_text(encoding="utf-8")
        pure = js[js.index("function buildStack"):js.index("if (typeof module")]
        assert "localStorage" not in js and "layout" not in pure and "document" not in pure
