"""Rooms: nodes that reach outside a run (MCP, skills, plugins, Leeth), loops over a list,
pauses, switched-off steps, the derived values and the ignore rules.

Nothing here starts a real MCP server, reads real skills or loads Leeth: the runner takes
an injected `integrations` object and the tools are replaced with fakes.
"""

import json

import pytest

from src.rooms import policy
from src.rooms.diagram import render_flow, render_mermaid
from src.rooms.engine import Engine, replayable
from src.rooms.integrations import Integrations, as_list, schema_params
from src.rooms.library import describe_room
from src.rooms.models import Node, Room, State
from src.rooms.runner import NodeRunner, derive_value, derive_result_values
from src.rooms.validator import room_warnings, validate_room


class FakeIntegrations:
    def __init__(self):
        self.calls = []

    def mcp(self, node, state):
        self.calls.append(("mcp", node.server, node.tool, dict(node.args)))
        return "Exit code: 0\nOUTPUT:\nasset_1\nasset_2\n"

    def skill(self, node, state):
        self.calls.append(("skill", node.skill))
        return "# Skill\nbody"

    def plugin(self, node, state):
        self.calls.append(("plugin", node.command, dict(node.args)))
        return "plugin said hi"

    def leeth(self, node, state):
        self.calls.append(("leeth", node.mode, node.instruction))
        return json.dumps({"yes": [0, 2], "no": [1]})


def _room(nodes, edges, **kw):
    return Room(room="t", budget={"max_iterations": 60, "max_tokens": 1000}, nodes=nodes, edges=edges, **kw)


# ── models ───────────────────────────────────────────────────────────────────────

class TestModels:
    def test_mcp_node(self):
        n = Node(id="a", type="mcp", server="unity", tool="find_assets", args={"q": "x"})
        assert n.server == "unity" and n.tool == "find_assets"

    def test_mcp_call_needs_tool(self):
        with pytest.raises(ValueError):
            Node(id="a", type="mcp", server="unity")

    def test_mcp_list_needs_no_tool(self):
        assert Node(id="a", type="mcp", server="unity", mode="list").mode == "list"

    def test_skill_needs_name(self):
        with pytest.raises(ValueError):
            Node(id="a", type="skill")

    def test_leeth_filter_needs_a_list(self):
        with pytest.raises(ValueError):
            Node(id="a", type="leeth", mode="filter", instruction="about animals?")

    def test_foreach_needs_list(self):
        with pytest.raises(ValueError):
            Node(id="a", type="foreach")

    def test_bad_ignore_rule_is_rejected(self):
        with pytest.raises(ValueError):
            Node(id="a", type="skill", skill="x", ignore=["nonsense"])

    def test_disabled_and_note_on_any_node(self):
        n = Node(id="a", type="tool", tool="run_command", disabled=True, note="later")
        assert n.disabled and n.note == "later"


# ── derived values ───────────────────────────────────────────────────────────────

class TestDerive:
    def v(self, text):
        return derive_result_values(text)

    def test_lines_count_first(self):
        vals = self.v("a\n\nb\nc")
        assert derive_value("lines", vals) == ["a", "b", "c"]
        assert derive_value("count", vals) == 3
        assert derive_value("first_line", vals) == "a"

    def test_numbers(self):
        vals = self.v("found 12 items, 3.5 s")
        assert derive_value("int", vals) == 12
        assert derive_value("float", vals) == 12.0

    def test_bool(self):
        assert derive_value("bool", self.v("Да")) is True
        assert derive_value("bool", self.v("no")) is False

    def test_json_and_path(self):
        vals = self.v('result: {"a": {"b": [10, 20]}}')
        assert derive_value("json", vals) == {"a": {"b": [10, 20]}}
        assert derive_value("json:a.b.1", vals) == 20
        assert derive_value("json:a.zzz", vals) is None

    def test_regex(self):
        vals = self.v("id=42; name=bob")
        assert derive_value(r"regex:id=(\d+)", vals) == "42"
        assert derive_value("regex:nothing", vals) == ""

    def test_unknown_spec(self):
        assert derive_value("wat", self.v("x")) is None


# ── policy ───────────────────────────────────────────────────────────────────────

SKILL = """# Title
intro
## Setup
setup text
### Deep
deep text
## Usage
usage text
```
## not a heading
```
"""


class TestPolicy:
    def test_parse(self):
        assert policy.parse_rule("mcp:unity").sub is None
        r = policy.parse_rule("mcp:unity.delete_*")
        assert (r.name, r.sub) == ("unity", "delete_*")
        r = policy.parse_rule("skill:docs#Setup")
        assert (r.name, r.sub) == ("docs", "Setup")
        with pytest.raises(policy.RuleError):
            policy.parse_rule("nope")

    def test_mcp_rules(self):
        p = policy.Policy(["mcp:unity.delete_*", "mcp:slack"])
        assert not p.mcp_server_allowed("slack")
        assert p.mcp_server_allowed("unity")
        assert not p.mcp_tool_allowed("unity", "delete_asset")
        assert p.mcp_tool_allowed("unity", "find_asset")

    def test_server_whitelist(self):
        p = policy.Policy([], mcp_servers=["unity"])
        assert p.mcp_server_allowed("unity") and not p.mcp_server_allowed("slack")

    def test_cut_section_with_children(self):
        out = policy.Policy(["skill:docs#Setup"]).cut_skill("docs", SKILL)
        assert "setup text" not in out and "deep text" not in out
        assert "usage text" in out and "intro" in out

    def test_heading_inside_fence_is_not_a_heading(self):
        out = policy.Policy(["skill:docs#Usage"]).cut_skill("docs", SKILL)
        assert "usage text" not in out and "setup text" in out
        # the fenced block belongs to the Usage section and goes with it
        assert "not a heading" not in out

    def test_skill_ignored_whole(self):
        p = policy.Policy(["skill:docs"])
        assert not p.skill_allowed("docs") and p.skill_allowed("other")

    def test_active_policy_is_enforced_and_removed(self):
        assert policy.mcp_tool_blocked("slack", "post") is None
        with policy.use(policy.Policy(["mcp:slack"])):
            assert "ignored" in policy.mcp_tool_blocked("slack", "post")
        assert policy.mcp_tool_blocked("slack", "post") is None

    def test_nested_policies_add_up(self):
        with policy.use(policy.Policy(["mcp:a"])):
            with policy.use(policy.Policy(["mcp:b"])):
                assert policy.mcp_tool_blocked("a", "x") and policy.mcp_tool_blocked("b", "x")
            assert policy.mcp_tool_blocked("b", "x") is None

    def test_validate_rules(self):
        assert policy.validate_rules(["mcp:x", "skill:y#z"]) == []
        assert len(policy.validate_rules(["mcp:", "oops"])) == 2


class TestEnforcedInTools:
    def test_call_mcp_tool_blocked(self, monkeypatch):
        from tools.misc_tools import call_mcp_tool
        with policy.use(policy.Policy(["mcp:unity.delete_*"])):
            out = call_mcp_tool("unity", "delete_asset", "{}")
        assert out.startswith("Error:") and "ignored" in out

    def test_read_skill_cut(self, monkeypatch):
        import skill_manager as sm
        from tools import skill_tools
        monkeypatch.setattr(sm.skill_manager, "read_skill", lambda name: SKILL)
        with policy.use(policy.Policy(["skill:docs#Setup"])):
            out = skill_tools.read_skill("docs")
        assert "setup text" not in out and "usage text" in out


# ── runner ───────────────────────────────────────────────────────────────────────

class TestRunnerBranches:
    def test_mcp_sets(self):
        fake = FakeIntegrations()
        node = Node(id="m", type="mcp", server="unity", tool="find", args={"q": "x"},
                    sets={"assets": "lines", "n": "count"})
        runner = NodeRunner(integrations=fake)
        st = State()
        runner.run_node(node, st)
        assert st.data["assets"] == ["asset_1", "asset_2"] and st.data["n"] == 2
        assert fake.calls == [("mcp", "unity", "find", {"q": "x"})]

    def test_args_are_templated(self):
        fake = FakeIntegrations()
        node = Node(id="m", type="mcp", server="u", tool="t", args={"q": "{{topic}}"})
        NodeRunner(integrations=fake).run_node(node, State(data={"topic": "cats"}))
        assert fake.calls[0][3] == {"q": "cats"}

    def test_confirm_declined(self):
        fake = FakeIntegrations()
        node = Node(id="m", type="mcp", server="u", tool="t", confirm=True, sets={"ok": "ok"})
        runner = NodeRunner(integrations=fake, confirm=lambda n, s: False)
        st = State()
        runner.run_node(node, st)
        assert fake.calls == [] and st.data["m.declined"] is True and st.data["ok"] is False

    def test_confirm_without_asker_declines(self):
        fake = FakeIntegrations()
        node = Node(id="p", type="plugin", command="x", confirm=True)
        NodeRunner(integrations=fake).run_node(node, State())
        assert fake.calls == []

    def test_confirm_accepted(self):
        fake = FakeIntegrations()
        node = Node(id="p", type="plugin", command="x", args={"a": 1}, confirm=True)
        runner = NodeRunner(integrations=fake, confirm=lambda n, s: True)
        st = State()
        runner.run_node(node, st)
        assert st.data["p.output"] == "plugin said hi"

    def test_skill_output(self):
        fake = FakeIntegrations()
        st = State()
        NodeRunner(integrations=fake).run_node(Node(id="s", type="skill", skill="docs"), st)
        assert st.data["s.output"].startswith("# Skill")

    def test_leeth_json_sets(self):
        fake = FakeIntegrations()
        node = Node(id="l", type="leeth", mode="filter", input_from="posts",
                    instruction="about animals?", sets={"yes": "json:yes"})
        st = State(data={"posts": ["a", "b", "c"]})
        NodeRunner(integrations=fake).run_node(node, st)
        assert st.data["yes"] == [0, 2]

    def test_no_integrations_is_an_error(self):
        with pytest.raises(RuntimeError):
            NodeRunner().run_node(Node(id="s", type="skill", skill="x"), State())

    def test_disabled_is_skipped_and_empties_sets(self):
        fake = FakeIntegrations()
        node = Node(id="m", type="mcp", server="u", tool="t", disabled=True,
                    sets={"rows": "lines", "n": "count", "ok": "ok"})
        st = State()
        NodeRunner(integrations=fake).run_node(node, st)
        assert fake.calls == []
        assert st.data["rows"] == [] and st.data["n"] == 0 and st.data["ok"] is False
        assert st.data["m.skipped"] is True

    def test_wait_uses_injected_sleep(self):
        slept = []
        NodeRunner(sleep=slept.append).run_node(Node(id="w", type="wait", seconds=2.5), State())
        assert slept == [2.5]


class TestForeachInRoom:
    def _room(self, **kw):
        return _room(
            [{"id": "each", "type": "foreach", "input_from": "items", "item": "it", **kw},
             {"id": "work", "type": "tool", "tool": "noop"},
             {"id": "after", "type": "set", "values": {"fin": "yes"}}],
            [{"from": "each", "if": "not state.each_done", "to": "work"},
             {"from": "each", "else": True, "to": "after"},
             {"from": "work", "to": "each"},
             {"from": "after", "to": "EXIT:done"}])

    def test_walks_the_list(self):
        seen = []
        runner = NodeRunner(tool_executor=lambda n, s: seen.append(s.data["it"]) or "ok")
        out = Engine().run(self._room(), State(data={"items": ["a", "b", "c"]}), runner.run_node)
        assert out.exit == "done" and seen == ["a", "b", "c"]
        assert out.state.data["each_count"] == 3 and out.state.data["fin"] == "yes"

    def test_max_items(self):
        seen = []
        runner = NodeRunner(tool_executor=lambda n, s: seen.append(s.data["it"]) or "ok")
        Engine().run(self._room(max_items=2), State(data={"items": ["a", "b", "c"]}), runner.run_node)
        assert seen == ["a", "b"]

    def test_empty_list_goes_straight_on(self):
        runner = NodeRunner(tool_executor=lambda n, s: "ok")
        out = Engine().run(self._room(), State(data={"items": []}), runner.run_node)
        assert out.exit == "done" and out.state.data["each_done"] is True

    def test_json_text_list(self):
        seen = []
        runner = NodeRunner(tool_executor=lambda n, s: seen.append(s.data["it"]) or "ok")
        Engine().run(self._room(), State(data={"items": '["x", "y"]'}), runner.run_node)
        assert seen == ["x", "y"]

    def test_item_fields_in_templates(self):
        node = Node(id="n", type="notify", message="{{it.title}} / {{rows.1}}")
        out = NodeRunner._materialize(node, State(data={"it": {"title": "T"}, "rows": ["a", "b"]}))
        assert out.message == "T / b"


class TestEngineBits:
    def test_replayable(self):
        assert replayable(Node(id="a", type="skill", skill="x"))
        assert replayable(Node(id="a", type="wait", seconds=1))
        assert replayable(Node(id="a", type="mcp", server="s", tool="t", disabled=True))
        assert not replayable(Node(id="a", type="mcp", server="s", tool="t"))
        assert replayable(Node(id="a", type="mcp", server="s", tool="t", idempotent=True))
        assert not replayable(Node(id="a", type="leeth", mode="delegate", instruction="x"))


# ── integrations (production class, with fakes behind it) ────────────────────────

class TestIntegrations:
    def test_as_list(self):
        assert as_list(None) == [] and as_list("") == []
        assert as_list("a\n\nb") == ["a", "b"]
        assert as_list('["a", 1]') == ["a", 1]
        assert as_list({"a": 1}) == [{"a": 1}]

    def test_schema_params(self):
        params = schema_params({"properties": {"q": {"type": "string", "description": "d"},
                                               "n": {"type": "integer", "default": 3}},
                                "required": ["q"]})
        assert [p["name"] for p in params] == ["q", "n"]
        assert params[0]["required"] and not params[1]["required"]

    def test_mcp_blocked_by_node_ignore(self, monkeypatch):
        node = Node(id="m", type="mcp", server="unity", tool="delete_asset", ignore=["mcp:unity.delete_*"])
        out = Integrations().mcp(node, State())
        assert out.startswith("Error:") and "ignored" in out

    def test_mcp_unconfigured_server(self, monkeypatch):
        monkeypatch.setattr(Integrations, "_configured_servers", staticmethod(lambda: {}))
        out = Integrations().mcp(Node(id="m", type="mcp", server="ghost", tool="t"), State())
        assert out.startswith("Error:") and "ghost" in out

    def test_mcp_call_passes_through(self, monkeypatch):
        import mcp_client as mc
        import tools.schemas as schemas
        monkeypatch.setattr(Integrations, "ensure_server", lambda self, name: "")
        monkeypatch.setattr(Integrations, "_required_params", staticmethod(lambda s, t: ["query"]))
        got = {}
        monkeypatch.setattr(schemas, "AVAILABLE_TOOLS", {
            "call_mcp_tool": lambda server, tool, arguments: got.update(s=server, t=tool, a=arguments) or "done"})
        node = Node(id="m", type="mcp", server="u", tool="find", input_from="q", args={"limit": 2})
        out = Integrations().mcp(node, State(data={"q": "cats"}))
        assert out == "done" and got["s"] == "u" and got["t"] == "find"
        assert json.loads(got["a"]) == {"limit": 2, "query": "cats"}

    def test_skill_cut_and_missing(self, monkeypatch):
        import skill_manager as sm
        monkeypatch.setattr(sm.skill_manager, "read_skill", lambda n: SKILL if n == "docs" else "")
        node = Node(id="s", type="skill", skill="docs", ignore=["skill:docs#Setup"])
        out = Integrations().skill(node, State())
        assert "setup text" not in out and "usage text" in out
        assert Integrations().skill(Node(id="s", type="skill", skill="zz"), State()).startswith("Error:")
        blocked = Node(id="s", type="skill", skill="docs", ignore=["skill:docs"])
        assert Integrations().skill(blocked, State()).startswith("Error:")

    def test_plugin_command_output_and_return(self, monkeypatch):
        import hook_manager as hm

        def command_hi(*words):
            print("printed", *words)
            return "returned"
        monkeypatch.setattr(hm.hook_manager, "get_custom_commands", lambda: {"hi": command_hi})
        out = Integrations().plugin(Node(id="p", type="plugin", command="hi", args={"a": "x"}), State())
        assert "printed x" in out and "returned" in out
        assert Integrations().plugin(Node(id="p", type="plugin", command="nope"), State()).startswith("Error:")

    def test_leeth_filter_uses_map_tool(self, monkeypatch):
        import tools.schemas as schemas
        got = {}
        monkeypatch.setattr(schemas, "AVAILABLE_TOOLS", {"leeth_map": lambda **kw: got.update(kw) or "{}"})
        node = Node(id="l", type="leeth", mode="filter", input_from="posts", instruction="about animals?")
        Integrations().leeth(node, State(data={"posts": ["a", "b"]}))
        assert got["mode"] == "yesno" and got["criterion"] == "about animals?"
        assert json.loads(got["items"]) == ["a", "b"]

    def test_leeth_empty_list(self):
        node = Node(id="l", type="leeth", mode="filter", input_from="posts", instruction="x")
        assert Integrations().leeth(node, State(data={"posts": []})).startswith("Error:")

    def test_leeth_delegate(self, monkeypatch):
        import tools.schemas as schemas
        monkeypatch.setattr(schemas, "AVAILABLE_TOOLS", {"delegate_to_leeth": lambda task: f"did {task}"})
        node = Node(id="l", type="leeth", instruction="sum up")
        assert Integrations().leeth(node, State()) == "did sum up"


# ── validator, diagram, library ──────────────────────────────────────────────────

class FakeCatalog:
    def mcp_servers(self):
        return {"unity": {"type": "stdio", "running": False}}

    def skills(self):
        return {"docs": "d"}

    def plugins(self):
        return {"p": {"commands": ["hi"], "hooks": []}}

    def leeth_available(self):
        return False


def _all_types_room():
    return _room(
        [{"id": "m", "type": "mcp", "server": "ghost", "tool": "t"},
         {"id": "s", "type": "skill", "skill": "missing"},
         {"id": "p", "type": "plugin", "command": "nope"},
         {"id": "l", "type": "leeth", "mode": "delegate", "instruction": "x"},
         {"id": "f", "type": "foreach", "input_from": "items"},
         {"id": "w", "type": "wait", "seconds": 1, "disabled": True, "note": "later"}],
        [{"from": "m", "to": "s"}, {"from": "s", "to": "p"}, {"from": "p", "to": "l"},
         {"from": "l", "to": "f"}, {"from": "f", "to": "w"}, {"from": "w", "to": "EXIT:done"}])


class TestValidatorWarnings:
    def test_valid_and_warns(self):
        room = _all_types_room()
        assert validate_room(room, set(), allow_spawn=False) == []
        text = "\n".join(room_warnings(room, FakeCatalog()))
        assert "ghost" in text and "missing" in text and "nope" in text
        assert "Leeth" in text and "foreach" in text.lower() or "f" in text

    def test_known_names_are_quiet(self):
        room = _room(
            [{"id": "m", "type": "mcp", "server": "unity", "tool": "t"},
             {"id": "s", "type": "skill", "skill": "docs"},
             {"id": "p", "type": "plugin", "command": "hi"}],
            [{"from": "m", "to": "s"}, {"from": "s", "to": "p"}, {"from": "p", "to": "EXIT:done"}])
        assert room_warnings(room, FakeCatalog()) == []

    def test_bad_ignore_rule_in_agent_is_an_error(self):
        with pytest.raises(ValueError):
            Node(id="a", type="agent", budget={"max_iterations": 1, "max_tokens": 10}, ignore=["bogus"])


class TestViews:
    def test_diagram_and_description_mention_new_nodes(self):
        room = _all_types_room()
        text = render_flow(room)
        for word in ("ghost", "missing", "nope", "Leeth", "ОТКЛЮЧЁН"):
            assert word in text
        desc = describe_room(room)
        assert "server=ghost" in desc and "skill=missing" in desc and "mode=delegate" in desc
        assert "[отключён]" in desc


# ── every Argent tool is reachable from a room ───────────────────────────────────

class TestAllTools:
    def _tool_room(self, tool):
        return _room([{"id": "a", "type": "tool", "tool": tool}], [{"from": "a", "to": "EXIT:done"}])

    def test_every_registered_tool_is_a_valid_step_except_runs_in_runs(self):
        from src.rooms.validator import NOT_IN_ROOMS
        from tools.schemas import AVAILABLE_TOOLS
        names = set(AVAILABLE_TOOLS)
        assert len(names) > 40
        for name in sorted(names - NOT_IN_ROOMS):
            assert validate_room(self._tool_room(name), names) == [], name
        for name in sorted(NOT_IN_ROOMS & names):
            assert any("another run" in e for e in validate_room(self._tool_room(name), names))

    def test_leeth_and_browser_tools_are_allowed_for_agents(self):
        from src.agent.subagents import FORBIDDEN_TOOLS
        for name in ("delegate_to_leeth", "leeth_map", "leeth_scan_page", "leeth_click_many",
                     "leeth_fill_form", "browser_open", "call_mcp_tool", "read_skill"):
            assert name not in FORBIDDEN_TOOLS, name


# ── an agent node's skills / mcp / ignore ────────────────────────────────────────

class TestAgentNode:
    def test_executor_passes_them_on(self):
        from src.rooms.runner import AgentExecutor
        seen = {}

        def run(prompt, max_iterations=None, **kw):
            seen.update(kw)
            return '{"x": 1}'
        node = Node(id="a", type="agent", budget={"max_iterations": 3, "max_tokens": 100}, writes=["x"],
                    skills=["docs"], mcp=["unity"], ignore=["mcp:unity.delete_*"])
        assert AgentExecutor(run)(node, {}) == {"x": 1}
        assert seen["skills"] == ["docs"] and seen["mcp_servers"] == ["unity"]
        assert seen["ignore"] == ["mcp:unity.delete_*"]

    def test_mcp_none_means_any_and_empty_means_none(self):
        from src.rooms.runner import AgentExecutor
        seen = []
        ex = AgentExecutor(lambda prompt, **kw: seen.append(kw) or "{}")
        ex(Node(id="a", type="agent", budget={"max_iterations": 1, "max_tokens": 10}), {})
        ex(Node(id="a", type="agent", budget={"max_iterations": 1, "max_tokens": 10}, mcp=[]), {})
        assert "mcp_servers" not in seen[0] and seen[1]["mcp_servers"] == []

    def test_skills_block_applies_the_rules(self, monkeypatch):
        import skill_manager as sm
        from src.rooms.runner import _skills_block
        monkeypatch.setattr(sm.skill_manager, "read_skill", lambda n: SKILL)
        block = _skills_block(["docs", "banned"], policy.Policy(["skill:docs#Setup", "skill:banned"]))
        assert "=== SKILL: docs ===" in block and "setup text" not in block and "banned" not in block


# ── Leeth click: what a scan result turns into ───────────────────────────────────

class TestLeethClick:
    SCAN = {"mode": "yesno", "yes": [
        {"post": 1, "text": "Мой кот", "buttons": [{"index": 4, "label": "Нравится"}, {"index": 5, "label": "В закладки"}]},
        {"post": 3, "text": "Собака", "buttons": [{"index": 9, "label": "Удалить"}, {"index": 10, "label": "В закладки"}]}],
        "unsure": [{"post": 7, "text": "Птица?", "buttons": [{"index": 14, "label": "В закладки"}]}]}

    def _node(self, **kw):
        args = kw.pop("args", {})
        return Node(id="c", type="leeth", mode="click", input_from="found", args=args, **kw)

    def test_posts_need_a_button_pattern(self):
        from src.rooms.integrations import click_indices
        idx, problem = click_indices(self._node(), State(data={"found": self.SCAN}))
        assert idx == [] and "args.button" in problem

    def test_button_pattern_picks_one_per_post(self):
        from src.rooms.integrations import click_indices
        node = self._node(args={"button": "закладк|bookmark"})
        assert click_indices(node, State(data={"found": self.SCAN})) == ([5, 10], "")

    def test_unsure_is_opt_in(self):
        from src.rooms.integrations import click_indices
        node = self._node(args={"button": "закладк", "include_unsure": True})
        assert click_indices(node, State(data={"found": self.SCAN}))[0] == [5, 10, 14]

    def test_link_items_and_plain_numbers(self):
        from src.rooms.integrations import click_indices
        links = {"yes": [{"index": 2, "label": "a"}, {"index": 3, "label": "b"}]}
        assert click_indices(self._node(), State(data={"found": links}))[0] == [2, 3]
        assert click_indices(self._node(), State(data={"found": "[1, 2, 2]"}))[0] == [1, 2]
        assert click_indices(self._node(), State(data={"found": "7, 8"}))[0] == [7, 8]

    def test_json_text_of_a_scan(self):
        from src.rooms.integrations import click_indices
        node = self._node(args={"button": "закладк"})
        text = "scan:\n" + json.dumps(self.SCAN, ensure_ascii=False)
        assert click_indices(node, State(data={"found": text}))[0] == [5, 10]

    def test_nothing_to_click_is_a_problem_not_a_click(self):
        from src.rooms.integrations import click_indices
        assert click_indices(self._node(), State(data={"found": {"yes": []}}))[1]

    def test_click_needs_a_source(self):
        with pytest.raises(ValueError):
            Node(id="c", type="leeth", mode="click")

    def test_whole_leeth_click_goes_through_the_tool(self, monkeypatch):
        import tools.schemas as schemas
        got = {}
        monkeypatch.setattr(schemas, "AVAILABLE_TOOLS", {"leeth_click_many": lambda **kw: got.update(kw) or "{}"})
        node = self._node(args={"button": "закладк"})
        Integrations().leeth(node, State(data={"found": self.SCAN}))
        assert json.loads(got["indices"]) == [5, 10]
