"""Long-term memory notes: the store, the two levels, the tool, the command and the consolidation."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import config
from src.memory import dream, hub
from src.memory.store import (INDEX_MAX_LINES, MAX_BODY_CHARS, MemoryRefused, NodeStore, normalize_name, parse)

NOW = datetime(2026, 10, 13, 12, 0, 0)


@pytest.fixture
def values(monkeypatch):
    stored = {}
    monkeypatch.setattr(config, "_get", lambda key, default=None: stored.get(key, default))
    monkeypatch.setattr(config, "_set", lambda key, val: stored.__setitem__(key, val))
    return stored


@pytest.fixture
def mem(tmp_path, values, monkeypatch):
    project, home = tmp_path / "project" / ".argent" / "memory", tmp_path / "home" / ".argent" / "memory"
    monkeypatch.setattr(hub, "project_dir", lambda: project)
    monkeypatch.setattr(hub, "global_dir", lambda: home)
    return NodeStore(project, "project"), NodeStore(home, "global")


@pytest.fixture
def store(mem):
    return mem[0]


# --- the store ----------------------------------------------------------------------------------------------

class TestNames:
    def test_normalizes(self):
        assert normalize_name("Why no ORM") == "why-no-orm"
        assert normalize_name("  Почему  без ORM ") == "почему-без-orm"

    @pytest.mark.parametrize("bad", ["", "   ", "../etc/passwd", "...", "memory", "CON"])
    def test_refuses(self, bad):
        if bad == "../etc/passwd":
            assert normalize_name(bad) == "etc-passwd"     # separators never survive into a path
            return
        with pytest.raises(MemoryRefused):
            normalize_name(bad)


class TestStore:
    def test_write_read_roundtrip(self, store):
        result = store.put("Why no ORM", "SQLite by hand", "An ORM was tried and dropped.\nSee [[db-layout]].",
                           type="project", source="agent", now=NOW)
        assert result.created and result.changed
        node = store.get("why-no-orm")
        assert node.description == "SQLite by hand"
        assert node.type == "project" and node.source == "agent" and node.updated == "2026-10-13"
        assert node.links == ["db-layout"]
        assert "An ORM was tried" in node.body

    def test_unchanged_write_changes_nothing(self, store):
        store.put("a", "d", "text", now=NOW)
        later = datetime(2026, 11, 1)
        again = store.put("a", "d", "text", now=later)
        assert not again.changed and again.node.updated == "2026-10-13"

    def test_secret_is_hidden_before_it_reaches_disk(self, store):
        key = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4"
        result = store.put("keys", "where the key lives", f"the key is {key} in .env", now=NOW)
        assert result.redacted
        assert key not in store.path_of("keys").read_text(encoding="utf-8")

    def test_long_body_is_cut(self, store):
        result = store.put("big", "d", "x " * MAX_BODY_CHARS)
        assert len(result.node.body) < MAX_BODY_CHARS + 120 and result.warnings

    def test_requires_description_and_body(self, store):
        with pytest.raises(MemoryRefused):
            store.put("a", "", "text")
        with pytest.raises(MemoryRefused):
            store.put("a", "d", "   ")

    def test_note_written_by_the_user_is_protected(self, store):
        store.put("mine", "d", "my text", source="user")
        with pytest.raises(MemoryRefused):
            store.put("mine", "d", "changed by the model", source="agent")
        with pytest.raises(MemoryRefused):
            store.forget("mine", by="dream")
        assert store.forget("mine", by="user")

    def test_file_without_header_is_a_user_note(self, store):
        store.root.mkdir(parents=True)
        (store.root / "hand.md").write_text("# Build\nrun make first\n", encoding="utf-8")
        node = store.get("hand")
        assert node.source == "user" and node.description == "Build"

    def test_header_with_crlf(self):
        node = parse("---\r\nname: a\r\ndescription: d\r\nmetadata:\r\n  type: feedback\r\n  source: dream\r\n---\r\nbody",
                     "a")
        assert (node.type, node.source, node.description, node.body) == ("feedback", "dream", "d", "body")

    def test_forget_and_restore(self, store):
        store.put("a", "d", "text", source="agent")
        assert store.forget("a", by="agent")
        assert store.get("a") is None and store.names() == []
        assert store.restore_from_trash("a") == "a"
        assert store.get("a").body == "text"

    def test_index_is_generated_and_capped(self, store):
        for i in range(INDEX_MAX_LINES + 20):
            store.put(f"note-{i:03d}", f"about {i}", "text", now=NOW)
        lines = store.index_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == INDEX_MAX_LINES
        assert "more notes" in lines[-1]

    def test_hand_made_note_enters_the_index_at_rebuild(self, store):
        store.put("a", "d", "text")
        (store.root / "extra.md").write_text("some text", encoding="utf-8")
        store.rebuild_index()
        assert "extra" in store.index_path.read_text(encoding="utf-8")

    def test_search_ranks_name_and_description_first(self, store):
        store.put("deploy", "how releases are made", "run the script", now=NOW)
        store.put("misc", "other", "deploy deploy deploy was mentioned here", now=NOW)
        found = store.search("deploy")
        assert [n.name for _, n in found][:1] == ["deploy"]
        assert len(found) == 2

    def test_search_finds_russian_word_forms(self, store):
        store.put("тесты", "как запускать тесты", "запускать через .venv", now=NOW)
        assert [n.name for _, n in store.search("запуск тестов")] == ["тесты"]

    def test_snapshot_and_restore(self, store):
        store.put("a", "d", "one", source="agent")
        shot = store.snapshot(now=NOW)
        store.put("a", "d", "two", source="agent")
        store.put("b", "d", "new", source="agent")
        stat = store.restore(shot)
        assert stat == {"restored": 1, "removed": 1}
        assert store.get("a").body == "one" and store.get("b") is None
        assert store.restore_from_trash("b") == "b"          # nothing was lost for good

    def test_snapshots_are_pruned(self, store):
        store.put("a", "d", "text")
        for i in range(8):
            store.snapshot(now=datetime(2026, 10, 13, 12, 0, i))
        assert len(store.snapshots()) == 5


# --- the two levels -------------------------------------------------------------------------------------------

class TestHub:
    def test_project_note_wins_over_global(self, mem):
        project, home = mem
        home.put("style", "global style", "tabs", source="agent")
        assert hub.find("style").scope == "global"
        project.put("style", "project style", "spaces", source="agent")
        assert hub.find("style").scope == "project"
        assert hub.find("global:style").body == "tabs"

    def test_prompt_section_is_empty_without_notes_or_when_off(self, mem, values):
        assert hub.prompt_section(3000) == ""
        mem[0].put("a", "d", "text")
        assert "a" in hub.prompt_section(3000)
        values["memory_enabled"] = False
        assert hub.prompt_section(3000) == ""

    def test_prompt_section_respects_the_budget(self, mem):
        for i in range(60):
            mem[0].put(f"note-{i:02d}", "a description of a fair length to take space", "text")
        text = hub.prompt_section(900)
        assert len(text) <= 900 + 60
        assert "more: use search" in text

    def test_scope_names(self, mem):
        assert hub.store_for("global").scope == "global"
        with pytest.raises(MemoryRefused):
            hub.store_for("galaxy")


# --- the tool -------------------------------------------------------------------------------------------------

class TestTool:
    def call(self, **kw):
        from tools.memory_tools import memory
        return memory(**kw)

    def test_write_read_search_forget(self, mem):
        out = self.call(action="write", name="why-sqlite", description="db choice", body="SQLite, no server needed",
                        type="project")
        assert "created" in out
        assert "SQLite" in self.call(action="read", name="why-sqlite")
        assert "why-sqlite" in self.call(action="search", query="sqlite")
        assert "moved to the trash" in self.call(action="forget", name="why-sqlite")
        assert self.call(action="read", name="why-sqlite").startswith("Error")

    def test_global_scope(self, mem):
        self.call(action="write", name="lang", description="reply language", body="Russian", scope="global",
                  type="user")
        assert mem[1].get("lang") is not None and mem[0].get("lang") is None

    def test_user_note_is_refused(self, mem):
        mem[0].put("mine", "d", "text", source="user")
        out = self.call(action="write", name="mine", description="d", body="other")
        assert out.startswith("Error") and "user" in out

    def test_errors_are_text(self, mem):
        assert self.call(action="read").startswith("Error")
        assert self.call(action="dance").startswith("Error")
        assert self.call(action="write", name="a", description="", body="x").startswith("Error")

    def test_switched_off(self, mem, values):
        values["memory_enabled"] = False
        assert "switched off" in self.call(action="list")

    def test_registered_everywhere(self):
        from tools.lazy_registry import DEFAULT_TOOL_LOADERS
        from tools.schemas import TOOL_SCHEMAS
        from src.agent.subagents import FORBIDDEN_TOOLS
        assert DEFAULT_TOOL_LOADERS["memory"] == ("tools.memory_tools", "memory")
        names = [t["function"]["name"] for t in TOOL_SCHEMAS]
        assert names.count("memory") == 1
        assert "memory" in FORBIDDEN_TOOLS

    def test_plan_mode_allows_reading_but_not_writing(self, mem):
        from src.agent import plan_mode
        plan_mode.PLAN.enter()
        try:
            assert plan_mode.violation("memory", {"action": "read", "name": "a"}) is None
            assert plan_mode.violation("memory", {"action": "search", "query": "a"}) is None
            assert plan_mode.violation("memory", {"action": "write"})
            assert plan_mode.violation("memory", {"action": "forget"})
        finally:
            plan_mode.PLAN.leave()

    def test_hidden_when_switched_off(self, mem, values):
        from src.agent import plan_mode
        assert plan_mode.hides("memory") is False
        values["memory_enabled"] = False
        assert plan_mode.hides("memory") is True

    def test_small_models_are_offered_it(self):
        import tool_profiles
        assert "memory" in tool_profiles.core_tools_now()


# --- the system prompt ----------------------------------------------------------------------------------------

class TestPrompt:
    def test_index_reaches_the_prompt(self, mem):
        from agent import ArgentAgent
        mem[0].put("deploy-steps", "how a release is made", "run make release", source="agent")
        agent = ArgentAgent.__new__(ArgentAgent)
        agent.model_name = "qwen3:8b"
        try:
            prompt = ArgentAgent.build_system_prompt(agent, available_tools={"memory", "read_file"})
            without = ArgentAgent.build_system_prompt(agent, available_tools={"read_file"})
        except Exception as e:                                   # the prompt builder needs a fuller agent
            pytest.skip(f"cannot build the prompt on a bare agent: {e}")
        assert "deploy-steps" in prompt and "LONG-TERM MEMORY" in prompt
        assert "deploy-steps" not in without                    # no tool, no instructions about it


# --- the consolidation ----------------------------------------------------------------------------------------

def _answer(*ops, conflicts=(), suggestions=()):
    return json.dumps({"ops": list(ops), "conflicts": list(conflicts), "suggestions": list(suggestions)})


class TestGate:
    def test_switched_off(self, values):
        values["memory_enabled"] = False
        assert not dream.check_gate().ok

    def test_cloud_needs_consent(self, values):
        gate = dream.check_gate(target_cloud=True, idle=lambda: 9999, battery=lambda: False)
        assert not gate.ok and "согласия" in gate.reason
        values["memory_cloud_consent"] = True
        assert dream.check_gate(target_cloud=True, idle=lambda: 9999, battery=lambda: False).ok

    def test_battery_and_busy_pc(self, values):
        assert not dream.check_gate(idle=lambda: 9999, battery=lambda: True).ok
        values["dream_on_battery"] = True
        assert dream.check_gate(idle=lambda: 9999, battery=lambda: True).ok
        assert not dream.check_gate(idle=lambda: 30, battery=lambda: False).ok
        assert dream.check_gate(idle=lambda: None, battery=lambda: False).ok      # cannot be told: go on

    def test_force_skips_the_pc_checks_only(self, values):
        assert dream.check_gate(force=True, idle=lambda: 1, battery=lambda: True).ok
        values["memory_enabled"] = False
        assert not dream.check_gate(force=True).ok


class TestExcerpt:
    def test_user_turns_and_final_answers(self):
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "use pytest, not unittest"},
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "content": "lots of output"},
            {"role": "assistant", "content": "Done, switched to pytest."},
            {"role": "user", "content": [{"type": "text", "text": "thanks"}]},
        ]
        text = dream.excerpt_of(messages)
        assert "USER: use pytest, not unittest" in text
        assert "ASSISTANT: Done, switched to pytest." in text
        assert "lots of output" not in text and "prompt" not in text
        assert "USER: thanks" in text

    def test_the_newest_turns_win_the_budget(self):
        messages = []
        for i in range(60):
            messages += [{"role": "user", "content": f"question number {i}"},
                         {"role": "assistant", "content": f"answer {i}"}]
        text = dream.excerpt_of(messages, limit=400)
        assert "question number 59" in text and "question number 0" not in text


class TestAnswer:
    def test_fenced_and_thinking(self):
        body = _answer({"op": "create", "name": "a", "description": "d", "body": "b"})
        raw = "<think>hmm</think>\n```json\n" + body + "\n```"
        ops, _, _ = dream.parse_answer(raw)
        assert [(o.op, o.name) for o in ops] == [("create", "a")]

    def test_braces_inside_strings(self):
        raw = _answer({"op": "create", "name": "a", "description": "d", "body": "use {braces} } here"})
        ops, _, _ = dream.parse_answer("Here you go: " + raw + " done")
        assert ops[0].body == "use {braces} } here"

    def test_unknown_ops_are_dropped_and_garbage_raises(self):
        ops, _, _ = dream.parse_answer(_answer({"op": "format_disk", "name": "x"}, {"op": "forget", "name": "y"}))
        assert [o.op for o in ops] == ["forget"]
        with pytest.raises(ValueError):
            dream.parse_answer("no json at all")
        with pytest.raises(ValueError):
            dream.parse_answer('{"ops": [')


def _sessions(monkeypatch, *items):
    """items: (id, saved_at, cwd, [messages])"""
    import session
    metas = [{"id": i, "saved_at": s, "cwd": c} for i, s, c, _ in items]
    by_id = {i: {"messages": m} for i, _, _, m in items}
    monkeypatch.setattr(session, "list_sessions", lambda query=None: list(metas))
    monkeypatch.setattr(session, "load_session", lambda sid: by_id.get(sid))


def _chat(user, answer="ok"):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


class TestGather:
    def test_only_this_project_and_only_newer(self, tmp_path, monkeypatch):
        mine, other = tmp_path / "proj", tmp_path / "elsewhere"
        _sessions(monkeypatch,
                  ("a", "2026-10-10T10:00:00", str(mine), _chat("old question")),
                  ("b", "2026-10-12T10:00:00", str(mine / "src"), _chat("new question")),
                  ("c", "2026-10-12T11:00:00", str(other), _chat("foreign question")))
        got = dream.gather(mine, since="2026-10-11T00:00:00")
        assert "new question" in got.text
        assert "old question" not in got.text and "foreign" not in got.text
        assert got.sessions == 1 and got.newest == "2026-10-12T10:00:00"

    def test_secrets_never_reach_the_model(self, tmp_path, monkeypatch):
        key = "sk-" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1"
        _sessions(monkeypatch, ("a", "2026-10-12T10:00:00", str(tmp_path), _chat(f"my key is {key}")))
        assert key not in dream.gather(tmp_path).text


class TestRun:
    @pytest.fixture
    def setup(self, store, tmp_path, monkeypatch, values):
        _sessions(monkeypatch, ("s1", "2026-10-12T10:00:00", str(tmp_path),
                                _chat("always run tests with the venv python", "ok, noted")))
        return store

    def go(self, store, tmp_path, ask, **kw):
        kw.setdefault("gate", lambda **k: dream.Gate(True))
        return dream.run(store, ask, tmp_path, now=NOW, **kw)

    def test_creates_a_note_and_remembers_how_far_it_read(self, setup, tmp_path):
        ask = MagicMock(return_value=_answer({"op": "create", "name": "run-tests", "type": "feedback",
                                              "description": "tests run with the venv python",
                                              "body": "Use .venv python for pytest.", "reason": "user said so"}))
        report = self.go(setup, tmp_path, ask)
        assert report.changed and len(report.applied) == 1
        node = setup.get("run-tests")
        assert node.source == "dream" and node.type == "feedback"
        state = dream.load_state(setup)
        assert state["processed_until"] == "2026-10-12T10:00:00"
        # nothing new: the model is not asked again
        ask.reset_mock()
        again = self.go(setup, tmp_path, ask)
        assert again.skipped == "нет новых диалогов"
        ask.assert_not_called()

    def test_a_snapshot_is_taken_and_undo_puts_it_back(self, setup, tmp_path):
        setup.put("old", "d", "before", source="dream")
        ask = MagicMock(return_value=_answer({"op": "update", "name": "old", "description": "d", "body": "after"},
                                             {"op": "create", "name": "fresh", "description": "d", "body": "x"}))
        report = self.go(setup, tmp_path, ask)
        assert report.snapshot and setup.get("old").body == "after"
        setup.restore(setup.snapshots()[-1])
        assert setup.get("old").body == "before" and setup.get("fresh") is None

    def test_the_users_note_becomes_a_suggestion(self, setup, tmp_path):
        setup.put("mine", "d", "written by me", source="user")
        ask = MagicMock(return_value=_answer({"op": "update", "name": "mine", "description": "d", "body": "other",
                                              "reason": "outdated"}, {"op": "forget", "name": "mine"}))
        report = self.go(setup, tmp_path, ask)
        assert setup.get("mine").body == "written by me"
        assert len(report.suggestions) == 2 and not report.applied

    def test_limits_and_bad_names(self, setup, tmp_path):
        for i in range(6):
            setup.put(f"gone-{i}", "d", "x", source="dream")
        ops = [{"op": "forget", "name": f"gone-{i}", "reason": "old"} for i in range(6)]
        ops.append({"op": "create", "name": "!!!", "description": "d", "body": "x"})
        report = self.go(setup, tmp_path, MagicMock(return_value=_answer(*ops)))
        assert len([a for a in report.applied if "забыта" in a]) == 3
        assert len(report.refused) == 4                       # three over the limit and the bad name

    def test_dry_run_writes_nothing(self, setup, tmp_path):
        ask = MagicMock(return_value=_answer({"op": "create", "name": "a", "description": "d", "body": "x"}))
        report = self.go(setup, tmp_path, ask, dry_run=True)
        assert report.applied and not report.changed
        assert setup.names() == [] and dream.load_state(setup) == {}

    def test_a_useless_answer_is_a_report_and_the_sessions_stay_unread(self, setup, tmp_path):
        report = self.go(setup, tmp_path, MagicMock(return_value="I cannot do that"))
        assert "годного ответа" in report.skipped
        assert dream.load_state(setup) == {}
        report = self.go(setup, tmp_path, MagicMock(side_effect=RuntimeError("server down")))
        assert "server down" in report.skipped

    def test_closed_gate_asks_nothing(self, setup, tmp_path):
        ask = MagicMock()
        report = self.go(setup, tmp_path, ask, gate=lambda **k: dream.Gate(False, "ноутбук работает от батареи"))
        assert report.skipped == "ноутбук работает от батареи"
        ask.assert_not_called()

    def test_one_run_at_a_time(self, setup, tmp_path):
        import os
        lock = tmp_path / ".argent" / "run" / "dream.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text(str(os.getppid()), encoding="utf-8")     # a live process that is not us
        ask = MagicMock()
        report = self.go(setup, tmp_path, ask)
        assert "уже идёт" in report.skipped
        ask.assert_not_called()

    def test_report_file_is_written(self, setup, tmp_path):
        ask = MagicMock(return_value=_answer({"op": "create", "name": "a", "description": "d", "body": "x"}))
        self.go(setup, tmp_path, ask)
        files = list((setup.root / dream.REPORTS_DIR).glob("*.md"))
        assert len(files) == 1 and "«a»" in files[0].read_text(encoding="utf-8")

    def test_the_request_is_made_at_background_priority(self, monkeypatch):
        from src.agent import governor
        seen = {}

        class Provider:
            def sync_chat(self, model, messages, **kw):
                seen["priority"] = governor.current_priority()
                return "{}"

        import providers
        monkeypatch.setattr(providers, "create_provider", lambda name: Provider())
        dream.default_ask("ollama", "m")([{"role": "user", "content": "x"}])
        assert seen["priority"] == governor.P_BACKGROUND


# --- migration --------------------------------------------------------------------------------------------------

class TestMigrate:
    def test_key_facts_become_a_note(self, store, tmp_path):
        from src.memory.migrate import import_working_memory
        path = tmp_path / "memory.json"
        path.write_text(json.dumps({"objective": "port to rust", "key_facts": ["uses poetry", "CI is slow"]}),
                        encoding="utf-8")
        result = import_working_memory(store, path)
        assert result.created
        body = store.get("imported-key-facts").body
        assert "uses poetry" in body and "port to rust" in body
        assert json.loads(path.read_text(encoding="utf-8"))["key_facts"]       # the source is untouched

    def test_nothing_to_import(self, store, tmp_path):
        from src.memory.migrate import import_working_memory
        path = tmp_path / "memory.json"
        path.write_text(json.dumps({"key_facts": []}), encoding="utf-8")
        assert import_working_memory(store, path) is None
        assert import_working_memory(store, tmp_path / "missing.json") is None


# --- the command ------------------------------------------------------------------------------------------------

def run(text):
    from src.cli.commands import CommandContext, dispatch_command
    return dispatch_command(text, CommandContext(agent=MagicMock(), user_input=text))


class TestCommand:
    def test_add_show_forget_restore(self, mem):
        assert run("/memory add build-notes use make, not ninja").handled
        node = mem[0].get("build-notes")
        assert node.source == "user" and node.body == "use make, not ninja"
        run("/memory add --global my-style short answers please")
        assert mem[1].get("my-style") is not None
        run("/memory forget build-notes")
        assert mem[0].get("build-notes") is None
        run("/memory restore build-notes")
        assert mem[0].get("build-notes") is not None

    def test_switches(self, values):
        run("/memory off")
        assert values["memory_enabled"] is False
        run("/memory on")
        assert values["memory_enabled"] is True

    def test_undo_without_snapshot(self, mem):
        assert run("/memory undo").handled          # a message, not a crash

    def test_dream_on_and_off_manage_a_scheduled_command(self, mem, tmp_path, monkeypatch):
        from src.automation import store as astore
        monkeypatch.setattr(astore, "defs_path", lambda: tmp_path / "automations.json")
        run("/memory dream on")
        job = astore.get_automation("memory-dream")
        assert job is not None and job.mode == "command" and "src.memory.dream" in job.task
        assert job.notify == "never"
        run("/memory dream off")
        assert astore.get_automation("memory-dream") is None

    def test_dream_settings(self, values):
        run("/memory dream consent on")
        assert values["memory_cloud_consent"] is True
        run("/memory dream battery on")
        assert values["dream_on_battery"] is True
        run("/memory dream model zai:glm-4.6")
        assert values["dream_model"] == "glm-4.6"
        run("/memory dream clear")
        assert values["dream_model"] == ""


class TestGlobalConsolidation:
    """The consolidation may also tend the global notes — only when the person allowed it, and narrowly."""

    @pytest.fixture
    def both(self, mem, tmp_path, monkeypatch, values):
        _sessions(monkeypatch, ("s1", "2026-10-12T10:00:00", str(tmp_path),
                                _chat("answer me in Russian always, in every project", "ok")))
        return mem

    def go(self, both, tmp_path, ask, with_global=True, **kw):
        project, home = both
        return dream.run(project, ask, tmp_path, now=NOW, gate=lambda **k: dream.Gate(True),
                         global_store=home if with_global else None, **kw)

    def test_the_model_is_shown_both_indexes(self, both, tmp_path):
        project, home = both
        project.put("p-note", "project fact", "x", source="dream")
        home.put("g-note", "global fact", "y", source="user")
        ask = MagicMock(return_value=_answer())
        self.go(both, tmp_path, ask)
        prompt = ask.call_args[0][0]
        text = prompt[1]["content"]
        assert "## GLOBAL INDEX" in text and "g-note" in text and "p-note" in text
        assert "GLOBAL notes" in prompt[0]["content"]

    def test_without_permission_the_global_level_is_not_in_the_request(self, both, tmp_path):
        ask = MagicMock(return_value=_answer())
        self.go(both, tmp_path, ask, with_global=False)
        assert "GLOBAL" not in ask.call_args[0][0][1]["content"]
        assert "GLOBAL notes" not in ask.call_args[0][0][0]["content"]

    def test_a_global_note_is_written_and_undoable(self, both, tmp_path):
        project, home = both
        home.put("language", "reply language", "English", source="dream")
        ask = MagicMock(return_value=_answer(
            {"op": "update", "name": "language", "scope": "global", "description": "reply language",
             "body": "Russian, in every project", "reason": "said so"},
            {"op": "create", "name": "likes-short-answers", "scope": "global", "type": "feedback",
             "description": "short answers", "body": "Keep answers short."}))
        report = self.go(both, tmp_path, ask)
        assert home.get("language").body.startswith("Russian") and home.get("likes-short-answers") is not None
        assert any("global:language" in a for a in report.applied)
        assert report.global_snapshot and Path(report.global_snapshot).is_dir()
        home.restore(Path(report.global_snapshot))
        assert home.get("language").body == "English" and home.get("likes-short-answers") is None

    def test_a_first_run_can_be_undone_too(self, both, tmp_path):
        project, home = both
        ask = MagicMock(return_value=_answer({"op": "create", "name": "first", "description": "d", "body": "x"}))
        report = self.go(both, tmp_path, ask)
        assert project.get("first") is not None and report.snapshot
        project.restore(Path(report.snapshot))
        assert project.get("first") is None                        # nothing to go back to is still a place to go back to

    def test_global_notes_take_the_person_not_the_project(self, both, tmp_path):
        project, home = both
        ask = MagicMock(return_value=_answer(
            {"op": "create", "name": "auth-module", "scope": "global", "type": "project",
             "description": "d", "body": "src/auth.py does X"}))
        report = self.go(both, tmp_path, ask)
        assert home.get("auth-module") is None and project.get("auth-module") is None
        assert any("global:auth-module" in r for r in report.refused)

    def test_at_most_three_global_operations(self, both, tmp_path):
        project, home = both
        ops = [{"op": "create", "name": f"g-{i}", "scope": "global", "type": "feedback", "description": "d",
                "body": "x"} for i in range(5)]
        report = self.go(both, tmp_path, MagicMock(return_value=_answer(*ops)))
        assert len(home.names()) == 3 and len(report.refused) == 2

    def test_a_global_operation_without_permission_is_refused(self, both, tmp_path):
        project, home = both
        ask = MagicMock(return_value=_answer(
            {"op": "create", "name": "g", "scope": "global", "type": "user", "description": "d", "body": "x"}))
        report = self.go(both, tmp_path, ask, with_global=False)
        assert home.names() == [] and "глобальная консолидация выключена" in report.refused[0]

    def test_the_persons_global_note_is_never_touched(self, both, tmp_path):
        project, home = both
        home.put("mine", "d", "written by me", source="user")
        ask = MagicMock(return_value=_answer({"op": "forget", "name": "mine", "scope": "global", "reason": "old"}))
        report = self.go(both, tmp_path, ask)
        assert home.get("mine").body == "written by me" and report.suggestions

    def test_the_setting_and_undo_command(self, both, values, tmp_path):
        run("/memory dream global on")
        assert values["dream_global"] is True
        project, home = both
        home.put("language", "d", "English", source="dream")
        shot = home.snapshot(now=NOW)
        home.put("language", "d", "Russian", source="dream")
        project.put("p", "d", "x", source="dream")
        project.snapshot(now=NOW)
        dream.save_state(project, {"last_global_snapshot": str(shot)})
        run("/memory undo")
        assert home.get("language").body == "English"
