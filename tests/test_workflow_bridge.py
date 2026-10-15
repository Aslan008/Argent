"""The TUI bridge and the workflow features: the state the frontend draws, and the prompt/stop hooks."""

import json
import sys
import time
from unittest.mock import MagicMock

import pytest

import agent as agent_module
import config
from agent import ArgentAgent
from src.agent import shell_hooks, todos
from src.agent.plan_mode import PLAN
from src.bridge.repl_bridge import ReplBridge


@pytest.fixture
def bridge_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".argent").mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(shell_hooks, "SHELL_HOOKS", shell_hooks.ShellHooks(project_dir=tmp_path, home=home))
    mem = MagicMock()
    mem.data = {}
    monkeypatch.setattr(agent_module, "memory", mem)
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)
    agent = ArgentAgent()
    bridge = ReplBridge(agent=agent)
    events = []
    bridge._event_sink = events.append
    bridge.session.set_event_sink(events.append)
    turns = []

    def fake_turn(text):
        """What the real _run_turn does to the frontend: run the turn, then say it is done."""
        turns.append(text)
        events.append({"type": "done"})

    monkeypatch.setattr(bridge.session, "_run_turn", fake_turn)
    return tmp_path, bridge, events, turns


def hook(tmp_path, event, body, name):
    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    path = tmp_path / ".argent" / "hooks.json"
    table = json.loads(path.read_text(encoding="utf-8"))["hooks"] if path.exists() else {}
    table.setdefault(event, []).append({"command": f'"{sys.executable}" "{script}"'})
    path.write_text(json.dumps({"hooks": table}), encoding="utf-8")
    shell_hooks.trust_file(path)


def send(bridge, events, text, timeout=20):
    events.clear()
    bridge.start_turn(text)
    end = time.time() + timeout
    while time.time() < end:
        if any(e.get("type") == "done" for e in events):
            return
        time.sleep(0.05)
    raise AssertionError(f"{text!r} never finished: {events}")


def test_the_state_carries_the_checklist_and_plan_mode(bridge_env):
    _, bridge, _, _ = bridge_env
    state = bridge.get_state()
    assert state["todos"] == "[]" and state["mode"] == "CHAT"
    todos.write([{"content": "read", "status": "completed"}, {"content": "edit", "status": "in_progress"}])
    PLAN.enter()
    state = bridge.get_state()
    assert state["mode"] == "PLAN"
    assert json.loads(state["todos"]) == [{"content": "read", "status": "completed"},
                                          {"content": "edit", "status": "in_progress"}]


def test_a_prompt_hook_adds_its_context_to_what_the_agent_receives(bridge_env):
    tmp_path, bridge, events, turns = bridge_env
    hook(tmp_path, "UserPromptSubmit", "print('branch: feature-x')", "ctx.py")
    send(bridge, events, "fix the bug")
    assert len(turns) == 1 and turns[0].startswith("fix the bug") and "branch: feature-x" in turns[0]


def test_a_prompt_hook_can_stop_the_message(bridge_env):
    tmp_path, bridge, events, turns = bridge_env
    hook(tmp_path, "UserPromptSubmit", "import sys; sys.stderr.write('release freeze'); sys.exit(2)", "freeze.py")
    send(bridge, events, "deploy now")
    assert turns == []
    assert any("release freeze" in str(e.get("text", "")) for e in events if e.get("type") == "notice")


def test_commands_are_not_decorated_by_the_prompt_hook(bridge_env):
    tmp_path, bridge, events, turns = bridge_env
    hook(tmp_path, "UserPromptSubmit", "print('extra context')", "ctx.py")
    send(bridge, events, "/plan status")
    assert turns == []                                   # a command, handled before any agent turn
    assert not any("extra context" in str(e) for e in events)


def test_the_stop_hook_speaks_after_the_turn(bridge_env):
    tmp_path, bridge, events, turns = bridge_env
    hook(tmp_path, "Stop", "print('turn finished: all good')", "stop.py")
    send(bridge, events, "do the thing")
    # The turn reports `done` itself; the Stop hook runs once it has returned, so its note follows.
    end = time.time() + 10
    while time.time() < end and not any(e.get("type") == "notice" for e in events):
        time.sleep(0.05)
    assert any("turn finished: all good" in str(e.get("text", "")) for e in events if e.get("type") == "notice")


def test_an_untrusted_project_hook_does_not_run_and_the_person_is_told(bridge_env):
    tmp_path, bridge, events, turns = bridge_env
    script = tmp_path / "ctx.py"
    script.write_text("print('should not run')", encoding="utf-8")
    (tmp_path / ".argent" / "hooks.json").write_text(
        json.dumps({"hooks": {"UserPromptSubmit": [{"command": f'"{sys.executable}" "{script}"'}]}}), encoding="utf-8")
    send(bridge, events, "hello")
    assert turns == ["hello"]
    assert shell_hooks.SHELL_HOOKS.status()["untrusted"] is not None
