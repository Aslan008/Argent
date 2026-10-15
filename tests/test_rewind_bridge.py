"""The TUI's "back to here": /back through the real bridge — events out, files and conversation restored."""

import subprocess
import time
from unittest.mock import MagicMock

import pytest

import agent as agent_module
import config
from agent import ArgentAgent
from src.agent.checkpoints import set_auto_checkpoint
from src.bridge.repl_bridge import ReplBridge


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _git("init", "-q")
    _git("config", "user.email", "t@argent.local")
    _git("config", "user.name", "T")
    (tmp_path / "app.py").write_text("v1\n", encoding="utf-8")
    _git("add", "-A")
    _git("commit", "-q", "-m", "initial")
    set_auto_checkpoint(True)
    mem = MagicMock()
    mem.data = {}
    monkeypatch.setattr(agent_module, "memory", mem)
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)
    agent = ArgentAgent()
    bridge = ReplBridge(agent=agent)
    events = []
    bridge._event_sink = events.append          # no start-up sequence, no watches: just the events
    bridge.session.set_event_sink(events.append)
    yield tmp_path, agent, bridge, events
    set_auto_checkpoint(False)


def run_command(bridge, events, command, timeout=20):
    events.clear()
    bridge.start_turn(command)
    end = time.time() + timeout
    while time.time() < end:
        if any(e.get("type") == "done" for e in events):
            return
        time.sleep(0.05)
    raise AssertionError(f"{command} never finished: {events}")


def fake_turn(agent, repo, text, write=None):
    agent._begin_turn_record(text)
    agent.messages.append({"role": "user", "content": text})
    agent._turn_checkpoint_done = False
    agent._turn_label = text
    if write:
        agent._maybe_turn_checkpoint("write_file")
        (repo / write[0]).write_text(write[1], encoding="utf-8")
    agent.messages.append({"role": "assistant", "content": "ok"})


def test_the_state_tells_the_frontend_when_there_is_something_to_go_back_to(setup):
    repo, agent, bridge, events = setup
    assert bridge.get_state()["can_rewind"] is False
    fake_turn(agent, repo, "правка", ("app.py", "v2\n"))
    assert bridge.get_state()["can_rewind"] is True


def test_back_restores_files_and_conversation_and_sends_the_text_to_the_prompt(setup):
    repo, agent, bridge, events = setup
    before = list(agent.messages)
    fake_turn(agent, repo, "поменяй app.py", ("app.py", "v2\n"))

    run_command(bridge, events, "/back -y")

    rewound = [e for e in events if e.get("type") == "rewound"]
    assert len(rewound) == 1 and rewound[0]["text"] == "поменяй app.py"
    assert "stash" in rewound[0]["detail"]
    assert (repo / "app.py").read_text(encoding="utf-8") == "v1\n"
    assert agent.messages == before
    assert bridge.get_state()["can_rewind"] is False


def test_back_with_nothing_to_go_back_to_is_an_error_not_a_crash(setup):
    repo, agent, bridge, events = setup
    run_command(bridge, events, "/back -y")
    assert not [e for e in events if e.get("type") == "rewound"]
    assert any("Нет сообщения" in str(e.get("text", "")) for e in events if e.get("type") in ("error", "notice", "system"))


def test_a_refused_file_rewind_changes_nothing_and_sends_no_event(setup):
    repo, agent, bridge, events = setup
    fake_turn(agent, repo, "правка", ("app.py", "v2\n"))
    _git("add", "-A")
    _git("commit", "-q", "-m", "the person committed it")
    snapshot = list(agent.messages)

    run_command(bridge, events, "/back -y")

    assert not [e for e in events if e.get("type") == "rewound"]
    assert agent.messages == snapshot and (repo / "app.py").read_text(encoding="utf-8") == "v2\n"


def test_the_button_asks_first_and_only_then_goes_back(setup):
    repo, agent, bridge, events = setup
    fake_turn(agent, repo, "поменяй app.py", ("app.py", "v2\n"))

    events.clear()
    bridge.start_turn("/back")                       # what the button sends
    end = time.time() + 20
    ask = None
    while time.time() < end and ask is None:
        ask = next((e for e in events if e.get("type") == "select_request"), None)
        time.sleep(0.05)
    assert ask is not None, events
    assert "поменяй app.py" in ask["title"] and "Да" in ask["options"]
    assert (repo / "app.py").read_text(encoding="utf-8") == "v2\n"       # nothing happened yet

    bridge.reply_select(ask["id"], "Нет")
    end = time.time() + 20
    while time.time() < end and not any(e.get("type") == "done" for e in events):
        time.sleep(0.05)
    assert not [e for e in events if e.get("type") == "rewound"]
    assert (repo / "app.py").read_text(encoding="utf-8") == "v2\n"       # declined: untouched

    run = events.clear()
    bridge.start_turn("/back")
    end = time.time() + 20
    ask = None
    while time.time() < end and ask is None:
        ask = next((e for e in events if e.get("type") == "select_request"), None)
        time.sleep(0.05)
    bridge.reply_select(ask["id"], "Да")
    end = time.time() + 20
    while time.time() < end and not any(e.get("type") == "rewound" for e in events):
        time.sleep(0.05)
    assert any(e.get("type") == "rewound" for e in events)
    assert (repo / "app.py").read_text(encoding="utf-8") == "v1\n"
