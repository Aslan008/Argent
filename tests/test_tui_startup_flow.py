"""The TUI opens the way the terminal does: ask what to do with the last session
FIRST, start the MCP servers only afterwards.

The bridge used to announce "last session: … (/resume)" as a one-line notice and
start every MCP server at once, in the background, while the user was still
deciding. A server's working directory is fixed when it is spawned and restoring a
session can change directory, so the servers were anchored to the wrong place —
the reason the REPL runs the question first.
"""

import os
import threading
import time

import pytest

import src.startup as startup
from src.bridge.repl_bridge import ReplBridge


@pytest.fixture
def with_prompts(monkeypatch):
    monkeypatch.delenv("ARGENT_NO_STARTUP_PROMPTS", raising=False)


class Recorder:
    """A fake TUI: records events and answers select requests from a script."""

    def __init__(self, bridge, answers):
        self.bridge = bridge
        self.answers = list(answers)
        self.events = []
        self.questions = []
        self.lock = threading.Lock()

    def __call__(self, event):
        with self.lock:
            self.events.append(event)
        if event.get("type") == "select_request":
            self.questions.append(event)
            answer = self.answers.pop(0) if self.answers else ""
            threading.Thread(
                target=lambda: self.bridge.reply_select(event["id"], answer), daemon=True).start()

    def texts(self):
        with self.lock:
            return [e.get("text", "") for e in self.events if e.get("type") in ("system", "notice", "error")]

    def wait_for(self, predicate, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if predicate(self.events):
                    return True
            time.sleep(0.05)
        return False


def _session(tmp_path, cwd=None):
    return {"id": "s1", "saved_at": "2026-10-06T22:00:00", "preview": "почини меню",
            "cwd": cwd if cwd is not None else os.getcwd()}


@pytest.fixture
def previous(monkeypatch, tmp_path):
    meta = _session(tmp_path)
    payload = {"id": "s1", "saved_at": meta["saved_at"], "model": "", "cwd": meta["cwd"],
               "messages": [{"role": "system", "content": "s"},
                            {"role": "user", "content": "почини меню"},
                            {"role": "assistant", "content": "Готово, меню починено."}]}
    monkeypatch.setattr(startup, "get_last_session", lambda: meta)
    monkeypatch.setattr(startup, "load_session", lambda sid: payload)
    monkeypatch.setattr(startup, "carried_over_memory", lambda: [])
    monkeypatch.setattr("config.CONFIG_FILE", tmp_path / "exists.json")      # not a first run
    (tmp_path / "exists.json").write_text("{}", encoding="utf-8")
    return payload


def _spy_services(monkeypatch):
    order = []
    monkeypatch.setattr("config.get_mcp_servers", lambda: [{"name": "srv", "command": "x"}])
    from mcp_client import mcp_client
    monkeypatch.setattr(mcp_client, "register_and_start",
                        lambda name, cfg: order.append("mcp") or f"Started {name}")
    return order


class TestTheQuestionComesFirst:
    def test_it_asks_what_to_do_with_the_last_session(self, with_prompts, previous, monkeypatch):
        _spy_services(monkeypatch)
        bridge = ReplBridge()
        tui = Recorder(bridge, ["Продолжить прошлую сессию"])
        bridge.set_event_sink(tui)
        assert tui.wait_for(lambda ev: any(e.get("type") == "select_request" for e in ev))
        question = tui.questions[0]
        assert question["title"] == "С чего начать?"
        assert "Продолжить прошлую сессию" in question["options"]
        assert "Начать с чистого листа (забыть всё выше)" in question["options"]

    def test_the_last_session_is_named_before_the_question(self, with_prompts, previous, monkeypatch):
        _spy_services(monkeypatch)
        bridge = ReplBridge()
        tui = Recorder(bridge, ["Продолжить прошлую сессию"])
        bridge.set_event_sink(tui)
        assert tui.wait_for(lambda ev: any(e.get("type") == "select_request" for e in ev))
        assert any("Прошлая сессия" in t and "почини меню" in t for t in tui.texts())

    def test_mcp_waits_for_the_answer(self, with_prompts, previous, monkeypatch):
        order = _spy_services(monkeypatch)
        bridge = ReplBridge()

        class Slow(Recorder):
            def __call__(self, event):
                if event.get("type") == "select_request":
                    # While the question is open, nothing may have started.
                    order.append(f"asked(mcp_started={'mcp' in order})")
                super().__call__(event)

        tui = Slow(bridge, ["Продолжить прошлую сессию"])
        bridge.set_event_sink(tui)
        assert tui.wait_for(lambda ev: any("MCP:" in e.get("text", "") for e in ev))
        assert order[0] == "asked(mcp_started=False)"
        assert order[-1] == "mcp"

    def test_the_servers_do_not_start_while_the_question_is_unanswered(self, with_prompts, previous, monkeypatch):
        order = _spy_services(monkeypatch)
        bridge = ReplBridge()
        tui = Recorder(bridge, [])
        tui.answers = None

        class Silent(Recorder):
            def __call__(self, event):
                with self.lock:
                    self.events.append(event)       # records, never answers

        silent = Silent(bridge, [])
        bridge.set_event_sink(silent)
        assert silent.wait_for(lambda ev: any(e.get("type") == "select_request" for e in ev))
        time.sleep(0.5)
        assert order == [], "an MCP server was started before the question was answered"
        # unblock the worker so the thread ends
        question = next(e for e in silent.events if e.get("type") == "select_request")
        bridge.reply_select(question["id"], "Начать с чистого листа (забыть всё выше)")


class TestTheAnswers:
    def test_continuing_restores_the_conversation(self, with_prompts, previous, monkeypatch):
        _spy_services(monkeypatch)
        bridge = ReplBridge()
        tui = Recorder(bridge, ["Продолжить прошлую сессию"])
        bridge.set_event_sink(tui)
        assert tui.wait_for(lambda ev: any("Восстановлено" in e.get("text", "") for e in ev))
        assert [m["role"] for m in bridge.agent.messages] == ["system", "user", "assistant"]
        assert bridge.agent.session_id == "s1"

    def test_continuing_puts_the_conversation_back_on_the_screen(self, with_prompts, previous, monkeypatch):
        """A full-screen transcript starts blank; the user must see what the model remembers."""
        _spy_services(monkeypatch)
        bridge = ReplBridge()
        tui = Recorder(bridge, ["Продолжить прошлую сессию"])
        bridge.set_event_sink(tui)
        assert tui.wait_for(lambda ev: any(e.get("type") == "history" for e in ev))
        history = next(e for e in tui.events if e.get("type") == "history")
        blocks = [(i["role"], i.get("text")) for i in history["items"] if i["role"] != "note"]
        assert blocks == [("user", "почини меню"), ("assistant", "Готово, меню починено.")]
        assert history["replace"] is False             # the start-up screen keeps its banner

    def test_a_clean_start_restores_nothing_and_clears_the_memory(self, with_prompts, previous, monkeypatch):
        _spy_services(monkeypatch)
        cleared = []
        monkeypatch.setattr("memory_manager.memory.clear", lambda: cleared.append(True))
        monkeypatch.setattr(startup, "carried_over_memory", lambda: ["1 фактов"])
        bridge = ReplBridge()
        before = list(bridge.agent.messages)
        tui = Recorder(bridge, ["Начать с чистого листа (забыть всё выше)"])
        bridge.set_event_sink(tui)
        assert tui.wait_for(lambda ev: any("Рабочая память очищена" in e.get("text", "") for e in ev))
        assert bridge.agent.messages == before
        assert cleared == [True]

    def test_dismissing_the_question_starts_clean_and_still_starts_the_servers(
            self, with_prompts, previous, monkeypatch):
        order = _spy_services(monkeypatch)
        bridge = ReplBridge()
        before = list(bridge.agent.messages)
        tui = Recorder(bridge, [""])                # Esc
        bridge.set_event_sink(tui)
        assert tui.wait_for(lambda ev: any("MCP:" in e.get("text", "") for e in ev))
        assert bridge.agent.messages == before
        assert order == ["mcp"]


class TestNothingToOffer:
    def test_with_no_previous_session_there_is_no_question(self, with_prompts, monkeypatch, tmp_path):
        order = _spy_services(monkeypatch)
        monkeypatch.setattr(startup, "get_last_session", lambda: None)
        monkeypatch.setattr(startup, "carried_over_memory", lambda: [])
        (tmp_path / "exists.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr("config.CONFIG_FILE", tmp_path / "exists.json")
        bridge = ReplBridge()
        tui = Recorder(bridge, [])
        bridge.set_event_sink(tui)
        assert tui.wait_for(lambda ev: any("MCP:" in e.get("text", "") for e in ev))
        assert tui.questions == []
        assert order == ["mcp"]

    def test_scripted_runs_skip_the_questions_entirely(self, monkeypatch, previous):
        """ARGENT_NO_STARTUP_PROMPTS=1: nobody is there to answer."""
        monkeypatch.setenv("ARGENT_NO_STARTUP_PROMPTS", "1")
        order = _spy_services(monkeypatch)
        bridge = ReplBridge()
        tui = Recorder(bridge, [])
        bridge.set_event_sink(tui)
        assert tui.wait_for(lambda ev: any("MCP:" in e.get("text", "") for e in ev))
        assert tui.questions == []


class TestFirstRun:
    def test_a_first_run_offers_the_setup_walkthrough(self, with_prompts, monkeypatch, tmp_path):
        _spy_services(monkeypatch)
        monkeypatch.setattr(startup, "get_last_session", lambda: None)
        monkeypatch.setattr(startup, "carried_over_memory", lambda: [])
        monkeypatch.setattr("config.CONFIG_FILE", tmp_path / "never_written.json")
        bridge = ReplBridge()
        tui = Recorder(bridge, [""])               # decline
        bridge.set_event_sink(tui)
        assert tui.wait_for(lambda ev: any("MCP:" in e.get("text", "") for e in ev))
        assert tui.questions and tui.questions[0]["title"] == "Настроить сейчас?"
        assert any("первый запуск" in t.lower() for t in tui.texts())
