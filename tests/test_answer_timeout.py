"""A question nobody answers must not freeze the turn until somebody comes back.

Before this, `AgentSession._approval_backend` and `ask_select` waited on an Event
with no timeout: leaving the keyboard with a prompt on screen stopped the worker
thread (and with it every later turn) until the person returned.
"""

import time

from src.bridge.session import AgentSession, DEFAULT_ANSWER_TIMEOUT, answer_timeout


def test_answer_timeout_is_configurable_and_forgiving(monkeypatch):
    monkeypatch.delenv("ARGENT_ANSWER_TIMEOUT", raising=False)
    assert answer_timeout() == DEFAULT_ANSWER_TIMEOUT

    monkeypatch.setenv("ARGENT_ANSWER_TIMEOUT", "2.5")
    assert answer_timeout() == 2.5

    # A typo must not turn the gate off or crash the turn.
    monkeypatch.setenv("ARGENT_ANSWER_TIMEOUT", "nonsense")
    assert answer_timeout() == DEFAULT_ANSWER_TIMEOUT


def test_an_unanswered_approval_is_denied_instead_of_waited_on_forever(monkeypatch):
    monkeypatch.setenv("ARGENT_ANSWER_TIMEOUT", "1")
    session = AgentSession()

    started = time.monotonic()
    decision = session._approval_backend("действие на рабочем столе: нажать «ОК»", False, None)
    waited = time.monotonic() - started

    assert decision == "deny"
    assert waited < 10, "the wait must end by itself"

    events = []
    while not session._out.empty():
        events.append(session._out.get_nowait())
    assert any(e.get("type") == "notice" and "не получено" in e.get("text", "") for e in events)
    assert not session._pending, "the unanswered request must not be left behind"


def test_an_unanswered_select_cancels_instead_of_hanging(monkeypatch):
    monkeypatch.setenv("ARGENT_ANSWER_TIMEOUT", "1")
    session = AgentSession()

    started = time.monotonic()
    value = session.ask_select("Выберите вариант", ["a", "b"])
    waited = time.monotonic() - started

    assert value is None
    assert waited < 10
    assert not session._pending
