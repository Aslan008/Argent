"""Окно присутствия: vibe спрашивает, только пока человек за клавиатурой.

Раньше vibe отклонял опасное действие сразу. Для ночного прогона это верно, для
прогона под присмотром — нет. Различие делает не режим, а присутствие: любое
действие человека продлевает окно, истекшее окно возвращает авто-отказ.
"""

import time

import pytest

from argent import presence


@pytest.fixture(autouse=True)
def fresh_window(monkeypatch):
    monkeypatch.setenv("ARGENT_PRESENCE_WINDOW", "300")
    monkeypatch.setenv("ARGENT_TUI", "1")
    presence.reset()
    yield
    presence.reset()


def test_a_fresh_session_is_attended():
    assert presence.is_attended()
    assert presence.remaining() > 290


def test_the_window_is_five_minutes_by_default(monkeypatch):
    monkeypatch.delenv("ARGENT_PRESENCE_WINDOW", raising=False)
    assert presence.window() == 300.0


def test_touching_the_window_extends_it():
    presence._last_touch -= 250          # four minutes of silence
    before = presence.remaining()
    assert before < 60

    presence.touch()
    assert presence.remaining() > 290, "действие человека должно вернуть окно"


def test_an_expired_window_is_not_attended(monkeypatch):
    monkeypatch.setenv("ARGENT_TUI", "1")
    monkeypatch.setenv("ARGENT_PRESENCE_WINDOW", "1")
    presence._last_touch -= 5            # nobody touched anything

    assert presence.remaining() == 0.0
    assert not presence.is_attended()
    assert "авто-отказ" in presence.describe()


def test_a_zero_window_never_asks(monkeypatch):
    monkeypatch.setenv("ARGENT_TUI", "1")
    monkeypatch.setenv("ARGENT_PRESENCE_WINDOW", "0")
    presence.reset()
    assert not presence.is_attended()


def test_the_plain_repl_counts_as_attended(monkeypatch):
    """В обычном REPL вопрос задаётся в том же терминале, там человек заведомо есть."""
    monkeypatch.delenv("ARGENT_TUI", raising=False)

    class _Terminal:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", _Terminal())
    assert presence.is_attended()


def test_a_non_interactive_run_is_not_attended(monkeypatch):
    """Скрипт и расписание — не человек: там vibe по-прежнему отказывает без вопроса."""
    monkeypatch.delenv("ARGENT_TUI", raising=False)

    class _Pipe:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", _Pipe())
    assert not presence.is_attended()


def test_the_indicator_says_what_vibe_will_do(monkeypatch):
    monkeypatch.setenv("ARGENT_TUI", "1")
    assert "жду решения" in presence.describe()

    monkeypatch.setenv("ARGENT_PRESENCE_WINDOW", "0")
    presence.reset()
    assert "авто-отказ" in presence.describe()


def test_a_broken_env_value_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("ARGENT_PRESENCE_WINDOW", "не число")
    assert presence.window() == presence.DEFAULT_WINDOW
