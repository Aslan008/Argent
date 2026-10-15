"""The word `ultracode` in a message switches the mode on for the session, and only when it is the word."""

import pytest

from src.agent import plan_mode
from src.agent.workflow import runtime, trigger


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    runtime.set_enabled(False)
    runtime.keyword_turn(False)
    monkeypatch.setattr("config.is_ultracode_keyword_enabled", lambda: True)
    monkeypatch.setattr(runtime, "say", lambda text: None)
    yield
    runtime.set_enabled(False)
    runtime.keyword_turn(False)


@pytest.mark.parametrize("text", [
    "ultracode: вынеси авторизацию в отдельный модуль и прогони тесты",
    "Сделай рефакторинг трёх модулей, ultracode",
    "Ultracode. Обнови зависимости и проверь сборку",
    "включи ULTRACODE для этой задачи",
])
def test_the_word_is_found(text):
    assert trigger.mentions(text)


@pytest.mark.parametrize("text", [
    "",
    "/ultracode on",
    "открой файл src/agent/ultracode.py",
    "посмотри `ultracode` в коде",
    "```\nultracode = 1\n```",
    "смотри https://example.com/ultracode/page",
    "переменная my_ultracode_flag",
    "ultracodes",
])
def test_other_uses_are_not_a_trigger(text):
    assert not trigger.mentions(text)


def test_a_message_switches_the_mode_on_and_the_model_is_told():
    assert trigger.on_user_message("ultracode: перепиши модуль и проверь тестами") is True
    assert runtime.enabled() and runtime.keyword_turn()
    assert "names Ultracode" in runtime.instructions()


def test_the_note_is_for_that_message_only():
    trigger.on_user_message("ultracode: перепиши модуль")
    trigger.on_user_message("спасибо, а теперь расскажи, как это работает")
    assert runtime.enabled(), "the mode itself stays on for the session"
    assert not runtime.keyword_turn()
    assert "names Ultracode" not in runtime.instructions()


def test_the_setting_turns_the_trigger_off(monkeypatch):
    monkeypatch.setattr("config.is_ultracode_keyword_enabled", lambda: False)
    assert trigger.on_user_message("ultracode: перепиши модуль") is False
    assert not runtime.enabled()


def test_plan_mode_does_not_get_it():
    plan_mode.PLAN.enter()
    try:
        assert trigger.on_user_message("ultracode: перепиши модуль") is False
        assert not runtime.enabled()
    finally:
        plan_mode.PLAN.leave()


def test_the_goal_without_the_keyword():
    assert trigger.strip("ultracode: перепиши модуль и проверь тесты") == "перепиши модуль и проверь тесты"
    assert trigger.strip("без ключевого слова") is None
