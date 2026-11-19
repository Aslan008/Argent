"""Правка файла показывается один раз, а не дважды.

TUI рисует карточку diff из собственного события (`MessageItem::Diff`), а Python
дополнительно печатал в stdout панель «Changes in …» — та же правка выглядела как
два поля подряд. Панель остаётся только для обычного REPL: там карточки нет, и она
единственный показ.
"""

from argent.tools import _helpers as helpers


def _run(monkeypatch, tui: bool):
    printed = []
    monkeypatch.setattr(helpers.console, "print", lambda *a, **k: printed.append(a))
    monkeypatch.setattr(helpers, "Syntax", lambda *a, **k: "syntax")
    monkeypatch.setattr(helpers, "Panel", lambda *a, **k: "panel")
    if tui:
        monkeypatch.setenv("ARGENT_TUI", "1")
    else:
        monkeypatch.delenv("ARGENT_TUI", raising=False)

    helpers.drain_diff_events()
    helpers._print_diff("a\nb\n", "a\nc\n", "x.py")
    return printed, helpers.drain_diff_events()


def test_the_tui_gets_the_diff_once_in_its_own_card(monkeypatch):
    printed, events = _run(monkeypatch, tui=True)

    assert printed == [], "в TUI панель «Changes in …» дублировала бы карточку"
    assert events, "событие diff должно уйти в интерфейс"
    assert "x.py" in str(events[0])


def test_the_plain_repl_keeps_the_panel(monkeypatch):
    printed, events = _run(monkeypatch, tui=False)

    assert printed, "в REPL панель — единственный показ правки"
    assert events, "и событие тоже остаётся: его читает тот же агент"


def test_an_unchanged_file_produces_nothing(monkeypatch):
    printed = []
    monkeypatch.setattr(helpers.console, "print", lambda *a, **k: printed.append(a))
    helpers.drain_diff_events()

    helpers._print_diff("same\n", "same\n", "x.py")
    assert printed == []
    assert helpers.drain_diff_events() == []
