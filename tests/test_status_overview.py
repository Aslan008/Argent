"""/status: one screen over every subsystem; reads only, and one broken subsystem must not hide the rest."""

import pytest

from src.cli import status_overview as so


def test_collect_returns_a_row_per_section_and_never_raises():
    rows = so.collect()
    assert len(rows) == len(so.SECTIONS)
    for name, level, text in rows:
        assert level in (so.OK, so.OFF, so.WARN, so.ATTENTION)
        assert name and text


def test_a_broken_section_is_reported_not_raised(monkeypatch):
    def boom():
        raise RuntimeError("kaput")
    monkeypatch.setattr(so, "SECTIONS", [so._todos, boom])
    rows = so.collect()
    assert rows[0][0] == "План дел"
    assert rows[1][1] == so.WARN and "kaput" in rows[1][2]


def test_what_waits_for_the_person_is_named_last():
    rows = [("A", so.OK, "x"), ("Комнаты", so.ATTENTION, "есть прогон")]
    lines = so.render(rows)
    assert "Ждёт вас" in lines[-1] and "Комнаты" in lines[-1]


def test_markup_in_a_value_is_not_interpreted():
    lines = so.render([("MCP", so.OK, "server [bold]x[/bold]")])
    assert "\\[bold]" in lines[0]            # escaped for rich


def test_desktop_on_with_no_allowed_apps_says_nothing_will_work(monkeypatch):
    from src.automation import desktop_guard
    monkeypatch.setattr(desktop_guard, "enabled", lambda: True)
    monkeypatch.setattr(desktop_guard, "allowed_apps", lambda: [])
    name, level, text = so._desktop()
    assert level == so.ATTENTION and "пуст" in text


def test_unfinished_room_run_is_flagged(monkeypatch, tmp_path):
    from src.rooms import session
    from src.rooms.journal import Journal
    j = Journal(tmp_path / "j.jsonl")
    j.append({"kind": "intent", "node": "a"})
    monkeypatch.setattr(session, "default_journal", lambda: j)
    name, level, text = so._rooms()
    assert level == so.ATTENTION and "resume" in text


def test_status_is_a_registered_command(capsys):
    from src.cli.commands import registry
    from src.cli.commands.base import CommandContext
    res = registry.dispatch("/status", CommandContext(agent=None))
    assert res.handled and res.should_continue
