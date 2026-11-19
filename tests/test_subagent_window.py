"""Субагенты: состояние для списка, лог по требованию и тишина в чате.

Раньше каждая строка воркера печаталась в чат через `print_system`, и рой из
четырёх агентов заливал разговор. Теперь строки копятся у запуска, наружу уходит
краткое состояние, а полный диалог отдаётся по требованию — когда его окно открыли.
"""

import json

import pytest

from src.agent import subagents
from src.agent.workflow import runtime


class _Proc:
    """A child process stand-in: its stdout produces the lines it printed."""

    def __init__(self, lines, returncode=0):
        self.stdout = iter([l.encode("utf-8") + b"\n" for l in lines])
        self.returncode = returncode

    def wait(self, *a, **k):
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        pass


@pytest.fixture
def manager(monkeypatch, tmp_path):
    mgr = subagents.SubagentManager()
    mgr._runs.clear()
    monkeypatch.setattr(subagents, "MANAGER", mgr)
    return mgr


def test_the_snapshot_is_short_and_the_detail_carries_the_log(manager):
    run = subagents.Run(id=3, role="coder", task="чинит парсер", isolation="none", cwd=".")
    manager._runs[3] = run
    run.lines = ["читаю src/parser.py", "запускаю pytest"]

    snap = manager.snapshot()[0]
    assert snap["id"] == 3 and snap["role"] == "coder"
    assert snap["last"] == "запускаю pytest"
    assert "lines" not in snap, "опрос состояния не должен тащить лог"

    detail = manager.detail(3)
    assert detail["lines"] == ["читаю src/parser.py", "запускаю pytest"]
    assert detail["task"] == "чинит парсер"
    assert manager.detail(99) is None


def test_the_lines_of_a_finished_worker_are_kept_in_a_ring(manager):
    run = subagents.Run(id=1, role="coder", task="t", isolation="none", cwd=".")
    manager._runs[1] = run
    run.lines = [f"строка {i}" for i in range(subagents.LOG_LINES_KEPT + 20)]
    # The manager trims on append; the payload must not grow without a bound.
    del run.lines[:-subagents.LOG_LINES_KEPT]
    assert len(manager.detail(1)["lines"]) == subagents.LOG_LINES_KEPT


def test_worker_lines_do_not_go_to_the_chat_in_the_tui(monkeypatch):
    said = []
    monkeypatch.setenv("ARGENT_TUI", "1")
    monkeypatch.setattr(runtime, "_say", lambda text: said.append(text))

    run = subagents.Run(id=3, role="coder", task="t", isolation="none", cwd=".")
    runtime._stream_worker(run, "читаю src/parser.py")
    assert said == [], "в TUI строки воркера живут в его окне, не в чате"


def test_the_plain_repl_still_shows_worker_lines(monkeypatch):
    said = []
    monkeypatch.delenv("ARGENT_TUI", raising=False)
    monkeypatch.setattr(runtime, "_say", lambda text: said.append(text))

    run = subagents.Run(id=3, role="coder", task="t", isolation="none", cwd=".")
    runtime._stream_worker(run, "читаю src/parser.py")
    assert len(said) == 1 and "coder #3" in said[0]


def test_a_worker_report_marker_is_not_streamed(monkeypatch):
    said = []
    monkeypatch.delenv("ARGENT_TUI", raising=False)
    monkeypatch.setattr(runtime, "_say", lambda text: said.append(text))

    run = subagents.Run(id=3, role="coder", task="t", isolation="none", cwd=".")
    runtime._stream_worker(run, "ARGENT_SUBAGENT_RESULT{}")
    assert said == []
