"""The TUI's turn snapshot is the REPL's snapshot, and it feeds the session totals.

Regression: the bridge accumulated usage through `src.cli.usage_tracker`, a
module that does not exist. The import failure was swallowed, so the running
"sum … tok · $…" never moved in the TUI — and nothing said so.
"""
import io

import pytest
from rich.console import Console

import ui
from src.bridge.session import AgentSession
from src.cli import cli_ui
from usage_tracker import usage as session_usage


class FakeAgent:
    _turn_tool_calls = 2
    _turn_local_agent_calls = 0

    def __init__(self, percent=10.0):
        self._percent = percent

    def get_context_usage(self):
        return {"tokens": 3100, "max": 32000, "percent": self._percent}

    def get_context_breakdown(self):
        return {"system": 900, "tools": 400, "history": 1800}


@pytest.fixture(autouse=True)
def _fresh_totals(monkeypatch):
    # The user's real config may select 'verbose'; the tests state their mode.
    import config
    monkeypatch.setattr(config, "get_turn_snapshot_mode", lambda: "compact")
    session_usage.reset()
    yield
    session_usage.reset()


USAGE = {"prompt": 1200, "completion": 340, "cached": 900, "cost": 0.0021}


def test_compact_snapshot_line_matches_the_repl_format():
    lines = cli_ui.turn_snapshot_lines(
        FakeAgent(), 8.2, timing={"first": 0.3, "tools": 1.0}, usage=USAGE,
        tool_calls=3, mode="compact")
    assert lines == [
        "  [dim]┄ 8.2s · 1.2k→340 tok · cached 900 (75%) · 3 tools · "
        "ctx 3.1k/32k (10%) · $0.0021[/dim]"
    ]


def test_snapshot_lines_are_pure():
    session_usage.reset()
    cli_ui.turn_snapshot_lines(FakeAgent(), 1.0, usage=USAGE, mode="compact")
    assert session_usage.requests == 0  # building the lines records nothing


def test_off_mode_is_the_minimal_line():
    lines = cli_ui.turn_snapshot_lines(FakeAgent(), 8.2, usage={}, mode="off")
    assert len(lines) == 1 and lines[0].startswith("[dim](⏱ 8.2s")


def test_verbose_mode_is_multiline():
    lines = cli_ui.turn_snapshot_lines(
        FakeAgent(), 8.2, timing={"first": 0.5, "tools": 2.0}, usage=USAGE,
        tool_calls=3, mode="verbose")
    assert len(lines) >= 3 and all(l.startswith("  [dim]┄ ") for l in lines)


def test_context_warning_over_85_percent():
    lines = cli_ui.turn_snapshot_lines(FakeAgent(92.0), 1.0, usage={}, mode="compact")
    assert any("Контекст заполнен на 92%" in l for l in lines)


def test_repl_printer_prints_exactly_those_lines(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(ui, "console", Console(file=buf, width=120, color_system=None,
                                               force_terminal=False))
    agent = FakeAgent()
    cli_ui._print_turn_snapshot(agent, 8.2, timing={"first": 0.3, "tools": 1.0},
                                usage=USAGE, tool_calls=3)
    printed = buf.getvalue()
    assert "┄ 8.2s · 1.2k→340 tok" in printed
    assert printed.endswith("\n\n")
    assert session_usage.requests == 1  # the REPL path records the turn


def test_session_snapshot_records_usage_once_and_returns_markup():
    sess = AgentSession(agent=FakeAgent())
    snap = {"total_s": 8.2, "ttft_s": 0.3, "tool_s": 1.0, "tool_calls": 3}
    out = sess._snapshot_markup(snap, dict(USAGE))
    assert session_usage.requests == 1
    assert session_usage.prompt == 1200 and session_usage.completion == 340
    assert session_usage.cost == pytest.approx(0.0021)
    assert out["lines"][0].startswith("  [dim]┄ 8.2s")
    assert out["context_line"] == ui.context_usage_markup(3100, 32000, 10.0)


def test_second_turn_shows_the_running_session_total():
    sess = AgentSession(agent=FakeAgent())
    snap = {"total_s": 1.0, "ttft_s": 0.1, "tool_s": 0.0, "tool_calls": 0}
    sess._snapshot_markup(snap, dict(USAGE))
    out = sess._snapshot_markup(snap, dict(USAGE))
    assert session_usage.requests == 2
    assert "sum 3.1k" in out["lines"][0]  # (1200+340)*2


def test_snapshot_without_usage_still_renders_and_records_nothing():
    sess = AgentSession(agent=FakeAgent())
    out = sess._snapshot_markup({"total_s": 2.0, "tool_calls": 0}, {})
    assert session_usage.requests == 0
    assert out["lines"] and "2.0s" in out["lines"][0]
