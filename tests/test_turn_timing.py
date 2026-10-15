"""Per-turn timing observer: splits model time from tool time on the chunk stream."""

import src.cli.cli_ui as cli_ui
from src.cli.cli_ui import _time_observe


def test_passes_chunks_unchanged_and_times(monkeypatch):
    # Controlled clock: each time.time() call returns the next value.
    clock = iter([100.5, 101.0, 105.0, 105.2])
    monkeypatch.setattr(cli_ui.time, "time", lambda: next(clock))

    timing = {"first": None, "tools": 0.0, "_ts": None}
    chunks = [
        {"type": "content", "content": "a"},   # first-chunk clock 100.5 -> first=0.5
        {"type": "tool_start", "name": "t"},    # clock 101.0 -> _ts
        {"type": "tool_end", "name": "t"},       # clock 105.0 -> tools += 4.0
        {"type": "content", "content": "b"},     # no clock read (first set, not a tool)
    ]
    out = list(_time_observe(iter(chunks), timing, start_time=100.0))

    assert out == chunks                 # stream is untouched
    assert timing["first"] == 0.5        # prefill/TTFT
    assert timing["tools"] == 4.0        # measured tool span


def test_no_tools_leaves_tool_time_zero(monkeypatch):
    monkeypatch.setattr(cli_ui.time, "time", lambda: 50.0)
    timing = {"first": None, "tools": 0.0, "_ts": None}
    list(_time_observe(iter([{"type": "content", "content": "x"}]), timing, start_time=49.0))
    assert timing["first"] == 1.0 and timing["tools"] == 0.0


def test_unpaired_tool_end_ignored(monkeypatch):
    monkeypatch.setattr(cli_ui.time, "time", lambda: 10.0)
    timing = {"first": None, "tools": 0.0, "_ts": None}
    # tool_end with no preceding tool_start must not crash or add time.
    list(_time_observe(iter([{"type": "tool_end", "name": "t"}]), timing, start_time=10.0))
    assert timing["tools"] == 0.0


def test_format_time_compact():
    from src.cli.cli_ui import _format_time_compact
    assert _format_time_compact(5.2) == "0:05"
    assert _format_time_compact(65) == "1:05"
    assert _format_time_compact(3665) == "1:01:05"


def test_format_time_human():
    from src.cli.cli_ui import _format_time_human
    assert _format_time_human(8.2) == "8.2s"
    assert _format_time_human(65) == "1m 5s"
    assert _format_time_human(3720) == "1h 2m"


def test_waiting_display_renders_live_timer():
    from src.cli.cli_ui import _WaitingDisplay
    from rich.console import Console
    d = _WaitingDisplay(start_turn_time=100.0)
    # Mock current time
    import src.cli.cli_ui as cli_ui
    orig_time = cli_ui.time.time
    try:
        cli_ui.time.time = lambda: 108.0
        items = list(d.__rich_console__(Console(width=80), None))
        assert len(items) == 1
        text_str = str(items[0].text)
        assert "Ожидание ответа..." in text_str
        assert "0:08" in text_str
    finally:
        cli_ui.time.time = orig_time

