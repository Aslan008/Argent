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
