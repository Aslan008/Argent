"""The TUI draws what the REPL prints, using the REPL's own renderer.

These tests pin the contract of `src.bridge.render`: every block is produced by
the same `ui.py` code the terminal uses, at exactly the requested width, without
touching the process-wide console.
"""
import re

import pytest
from rich.console import Console

import ui
from src.bridge import render
from src.bridge.render import make_console, render_block

SGR = re.compile(r"\x1b\[[0-9;]*m")
OSC8 = re.compile(r"\x1b\]8;[^\x1b]*\x1b\\")


def plain(ansi: str) -> str:
    return SGR.sub("", OSC8.sub("", ansi))


def via_ui(fn, width, *a, **k) -> str:
    """Reference: call the real ui function against a same-shaped console."""
    out = make_console(width)
    fn(*a, out=out, **k)
    return out.file.getvalue()


# ── every block equals the REPL function it wraps ────────────────────────────

def test_assistant_header_is_the_repl_header():
    assert render_block("assistant_header", {}, 80) == via_ui(ui.print_assistant_header, 80)
    assert plain(render_block("assistant_header", {}, 80)) == "✦ Argent\n"


def test_tool_end_compact_matches_the_repl():
    data = {"name": "read_file", "result": "a\nb\nc", "duration": 0.028,
            "args": {"file_path": "agent.py", "offset": 10, "limit": 40}}
    got = render_block("tool_end", data, 100)
    ref = via_ui(ui.print_tool_end, 100, "read_file", "a\nb\nc", duration=0.028,
                 args=data["args"], expanded=False)
    assert got == ref
    text = plain(got)
    assert "│ read_file: agent.py:10-50" in text
    assert "↳ 3 строк · a · 28ms" in text


def test_tool_end_expanded_shows_the_result_panel():
    data = {"name": "read_file", "result": "alpha\nbeta", "duration": 1.4,
            "args": {"file_path": "x.py"}, "expanded": True}
    text = plain(render_block("tool_end", data, 100))
    assert "finished (1.4s)" in text and "alpha" in text and "beta" in text
    assert "╭" in text and "╰" in text


def test_tool_error_is_never_collapsed():
    data = {"name": "read_file", "result": "Error: boom", "duration": 0.004,
            "args": {"file_path": "x.py"}, "expanded": False}
    text = plain(render_block("tool_end", data, 100))
    assert "✗" in text and "failed" in text and "Error: boom" in text


def test_markdown_chunk_is_the_committed_block():
    md = "# Title\n\ntext **bold**\n\n```python\nx = 1\n```"
    got = render_block("markdown", {"text": md}, 90)
    out = make_console(90)
    ui.print_committed_markdown(md, out=out)
    assert got == out.file.getvalue()
    text = plain(got)
    assert "Title" in text and "── python ──" in text and "x = 1" in text
    assert got.endswith("\n\n")  # one blank line closes a committed chunk


def test_empty_markdown_chunk_prints_nothing():
    assert render_block("markdown", {"text": "\n\n"}, 80) == ""


def test_code_is_highlighted_with_the_repl_theme():
    got = render_block("markdown", {"text": "```python\ndef f():\n    pass\n```"}, 80)
    # One Dark keyword colour from the legacy renderer, on the #181a1f code bg.
    assert "38;2;198;120;221" in got and "48;2;24;26;31" in got


def test_reasoning_summary_and_full_modes():
    summary = plain(render_block(
        "reasoning", {"text": "first\nsecond line is the conclusion", "duration": 3.2,
                      "mode": "summary"}, 100))
    assert summary.strip().startswith("◇ Анализ · 3.2s (")
    full = plain(render_block(
        "reasoning", {"text": "first\nsecond", "duration": 3.2, "mode": "full"}, 100))
    assert "◇ Анализ" in full and "first" in full and "second" in full


def test_rethink_block():
    text = plain(render_block("rethink", {"text": "check edge", "duration": 1.1,
                                         "mode": "summary"}, 100))
    assert "◈ Перепроверка · 1.1s" in text


def test_diff_block_counts_changes():
    diff = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new"
    text = plain(render_block("diff", {"file": "src/f.py", "diff": diff}, 100))
    assert "diff f.py (+1, -1)" in text and "-old" in text and "+new" in text


def test_error_has_one_prefix_and_a_hint_line_when_known():
    text = plain(render_block("error", {"text": "boom"}, 100))
    assert text.startswith("Ошибка: boom")
    twice = plain(render_block("error", {"text": "Error: boom"}, 100))
    assert twice.count("Error") == 1 and "Ошибка: Error" not in twice


def test_system_text_is_escaped_unless_it_is_markup():
    assert plain(render_block("system", {"text": "[System: auto-continue]"}, 80)).strip() \
        == "[System: auto-continue]"
    assert plain(render_block("system", {"text": "[dim]quiet[/dim]"}, 80)).strip() == "quiet"


def test_checkpoint_and_banner_and_context_lines():
    assert plain(render_block("checkpoint", {"sha": "abc123"}, 80)).strip() \
        == "◇ Чекпоинт abc123 создан — откат: /rewind"
    banner = plain(render_block("banner", {"version": "9.9"}, 80))
    assert "Argent Coder v9.9" in banner and banner.strip().startswith("─")
    ctx = plain(render_block("context_usage",
                             {"tokens": 3100, "max": 32768, "percent": 9.5}, 80))
    assert ctx.strip() == "Context: 9.5% (3100/32768 tokens)"


def test_markup_lines_with_trailing_blank():
    got = plain(render_block("markup", {"lines": ["[dim]  ┄ 8.2s[/dim]"],
                                        "blank_after": True}, 80))
    assert got == "  ┄ 8.2s\n\n"


# ── contract: width, isolation, errors ──────────────────────────────────────

@pytest.mark.parametrize("width", [20, 40, 80, 120])
def test_no_row_is_wider_than_requested(width):
    md = ("# Heading\n\n" + "word " * 80 + "\n\n- item one\n- item two\n\n"
          "| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nprint('x')\n```")
    for kind, data in (
        ("markdown", {"text": md}),
        ("tool_end", {"name": "run_command", "result": "x\n" * 30,
                      "args": {"command": "echo " + "y" * 200}, "expanded": True}),
        ("diff", {"file": "f.py", "diff": "+" + "z" * 300}),
    ):
        for row in plain(render_block(kind, data, width)).split("\n"):
            assert len(row) <= width, (kind, width, len(row), row[:40])


def test_width_is_clamped_instead_of_crashing():
    assert render_block("assistant_header", {}, 1)   # below the floor
    assert render_block("assistant_header", {}, 10_000)  # above the ceiling


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        render_block("nope", {}, 80)


def test_rendering_does_not_touch_the_global_console(capsys):
    before = ui.console
    before_file = ui.console.file
    render_block("tool_end", {"name": "x", "result": "y", "args": {}}, 80)
    render_block("markdown", {"text": "hello"}, 80)
    assert ui.console is before and ui.console.file is before_file
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_make_console_emits_colour_but_no_cursor_control():
    out = make_console(60)
    out.print("[bold red]x[/bold red]")
    raw = out.file.getvalue()
    assert "\x1b[1;31mx" in raw or "\x1b[1;38;2" in raw
    assert not re.search(r"\x1b\[[0-9;?]*[A-HJKSTfhl]", raw)  # no cursor/erase/mode


def test_ui_config_exposes_the_theme():
    cfg = render.ui_config()
    assert cfg["colors"]["user_prompt_symbol"] == ui.c["user_prompt_symbol"]
    assert cfg["content_width"] == ui.s["content_width"]
    assert cfg["thinking_mode"] in ("summary", "full")


# ── approval header ─────────────────────────────────────────────────────────

def test_approval_header_escapes_markup_in_the_action():
    markup = ui.approval_header_markup("pytest tests/[abc]", False)
    out = Console(width=80, file=__import__("io").StringIO(), color_system=None)
    out.print(markup)  # must not raise MarkupError nor swallow "[abc]"
    assert "pytest tests/[abc]" in out.file.getvalue()


def test_approval_header_flags_destructive_actions():
    assert "▲" in ui.approval_header_markup("rm x", True)
    assert "◇" in ui.approval_header_markup("ls", False)


# ── tool summary (the premium card's second row) ─────────────────────────────

def test_tool_summary_is_the_repls_wording_without_its_gutter():
    data = {"name": "read_file", "result": "a\nb\nc", "args": {"file_path": "agent.py"}}
    got = plain(render_block("tool_summary", data, 100))
    assert got == ui._format_tool_summary("read_file", "a\nb\nc", {"file_path": "agent.py"})
    assert got == "3 строк · a"
    assert "\n" not in got, "one row, no trailing newline"


def test_tool_summary_keeps_brackets_literal():
    # A result such as `[dim]` must not be read as Rich markup.
    got = plain(render_block("tool_summary", {"name": "run_command", "result": "[bold]hi[/bold]"}, 80))
    assert got == "[bold]hi[/bold]"


def test_every_block_kind_the_tui_asks_for_exists():
    for kind in ("tool_header", "tool_summary", "markdown", "reasoning", "rethink", "tool_end", "system"):
        assert kind in render.BLOCKS
