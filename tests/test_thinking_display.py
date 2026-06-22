"""The reasoning ('Анализ') live preview must stay bounded to the screen.

Same root cause as the content duplication: a transient Live that scrolls
off-screen strands a partial copy in the scrollback. Bounding the preview to
the terminal height keeps the transient Live erasable; the full reasoning is
committed once afterwards via print_reasoning.
"""

from rich.console import Console

from src.cli.cli_ui import _ThinkingDisplay


def _body(display: _ThinkingDisplay, console: Console) -> str:
    items = list(display.__rich_console__(console, None))
    texts = [it.plain for it in items if hasattr(it, "plain")]
    # items are: header ("\nАнализ:\n"), the reasoning body, a blank line.
    candidates = [t for t in texts if t.strip() and not t.strip().startswith("Анализ")]
    return candidates[0] if candidates else ""


def test_preview_bounded_to_terminal_height():
    d = _ThinkingDisplay("bold", "dim")
    d.thinking_text = "\n".join(f"reasoning line {i}" for i in range(500))
    console = Console(width=80)
    body = _body(d, console)
    max_lines = max(3, (console.size.height or 24) - 6)
    assert body.count("\n") + 1 <= max_lines


def test_preview_keeps_the_tail_not_the_head():
    d = _ThinkingDisplay("bold", "dim")
    d.thinking_text = "\n".join(f"reasoning line {i}" for i in range(500))
    body = _body(d, Console(width=80))
    assert "reasoning line 499" in body   # most recent kept
    assert "reasoning line 0" not in body  # oldest dropped


def test_short_reasoning_is_shown_whole():
    d = _ThinkingDisplay("bold", "dim")
    d.thinking_text = "one\ntwo\nthree"
    body = _body(d, Console(width=80))
    assert body == "one\ntwo\nthree"
