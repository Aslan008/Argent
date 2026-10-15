"""Render the REPL's visual blocks to ANSI text for the TUI frontend.

Why this exists: the REPL's look is defined by Rich (`ui.py`) and Pygments. A
second, hand-written renderer in another language can only approximate that —
and silently drifts the moment `ui.py` or `theme.yaml` changes. So the TUI does
not re-implement any block: it asks for it here, and this module prints the very
same `ui.print_*` functions into a capture console of the requested width.

Nothing in this module swaps a global. Every call gets its own `Console`, so a
render on the UI thread can never misroute a print from an agent thread.

The output is plain ANSI (SGR only: truecolor, bold/dim/italic/underline/reverse),
one `\\n` per row, already wrapped to `width`. The frontend must therefore draw
it at exactly that width and ask again when the width changes.
"""

import io
from typing import Any, Callable, Dict

from rich.console import Console

import ui

# A block is never taller than this when measured by a Console (only Live
# displays look at height; nothing here does). Fixed so output is reproducible.
_CAPTURE_HEIGHT = 40
_MIN_WIDTH = 20
_MAX_WIDTH = 500


def make_console(width: int) -> Console:
    """A Console that writes truecolor ANSI into memory at exactly `width` columns.

    `force_terminal=False` + an explicit `color_system` keeps Rich from emitting
    cursor control sequences while still colouring the text; `legacy_windows` is
    off so the Windows console path (which would try to write to a real handle)
    is never taken.
    """
    width = max(_MIN_WIDTH, min(int(width), _MAX_WIDTH))
    return Console(
        file=io.StringIO(),
        theme=ui.custom_theme,
        safe_box=True,
        width=width,
        height=_CAPTURE_HEIGHT,
        force_terminal=False,
        color_system="truecolor",
        legacy_windows=False,
    )


# ── block renderers ─────────────────────────────────────────────────────────
# Each takes (console, data) and prints exactly what the REPL would print.

def _user_markup(out: Console, d: Dict[str, Any]) -> None:
    ui.print_user_prompt(str(d.get("text", "")), out=out)


def _assistant_header(out: Console, d: Dict[str, Any]) -> None:
    ui.print_assistant_header(out=out)


def _markdown(out: Console, d: Dict[str, Any]) -> None:
    text = str(d.get("text", ""))
    if d.get("style") == "premium":
        from src.bridge import premium_markdown
        premium_markdown.print_committed(text, out)
        return
    ui.print_committed_markdown(text, out=out)


def _reasoning(out: Console, d: Dict[str, Any]) -> None:
    ui.print_reasoning(
        str(d.get("text", "")), duration=d.get("duration"), out=out, mode=d.get("mode"),
    )


def _rethink(out: Console, d: Dict[str, Any]) -> None:
    ui.print_rethink(
        str(d.get("text", "")), duration=d.get("duration"), out=out, mode=d.get("mode"),
    )


def _tool_end(out: Console, d: Dict[str, Any]) -> None:
    ui.print_tool_end(
        str(d.get("name", "")),
        str(d.get("result", "")),
        duration=d.get("duration"),
        args=d.get("args") or {},
        out=out,
        expanded=d.get("expanded"),
    )


def _diff(out: Console, d: Dict[str, Any]) -> None:
    ui.print_inline_diff(str(d.get("file", "")), str(d.get("diff", "")), out=out)


def _error(out: Console, d: Dict[str, Any]) -> None:
    ui.print_error(str(d.get("text", "")), out=out)


def _system(out: Console, d: Dict[str, Any]) -> None:
    ui.print_system(str(d.get("text", "")), out=out)


def _checkpoint(out: Console, d: Dict[str, Any]) -> None:
    ui.print_checkpoint(str(d.get("sha", "")), out=out)


def _banner(out: Console, d: Dict[str, Any]) -> None:
    ui.print_banner(str(d.get("version", "?")), out=out)


def _context_usage(out: Console, d: Dict[str, Any]) -> None:
    ui.print_context_usage(
        int(d.get("tokens", 0) or 0), int(d.get("max", 0) or 0),
        float(d.get("percent", 0.0) or 0.0), out=out,
    )


def _approval_header(out: Console, d: Dict[str, Any]) -> None:
    ui.safe_print(ui.approval_header_markup(str(d.get("action", "")),
                                            bool(d.get("destructive"))), out=out)


def _tool_header(out: Console, d: Dict[str, Any]) -> None:
    """The coloured `name: target` header on ONE row, no trailing newline —
    the live tool spinner embeds it in its own line."""
    markup = ui._format_tool_header_markup(str(d.get("name", "")), d.get("args") or {})
    out.print(markup, end="")


def _tool_summary(out: Console, d: Dict[str, Any]) -> None:
    """The one-line outcome of a finished tool (`120 строк · import os`), as plain text.

    The premium look lays a tool call out as a card of its own; it needs the REPL's
    wording for the result (`ui._format_tool_summary`) without the REPL's gutter and
    colours, so the words stay the same whichever look draws them."""
    summary = ui._format_tool_summary(
        str(d.get("name", "")), str(d.get("result", "")), d.get("args") or {}
    )
    out.print(summary, end="", markup=False, highlight=False, soft_wrap=True)


def _markup(out: Console, d: Dict[str, Any]) -> None:
    """Pre-built Rich-markup lines (turn snapshot, interrupt notice, ...)."""
    for line in d.get("lines", []):
        ui.safe_print(str(line), out=out)
    if d.get("blank_after"):
        ui.safe_print("", out=out)


BLOCKS: Dict[str, Callable[[Console, Dict[str, Any]], None]] = {
    "user_markup": _user_markup,
    "assistant_header": _assistant_header,
    "markdown": _markdown,
    "reasoning": _reasoning,
    "rethink": _rethink,
    "tool_end": _tool_end,
    "diff": _diff,
    "error": _error,
    "system": _system,
    "checkpoint": _checkpoint,
    "banner": _banner,
    "context_usage": _context_usage,
    "approval_header": _approval_header,
    "tool_header": _tool_header,
    "tool_summary": _tool_summary,
    "markup": _markup,
}


def render_block(kind: str, data: Dict[str, Any], width: int) -> str:
    """Render one block to ANSI at `width` columns. Raises on an unknown kind."""
    fn = BLOCKS.get(kind)
    if fn is None:
        raise ValueError(f"unknown block kind: {kind!r}")
    out = make_console(width)
    fn(out, data or {})
    return out.file.getvalue()


def ui_config() -> Dict[str, Any]:
    """The look-and-feel settings the frontend's native (live) widgets need.

    Colours come from the same `theme.yaml` the REPL reads, so the TUI follows a
    user's theme instead of hard-coding its own palette.
    """
    cfg: Dict[str, Any] = {
        "colors": dict(ui.c),
        "content_width": int(ui.s.get("content_width", 110)),
    }
    try:
        from config import get_thinking_display_mode, get_show_tool_results
        cfg["thinking_mode"] = get_thinking_display_mode()
        cfg["show_tool_results"] = bool(get_show_tool_results())
    except Exception:
        cfg["thinking_mode"] = "summary"
        cfg["show_tool_results"] = False
    try:
        from config import get_ui_style
        cfg["style"] = get_ui_style()
    except Exception:
        cfg["style"] = "premium"
    return cfg
