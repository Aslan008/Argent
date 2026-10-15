"""The premium look of an answer's markdown, for the TUI (``ui_style = premium``).

The REPL prints an answer's markdown with Rich's defaults: a centred underlined
heading, a bare ``── python ──`` rule over code, hyphens for a divider. That is
faithful and readable, and the classic look keeps it exactly. The premium look
dresses the same markdown as designed blocks — and only draws them differently:
the text, the order and the line breaks are the same, so nothing is lost by
switching.

* ``#`` / ``##`` / ``###`` get an accent bar and, for the first two, a rule under
  them; ``####`` and below are quiet bold;
* a fenced block is a card: a language label over syntax-highlighted code on a
  tinted panel, with a blank row of padding under it;
* a quote has an accent bar, a divider is a thin rule, a table has rounded
  corners and an accent header;
* ``- [x]`` / ``- [ ]`` become ``☑`` / ``☐``;
* inline code is a chip: its own colour on a slightly lighter background.

Colours follow the same palette as the native widgets in the Rust frontend
(``theme::Palette``), with the accent taken from the user's ``theme.yaml``.
"""

from __future__ import annotations

import re
from typing import List

from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import BlockQuote, Heading, HorizontalRule, Markdown, TableElement
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

import ui

TEXT = "#e6edf3"
MUTED = "#8b949e"
BORDER = "#30363d"
SURFACE = "#161b22"
CHIP = "#1c2128"
CODE = "#79c0ff"


def _accent() -> str:
    """The theme's prompt colour, or the shipped blue when it is not a hex value."""
    for word in reversed(str(ui.c.get("user_prompt_symbol", "")).split()):
        if re.fullmatch(r"#[0-9a-fA-F]{6}", word):
            return word
    return "#58a6ff"


ACCENT = _accent()

THEME = Theme({
    "markdown.h1": f"bold {TEXT}",
    "markdown.h2": f"bold {TEXT}",
    "markdown.h3": f"bold {ACCENT}",
    "markdown.h4": f"bold {MUTED}",
    "markdown.h5": f"bold {MUTED}",
    "markdown.h6": f"italic {MUTED}",
    "markdown.code": f"{CODE} on {CHIP}",
    "markdown.link": f"underline {ACCENT}",
    "markdown.link_url": f"underline {ACCENT}",
    "markdown.block_quote": f"italic {MUTED}",
    "markdown.item.bullet": f"bold {ACCENT}",
    "markdown.item.number": f"bold {ACCENT}",
    "markdown.hr": BORDER,
    "markdown.table.header": f"bold {ACCENT}",
    "markdown.table.border": BORDER,
})


def _rule_width(options: ConsoleOptions) -> int:
    return max(8, min(options.max_width, int(ui.s.get("content_width", 110))))


class _Heading(Heading):
    """A heading with an accent bar instead of centring and underlining."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        level = int(self.tag[1:]) if self.tag[1:].isdigit() else 3
        body = self.text.copy()
        body.justify = "left"
        # Markdown already puts a blank row between blocks; none is added here.
        if level <= 2:
            yield Text.assemble(("▌ " if level == 1 else "▎ ", f"bold {ACCENT}"), body)
            yield Text(("━" if level == 1 else "─") * _rule_width(options), style=BORDER)
        elif level == 3:
            yield Text.assemble(("› ", f"bold {ACCENT}"), body)
        else:
            yield Text.assemble(("  ", ""), body)


class _Quote(BlockQuote):
    """A quote: an accent bar down the left, the text muted and italic."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        render_options = options.update(width=options.max_width - 3)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        bar = Segment("▎ ", Style(color=ACCENT))
        new_line = Segment("\n")
        for line in lines:
            yield bar
            yield from line
            yield new_line


class _Rule(HorizontalRule):
    """A divider: a thin rule, not a row of hyphens."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield Text("─" * _rule_width(options), style=console.get_style("markdown.hr", default="none"))
        yield Text()


class _Table(TableElement):
    """A table with rounded corners; the header takes the accent."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        table = Table(
            box=box.ROUNDED,
            pad_edge=False,
            style="markdown.table.border",
            show_edge=True,
            collapse_padding=False,
        )
        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                heading = column.content.copy()
                heading.stylize("markdown.table.header")
                table.add_column(heading)
        if self.body is not None:
            for row in self.body.rows:
                table.add_row(*[element.content for element in row.cells])
        yield table


class PremiumMarkdown(Markdown):
    """Rich's Markdown with the premium blocks swapped in."""

    elements = {
        **Markdown.elements,
        "heading_open": _Heading,
        "blockquote_open": _Quote,
        "hr": _Rule,
        "table_open": _Table,
    }


class CodeCard:
    """A fenced block as a card: the language, then the highlighted code on a panel."""

    def __init__(self, lang: str, code: str):
        self.lang = lang or "text"
        self.code = code

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width
        label = Text.assemble((" ▍", f"bold {ACCENT}"), (f" {self.lang}", f"bold {MUTED}"))
        label.pad_right(max(0, width - label.cell_len))
        label.stylize(f"on {SURFACE}")
        yield label
        yield Syntax(
            self.code,
            self.lang,
            theme=ui.s["syntax_theme"],
            line_numbers=False,
            word_wrap=True,
            background_color=SURFACE,
            padding=(0, 2),
        )
        yield Text(" " * width, style=f"on {SURFACE}")


# The language is the first word of the info string: ``c++``, ``c#`` and ``objective-c``
# are all real, and whatever follows it (``python title=x``) stays out of the label.
_FENCE = re.compile(r"```([\w+#.-]*)[^\n]*\n(.*?)```", re.DOTALL)
_TASK = re.compile(r"(?m)^(\s*(?:[-*+]|\d+[.)])\s+)\[( |x|X)\](?=\s)")


def _task_boxes(text: str) -> str:
    """``- [x] done`` -> ``- ☑ done``; Rich's markdown has no task lists of its own."""
    return _TASK.sub(lambda m: m.group(1) + ("☐" if m.group(2) == " " else "☑"), text)


def renderables(md: str) -> List[object]:
    """What to print for one committed chunk of an answer: prose, and a card per fence."""
    md = md.strip("\n")
    out: List[object] = []
    last = 0
    for match in _FENCE.finditer(md):
        before = md[last:match.start()]
        if before.strip():
            out.append(PremiumMarkdown(_task_boxes(before.strip()), code_theme=ui.s["syntax_theme"]))
        out.append(CodeCard(match.group(1) or "text", match.group(2).rstrip("\n")))
        last = match.end()
    rest = md[last:]
    if rest.strip():
        out.append(PremiumMarkdown(_task_boxes(rest.strip()), code_theme=ui.s["syntax_theme"]))
    return out


# Rich starts some blocks (a list, a table) with a blank row of their own. Every chunk
# already ends with one, so printed back to back they leave two; the premium look
# keeps the single row between blocks.
_LEADING_BLANK_ROWS = re.compile(r"\A(?: *\n)+")
_TRAILING_BLANK_ROWS = re.compile(r"\n(?: *\n)*\Z")


def print_committed(md: str, out: Console) -> None:
    """The premium counterpart of ``ui.print_committed_markdown``."""
    if not md.strip("\n"):
        return
    with out.capture() as captured:
        with out.use_theme(THEME):
            for element in renderables(md):
                ui.safe_print(element, out=out)
    body = _LEADING_BLANK_ROWS.sub("", captured.get(), count=1)
    # A divider asks for a blank row of its own; the one printed below is enough.
    out.file.write(_TRAILING_BLANK_ROWS.sub("\n", body, count=1))
    ui.safe_print("", out=out)


__all__ = ["CodeCard", "PremiumMarkdown", "renderables", "print_committed", "THEME", "ACCENT"]
