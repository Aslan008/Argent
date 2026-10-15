"""The premium look of an answer's markdown, and the setting that turns it on.

Premium dresses the same markdown as designed blocks; it must never change what the
text says. The classic look, which the terminal REPL and the Rust parity tests are
held to, must stay exactly what it was.
"""

import io
import re

import pytest
from rich.console import Console

import config
import ui
from src.bridge import premium_markdown, render

ANSI = re.compile(r"\x1b\[[0-9;:]*m|\x1b\]8;[^\x1b]*\x1b\\")

SAMPLE = """# Заголовок

Абзац с **жирным**, *курсивом* и `inline_code`.

## Второй уровень
### Третий уровень
#### Четвёртый
> Цитата

- один
- два
1. раз
2. два

---

| Имя | Значение |
|-----|----------|
| a   | 1        |

```python
def f(x):
    return x * 2
```

- [x] сделано
- [ ] не сделано
"""


def plain(ansi: str) -> str:
    return ANSI.sub("", ansi)


def premium(text=SAMPLE, width=80) -> str:
    return render.render_block("markdown", {"text": text, "style": "premium"}, width)


def classic(text=SAMPLE, width=80) -> str:
    return render.render_block("markdown", {"text": text}, width)


class TestHeadings:
    def test_levels_are_told_apart(self):
        rows = plain(premium()).split("\n")
        assert any(r.startswith("▌ Заголовок") for r in rows), "h1: accent bar"
        assert any(r.startswith("▎ Второй уровень") for r in rows), "h2: slimmer bar"
        assert any(r.startswith("› Третий уровень") for r in rows), "h3: a chevron"
        assert any(r.strip() == "Четвёртый" for r in rows), "h4 and below: quiet bold, no mark"

    def test_the_first_two_levels_get_a_rule_under_them(self):
        rows = plain(premium()).split("\n")
        h1 = next(i for i, r in enumerate(rows) if r.startswith("▌ Заголовок"))
        h2 = next(i for i, r in enumerate(rows) if r.startswith("▎ Второй"))
        assert set(rows[h1 + 1]) == {"━"}
        assert set(rows[h2 + 1]) == {"─"}

    def test_a_heading_is_not_centred_any_more(self):
        first = plain(premium("# Центр")).split("\n")[0]
        assert first.startswith("▌ Центр")

    def test_the_accent_comes_from_the_theme(self):
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", premium_markdown.ACCENT)
        r, g, b = (int(premium_markdown.ACCENT[i:i + 2], 16) for i in (1, 3, 5))
        assert f"38;2;{r};{g};{b}" in premium("# X")


class TestBlocks:
    def test_a_fenced_block_is_a_card_with_its_language(self):
        out = premium()
        rows = plain(out).split("\n")
        label = next(i for i, r in enumerate(rows) if "▍ python" in r)
        assert "def f(x):" in rows[label + 1]
        assert "return x * 2" in rows[label + 2]
        assert "48;2;22;27;34" in out, "the card is a tinted panel"

    def test_every_card_row_spans_the_whole_width(self):
        rows = plain(premium(width=60)).split("\n")
        label = next(i for i, r in enumerate(rows) if "▍ python" in r)
        for r in rows[label:label + 4]:
            assert len(r) == 60, repr(r)

    def test_an_unknown_language_still_renders(self):
        out = plain(premium("```zzz-not-a-lang\nsome text\n```"))
        assert "some text" in out and "zzz-not-a-lang" in out

    def test_a_fence_without_a_language_is_text(self):
        assert "▍ text" in plain(premium("```\nplain\n```"))

    def test_a_quote_has_a_bar(self):
        assert any(r.startswith("▎ Цитата") for r in plain(premium()).split("\n"))

    def test_a_divider_is_a_thin_rule(self):
        rows = plain(premium("above\n\n---\n\nbelow")).split("\n")
        assert any(r and set(r) == {"─"} for r in rows)

    def test_a_table_has_rounded_corners_and_all_its_cells(self):
        out = plain(premium())
        assert "╭" in out and "╰" in out
        for cell in ("Имя", "Значение", "a", "1"):
            assert cell in out

    def test_task_lists_become_boxes(self):
        out = plain(premium())
        assert "☑ сделано" in out and "☐ не сделано" in out
        assert "[x]" not in out and "[ ]" not in out

    def test_a_checkbox_inside_a_code_fence_is_left_alone(self):
        out = plain(premium("```\n- [x] literal\n```"))
        assert "- [x] literal" in out

    def test_inline_code_is_a_chip(self):
        assert "48;2;28;33;40" in premium("`chip`"), "its own background"

    def test_links_keep_their_text(self):
        assert "ссылкой" in plain(premium("[ссылкой](https://example.com)"))


class TestNothingIsLost:
    def test_the_words_are_the_same_in_both_looks(self):
        """Premium only draws differently. Take away the decoration and every word
        the classic look prints is still there."""
        words = lambda s: set(re.findall(r"\w+", plain(s)))
        classic_words = words(classic())
        premium_words = words(premium())
        missing = classic_words - premium_words
        # The task boxes replace the literal "x" in [x]; nothing else may go.
        assert missing <= {"x"}, missing

    def test_no_row_is_wider_than_the_console(self):
        for width in (30, 60, 100):
            for row in plain(premium(width=width)).split("\n"):
                assert len(row) <= width, (width, row)

    def test_empty_and_blank_input_print_nothing(self):
        assert premium("") == ""
        assert premium("\n\n") == ""

    def test_plain_prose_still_renders(self):
        assert plain(premium("just a sentence")).strip() == "just a sentence"

    def test_it_survives_hostile_markdown(self):
        for text in ("#" * 30, "```", "| a |\n|---|\n", ">" * 40, "- [x", "```python\nunterminated"):
            premium(text)


class TestTheClassicLookIsUntouched:
    def test_no_style_means_the_original_renderer(self):
        buf = Console(file=io.StringIO(), theme=ui.custom_theme, safe_box=True, width=80,
                      force_terminal=False, color_system="truecolor", legacy_windows=False)
        ui.print_committed_markdown(SAMPLE, out=buf)
        assert classic() == buf.file.getvalue()

    def test_classic_style_is_the_original_renderer_too(self):
        assert render.render_block("markdown", {"text": SAMPLE, "style": "classic"}, 80) == classic()

    def test_the_premium_theme_does_not_leak_into_later_blocks(self):
        before = classic()
        premium()
        assert classic() == before


class TestTheSetting:
    def test_the_default_is_premium(self, monkeypatch):
        monkeypatch.delenv("ARGENT_UI_STYLE", raising=False)
        assert config.get_ui_style() == "premium"

    def test_it_is_saved_and_read_back(self, monkeypatch):
        monkeypatch.delenv("ARGENT_UI_STYLE", raising=False)
        config.set_ui_style("classic")
        assert config.get_ui_style() == "classic"
        config.set_ui_style(" Premium ")
        assert config.get_ui_style() == "premium"

    def test_a_bad_value_is_refused(self):
        with pytest.raises(ValueError):
            config.set_ui_style("neon")

    def test_a_bad_saved_value_falls_back(self, monkeypatch):
        monkeypatch.delenv("ARGENT_UI_STYLE", raising=False)
        config._set("ui_style", "garbage")
        assert config.get_ui_style() == "premium"

    def test_the_environment_wins_for_one_run(self, monkeypatch):
        config.set_ui_style("premium")
        monkeypatch.setenv("ARGENT_UI_STYLE", "classic")
        assert config.get_ui_style() == "classic"

    def test_the_frontend_is_told(self, monkeypatch):
        monkeypatch.delenv("ARGENT_UI_STYLE", raising=False)
        config.set_ui_style("classic")
        assert render.ui_config()["style"] == "classic"


class TestTheCommand:
    @pytest.fixture
    def run(self, monkeypatch):
        from src.cli.commands import dispatch_command
        from src.cli.commands.base import CommandContext

        printed = []
        monkeypatch.setattr("src.cli.commands.config_commands.print_system", lambda t, *a, **k: printed.append(str(t)))
        monkeypatch.setattr("src.cli.commands.config_commands.print_error", lambda t, *a, **k: printed.append(str(t)))
        monkeypatch.delenv("ARGENT_UI_STYLE", raising=False)
        monkeypatch.delenv("ARGENT_TUI", raising=False)

        def _run(text):
            return dispatch_command(text, CommandContext(agent=None, user_input=text)), printed
        return _run

    def test_it_is_offered_on_tab(self):
        from src.cli.commands import completion_catalog
        names = {e["name"] for e in completion_catalog()}
        assert {"/ui", "/ui premium", "/ui classic"} <= names

    def test_without_an_argument_it_says_what_is_set(self, run):
        res, printed = run("/ui")
        assert res.handled
        assert any("premium" in p for p in printed)

    def test_it_switches_and_saves(self, run):
        res, printed = run("/ui classic")
        assert res.handled
        assert config.get_ui_style() == "classic"

    def test_an_unknown_style_changes_nothing(self, run):
        config.set_ui_style("premium")
        _, printed = run("/ui neon")
        assert config.get_ui_style() == "premium"
        assert any("Неизвестный стиль" in p for p in printed)

    def test_the_running_tui_is_told_at_once(self, run, monkeypatch):
        from src.bridge.repl_bridge import ReplBridge
        seen = []
        bridge = ReplBridge()
        bridge.set_event_sink(seen.append)
        monkeypatch.setenv("ARGENT_TUI", "1")
        run("/ui classic")
        assert {"type": "ui_style", "style": "classic"} in seen


class TestSpacing:
    """Chunks are printed back to back; the gap between blocks is one blank row."""

    CHUNKS = ["para", "- a\n- b", "> quote", "```py\nx = 1\n```", "| a | b |\n|---|---|\n| 1 | 2 |",
              "# H", "1. one\n2. two", "---"]

    @pytest.mark.parametrize("text", CHUNKS)
    def test_no_chunk_starts_with_a_blank_row(self, text):
        first = plain(premium(text, width=40)).split("\n")[0]
        assert first.strip(), f"{text!r} starts with a blank row"

    @pytest.mark.parametrize("text", CHUNKS)
    def test_every_chunk_ends_with_exactly_one_blank_row(self, text):
        # Measured on the raw rows: a code card's last row is tinted padding, which is
        # blank to the eye but not an empty row. [..., last row, the one blank row, ""]
        rows = premium(text, width=40).split("\n")
        assert rows[-1] == "" and rows[-2] == "" and rows[-3] != "", f"{text!r}: {rows[-4:]!r}"

    def test_two_chunks_leave_one_blank_row_between_them(self):
        both = plain(premium("- a\n- b", width=40) + premium("> q", width=40)).split("\n")
        i = next(k for k, r in enumerate(both) if "• b" in r)
        assert both[i + 1] == "" and "▎ q" in both[i + 2]
