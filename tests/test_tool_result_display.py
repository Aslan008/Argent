"""A tool result is written for the model, not for the person watching.

read_file returns the file. grep_search returns every match. Printing all of it
after every call pushed the conversation off the screen — you scrolled past
your own work to find the answer, and a five-step task became a wall.

The compact line keeps what a reader actually wants: which tool ran and the
shape of what came back. /results (or /debug) brings the panel back, and an
error is never collapsed — a failure you cannot see is one you discover three
turns later from the model's behaviour.
"""

import pytest

import ui


@pytest.fixture(autouse=True)
def compact_by_default(monkeypatch):
    monkeypatch.setattr("config.get_debug_mode", lambda: False)
    monkeypatch.setattr("config.get_show_tool_results", lambda: False)


@pytest.fixture
def printed(monkeypatch):
    out = []
    monkeypatch.setattr(ui, "safe_print", lambda x: out.append(x))
    return out


class TestTheCompactLine:
    def test_a_long_result_becomes_one_line(self, printed):
        ui.print_tool_end("read_file", "\n".join(f"line {i}" for i in range(400)))
        assert len(printed) == 1
        assert "read_file" in printed[0] and "400 строк" in printed[0]

    def test_a_short_result_is_shown_as_it_is(self, printed):
        ui.print_tool_end("write_file", "Successfully wrote to 'src/foo.py'.")
        assert "Successfully wrote to 'src/foo.py'." in printed[0]

    def test_an_empty_result_says_so(self, printed):
        ui.print_tool_end("run_command", "   \n\n")
        assert "пусто" in printed[0]

    def test_a_very_long_single_line_is_cut(self, printed):
        ui.print_tool_end("grep_search", "x" * 5_000)
        assert len(printed[0]) < 200

    def test_markup_in_a_result_cannot_break_the_render(self, printed):
        """A result containing [bold] is data, not formatting."""
        ui.print_tool_end("read_file", "config = [bold red]not markup[/]")
        assert "not markup" in printed[0]


class TestWhatIsNeverCollapsed:
    def test_an_error_keeps_its_panel(self, printed):
        ui.print_tool_end("replace_in_file", "Error: The target text appears 3 times.")
        assert len(printed) == 1
        assert not str(printed[0]).startswith("  [dim]✓")     # a Panel, not a line

    def test_debug_mode_keeps_the_panel(self, printed, monkeypatch):
        monkeypatch.setattr("config.get_debug_mode", lambda: True)
        ui.print_tool_end("read_file", "\n".join(str(i) for i in range(50)))
        assert not str(printed[0]).startswith("  [dim]✓")

    def test_the_setting_brings_the_panel_back(self, printed, monkeypatch):
        monkeypatch.setattr("config.get_show_tool_results", lambda: True)
        ui.print_tool_end("read_file", "a\nb\nc")
        assert not str(printed[0]).startswith("  [dim]✓")

    def test_an_interactive_tool_prints_nothing_either_way(self, printed):
        ui.print_tool_end("ask_user_questions", "answered")
        assert printed == []


class TestTheSummary:
    @pytest.mark.parametrize("text,expected", [
        ("one line", "one line"),
        ("", "пусто"),
        ("first\nsecond\nthird", "3 строк"),
    ])
    def test_shapes(self, text, expected):
        assert expected in ui._tool_result_summary(text)

    def test_it_skips_leading_blank_lines_for_the_preview(self):
        assert "real content" in ui._tool_result_summary("\n\nreal content\nmore")
