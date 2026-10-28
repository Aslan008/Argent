"""The two listings people actually read must be grouped, not one long wall.

`/help` printed every command in a single table with a repeated "category" column, and
`/tools` printed 59 tick-boxes in registry order. Both were scannable only if you
already knew what you were looking for. These tests hold the shape: one table per
category in `/help`, a group separator per `TOOL_GROUPS` entry in `/tools`.
"""

import pytest
from questionary.prompts.common import Choice, Separator
from rich.console import Console


class TestHelpIsGrouped:
    def test_each_category_gets_its_own_table(self, monkeypatch):
        import src.cli.commands.config_commands as cc

        recorder = Console(record=True, width=140)
        monkeypatch.setattr(cc, "console", recorder)
        cc.handle_help(None)
        text = recorder.export_text()

        categories = {e.category or "general" for e in cc.registry._entries}
        missing = sorted(c for c in categories if c not in text)
        assert not missing, f"/help shows no section for: {missing}"

    def test_it_no_longer_repeats_a_category_column(self, monkeypatch):
        import src.cli.commands.config_commands as cc

        recorder = Console(record=True, width=140)
        monkeypatch.setattr(cc, "console", recorder)
        cc.handle_help(None)
        assert "Категория" not in recorder.export_text()

    def test_every_command_appears_exactly_once(self, monkeypatch):
        import re

        import src.cli.commands.config_commands as cc

        recorder = Console(record=True, width=140)
        monkeypatch.setattr(cc, "console", recorder)
        cc.handle_help(None)
        text = recorder.export_text()
        for entry in cc.registry._entries:
            # Only the first COLUMN counts: a command's own help text may mention it
            # (`/copy` documents `/copy [code|all]`), and that is not a second row.
            rows = [ln for ln in text.splitlines()
                    if re.match(rf"^\s*[│|]\s*{re.escape(entry.name)}(?![\w/])", ln)]
            assert len(rows) == 1, f"{entry.name} listed {len(rows)} times"


def _tools(choices):
    """The tool names among the prompt's choices.

    `Separator` SUBCLASSES `Choice` (adding only `line`/`title`), so an isinstance
    check on Choice alone counts the group headers as tools.
    """
    return [c.value for c in choices
            if isinstance(c, Choice) and not isinstance(c, Separator)]


class TestToolsAreGrouped:
    @pytest.fixture
    def captured(self, monkeypatch):
        """Answer the prompt with 'cancelled' and keep the choices we were given."""
        import questionary

        import src.cli.commands.agent_commands as ac

        box = {}

        class _Answer:
            def ask(self):
                return None          # Esc: change nothing

        def _checkbox(prompt, choices):
            box["choices"] = choices
            return _Answer()

        monkeypatch.setattr(questionary, "checkbox", _checkbox)
        monkeypatch.setattr(ac, "get_disabled_tools", lambda: [])
        ac.handle_tools(None)
        return box["choices"]

    def test_there_is_a_separator_for_each_group(self, captured):
        from argent.tools import TOOL_GROUPS

        labels = " ".join(str(c.line) for c in captured if isinstance(c, Separator))
        for group, members in TOOL_GROUPS.items():
            if set(members):
                assert group in labels, f"no separator for group '{group}'"

    def test_aliases_are_gone_from_the_api(self, captured):
        from argent.tools.schemas import TOOL_SCHEMAS

        advertised = {s["function"]["name"] for s in TOOL_SCHEMAS}
        offered = set(_tools(captured))
        assert offered == advertised, "the list must be exactly the advertised tools"

    def test_every_advertised_tool_is_offered_once(self, captured):
        from argent.tools import AVAILABLE_TOOLS

        offered = _tools(captured)
        expected = list(AVAILABLE_TOOLS)
        assert sorted(offered) == sorted(expected)

    def test_the_original_order_is_not_the_registry_order(self, captured):
        """Grouping must actually regroup: a list ordered by group differs from the
        registry's own order."""
        from argent.tools import TOOL_GROUPS

        offered = _tools(captured)
        order = {name: i for i, names in enumerate(TOOL_GROUPS.values()) for name in names}
        grouped = [order[n] for n in offered]
        assert grouped == sorted(grouped), "tools are not ordered by group"
