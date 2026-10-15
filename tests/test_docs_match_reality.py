"""Documentation must describe the program that exists.

The README listed six commands that had been removed (/enable_rag,
/disable_rag, /rag_provider, /obsidian, /setup_terminal, /undo_all), and
/doctor and the RAG hints pointed users at two of them. Nothing about that is
visible at runtime — you only find out when a user types the command — so it is
pinned here.

What "the program that exists" means is the command registry: it is what
dispatch consults. These tests used to find the commands by running regular
expressions over main.py and command_handler.py for `cmd == "/x"` idioms. When
dispatch moved into the registry those idioms disappeared, the extracted set
shrank to almost nothing, and the guards went on passing or failing for reasons
that had nothing to do with the commands — while /verbose, /results, /debug and
/max_tokens lost their handlers without a single failure. Reading the registry
directly cannot go stale that way.
"""

import re
from pathlib import Path

import pytest

from src.cli.commands import completion_catalog, registry

_ROOT = Path(__file__).resolve().parents[1]


def _primary_commands() -> set:
    return {entry.name for entry in registry._entries}


def _handled_commands() -> set:
    """Every spelling that dispatches: names and aliases."""
    handled = set()
    for entry in registry._entries:
        handled.add(entry.name)
        handled.update(entry.aliases)
    return handled


def _dispatches(text: str) -> bool:
    return any(entry.matches(text) for entry in registry._entries)


def _readme() -> str:
    return (_ROOT / "README.md").read_text(encoding="utf-8")


def _documented_commands() -> set:
    """Any command named in backticks, not only the ones opening a bullet:
    /quit and /stop are documented on the /exit and /jobs lines."""
    return set(re.findall(r"`(/[a-z_]+)", _readme()))


class TestReadme:
    def test_every_documented_command_exists(self):
        ghosts = sorted(_documented_commands() - _handled_commands())
        assert not ghosts, f"README documents commands that do not exist: {ghosts}"

    def test_every_command_is_documented(self):
        """A command the README never mentions is a feature nobody finds."""
        documented = _documented_commands()

        def mentioned(spelling: str) -> bool:     # "/skill import" is found as "/skill"
            return spelling.split()[0] in documented

        missing = sorted(
            entry.name for entry in registry._entries
            if entry.name.startswith("/")
            and not mentioned(entry.name)
            and not any(mentioned(alias) for alias in entry.aliases)
        )
        assert not missing, f"README не упоминает: {missing}"

    def test_the_headline_features_are_documented(self):
        """Features shipped without a mention may as well not exist."""
        readme = _readme()
        for command in ("/rewind", "/vibe", "/tasks", "/search"):
            assert command in readme, f"{command} is undocumented"

    def test_both_language_sections_list_the_new_commands(self):
        readme = _readme()
        for command in ("/rewind", "/vibe", "/tasks", "/search"):
            assert readme.count(f"- `{command}") >= 2, \
                f"{command} is missing from one of the language sections"

    def test_both_language_sections_list_the_same_commands(self):
        """The Russian list used to lag the English one."""
        english, russian = _readme().split("## 🛠 Команды", 1)
        english = english.split("## 🛠 Commands", 1)[1]
        in_english = set(re.findall(r"^- `(/[a-z_]+)", english, re.M))
        in_russian = set(re.findall(r"^- `(/[a-z_]+)", russian, re.M))
        assert in_english == in_russian, (
            f"только в английском: {sorted(in_english - in_russian)}; "
            f"только в русском: {sorted(in_russian - in_english)}")

    def test_it_does_not_advertise_removed_features(self):
        """The desktop GUI (`argent_server.py`, `desktop/`) and the Obsidian
        integration were deleted; a README that still sells them is wrong."""
        readme = _readme()
        for removed in ("argent_server.py", "Tauri", "Obsidian"):
            assert removed not in readme, f"README still mentions the removed {removed}"
        assert not (_ROOT / "argent_server.py").exists()


class TestDiscoverability:
    """A command you cannot find is a feature you do not have.

    /kb — add, index and search external documentation, the thing that makes
    semantic_search useful on a Unity or library codebase — was dispatched but
    appeared in none of the three places a user looks: Tab completion, /help,
    the README. It had been invisible since it shipped.
    """

    def test_every_command_is_offered_on_tab(self):
        offered = {item["name"] for item in completion_catalog()}
        missing = sorted(_handled_commands() - offered)
        assert not missing, f"эти команды работают, но не предлагаются по Tab: {missing}"

    def test_tab_never_offers_something_that_does_not_run(self):
        """Worse than hiding a command: an offered one falls through to the
        model as an ordinary message."""
        ghosts = sorted(item["name"] for item in completion_catalog()
                        if not _dispatches(item["name"]))
        assert not ghosts, f"Tab предлагает несуществующие команды: {ghosts}"

    def test_every_offered_command_describes_itself(self):
        undescribed = sorted(item["name"] for item in completion_catalog()
                             if not item["help"].strip())
        assert not undescribed, f"в подсказке Tab у них пустое описание: {undescribed}"

    def test_every_command_has_a_description_for_help(self):
        """/help prints each entry's help_text."""
        blank = sorted(entry.name for entry in registry._entries
                       if not (entry.help_text or "").strip())
        assert not blank, f"/help показывает пустую строку для: {blank}"

    def test_the_catalog_lists_each_name_once(self):
        names = [item["name"] for item in completion_catalog()]
        assert len(names) == len(set(names))


class TestInProgramHints:
    @pytest.mark.parametrize("source", ["doctor.py", "rag_engine.py", "command_handler.py"])
    def test_hints_do_not_point_at_removed_commands(self, source):
        """A message telling the user to run a command that was deleted is
        worse than no message."""
        text = (_ROOT / source).read_text(encoding="utf-8")
        handled = _handled_commands()
        # Only check commands mentioned in user-facing strings.
        for command in set(re.findall(r"/(?:enable_rag|disable_rag|rag_provider|"
                                      r"obsidian|setup_terminal|undo_all)\b", text)):
            assert command in handled, f"{source} points at removed command {command}"


class TestRestoredCommands:
    """/verbose, /results, /debug and /max_tokens were dispatched by the old
    handler chain, lost in the move to the registry, and kept in the README, the
    Tab menu and the config setters."""

    @pytest.mark.parametrize("command", ["/verbose", "/results", "/debug",
                                         "/max_tokens", "/maxtokens"])
    def test_they_dispatch(self, command):
        assert _dispatches(command)
