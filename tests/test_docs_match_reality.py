"""Documentation must describe the program that exists.

The README listed six commands that had been removed (/enable_rag,
/disable_rag, /rag_provider, /obsidian, /setup_terminal, /undo_all), and
/doctor and the RAG hints pointed users at two of them. Nothing about that is
visible at runtime — you only find out when a user types the command — so it is
pinned here.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _handled_commands() -> set:
    """Slash commands main.py / command_handler.py actually dispatch."""
    src = ""
    for name in ("main.py", "command_handler.py"):
        src += (_ROOT / name).read_text(encoding="utf-8")
    found = set()
    found |= set(re.findall(r'user_input\.strip\(\)\s*==\s*"(/[a-z_]+)"', src))
    # Both `user_input.startswith(...)` and `user_input.strip().startswith(...)`:
    # a command dispatched through the second form is no less real, and missing
    # it made the README look wrong when only the dispatch style had changed.
    # A two-word dispatch (`/skill import`) is a command too; matching only up
    # to the closing quote used to drop it, which made it look undispatched.
    found |= set(re.findall(r'user_input(?:\.strip\(\))?\.startswith\("(/[a-z_]+)', src))
    found |= set(re.findall(r'cmd\s*==\s*"(/[a-z_]+)"', src))
    # Commands with subcommands are dispatched by prefix (e.g. /mcp add …).
    found |= set(re.findall(r'cmd\.startswith\("(/[a-z_]+)"', src))
    for group in re.findall(r'cmd in \(([^)]+)\)', src):
        found |= set(re.findall(r'"(/[a-z_]+)"', group))
    return found


def _documented_commands() -> set:
    """Any command named in backticks, not only the ones opening a bullet:
    /quit and /stop are documented on the /exit and /jobs lines."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    return set(re.findall(r"`(/[a-z_]+)", readme))


def _completer_commands() -> set:
    """What the REPL offers on Tab."""
    src = (_ROOT / "main.py").read_text(encoding="utf-8")
    block = re.search(r"builtin_cmds = \[(.*?)\n    \]", src, re.S)
    return set(re.findall(r"'(/[a-z_]+)", block.group(1)))


def _commands_with_help() -> set:
    """What the completer can describe while you type."""
    src = (_ROOT / "src" / "cli" / "cli_prompt.py").read_text(encoding="utf-8")
    block = re.search(r"COMMAND_HELP = \{(.*?)\n\}", src, re.S)
    return set(re.findall(r'"(/[a-z_]+)"\s*:', block.group(1)))


# Spellings that exist only as an alias of a documented command.
_ALIASES = {"/temperature": "/temp", "/skills": "/skill"}


class TestReadme:
    def test_every_documented_command_exists(self):
        ghosts = sorted(_documented_commands() - _handled_commands())
        assert not ghosts, f"README documents commands that do not exist: {ghosts}"

    def test_the_headline_features_are_documented(self):
        """Features shipped without a mention may as well not exist."""
        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        for command in ("/rewind", "/vibe", "/tasks", "/search"):
            assert command in readme, f"{command} is undocumented"

    def test_both_language_sections_list_the_new_commands(self):
        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        for command in ("/rewind", "/vibe", "/tasks", "/search"):
            assert readme.count(f"- `{command}") >= 2, \
                f"{command} is missing from one of the language sections"


class TestDiscoverability:
    """A command you cannot find is a feature you do not have.

    /kb — add, index and search external documentation, the thing that makes
    semantic_search useful on a Unity or library codebase — was dispatched by
    command_handler and appeared in none of the three places a user looks: Tab
    completion, /help, the README. It had been invisible since it shipped.
    """

    def test_every_command_is_offered_on_tab(self):
        missing = sorted(_handled_commands() - _completer_commands() - set(_ALIASES))
        assert not missing, f"эти команды работают, но не предлагаются по Tab: {missing}"

    def test_tab_never_offers_something_that_does_not_run(self):
        """Worse than hiding a command: an offered one falls through to the
        model as an ordinary message."""
        ghosts = sorted(_completer_commands() - _handled_commands())
        assert not ghosts, f"Tab предлагает несуществующие команды: {ghosts}"

    def test_every_offered_command_describes_itself(self):
        undescribed = sorted(_completer_commands() - _commands_with_help())
        assert not undescribed, f"в подсказке Tab у них пустое описание: {undescribed}"

    def test_help_text_lists_every_command(self):
        help_source = (_ROOT / "command_handler.py").read_text(encoding="utf-8")
        block = help_source.split('elif cmd == "/help":', 1)[1].split("custom_cmds", 1)[0]
        listed = set(re.findall(r"`(/[a-z_]+)", block))
        missing = sorted(_handled_commands() - listed - set(_ALIASES))
        assert not missing, f"/help не упоминает: {missing}"

    def test_readme_lists_every_command(self):
        missing = sorted(_handled_commands() - _documented_commands() - set(_ALIASES))
        assert not missing, f"README не упоминает: {missing}"


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
