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
    found |= set(re.findall(r'user_input\.startswith\("(/[a-z_]+)"', src))
    found |= set(re.findall(r'cmd\s*==\s*"(/[a-z_]+)"', src))
    # Commands with subcommands are dispatched by prefix (e.g. /mcp add …).
    found |= set(re.findall(r'cmd\.startswith\("(/[a-z_]+)"', src))
    for group in re.findall(r'cmd in \(([^)]+)\)', src):
        found |= set(re.findall(r'"(/[a-z_]+)"', group))
    return found


def _documented_commands() -> set:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    return set(re.findall(r"^- `(/[a-z_]+)", readme, re.M))


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
