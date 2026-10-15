"""Argent CLI Command System.
Provides modular command dispatching, domain handlers, and unified registry.
"""

from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry, CommandRegistry

# Import all handler modules to register commands
from src.cli.commands import session_commands
from src.cli.commands import project_commands
from src.cli.commands import agent_commands
from src.cli.commands import config_commands
from src.cli.commands import queue_commands
from src.cli.commands import backup_commands
from src.cli.commands import plan_commands
from src.cli.commands import permission_commands
from src.cli.commands import review_commands
from src.cli.commands import subagent_commands
from src.cli.commands import memory_commands
from src.cli.commands import desktop_commands


def get_registry() -> CommandRegistry:
    return registry


def completion_catalog() -> list:
    """Every command name worth offering on Tab: {"name", "help", "category"}.

    One list for both front-ends (the prompt_toolkit REPL and the TUI), built
    from the registry (what really dispatches: names and aliases) and from
    COMMAND_HELP (the described subcommand variants, e.g. `/sound set done`).
    Nothing is typed by hand a second time, so a command removed from the
    registry stops being offered, and one added to it is offered at once. The
    REPL kept its own hand-written list for a long time and went on advertising
    four commands that no longer existed.
    """
    seen: set = set()
    out: list = []

    def add(name: str, help_text: str, category: str) -> None:
        if name and name not in seen:
            seen.add(name)
            out.append({"name": name, "help": help_text or "", "category": category})

    for entry in registry._entries:
        add(entry.name, entry.help_text, entry.category or "general")
        for alias in entry.aliases:
            add(alias, f"(алиас {entry.name})", entry.category or "general")

    try:
        from src.cli.cli_prompt import COMMAND_HELP
    except Exception:
        COMMAND_HELP = {}
    for key, desc in COMMAND_HELP.items():
        head = str(key).strip()
        if head.startswith("/") or head == "!":
            add(head, str(desc), "subcommand")
    return out


def dispatch_command(user_input: str, ctx: CommandContext) -> CommandResult:
    """Convenience entry point to dispatch user input to registered command handlers."""
    return registry.dispatch(user_input, ctx)


__all__ = [
    "CommandContext",
    "CommandResult",
    "CommandRegistry",
    "registry",
    "get_registry",
    "completion_catalog",
    "dispatch_command",
]
