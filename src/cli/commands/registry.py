"""Registry and dispatcher for Argent CLI commands."""

from typing import Callable, Dict, List, Optional
from src.cli.commands.base import CommandContext, CommandResult
from logger import get_logger

logger = get_logger("command_registry")

CommandHandler = Callable[[CommandContext], CommandResult]


class CommandEntry:
    def __init__(
        self,
        name: str,
        handler: CommandHandler,
        aliases: Optional[List[str]] = None,
        help_text: str = "",
        category: str = "general",
        is_prefix: bool = False,
    ):
        self.name = name.strip().lower()
        self.handler = handler
        self.aliases = [a.strip().lower() for a in (aliases or [])]
        self.help_text = help_text
        self.category = category
        self.is_prefix = is_prefix

    def matches(self, cmd_line: str) -> bool:
        cmd_clean = cmd_line.strip()
        if not cmd_clean:
            return False

        first_token = cmd_clean.split()[0].lower() if cmd_clean else ""

        # Exact token match (e.g. "/vibe", "/model", "/diff")
        if first_token == self.name or first_token in self.aliases:
            return True

        # Special shell escape prefix "!" (e.g. "!dir" or "!ls -la")
        if self.name == "!" or "!" in self.aliases:
            if cmd_clean.startswith("!"):
                return True

        # Multi-word command matching (e.g. "/skill import <repo>")
        cmd_lower = cmd_clean.lower()
        if " " in self.name:
            if cmd_lower == self.name or cmd_lower.startswith(self.name + " "):
                return True
        for alias in self.aliases:
            if " " in alias:
                if cmd_lower == alias or cmd_lower.startswith(alias + " "):
                    return True

        return False




class CommandRegistry:
    """Central registry of all slash commands in Argent."""

    def __init__(self):
        self._entries: List[CommandEntry] = []
        self._by_name: Dict[str, CommandEntry] = {}

    def register(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
        help_text: str = "",
        category: str = "general",
        is_prefix: bool = False,
    ) -> Callable[[CommandHandler], CommandHandler]:
        """Decorator to register a command handler."""
        def decorator(handler: CommandHandler) -> CommandHandler:
            entry = CommandEntry(
                name=name,
                handler=handler,
                aliases=aliases,
                help_text=help_text,
                category=category,
                is_prefix=is_prefix,
            )
            self._entries.append(entry)
            self._by_name[name] = entry
            for alias in (aliases or []):
                self._by_name[alias] = entry
            return handler
        return decorator

    def add_entry(self, entry: CommandEntry) -> None:
        self._entries.append(entry)
        self._by_name[entry.name] = entry
        for alias in entry.aliases:
            self._by_name[alias] = entry

    def dispatch(self, user_input: str, ctx: CommandContext) -> CommandResult:
        """Find matching command and execute handler."""
        cmd_clean = user_input.strip()
        if not cmd_clean:
            return CommandResult.not_handled()

        ctx.user_input = user_input

        # Check exact and prefix matches
        for entry in self._entries:
            if entry.matches(cmd_clean):
                try:
                    res = entry.handler(ctx)
                    if isinstance(res, CommandResult):
                        return res
                    if isinstance(res, bool):
                        return CommandResult(handled=True, should_exit=res, should_continue=True)
                    return CommandResult.continue_loop()
                except Exception as e:
                    logger.exception("Error executing command '%s': %s", entry.name, e)
                    from ui import print_error
                    print_error(f"Ошибка при выполнении команды {entry.name}: {e}")
                    return CommandResult.continue_loop()

        return CommandResult.not_handled()

    def get_all_commands(self) -> List[str]:
        """Return list of all primary command names and aliases."""
        cmds = set()
        for entry in self._entries:
            cmds.add(entry.name)
            for a in entry.aliases:
                cmds.add(a)
        return sorted(list(cmds))


# Global default registry instance
registry = CommandRegistry()
