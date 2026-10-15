"""Command dispatcher base types and interfaces for Argent CLI."""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class CommandContext:
    """Context passed to all CLI command handlers."""
    agent: Any
    user_input: str = ""
    ui_state: Any = None
    turn_counter: int = 0
    is_project_mode: bool = False
    project_iterations: int = 0
    vibe_mode: bool = False
    prompt_session: Any = None
    extra: dict = field(default_factory=dict)


@dataclass
class CommandResult:
    """Result returned by a command handler."""
    handled: bool = True
    should_exit: bool = False
    should_continue: bool = True
    new_input: Optional[str] = None
    active_tools: Optional[List[str]] = None
    turn_counter: Optional[int] = None
    vibe_mode: Optional[bool] = None
    is_project_mode: Optional[bool] = None
    project_iterations: Optional[int] = None

    @classmethod
    def continue_loop(cls, **kwargs) -> "CommandResult":
        """Convenience factory: command handled, stay in REPL loop."""
        return cls(handled=True, should_continue=True, **kwargs)

    @classmethod
    def exit_repl(cls) -> "CommandResult":
        """Convenience factory: command handled, exit REPL."""
        return cls(handled=True, should_exit=True, should_continue=False)

    @classmethod
    def pass_to_agent(cls, new_input: Optional[str] = None, active_tools: Optional[List[str]] = None, **kwargs) -> "CommandResult":
        """Convenience factory: command transforms input, proceed to agent execution."""
        return cls(handled=True, should_continue=False, new_input=new_input, active_tools=active_tools, **kwargs)

    @classmethod
    def not_handled(cls) -> "CommandResult":
        """Convenience factory: command was not recognized or handled."""
        return cls(handled=False, should_continue=True)
