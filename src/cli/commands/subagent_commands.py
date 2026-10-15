"""/agents: the sub-agents running beside you (see src.agent.subagents)."""

from rich.syntax import Syntax

from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from ui import console, print_error, print_system

_USAGE = ("/agents — список; /agents <N> — отчёт; /agents diff <N> — патч; /agents apply <N> — применить правки "
          "изолированного подагента; /agents discard <N> — выбросить; /agents stop <N> — остановить")


@registry.register("/agents", aliases=["/subagents"],
                   help_text="Подагенты: список, отчёт, патч, применить или выбросить их правки "
                             "(/agents [N | diff N | apply N | discard N | stop N])",
                   category="agent", is_prefix=True)
def handle_agents(ctx: CommandContext) -> CommandResult:
    from src.agent.subagents import MANAGER, describe, table

    words = ctx.user_input.strip().split()[1:]
    if not words:
        print_system(table(MANAGER.all()) + ("\n[dim]" + _USAGE + "[/dim]" if MANAGER.all() else ""))
        return CommandResult.continue_loop()

    action, rest = (words[0].lower(), words[1:]) if not words[0].lstrip("#").isdigit() else ("result", words)
    run = MANAGER.get(rest[0]) if rest else None
    if run is None:
        print_error(f"Нет подагента {rest[0] if rest else ''}".strip() + ". " + _USAGE)
        return CommandResult.continue_loop()

    if action in ("result", "show", "report"):
        print_system(describe(run))
    elif action == "diff":
        if run.patch is None or not run.patch.is_file():
            print_system("У этого подагента нет правок.")
        else:
            console.print(Syntax(run.patch.read_text(encoding="utf-8", errors="replace"), "diff",
                                 theme="monokai", word_wrap=True))
    elif action == "apply":
        problem = MANAGER.apply(run)
        if problem:
            print_error(f"Патч не применился: {problem}")
        else:
            print_system(f"Правки подагента #{run.id} применены к рабочему дереву.")
    elif action in ("discard", "drop"):
        MANAGER.discard(run)
        print_system(f"Правки подагента #{run.id} выброшены.")
    elif action == "stop":
        print_system(f"Подагент #{run.id} остановлен." if MANAGER.stop(run) else "Он не запущен.")
    else:
        print_error(_USAGE)
    return CommandResult.continue_loop()
