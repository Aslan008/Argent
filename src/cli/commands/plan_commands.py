"""/plan and /todo: work in steps, and look before changing anything."""

from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from ui import print_error, print_system


@registry.register("/plan", help_text="Режим плана: агент только изучает проект и показывает план, ничего не меняя "
                                     "(/plan [задача] | off | status)",
                   category="agent", is_prefix=True)
def handle_plan(ctx: CommandContext) -> CommandResult:
    from src.agent.plan_mode import PLAN

    arg = ctx.user_input.strip()[len("/plan"):].strip()
    low = arg.lower()

    if low in ("off", "выкл", "stop"):
        if PLAN.leave():
            print_system("Режим плана выключен: инструменты правки снова доступны.")
        else:
            print_system("Режим плана и так выключен.")
        return CommandResult.continue_loop()

    if low == "status":
        print_system("Режим плана: [bold]ВКЛЮЧЁН[/bold] — правки заблокированы до утверждения плана."
                     if PLAN.active else "Режим плана: выключен.")
        return CommandResult.continue_loop()

    PLAN.enter()
    if not arg or low == "on":
        print_system("Режим плана [bold]включён[/bold]: агент изучает проект, ничего не меняет, и в конце "
                     "предложит план на утверждение. Выйти: /plan off")
        return CommandResult.continue_loop()

    print_system("Режим плана [bold]включён[/bold]: сначала план, правок не будет до вашего утверждения.")
    return CommandResult.pass_to_agent(new_input=arg)


@registry.register("/todo", aliases=["/todos"],
                   help_text="Список задач агента на текущую работу (/todo | clear)",
                   category="agent", is_prefix=True)
def handle_todo(ctx: CommandContext) -> CommandResult:
    from src.agent.todos import TODOS, progress, render

    arg = ctx.user_input.strip().split(None, 1)
    arg = arg[1].strip().lower() if len(arg) > 1 else ""
    if arg in ("clear", "очистить", "сброс"):
        TODOS.clear("main")
        print_system("Список задач очищен.")
        return CommandResult.continue_loop()
    items = TODOS.get("main")
    if not items:
        print_error("Список задач пуст — агент заведёт его сам, когда возьмётся за многошаговую работу.")
        return CommandResult.continue_loop()
    print_system(f"Список задач ({progress(items)} выполнено):\n{render(items)}")
    return CommandResult.continue_loop()
