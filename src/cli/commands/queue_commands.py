"""Queue management slash commands for Argent CLI (/queue)."""

from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from src.cli.input_queue import get_input_queue
from ui import console, print_system, print_error
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text


@registry.register(
    "/queue",
    aliases=["/qmsg"],
    help_text="Очередь сообщений: list / rm <номер> / edit <номер> <текст> / clear",
    category="config",
    is_prefix=True,
)
def handle_queue(ctx: CommandContext) -> CommandResult:
    """Handle /queue commands for inspecting, editing, and deleting queued messages."""
    queue = get_input_queue()
    parts = ctx.user_input.strip().split(maxsplit=2)
    subcmd = parts[1].lower() if len(parts) > 1 else "list"

    if subcmd in ("list", "ls", "show"):
        items = queue.items()
        if not items:
            print_system(
                "Очередь сообщений пуста.\n"
                "[dim]Вы можете печатать текст и нажимать Enter прямо во время работы агента, "
                "чтобы ставить задачи в очередь.[/dim]"
            )
            return CommandResult.continue_loop()

        lines = []
        for i, item in enumerate(items, 1):
            short = item.replace("\n", " ↵ ")
            lines.append(f"[bold cyan]{i}.[/bold cyan] [white]{escape(short)}[/white]")

        panel = Panel(
            Text.from_markup("\n".join(lines)),
            title=f"[bold cyan]Очередь сообщений ({len(items)})[/bold cyan]",
            subtitle="[dim]/queue rm <№> · /queue edit <№> <текст> · /queue clear[/dim]",
            border_style="cyan dim",
            padding=(0, 1),
        )
        console.print(panel)
        return CommandResult.continue_loop()

    elif subcmd in ("clear", "clean", "reset"):
        cleared = queue.clear()
        print_system(f"Очередь очищена (удалено {len(cleared)} сообщений).")
        return CommandResult.continue_loop()

    elif subcmd in ("rm", "remove", "del", "pop"):
        if len(parts) < 3:
            print_error("Укажите номер сообщения для удаления: /queue rm <номер>")
            return CommandResult.continue_loop()
        try:
            num = int(parts[2].strip())
            removed = queue.remove(num - 1)
            if removed is not None:
                short = removed.replace("\n", " ")
                if len(short) > 60:
                    short = short[:57] + "…"
                print_system(f"Сообщение {num} удалено из очереди: [dim]'{escape(short)}'[/dim]")
            else:
                print_error(f"Сообщение с номером {num} не найдено в очереди (всего: {queue.count()}).")
        except ValueError:
            print_error(f"Неверный номер сообщения: '{parts[2]}'")
        return CommandResult.continue_loop()

    elif subcmd in ("edit", "change"):
        if len(parts) < 3:
            print_error("Формат: /queue edit <номер> <новый текст>")
            return CommandResult.continue_loop()
        rest = parts[2].strip().split(maxsplit=1)
        try:
            num = int(rest[0].strip())
            if len(rest) < 2 or not rest[1].strip():
                print_error("Укажите новый текст сообщения: /queue edit <номер> <новый текст>")
                return CommandResult.continue_loop()
            new_text = rest[1].strip()
            ok = queue.edit(num - 1, new_text)
            if ok:
                print_system(f"Сообщение {num} в очереди обновлено.")
            else:
                print_error(f"Сообщение с номером {num} не найдено в очереди (всего: {queue.count()}).")
        except ValueError:
            print_error(f"Неверный номер сообщения: '{rest[0]}'")
        return CommandResult.continue_loop()

    elif subcmd in ("add", "push"):
        if len(parts) < 3:
            print_error("Укажите текст сообщения: /queue add <текст>")
            return CommandResult.continue_loop()
        text = parts[2].strip()
        idx = queue.push(text)
        print_system(f"Сообщение добавлено в очередь под номером {idx}.")
        return CommandResult.continue_loop()

    else:
        print_error(
            f"Неизвестная подкоманда очереди: '{subcmd}'\n"
            "Доступные действия: /queue [list], /queue rm <№>, /queue edit <№> <текст>, /queue clear, /queue add <текст>"
        )
        return CommandResult.continue_loop()
