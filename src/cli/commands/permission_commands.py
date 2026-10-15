"""/permissions: standing allow / ask / deny rules (see src.agent.permissions)."""

from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from ui import print_error, print_system

_USAGE = (
    "Правила разрешений (.argent/permissions.json для проекта, ~/.argent/permissions.json везде):\n"
    "  /permissions                       — показать правила\n"
    "  /permissions allow run_command(git status*)   — команда выполняется без вопроса\n"
    "  /permissions ask   run_command(git push*)     — всегда спрашивать, даже в авто-режиме\n"
    "  /permissions deny  write_file(.env*)          — запретить полностью\n"
    "  /permissions remove <правило>      — убрать\n"
    "  /permissions trust                 — доверить правилам allow из файла проекта\n"
    "Добавьте --global, чтобы правило действовало во всех проектах."
)


@registry.register("/permissions", aliases=["/perms"],
                   help_text="Постоянные правила: что разрешено, что спрашивать, что запрещено "
                             "(/permissions allow|ask|deny <Tool(шаблон)>)",
                   category="agent", is_prefix=True)
def handle_permissions(ctx: CommandContext) -> CommandResult:
    from src.agent.permissions import KINDS, PERMISSIONS

    words = ctx.user_input.strip().split(None, 1)
    rest = words[1].strip() if len(words) > 1 else ""
    head, _, tail = rest.partition(" ")
    head = head.lower()
    scope = "project"
    if tail.rstrip().endswith("--global"):
        scope, tail = "global", tail.rstrip()[:-len("--global")].strip()
    tail = tail.strip()

    if head in KINDS:
        if not tail:
            print_error(f"Какое правило? Пример: /permissions {head} run_command(git status*)")
            return CommandResult.continue_loop()
        try:
            rule = PERMISSIONS.add(head, tail, scope)
        except ValueError as e:
            print_error(str(e))
            return CommandResult.continue_loop()
        where = "всех проектов" if scope == "global" else "этого проекта"
        print_system(f"Правило [bold]{head}[/bold] {rule.raw} добавлено для {where}.")
        if PERMISSIONS.untrusted():
            print_system("[yellow]Файл правил проекта пришёл не от вас и не доверен: его allow-правила не "
                         "действуют. Прочитайте его и выполните /permissions trust[/yellow]")
        return CommandResult.continue_loop()

    if head in ("remove", "rm", "del", "delete"):
        if not tail:
            print_error("Какое правило убрать? Скопируйте его из /permissions.")
            return CommandResult.continue_loop()
        removed = PERMISSIONS.remove(tail, scope if "--global" in rest else None)
        print_system(f"Убрано правил: {removed}." if removed else "Такого правила нет.")
        return CommandResult.continue_loop()

    if head == "trust":
        path = PERMISSIONS.trust()
        print_system(f"Файл правил {path} доверен." if path else "В проекте нет файла правил.")
        return CommandResult.continue_loop()

    if head in ("help", "?"):
        print_system(_USAGE)
        return CommandResult.continue_loop()

    rules = PERMISSIONS.rules()
    if not rules and not PERMISSIONS.untrusted() and not PERMISSIONS.notes():
        print_system("Правил пока нет.\n" + _USAGE)
        return CommandResult.continue_loop()
    for kind in KINDS:
        for rule in (r for r in rules if r.kind == kind):
            print_system(f"  {kind:<5} [{rule.source}] {rule.raw}")
    for note in PERMISSIONS.notes():
        print_system(f"  [yellow]! {note}[/yellow]")
    if PERMISSIONS.untrusted():
        print_system(f"[yellow]Файл {PERMISSIONS.untrusted()} не доверен — его allow-правила не действуют "
                     "(deny и ask действуют). /permissions trust[/yellow]")
    return CommandResult.continue_loop()
