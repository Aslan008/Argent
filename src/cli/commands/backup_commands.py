"""/backup: a zip of your rooms, automations and memory notes, and the way back (see src/project/backup.py)."""

from pathlib import Path

from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from ui import print_error, print_system

_USAGE = "/backup [create [файл]] | list | show <N|файл> | restore <N|файл>"


def _size(n: int) -> str:
    return f"{n / 1024:.0f} КБ" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} МБ"


def _pick(spec: str, backups: list):
    """A file from `list` by its number, or a path."""
    spec = spec.strip().strip('"')
    if spec.isdigit():
        n = int(spec)
        return backups[n - 1] if 1 <= n <= len(backups) else None
    path = Path(spec)
    return path if path.is_file() else None


@registry.register("/backup", help_text="Копия ваших комнат, автоматизаций и памяти одним файлом: "
                                        "create / list / show / restore", category="agent", is_prefix=True)
def handle_backup(ctx: CommandContext) -> CommandResult:
    from src.project import backup

    arg = ctx.user_input.strip()[len("/backup"):].strip()
    word, _, rest = arg.partition(" ")
    word, rest = word.lower(), rest.strip()

    if word in ("", "create", "new", "make"):
        for title, files, size in backup.preview():
            print_system(f"  {title}: {files} файлов, {_size(size)}")
        try:
            path, count, total = backup.create(Path(rest.strip('"')) if rest else None)
        except OSError as e:
            print_error(f"Не удалось записать копию: {e}")
            return CommandResult.continue_loop()
        print_system(f"Готово: {path} ({count} файлов, {_size(total)} без сжатия).")
        print_system("[dim]В копии нет ключей и настроек (конфиг не читается), но комнаты и заметки лежат как есть: "
                     "не публикуйте файл. Вернуть: /backup restore <номер|файл>[/dim]")
    elif word in ("list", "ls"):
        found = backup.list_backups()
        if not found:
            print_system("Копий пока нет. /backup — сделать.")
        for n, p in enumerate(found, 1):
            print_system(f"  {n}. {p.name}  ({_size(p.stat().st_size)})")
        if found:
            print_system(f"[dim]Папка: {backup.default_folder()}[/dim]")
    elif word in ("show", "info", "restore"):
        found = backup.list_backups()
        path = _pick(rest, found) if rest else (found[0] if found and word == "show" else None)
        if path is None:
            print_error("Укажите номер из /backup list или путь к файлу.")
            return CommandResult.continue_loop()
        try:
            if word in ("show", "info"):
                data = backup.inspect(path)
                print_system(f"{path.name}: создана {data.get('created', '?')}")
                for label, names in data["roots"].items():
                    print_system(f"  {label}: {len(names)} файлов")
            else:
                import questionary

                def ask(text: str) -> bool:
                    return bool(questionary.confirm(text, default=False).ask())

                count, safety = backup.restore(path, confirm=ask)
                if not count:
                    print_system("Отменено.")
                else:
                    print_system(f"Возвращено файлов: {count}.")
                    if safety:
                        print_system(f"[dim]То, что было заменено, сохранено: {safety}[/dim]")
        except backup.BackupError as e:
            print_error(str(e))
    else:
        print_error("Использование: " + _USAGE)
    return CommandResult.continue_loop()
