"""/memory: the long-term notes (see src/memory)."""

import sys

from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from ui import print_error, print_system

DREAM_TASK_NAME = "memory-dream"
DREAM_SCHEDULE = "every 3h"
_USAGE = ("/memory [list] | show <имя> | search <слова> | add <имя> <текст> [--global] | forget <имя> | "
          "restore <имя> | migrate | undo | on | off | dream [status|run|on|off|consent on|off|battery on|off|model <п:м>|clear]")


def _store_line(store) -> str:
    return f"{store.scope}: {len(store)} заметок · {store.root}"


def _dream_command() -> str:
    return f'"{sys.executable}" -m src.memory.dream'


def _dream_status() -> str:
    import config
    from src.automation import store as astore
    from src.memory import dream, hub
    job = astore.get_automation(DREAM_TASK_NAME)
    store = hub.store_for("project")
    state = dream.load_state(store)
    provider, model, cloud = dream.resolve_target()
    lines = [
        "Консолидация: " + ("[green]включена[/green]" if job and job.enabled else "[yellow]выключена[/yellow]")
        + (f" · расписание: {job.schedule}" if job else ""),
        f"Модель: [bold]{provider}:{model}[/bold] · " + ("облако" if cloud else "ваш компьютер"),
    ]
    if cloud:
        lines.append("Согласие на отправку выдержек в облако: "
                     + ("[green]дано[/green]" if config.is_memory_cloud_consented() else "[yellow]не дано[/yellow]"))
    lines.append("От батареи: " + ("разрешена" if config.is_dream_on_battery_allowed() else "нет (только от сети)"))
    lines.append("Глобальные заметки: " + ("консолидируются (/memory dream global off)" if config.is_dream_global_enabled()
                                          else "не трогаются (/memory dream global on)"))
    if state.get("last_run"):
        lines.append(f"Последний запуск: {state['last_run']} · изменений: {state.get('last_changes', 0)}")
    reports = sorted((store.root / dream.REPORTS_DIR).glob("*.md")) if (store.root / dream.REPORTS_DIR).exists() else []
    if reports:
        lines.append(f"Отчёт: {reports[-1]}")
    return "\n".join(lines)


def _dream(rest: str) -> None:
    import config
    from src.automation import store as astore
    from src.memory import dream, hub

    word, _, arg = rest.partition(" ")
    word, arg = word.lower().strip(), arg.strip()
    if word in ("", "status"):
        print_system(_dream_status())
    elif word == "run":
        provider, model, cloud = dream.resolve_target()
        flags = arg.lower().split()
        from project_paths import project_root_or_cwd
        from pathlib import Path
        store = hub.store_for("project")
        print_system("Консолидация идёт (запросы к модели — с низким приоритетом, может занять несколько минут)…")
        # asked for by the person at the keyboard: the idle and power checks are for the unattended runs
        report = dream.run(store, dream.default_ask(provider, model), Path(project_root_or_cwd()),
                           force=True, dry_run="--dry-run" in flags, target_cloud=cloud,
                           global_store=hub.store_for("global") if config.is_dream_global_enabled() else None)
        print_system(report.text())
    elif word == "on":
        job = astore.Automation(name=DREAM_TASK_NAME, task=_dream_command(), schedule=DREAM_SCHEDULE, mode="command",
                                notify="never", timeout_minutes=60, retries=0, max_failures=0)
        problems = astore.validate_automation(job)
        if problems:
            print_error("Не удалось включить: " + "; ".join(problems))
            return
        astore.upsert_automation(job)
        print_system(f"Консолидация включена: проверка {DREAM_SCHEDULE}, работает, только когда ПК простаивает "
                     "и подключён к питанию. Отчёты: /memory dream status")
    elif word == "off":
        print_system("Консолидация выключена." if astore.remove_automation(DREAM_TASK_NAME)
                     else "Консолидация и не была включена.")
    elif word == "consent" and arg.lower() in ("on", "off"):
        config.set_memory_cloud_consented(arg.lower() == "on")
        print_system("Отправка выдержек из диалогов облачной модели: "
                     + ("[bold]разрешена[/bold]" if arg.lower() == "on" else "[bold]запрещена[/bold]"))
    elif word == "global" and arg.lower() in ("on", "off"):
        config.set_dream_global_enabled(arg.lower() == "on")
        print_system("Консолидация глобальных заметок: "
                     + ("[bold]разрешена[/bold] (не больше 3 операций за раз, только о вас, а не о проекте)"
                        if arg.lower() == "on" else "[bold]запрещена[/bold] (трогает только заметки проекта)"))
    elif word == "battery" and arg.lower() in ("on", "off"):
        config.set_dream_on_battery_allowed(arg.lower() == "on")
        print_system("Консолидация от батареи: " + ("разрешена" if arg.lower() == "on" else "запрещена"))
    elif word == "model" and arg:
        from src.agent.critic import parse_critic_model
        provider, model = parse_critic_model(arg)
        config.set_dream_model(model)
        config.set_dream_provider(provider)
        print_system(f"Модель консолидации: [bold]{model}[/bold]" + (f" (провайдер: {provider})" if provider else ""))
    elif word in ("clear", "reset"):
        config.set_dream_model("")
        config.set_dream_provider("")
        print_system("Модель консолидации сброшена: используется основная.")
    else:
        print_error("Использование: /memory dream [status] | run [--dry-run] | on | off | consent on|off | "
                    "battery on|off | global on|off | model <провайдер>:<модель> | clear")


@registry.register("/memory", help_text="Долговременная память: заметки проекта и глобальные "
                                        "(list / show / search / add / forget / restore / migrate / undo / dream)",
                   category="agent", is_prefix=True)
def handle_memory(ctx: CommandContext) -> CommandResult:
    import config
    from src.memory import hub
    from src.memory.migrate import import_working_memory
    from src.memory.store import MemoryRefused

    arg = ctx.user_input.strip()[len("/memory"):].strip()
    word, _, rest = arg.partition(" ")
    word, rest = word.lower(), rest.strip()
    try:
        if word in ("", "list", "ls"):
            lines = [("[green]включена[/green]" if hub.enabled() else "[yellow]выключена (/memory on)[/yellow]")]
            for store in hub.stores():
                lines.append(f"[bold]{_store_line(store)}[/bold]")
                for node in sorted(store.all(), key=lambda n: n.name):
                    mark = "" if node.source != "user" else " [dim](ваша)[/dim]"
                    lines.append(f"  • {node.name}{mark} — {node.description}")
            lines.append(f"[dim]{_USAGE}[/dim]")
            print_system("\n".join(lines))
        elif word in ("show", "read") and rest:
            node = hub.find(rest)
            if node is None:
                print_error(f"Нет заметки «{rest}».")
            else:
                print_system(f"[bold]{node.scope}:{node.name}[/bold] · {node.type} · автор: {node.source} · "
                             f"{node.updated or 'без даты'}\n{node.description}\n\n{node.body}")
        elif word in ("search", "find") and rest:
            found = hub.search(rest, limit=8)
            if not found:
                print_system("Ничего не найдено.")
            for node in found:
                print_system(f"[bold]{node.scope}:{node.name}[/bold] — {node.description}")
        elif word == "add" and rest:
            scope = "global" if "--global" in rest.split() else "project"
            rest = " ".join(w for w in rest.split() if w != "--global")
            name, _, text = rest.partition(" ")
            if not text.strip():
                print_error("Использование: /memory add <имя> <текст> [--global]")
            else:
                first = text.strip().split("\n")[0][:120]
                result = hub.store_for(scope).put(name, first, text.strip(), type="user" if scope == "global" else "project",
                                                  source="user")
                print_system(f"Заметка {scope}:{result.node.name} {'создана' if result.created else 'обновлена'}."
                             + (f" ({'; '.join(result.warnings)})" if result.warnings else ""))
        elif word in ("forget", "rm", "delete") and rest:
            scope, bare = hub.split_scope(rest)
            done = False
            for store in ([hub.store_for(scope)] if scope else hub.stores()):
                if store.forget(bare, by="user"):
                    print_system(f"Заметка {store.scope}:{bare} в корзине. Вернуть: /memory restore {bare}")
                    done = True
                    break
            if not done:
                print_error(f"Нет заметки «{rest}».")
        elif word == "restore" and rest:
            scope, bare = hub.split_scope(rest)
            for store in ([hub.store_for(scope)] if scope else hub.stores()):
                if store.restore_from_trash(bare):
                    print_system(f"Заметка {store.scope}:{bare} возвращена.")
                    break
            else:
                print_error(f"В корзине нет заметки «{rest}».")
        elif word == "undo":
            from pathlib import Path
            from src.memory import dream
            store = hub.store_for("project")
            shots = store.snapshots()
            if not shots:
                print_error("Нет снимков памяти: консолидация ещё ничего не меняла.")
            else:
                stat = store.restore(shots[-1])
                print_system(f"Память возвращена к снимку {shots[-1].name}: заметок восстановлено {stat['restored']}, "
                             f"убрано в корзину {stat['removed']}.")
            # the last run may also have changed global notes: its own record says which snapshot is theirs
            shot = str(dream.load_state(store).get("last_global_snapshot") or "")
            if shot and Path(shot).is_dir():
                stat = hub.store_for("global").restore(Path(shot))
                print_system(f"Глобальные заметки возвращены к снимку {Path(shot).name}: восстановлено "
                             f"{stat['restored']}, убрано в корзину {stat['removed']}.")
        elif word == "migrate":
            from memory_manager import resolve_memory_file
            result = import_working_memory(hub.store_for("project"), resolve_memory_file())
            print_system("Факты рабочей памяти импортированы в заметку imported-key-facts." if result
                         else "В рабочей памяти нет фактов для импорта.")
        elif word == "on":
            config.set_memory_enabled(True)
            print_system("Долговременная память включена.")
        elif word == "off":
            config.set_memory_enabled(False)
            print_system("Долговременная память выключена: индекс не попадает в промпт, инструмент скрыт.")
        elif word == "dream":
            _dream(rest)
        else:
            print_error("Использование: " + _USAGE)
    except MemoryRefused as e:
        print_error(str(e))
    return CommandResult.continue_loop()
