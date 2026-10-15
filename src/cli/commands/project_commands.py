"""Project-related command handlers for Argent CLI (/init, /research, /cd, /undo, /diff, /changes, /commit, !)."""

import os
import subprocess
from pathlib import Path
import questionary
from rich.syntax import Syntax

from file_tracker import undo, get_diff, get_pending_changes
from src.agent.shell import run_text
from src.cli.cd_path import clean_cd_path
from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from ui import console, print_system, print_error


@registry.register("/init", help_text="Изучить проект и создать .argent/AGENTS.md", category="project", is_prefix=True)
def handle_init(ctx: CommandContext) -> CommandResult:
    new_input = (
        "Onboard to THIS project and create a professional `.argent/AGENTS.md` — it is "
        "your persistent memory about this codebase and is loaded into your context every "
        "session, so it must be accurate and concise.\n\n"
        "STRICT workflow:\n"
        "1. Call `analyze_project()` first to get the structure, languages, entry points and commands.\n"
        "2. Read the few key files you need (README, the main entry point, core config) to "
        "understand the real architecture — do not guess.\n"
        "3. Write `.argent/AGENTS.md` with `write_file`. Keep it DENSE and under ~150 lines, with:\n"
        "   - `## Overview` — what the project is, in 2-3 sentences.\n"
        "   - `## Architecture` — main modules/layers and how they relate.\n"
        "   - `## Build / Run / Test` — exact commands.\n"
        "   - `## Conventions` — code style, patterns and rules a contributor must follow.\n"
        "   - `## Key files` — the most important files and what each does.\n"
        "4. Tell the user what you wrote and remind them they can edit the file by hand.\n"
        "Do NOT dump full file listings or trivia — this file costs context every turn, keep it tight."
    )
    active_tools = [
        "analyze_project", "read_file", "get_file_outline", "list_directory",
        "grep_search", "search_files", "write_file",
    ]
    print_system("Изучаю проект и создаю .argent/AGENTS.md...")
    return CommandResult.pass_to_agent(new_input=new_input, active_tools=active_tools)


@registry.register("/research", help_text="Автономный агент глубокого исследования темы", category="project", is_prefix=True)
def handle_research(ctx: CommandContext) -> CommandResult:
    parts = ctx.user_input.split(" ", 1)
    if len(parts) < 2:
        print_error("Укажите тему. Пример: /research Unity DOTS")
        return CommandResult.continue_loop()
    topic = parts[1].strip()
    active_tools = ["run_deep_research"]
    new_input = (
        f"Please act as an autonomous Research Agent for the topic: '{topic}'.\n\n"
        f"Your STRICT workflow is:\n"
        f"1. Call `run_deep_research(topic=\"{topic}\", plan_only=True)` to create and review a comprehensive research plan.\n"
        f"2. Proceed to run `run_deep_research(topic=\"{topic}\", plan_only=False)`.\n"
        f"3. Summarize the key findings, conclusions, and where the final report was saved in artifacts.\n"
        f"Do not guess answers without gathering verified evidence."
    )
    print_system(f"Запускаю глубокое исследование по теме: [bold cyan]{topic}[/bold cyan]...")
    return CommandResult.pass_to_agent(new_input=new_input, active_tools=active_tools)


@registry.register("/cd", help_text="Сменить рабочую директорию", category="project", is_prefix=True)
def handle_cd(ctx: CommandContext) -> CommandResult:
    parts = ctx.user_input.split(" ", 1)
    if len(parts) < 2:
        print_system(f"Рабочая директория: [bold cyan]{os.getcwd()}[/bold cyan]")
        print_system('Использование: /cd <путь> (путь с пробелами — в кавычках: /cd "<путь>")')
        return CommandResult.continue_loop()
    target_dir = clean_cd_path(parts[1])
    if not target_dir:
        print_system(f"Рабочая директория: [bold cyan]{os.getcwd()}[/bold cyan]")
        print_system('Использование: /cd <путь> (путь с пробелами — в кавычках: /cd "<путь>")')
        return CommandResult.continue_loop()
    try:
        resolved = Path(target_dir).expanduser().resolve()
        if not resolved.exists():
            print_error(f"Директории не существует: {resolved}")
            return CommandResult.continue_loop()
        if not resolved.is_dir():
            print_error(f"Это не директория: {resolved}")
            return CommandResult.continue_loop()
        os.chdir(resolved)
        try:
            from memory_manager import memory
            memory._mgr = None
        except Exception:
            pass
        print_system(f"Рабочая директория изменена: [bold green]{os.getcwd()}[/bold green]")
    except Exception as e:
        print_error(f"Не удалось сменить директорию: {e}")
    return CommandResult.continue_loop()



@registry.register("/undo", help_text="Откатить изменения файла", category="project", is_prefix=True)
def handle_undo(ctx: CommandContext) -> CommandResult:
    parts = ctx.user_input.split(" ", 1)
    if len(parts) < 2:
        print_error("Использование: /undo <путь_к_файлу>")
        return CommandResult.continue_loop()
    result = undo(parts[1].strip())
    print_system(result)
    return CommandResult.continue_loop()


@registry.register("/diff", help_text="Показать diff изменений файла", category="project", is_prefix=True)
def handle_diff(ctx: CommandContext) -> CommandResult:
    parts = ctx.user_input.split(" ", 1)
    if len(parts) < 2:
        print_error("Использование: /diff <путь_к_файлу>")
        return CommandResult.continue_loop()
    result = get_diff(parts[1].strip())
    if result.startswith("Error") or result.startswith("No") or result.startswith("File"):
        print_system(result)
    else:
        syntax = Syntax(result, "diff", theme="monokai", word_wrap=True)
        console.print(syntax)
    return CommandResult.continue_loop()


@registry.register("/changes", help_text="Показать список отслеживаемых измененных файлов", category="project")
def handle_changes(ctx: CommandContext) -> CommandResult:
    changes = get_pending_changes()
    if not changes:
        print_system("В этой сессии файлы не менялись.")
    else:
        print_system("[bold cyan]Tracked File Changes:[/bold cyan]")
        for ch in changes:
            print_system(f"  - {ch['key']} ({ch['snapshot_count']} snapshots)")
        print_system("\nПосмотреть изменения: /diff <путь>, откатить: /undo <путь>.")
    return CommandResult.continue_loop()


@registry.register("/commit", help_text="Создать git коммит с автогенерацией описания", category="project", is_prefix=True)
def handle_commit(ctx: CommandContext) -> CommandResult:
    try:
        git_check = run_text("git rev-parse --git-dir", capture_output=True)
        if git_check.returncode != 0:
            print_error("Это не git-репозиторий. Перейдите в проект с git.")
            return CommandResult.continue_loop()

        staged_diff = run_text(["git", "diff", "--cached"], capture_output=True).stdout
        if not staged_diff.strip():
            print_error("No staged changes found. Use 'git add' first.")
            return CommandResult.continue_loop()

        from config import get_critic_auto
        if get_critic_auto():
            from agent import run_plan_critique
            from src.agent.critic import parse_critic_verdict, CriticVerdict, sanitize_and_truncate_diff
            from memory_manager import memory
            print_system("[dim]Критик смотрит подготовленный дифф перед коммитом…[/dim]")
            clean_diff = sanitize_and_truncate_diff(staged_diff)
            try:
                review = run_plan_critique(
                    clean_diff, goal=memory.data.get("objective") or "",
                    what="staged git diff (about to be committed)",
                )
                verdict = parse_critic_verdict(review)
                verdict_badge = {
                    CriticVerdict.SOLID: "[bold green]VERDICT: SOLID[/bold green]",
                    CriticVerdict.FIXABLE: "[bold yellow]VERDICT: FIXABLE[/bold yellow]",
                    CriticVerdict.RETHINK: "[bold red]VERDICT: RETHINK[/bold red]",
                }.get(verdict, "")
                if verdict_badge:
                    print_system(f"[bold #79c0ff]── Critic ──[/bold #79c0ff] {verdict_badge}")
                else:
                    print_system("[bold #79c0ff]── Critic ──[/bold #79c0ff]")
                print_system(review or "[critic returned nothing]")

                if verdict == CriticVerdict.RETHINK:
                    proceed = questionary.confirm(
                        "Критик выставил вердикт RETHINK (критические проблемы в подготовленном диффе). Вы уверены, что хотите закоммитить?",
                        default=False,
                    ).ask()
                    if not proceed:
                        print_system("[yellow]Коммит отменён пользователем для доработки диффа.[/yellow]")
                        return CommandResult.continue_loop()
            except Exception as e:
                print_error(f"Критик упал (коммит продолжается): {e}")

        print_system("Generating commit message based on staged changes...")
        commit_prompt = (
            "You are a Senior Developer. Generate a concise, professional Git commit message following Conventional Commits "
            "specification based on the following diff. Only output the commit message, nothing else.\n\n"
            f"{staged_diff}"
        )
        gen_message = ""
        try:
            from providers import create_service_provider
            provider, svc_model = create_service_provider()
            res = provider.sync_chat(
                model=svc_model,
                messages=[{"role": "user", "content": commit_prompt}]
            )
            if isinstance(res, str):
                gen_message = res.strip().strip('"').strip("'")
            elif isinstance(res, dict):
                gen_message = res.get("text", str(res))
            elif res:
                gen_message = str(res)
        except Exception as e:
            print_error(f"Не удалось сгенерировать сообщение коммита: {e}")
            return CommandResult.continue_loop()

        if not gen_message:
            print_error("Не удалось сгенерировать сообщение коммита.")
            return CommandResult.continue_loop()

        print_system(f"Suggested commit message:\n[bold cyan]{gen_message}[/bold cyan]")
        approved = questionary.confirm("Do you want to commit with this message?").ask()
        if approved:
            subprocess.run(["git", "commit", "-m", gen_message], check=True)
            print_system("Commited successfully!")
        else:
            custom_msg = questionary.text("Enter custom commit message (leave empty to prevent commit):").ask()
            if custom_msg:
                subprocess.run(["git", "commit", "-m", custom_msg], check=True)
                print_system("Коммит создан с вашим сообщением.")
            else:
                print_system("Commit aborted.")
    except FileNotFoundError:
        print_error("Git не установлен или не найден в PATH.")
    except Exception as e:
        print_error(f"Ошибка при коммите: {e}")

    return CommandResult.continue_loop()


@registry.register("!", help_text="Выполнить команду оболочки", category="project", is_prefix=True)
def handle_shell_escape(ctx: CommandContext) -> CommandResult:
    cmd = ctx.user_input[1:].strip()
    if not cmd:
        return CommandResult.continue_loop()
    print_system(f"Running local command: {cmd}")

    try:
        import shlex
        try:
            cmd_parts = shlex.split(cmd, posix=(os.name != 'nt'))
        except ValueError:
            cmd_parts = cmd.split()
        result = run_text(
            cmd_parts,
            shell=False,
            capture_output=True,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()

        if out:
            print(f"{out}")
        if err:
            print_error(f"{err}")

        if result.returncode != 0:
            print_error(f"Команда завершилась с кодом {result.returncode}.")
            ask_fix = questionary.confirm("Would you like Argent to help fix this error?").ask()
            if ask_fix:
                user_input = (
                    f"I ran the command `{cmd}` and it failed with exit code {result.returncode}.\n"
                    f"STDOUT:\n{out}\n"
                    f"STDERR:\n{err}\n"
                    f"Please analyze the error and help me fix it."
                )
                return CommandResult.pass_to_agent(new_input=user_input)
            else:
                return CommandResult.continue_loop()
    except Exception as e:
        print_error(f"Ошибка выполнения: {e}")

    return CommandResult.continue_loop()
