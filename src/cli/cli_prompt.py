"""
Interactive prompt session for Argent: command completion with inline help,
a live status bar (bottom toolbar) and on-disk input history.

Kept separate from main.py so the REPL wiring stays small and this can be
unit-tested in isolation.
"""

import os
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, InMemoryHistory


# One-line help shown next to a command while completing. Keys are the base
# commands; subcommand variants inherit the base command's description.
COMMAND_HELP = {
    "/help": "Показать список команд",
    "/provider": "Выбрать API-провайдера и эндпоинт",
    "/model": "Выбрать активную модель (с поиском)",
    "/clear": "Очистить историю диалога",
    "/research": "Автономное веб-исследование темы",
    "/rag_toggle": "Вкл/выкл семантический поиск при старте",
    "/hooks": "Папка плагинов (хуков)",
    "/tools": "Включить/выключить инструменты ИИ",
    "/save": "Экспортировать диалог в Markdown",
    "/project": "Построить новый проект с нуля",
    "/work": "Автономно изменить существующий код",
    "/commit": "Сгенерировать commit и закоммитить",
    "/sessions": "Список сохранённых сессий",
    "/load": "Восстановить сессию по номеру",
    "/copy": "Скопировать блок кода №n",
    "/logs": "Просмотр логов",
    "/skills": "Список навыков ИИ",
    "/auto": "Полностью автономный режим",
    "/verbose": "Переключить индикаторы статуса",
    "/debug": "Подробные логи инструментов",
    "/browser": "Настройка автоматизации браузера",
    "/mcp": "Управление MCP-серверами",
    "/doctor": "Самодиагностика окружения",
    "/thinking": "Удаление reasoning из истории",
    "/temp": "Температура генерации (0.0–2.0)",
    "/temperature": "Температура генерации (0.0–2.0)",
    "/cd": "Сменить рабочую директорию",
    "/undo": "Откатить файл к версии до правки ИИ",
    "/diff": "Показать изменения файла",
    "/changes": "Файлы, изменённые ИИ в сессии",
    "/stats": "Диагностика сессии",
    "/exit": "Выйти",
    "/quit": "Выйти",
}


def _describe(command: str) -> str:
    """Help text for a command or one of its subcommand variants."""
    if command in COMMAND_HELP:
        return COMMAND_HELP[command]
    base = "/" + command.lstrip("/").split(" ", 1)[0]
    return COMMAND_HELP.get(base, "")


class ArgentCommandCompleter(Completer):
    """Completes slash commands, showing a one-line description for each.

    Only activates while the input is a single token starting with '/', so it
    never interferes with normal multi-word prompts to the model.
    """

    def __init__(self, get_commands):
        self._get_commands = get_commands

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        stripped = text.lstrip()
        # Only complete a leading slash-command token (no spaces typed yet).
        if not stripped.startswith("/") or " " in stripped:
            return
        word = stripped.lower()
        for cmd in self._get_commands():
            if cmd.lower().startswith(word):
                yield Completion(
                    cmd,
                    start_position=-len(stripped),
                    display=cmd,
                    display_meta=_describe(cmd),
                )


_STRATEGY_TIER = {
    "TinyLocalStrategy": "tiny",
    "StandardLocalStrategy": "local",
    "CloudStrategy": "cloud",
}


def _tier_label(agent) -> str:
    return _STRATEGY_TIER.get(type(getattr(agent, "strategy", None)).__name__, "?")


def build_bottom_toolbar(agent, ui_state):
    """Return a callable rendering the live status bar.

    Reads current state on every keystroke (cheap fields only — no token
    counting), so it always reflects the active model/provider/mode.
    """
    def _toolbar():
        try:
            provider = getattr(agent, "provider", "?")
            model = getattr(agent, "model_name", "?")
            tier = _tier_label(agent)
            mode = ui_state.get("mode", "CHAT")
            cwd = os.path.basename(os.getcwd()) or os.getcwd()
            # Running session token/cost total, once anything has been spent.
            usage_str = ""
            try:
                from usage_tracker import usage as session_usage
                if session_usage.requests:
                    usage_str = f"| {session_usage.format_session()} "
            except Exception:
                usage_str = ""
            # ASCII separators only: legacy Windows consoles (cp1251) choke on
            # box-drawing chars and emoji.
            return HTML(
                f" <b>{mode}</b> | {provider}:{model} | tier:{tier} | cwd:{cwd} "
                f"{usage_str}| <style fg='#888888'>/help</style> "
            )
        except Exception:
            return ""
    return _toolbar


def build_prompt_session(get_commands, agent, ui_state, history_path=None) -> PromptSession:
    """Create the PromptSession used by the main REPL loop."""
    if history_path:
        try:
            Path(history_path).parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(history_path))
        except Exception:
            history = InMemoryHistory()
    else:
        history = InMemoryHistory()

    return PromptSession(
        completer=ArgentCommandCompleter(get_commands),
        history=history,
        bottom_toolbar=build_bottom_toolbar(agent, ui_state),
        complete_while_typing=True,
    )
