"""The two prompt files a person edits, and the editors they can be opened in.

There is no built-in text editor on purpose: the field where a long Markdown file
would be edited is the same input box that carries every keystroke of a turn, and a
bug there costs more than it saves. The file is opened in whatever the person
already uses — Notepad for a quick look, VS Code or Sublime when it is real work.

Project prompt:  <project>/AGENTS.md
Global prompt:   ~/.argent/AGENTS.md
Both are read by `build_system_prompt`; the project one wins where they overlap.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

TEMPLATE = """# AGENTS.md — правила для агента в этом проекте

## Команды
- Тесты:
- Сборка:
- Запуск:

## Структура
-

## Соглашения
- 
"""

#: Editors we can start without asking the system what `.md` means. `system` is the
#: association Windows already has (often an IDE, which is why it is not the default).
EDITORS = {
    "notepad": "Блокнот",
    "code": "VS Code",
    "subl": "Sublime Text",
    "system": "Системная программа",
}
DEFAULT_EDITOR = "notepad"


def project_path() -> Path:
    return Path.cwd() / "AGENTS.md"


def global_path() -> Path:
    base = Path(os.environ.get("ARGENT_HOME") or (Path.home() / ".argent"))
    return base / "AGENTS.md"


def path_for(which: str) -> Path:
    return global_path() if str(which).strip().lower() in ("global", "глобальный") else project_path()


def describe(which: str) -> str:
    """`AGENTS.md — есть, 34 строки` or `— нет`, for the status line."""
    path = path_for(which)
    if not path.exists():
        return f"{path.name} — нет ({path.parent})"
    try:
        lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except Exception:
        lines = 0
    return f"{path.name} — есть, {lines} строк ({path.parent})"


def read(which: str) -> str:
    path = path_for(which)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"(не прочитать: {e})"


def create(which: str, force: bool = False) -> str:
    """Make the file from the template. Refuses to overwrite unless `force`."""
    path = path_for(which)
    if path.exists() and not force:
        return f"{path} уже есть — не перезаписываю"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE, encoding="utf-8")
    return f"Создан {path}"


def _launch(editor: str, path: Path) -> Optional[str]:
    """Start the editor; None on success, otherwise a human-readable reason."""
    editor = (editor or DEFAULT_EDITOR).strip().lower()
    try:
        if editor == "system":
            os.startfile(str(path))                     # type: ignore[attr-defined]
            return None
        exe = {"notepad": "notepad.exe", "code": "code", "subl": "subl"}.get(editor)
        if exe is None:
            return f"неизвестный редактор '{editor}'"
        found = shutil.which(exe)
        if found is None:
            return f"'{exe}' не найден в PATH"
        subprocess.Popen([found, str(path)])            # noqa: S603 - fixed argv, no shell
        return None
    except Exception as e:
        return str(e)


def open_in(which: str, editor: Optional[str] = None, force_create: bool = False) -> str:
    """Open the prompt file for editing. Returns the line to show the person."""
    path = path_for(which)
    if not path.exists():
        if not force_create:
            return f"{path} нет. Создать: /prompt new {str(which).lower()}"
        create(which)
    chosen = (editor or os.environ.get("ARGENT_EDITOR_MD") or DEFAULT_EDITOR).strip().lower()
    problem = _launch(chosen, path)
    if problem is None:
        return f"Открыт в {EDITORS.get(chosen, chosen)}: {path}"
    # A missing editor must not leave the person with nothing: fall back to the
    # system association, and say what happened.
    fallback = _launch("system", path)
    if fallback is None:
        return f"'{chosen}' не сработал ({problem}), открыл системной программой: {path}"
    return f"Не удалось открыть {path}: {problem}"


def editor_names() -> List[str]:
    return list(EDITORS.keys())
