"""
Specialist Argument Extractors for Argent MicroModels.

Once the IntentRouter identifies the intended tool, the corresponding specialist
extracts cleanly structured arguments (file paths, search queries, commands, math expressions)
in microsecond time on CPU without calling an external LLM.
"""

import re
from typing import Any, Dict, Optional


class BaseSpecialist:
    """Base class for tool argument extraction specialists."""

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class ReadFileSpecialist(BaseSpecialist):
    """Extracts file_path and optional line boundaries for read_file."""

    # Strip routine command prefixes
    PREFIXES = [
        r"^(?:прочитай(?:\s+содержимое)?(?:\s+файл[а-я]*)?)\s+",
        r"^(?:покажи(?:\s+содержимое)?(?:\s+файл[а-я]*)?)\s+",
        r"^(?:открой(?:\s+файл[а-я]*)?)\s+",
        r"^(?:выведи(?:\s+на\s+экран)?(?:\s+файл[а-я]*)?)\s+",
        r"^(?:глянь\s+в(?:\s+файл[а-я]*)?)\s+",
        r"^(?:посмотри(?:\s+файл[а-я]*)?)\s+",
        r"^(?:покажи\s+код(?:\s+из)?(?:\s+файл[а-я]*)?)\s+",
        r"^(?:считай(?:\s+файл[а-я]*)?)\s+",
        r"^(?:read(?:\s+content\s+of)?(?:\s+file)?)\s+",
        r"^(?:open(?:\s+file)?)\s+",
        r"^(?:show(?:\s+content\s+of)?(?:\s+file)?)\s+",
        r"^(?:cat|display|view|inspect)\s+",
        r"^(?:view\s+source\s+of)\s+",
    ]

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        text = query.strip()

        # Remove prefix
        for pat in self.PREFIXES:
            new_text, n = re.subn(pat, "", text, count=1, flags=re.IGNORECASE)
            if n > 0:
                text = new_text.strip()
                break

        # Check for line ranges e.g. "lines 10-50", "строки 20-40", ":10:50"
        line_match = re.search(r"(?:строки|lines|l)\s+(\d+)(?:\s*-\s*|\s+до\s+)(\d+)", text, re.IGNORECASE)
        start_line = None
        line_count = None
        if line_match:
            start_line = int(line_match.group(1))
            end_line = int(line_match.group(2))
            line_count = max(1, end_line - start_line + 1)
            text = text[: line_match.start()] + text[line_match.end() :]
            text = text.strip()

        # Extract file path: could be quoted or unquoted
        quote_match = re.search(r"['\"]([^'\"]+)['\"]", text)
        if quote_match:
            raw_path = quote_match.group(1).strip()
        else:
            # 1. Try finding Windows drive path: e.g. C:\Users\test\main.py, C:/path/file.py
            win_match = re.search(r"([a-zA-Z]:[\\/][a-zA-Z0-9_\-\.\/\\]+(?::\d+(?::\d+)?)?)", text)
            # 2. Try finding path with extension: e.g. src/agent.py, config.json, test_foo.py
            ext_match = re.search(r"([a-zA-Z0-9_\-\.\/\\]+\.[a-zA-Z0-9_]{1,6}(?::\d+(?::\d+)?)?)", text)
            # 3. Try finding path with slash: e.g. src/utils, path/to/dir
            slash_match = re.search(r"([a-zA-Z0-9_\-\.]+(?:[/\\][a-zA-Z0-9_\-\.]+)+(?::\d+(?::\d+)?)?)", text)

            if win_match:
                raw_path = win_match.group(1)
            elif ext_match:
                raw_path = ext_match.group(1)
            elif slash_match:
                raw_path = slash_match.group(1)
            else:
                # Filter out Russian/English noise words before taking first path token
                stopwords = {"в", "из", "на", "файл", "файла", "файле", "файлу", "file", "сначала", "потом", "нужно", "открыть"}
                words = [p.strip(" ,;:") for p in text.split() if p.strip(" ,;:").lower() not in stopwords]
                if not words:
                    return None
                raw_path = words[0]

        raw_path = raw_path.strip().strip("'\"")
        if not raw_path:
            return None

        # Check for colon-separated line range in path e.g. "foo.py:10:50" or "foo.py:10"
        colon_match = re.search(r":(\d+)(?::(\d+))?$", raw_path)
        if colon_match:
            raw_path = raw_path[: colon_match.start()]
            if start_line is None:
                start_line = int(colon_match.group(1))
                if colon_match.group(2):
                    end_val = int(colon_match.group(2))
                    line_count = max(1, end_val - start_line + 1)

        # Normalize Windows / POSIX slashes
        norm_path = raw_path.replace("\\", "/")

        args: Dict[str, Any] = {"file_path": norm_path}
        if start_line is not None:
            args["start_line"] = start_line
        if line_count is not None:
            args["line_count"] = line_count

        return args


class GrepSearchSpecialist(BaseSpecialist):
    """Extracts search query and target directory for grep_search."""

    PREFIXES = [
        r"^(?:найди\s+где\s+(?:используется|объявлен|вызывается|упоминается))\s+",
        r"^(?:поищи(?:\s+в\s+коде)?)\s+",
        r"^(?:найди\s+вхождения)\s+",
        r"^(?:поиск\s+по\s+коду)\s+",
        r"^(?:прогрепай)\s+",
        r"^(?:где\s+(?:объявлен|вызывается|находится|упоминается))\s+",
        r"^(?:найди|поищи)\s+",
        r"^(?:find\s+where)\s+",
        r"^(?:search\s+for|grep\s+for|search\s+in\s+code\s+for)\s+",
        r"^(?:find\s+references\s+to|where\s+is|find\s+all\s+occurrences\s+of)\s+",
        r"^(?:locate|grep)\s+",
    ]

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        text = query.strip()

        for pat in self.PREFIXES:
            new_text, n = re.subn(pat, "", text, count=1, flags=re.IGNORECASE)
            if n > 0:
                text = new_text.strip()
                break

        # Check for quoted query: e.g. найди 'def run' в src/
        quote_match = re.search(r"['\"]([^'\"]+)['\"]", text)
        target_path = "."

        # Check for "in <dir>" or "в <dir>"
        in_match = re.search(
            r"\s+(?:в\s+папке|в\s+директории|в|in\s+folder|in\s+dir|in)\s+([a-zA-Z0-9_\-\./\\]+)\s*$",
            text,
            re.IGNORECASE,
        )
        if in_match:
            target_path = in_match.group(1).replace("\\", "/")
            text = text[: in_match.start()].strip()

        if quote_match:
            q_str = quote_match.group(1).strip()
        else:
            q_str = text.strip()

        q_str = q_str.strip("'\"")
        if not q_str:
            return None

        return {"query": q_str, "path": target_path}


class SearchFilesSpecialist(BaseSpecialist):
    """Extracts pattern and directory for search_files."""

    PREFIXES = [
        r"^(?:найди\s+файлы(?:\s+с\s+именем)?)\s+",
        r"^(?:поищи\s+файлы(?:\s+с\s+именем)?)\s+",
        r"^(?:список\s+файлов\s+по\s+маске)\s+",
        r"^(?:где\s+лежат\s+файлы)\s+",
        r"^(?:поиск\s+файлов)\s+",
        r"^(?:найди\s+все\s+файлы)\s+",
        r"^(?:find\s+files\s+matching)\s+",
        r"^(?:search\s+for\s+files\s+named)\s+",
        r"^(?:locate\s+files\s+with\s+pattern)\s+",
        r"^(?:find\s+all\s+files|search\s+files)\s+",
        r"^(?:find\s+file\s+by\s+name|glob\s+for)\s+",
    ]

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        text = query.strip()
        for pat in self.PREFIXES:
            new_text, n = re.subn(pat, "", text, count=1, flags=re.IGNORECASE)
            if n > 0:
                text = new_text.strip()
                break

        target_dir = "."
        in_match = re.search(
            r"\s+(?:в\s+папке|в\s+директории|в|in\s+folder|in\s+dir|in)\s+([a-zA-Z0-9_\-\./\\]+)\s*$",
            text,
            re.IGNORECASE,
        )
        if in_match:
            target_dir = in_match.group(1).replace("\\", "/")
            text = text[: in_match.start()].strip()

        pattern = text.strip().strip("'\"")
        if not pattern:
            return None

        return {"pattern": pattern, "directory": target_dir}


class RunCommandSpecialist(BaseSpecialist):
    """Extracts and validates shell commands for run_command."""

    PREFIXES = [
        r"^(?:запусти\s+команду)\s+",
        r"^(?:выполни\s+команду)\s+",
        r"^(?:запусти\s+в\s+терминале)\s+",
        r"^(?:выполни\s+в\s+шелле)\s+",
        r"^(?:запусти|выполни|прогони)\s+",
        r"^(?:run\s+command|execute\s+command)\s+",
        r"^(?:run\s+in\s+terminal|run\s+shell\s+command)\s+",
        r"^(?:run|execute|exec)\s+",
    ]

    SAFE_COMMAND_PREFIXES = (
        "git ",
        "pytest",
        "python -m pytest",
        "python -m unittest",
        "python --version",
        "python -V",
        "node --version",
        "npm test",
        "npm run test",
        "npm run build",
        "cargo test",
        "cargo check",
        "go test",
        "dir",
        "ls",
        "ruff check",
        "flake8",
        "mypy",
        "black --check",
    )

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        text = query.strip()
        for pat in self.PREFIXES:
            new_text, n = re.subn(pat, "", text, count=1, flags=re.IGNORECASE)
            if n > 0:
                text = new_text.strip()
                break

        cmd = text.strip().strip("'\"`")
        if not cmd:
            return None

        # Check if safe for immediate fast-path execution
        is_safe = any(cmd.lower().startswith(prefix) for prefix in self.SAFE_COMMAND_PREFIXES)

        return {"command": cmd, "is_safe": is_safe}


class ListDirectorySpecialist(BaseSpecialist):
    """Extracts target directory for list_directory."""

    PREFIXES = [
        r"^(?:покажи\s+файлы\s+в)\s+",
        r"^(?:что\s+лежит\s+в\s+папке)\s+",
        r"^(?:список\s+файлов\s+в\s+директории)\s+",
        r"^(?:покажи\s+содержимое\s+папки)\s+",
        r"^(?:выведи\s+структуру\s+папки)\s+",
        r"^(?:глянь\s+что\s+в\s+папке)\s+",
        r"^(?:покажи\s+директорию|покажи\s+каталог)\s+",
        r"^(?:list\s+files\s+in)\s+",
        r"^(?:show\s+contents\s+of\s+folder)\s+",
        r"^(?:list\s+directory|show\s+files\s+in)\s+",
        r"^(?:inspect\s+folder|list\s+contents\s+of)\s+",
        r"^(?:ls|dir)\s+",
    ]

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        text = query.strip()
        for pat in self.PREFIXES:
            new_text, n = re.subn(pat, "", text, count=1, flags=re.IGNORECASE)
            if n > 0:
                text = new_text.strip()
                break

        d_path = text.strip().strip("'\"")
        if not d_path:
            d_path = "."

        return {"path": d_path.replace("\\", "/")}


class CalculateSpecialist(BaseSpecialist):
    """Extracts mathematical expression for calculate."""

    PREFIXES = [
        r"^(?:посчитай|сколько\s+будет|вычисли|расчитай|каков\s+результат|реши\s+пример)\s+",
        r"^(?:calculate|calc|compute|evaluate|what\s+is|solve)\s+",
    ]

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        text = query.strip()
        has_prefix = False
        for pat in self.PREFIXES:
            new_text, n = re.subn(pat, "", text, count=1, flags=re.IGNORECASE)
            if n > 0:
                text = new_text.strip()
                has_prefix = True
                break

        expr = text.strip().rstrip("?").strip()
        if not expr:
            return None

        # Check for natural language prose (words with 4+ letters)
        prose_words = re.findall(r"[a-zA-Zа-яА-ЯёЁ]{4,}", expr)
        # If there are multiple prose words, it is conversational/instructional text, not a math expression
        if len(prose_words) >= 2:
            return None

        # Without an explicit calculate prefix, expression must contain math operators or functions
        if not has_prefix:
            has_math_op = bool(re.search(r"[\d][\s]*[\+\-\*\/\^\%][\s]*[\d]", expr))
            has_math_func = bool(re.search(r"\b(?:sqrt|sin|cos|tan|log|pow|abs)\b", expr, re.IGNORECASE))
            if not (has_math_op or has_math_func):
                return None

        # Clean math characters
        return {"expression": expr}


class SearchWebSpecialist(BaseSpecialist):
    """Extracts search query for search_web."""

    PREFIXES = [
        r"^(?:найди\s+в\s+интернете|погугли|поищи\s+в\s+сети|найди\s+инфу\s+про|поищи\s+в\s+гугле)\s+",
        r"^(?:поиск\s+в\s+яндексе|пробей\s+в\s+интернете|найди\s+информацию\s+о)\s+",
        r"^(?:search\s+the\s+web\s+for|google|look\s+up\s+online|search\s+internet\s+for)\s+",
        r"^(?:find\s+info\s+on|web\s+search\s+for|search\s+duckduckgo\s+for)\s+",
    ]

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        text = query.strip()
        for pat in self.PREFIXES:
            new_text, n = re.subn(pat, "", text, count=1, flags=re.IGNORECASE)
            if n > 0:
                text = new_text.strip()
                break

        q = text.strip().strip("'\"")
        if not q:
            return None
        return {"query": q}


class ReadWebpageSpecialist(BaseSpecialist):
    """Extracts target URL for read_webpage."""

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        # Find first http or https URL
        url_match = re.search(r"https?://[^\s'\"<>]+", query)
        if url_match:
            return {"url": url_match.group(0)}
        return None


class DeleteFileSpecialist(BaseSpecialist):
    """Extracts file_path for delete_file."""

    PREFIXES = [
        r"^(?:удали\s+файл|сотри\s+файл|удали\s+временный\s+файл|снеси\s+файл|уничтожь\s+файл|удали|убери)\s+",
        r"^(?:delete\s+file|remove\s+file|erase\s+file|drop\s+file|delete|remove|rm|unlink)\s+",
    ]

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        text = query.strip()
        for pat in self.PREFIXES:
            new_text, n = re.subn(pat, "", text, count=1, flags=re.IGNORECASE)
            if n > 0:
                text = new_text.strip()
                break

        path = text.strip().strip("'\"")
        if not path:
            return None
        return {"file_path": path.replace("\\", "/")}


class CreateDirectorySpecialist(BaseSpecialist):
    """Extracts path for create_directory."""

    PREFIXES = [
        r"^(?:создай\s+папку|создай\s+директорию|сделай\s+директорию|создай\s+каталог|сделай\s+папку|mkdir)\s+",
        r"^(?:create\s+directory|create\s+folder|make\s+directory|make\s+folder|create\s+dir|mkdir)\s+",
    ]

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        text = query.strip()
        for pat in self.PREFIXES:
            new_text, n = re.subn(pat, "", text, count=1, flags=re.IGNORECASE)
            if n > 0:
                text = new_text.strip()
                break

        path = text.strip().strip("'\"")
        if not path:
            return None
        return {"path": path.replace("\\", "/")}


class ReadGitDiffSpecialist(BaseSpecialist):
    """Extracts optional file filter for read_git_diff."""

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        # Check if a specific file was requested, e.g. "git diff src/agent.py"
        text = query.strip()
        m = re.search(r"(?:diff|дифф)\s+([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)", text, re.IGNORECASE)
        if m:
            return {"file": m.group(1).replace("\\", "/")}
        return {}


class NoArgSpecialist(BaseSpecialist):
    """Specialist for tools requiring no arguments (e.g. list_desktop_windows, list_skills, git_rollback)."""

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        return {}


class SymbolSpecialist(BaseSpecialist):
    """Extracts symbol name for find_definition, find_references, etc."""

    PREFIXES = [
        r"^(?:где\s+(?:определен|объявлен|вызывается))\s+",
        r"^(?:найди\s+(?:объявление|определение|ссылки|реализации))\s+",
        r"^(?:find\s+(?:definition|references|implementations)\s+of)\s+",
        r"^(?:where\s+is\s+defined)\s+",
    ]

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        text = query.strip()
        for pat in self.PREFIXES:
            new_text, n = re.subn(pat, "", text, count=1, flags=re.IGNORECASE)
            if n > 0:
                text = new_text.strip()
                break
        words = text.split()
        if not words:
            return None
        # Extract last or quoted word
        quote_m = re.search(r"['\"]([^'\"]+)['\"]", text)
        if quote_m:
            sym = quote_m.group(1)
        else:
            sym = words[-1].strip(" ,;:")
        return {"symbol": sym}


class WindowSpecialist(BaseSpecialist):
    """Extracts window title for capture_desktop_window."""

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"(?:окна|окно|window)\s+['\"]?([^'\"]+)['\"]?", query, re.IGNORECASE)
        if m:
            return {"title": m.group(1).strip()}
        # Fallback to last token
        parts = query.split()
        return {"title": parts[-1].strip()} if parts else None


class SkillSpecialist(BaseSpecialist):
    """Extracts skill name for read_skill."""

    def extract(self, query: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"(?:скилл|скилла|skill)\s+([a-zA-Z0-9_\-]+)", query, re.IGNORECASE)
        if m:
            return {"skill_name": m.group(1).strip()}
        return None


# Registry of specialists for all routine and direct tools
SPECIALISTS: Dict[str, BaseSpecialist] = {
    # File & Search
    "read_file": ReadFileSpecialist(),
    "grep_search": GrepSearchSpecialist(),
    "search_files": SearchFilesSpecialist(),
    "run_command": RunCommandSpecialist(),
    "run_admin_command": RunCommandSpecialist(),
    "list_directory": ListDirectorySpecialist(),
    "calculate": CalculateSpecialist(),
    "search_web": SearchWebSpecialist(),
    "read_webpage": ReadWebpageSpecialist(),
    "delete_file": DeleteFileSpecialist(),
    "create_directory": CreateDirectorySpecialist(),
    "read_git_diff": ReadGitDiffSpecialist(),

    # Code Navigation & Diagnostics
    "find_definition": SymbolSpecialist(),
    "find_references": SymbolSpecialist(),
    "find_implementations": SymbolSpecialist(),
    "search_workspace_symbols": SymbolSpecialist(),
    "get_call_hierarchy": SymbolSpecialist(),
    "get_file_outline": ReadFileSpecialist(),
    "check_code": ReadFileSpecialist(),
    "run_deep_linter": ReadFileSpecialist(),
    "analyze_project": NoArgSpecialist(),

    # Desktop & Windows
    "list_desktop_windows": NoArgSpecialist(),
    "capture_desktop_window": WindowSpecialist(),
    "capture_desktop_screen": NoArgSpecialist(),
    "get_clipboard_image": NoArgSpecialist(),

    # Git & Checkpoints
    "git_checkpoint": NoArgSpecialist(),
    "git_rollback": NoArgSpecialist(),

    # Skills & MCP
    "list_skills": NoArgSpecialist(),
    "read_skill": SkillSpecialist(),
    "list_mcp_tools": NoArgSpecialist(),

    # Background Tasks
    "list_background_commands": NoArgSpecialist(),
    "wait_heartbeat": NoArgSpecialist(),

    # Browser Direct Tools
    "browser_open": ReadWebpageSpecialist(),
    "browser_state": NoArgSpecialist(),
    "browser_screenshot": NoArgSpecialist(),
    "browser_close": NoArgSpecialist(),
    "browser_reload": NoArgSpecialist(),
    "browser_back": NoArgSpecialist(),
    "browser_get_content": NoArgSpecialist(),
    "view_image": ReadFileSpecialist(),
}


def extract_specialist_args(intent: str, query: str) -> Optional[Dict[str, Any]]:
    """Dispatches query to the specialist for the given intent.

    If tool has a registered specialist, extracts arguments.
    If tool has no specialist, returns None to ensure LLM safely handles argument structuring.
    """
    spec = SPECIALISTS.get(intent)
    if spec is None:
        return None
    return spec.extract(query)
