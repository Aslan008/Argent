"""What Argent says in Telegram, and how it looks.

Three jobs, none of which touch the network:

* ``format_html`` turns the Markdown a model writes into the HTML subset Telegram
  accepts (headings, lists, tables, quotes, code, links), and ``repair_html`` makes
  sure the tags it emits are always balanced -- a single stray ``</b>`` would make
  Telegram reject the whole message, and the user would get raw tags instead.
* ``split_html`` cuts a long message into pieces that each fit Telegram's limit
  *without* cutting a tag or an entity in half: a code block that spans two
  messages is closed at the end of the first and reopened at the top of the second.
* ``tool_label`` and the small formatters (durations, token counts, a context bar)
  are shared by the live status panel and the result footer.
"""

import html
import os
import re
from typing import Any, Dict, List, Optional, Tuple

MAX_HTML = 4096          # Telegram's hard limit for one message's text
_CLOSE_MARGIN = 120      # room kept for the closing tags a cut may have to add

_TAGS = ("b", "i", "u", "s", "a", "code", "pre", "blockquote", "tg-spoiler")
_TAG_RE = re.compile(r"<(/?)([a-z][a-z-]*)([^<>]*)>")
_ENTITY_TAIL = re.compile(r"&#?\w{0,10}$")


# ─── small formatters ─────────────────────────────────────────────────────────

def esc(text: Any) -> str:
    """Escape for Telegram HTML (quotes stay as they are: they need no escaping in text)."""
    return html.escape(str(text if text is not None else ""), quote=False)


def one_line(text: Any, limit: int = 60) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: max(1, limit - 1)].rstrip() + "…"


def short_path(path: Any, limit: int = 44) -> str:
    p = str(path or "").replace("\\", "/")
    try:                                   # inside the project a relative path says more than a drive letter
        root = os.getcwd().replace("\\", "/").rstrip("/") + "/"
        if p.lower().startswith(root.lower()):
            p = p[len(root):]
    except OSError:
        pass
    if len(p) <= limit:
        return p
    parts = [x for x in p.split("/") if x]
    tail = "/".join(parts[-2:]) if len(parts) > 1 else p
    return ("…/" + tail) if len(tail) + 2 <= limit else "…" + p[-(limit - 1):]


def fmt_duration(seconds: Optional[float]) -> str:
    """42s -> '42 с', 72s -> '1 мин 12 с', 3900s -> '1 ч 05 мин'."""
    if seconds is None:
        return ""
    s = max(0, int(round(float(seconds))))
    if s < 60:
        return f"{s} с"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m} мин {sec:02d} с" if sec else f"{m} мин"
    h, m = divmod(m, 60)
    return f"{h} ч {m:02d} мин"


def fmt_tokens(n: Any) -> str:
    try:
        v = float(n)
    except (TypeError, ValueError):
        return "?"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1000:
        return f"{v / 1000:.1f}k" if v < 100_000 else f"{v / 1000:.0f}k"
    return str(int(v))


def bar(percent: Any, width: int = 10) -> str:
    """▰▰▰▱▱▱▱▱▱▱ for a 0..100 percentage."""
    try:
        p = max(0.0, min(100.0, float(percent)))
    except (TypeError, ValueError):
        p = 0.0
    filled = int(round(p / 100 * width))
    return "▰" * filled + "▱" * (width - filled)


def steps_word(n: int) -> str:
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return "шаг"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "шага"
    return "шагов"


def turn_footer(snap: Optional[Dict[str, Any]]) -> str:
    """One quiet line under an answer: time, tool calls, tokens, context fill, cost."""
    if not snap:
        return ""
    bits: List[str] = []
    if snap.get("total_s") is not None:
        bits.append("⏱ " + fmt_duration(snap["total_s"]))
    if snap.get("tool_calls"):
        bits.append(f"🛠 {snap['tool_calls']}")
    pt, ct = snap.get("prompt_tokens"), snap.get("completion_tokens")
    if pt or ct:
        bits.append(f"↑{fmt_tokens(pt or 0)} ↓{fmt_tokens(ct or 0)}")
    if snap.get("ctx_percent") is not None:
        bits.append(f"ctx {snap['ctx_percent']:.0f}%" if isinstance(snap["ctx_percent"], (int, float)) else f"ctx {snap['ctx_percent']}%")
    if isinstance(snap.get("cost"), (int, float)) and snap["cost"] > 0:
        bits.append(f"${snap['cost']:.3f}")
    return "<i>" + esc("  ·  ".join(bits)) + "</i>" if bits else ""


# ─── tools, as a person would name them ───────────────────────────────────────

def _a(args: Dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = (args or {}).get(k)
        if v not in (None, "", [], {}):
            return str(v)
    return ""


def _host(url: str) -> str:
    m = re.match(r"^\w+://([^/\s?#]+)([^\s?#]*)", url or "")
    if not m:
        return one_line(url, 48)
    path = m.group(2).rstrip("/")
    return one_line(m.group(1) + path, 48)


# name -> (emoji, verb, how to pick the detail)
_TOOLS: Dict[str, Tuple[str, str, Any]] = {
    "read_file": ("📖", "Чтение", lambda a: short_path(_a(a, "file_path", "path"))),
    "write_file": ("📝", "Запись", lambda a: short_path(_a(a, "file_path", "path"))),
    "replace_in_file": ("✏️", "Правка", lambda a: short_path(_a(a, "file_path", "path"))),
    "delete_file": ("🗑", "Удаление", lambda a: short_path(_a(a, "file_path", "path"))),
    "create_directory": ("📁", "Создание папки", lambda a: short_path(_a(a, "dir_path", "path"))),
    "move_file": ("📦", "Перенос", lambda a: one_line(f"{short_path(_a(a, 'source'), 24)} → {short_path(_a(a, 'destination'), 24)}", 52)),
    "copy_file": ("📋", "Копирование", lambda a: one_line(f"{short_path(_a(a, 'source'), 24)} → {short_path(_a(a, 'destination'), 24)}", 52)),
    "list_directory": ("📂", "Просмотр папки", lambda a: short_path(_a(a, "dir_path", "path") or ".")),
    "search_files": ("🔎", "Поиск файлов", lambda a: one_line(_a(a, "name_contains", "content_contains", "pattern"), 48)),
    "grep_search": ("🔎", "Поиск в коде", lambda a: one_line(_a(a, "pattern", "query", "symbol"), 48)),
    "semantic_search": ("🧭", "Смысловой поиск", lambda a: one_line(_a(a, "query"), 48)),
    "get_file_outline": ("🧩", "Структура файла", lambda a: short_path(_a(a, "file_path"))),
    "analyze_project": ("🗺", "Анализ проекта", lambda a: short_path(_a(a, "path"))),
    "run_deep_linter": ("🧹", "Линтер", lambda a: short_path(_a(a, "path"))),
    "run_command": ("⚡", "Команда", lambda a: one_line(_a(a, "command"), 56)),
    "run_admin_command": ("🛡", "Команда (админ)", lambda a: one_line(_a(a, "command"), 56)),
    "start_background_command": ("🚀", "Фоновая команда", lambda a: one_line(_a(a, "command"), 52)),
    "read_background_command": ("📜", "Вывод фоновой команды", lambda a: _a(a, "pid")),
    "send_background_command": ("⌨️", "Ввод в фоновую команду", lambda a: _a(a, "pid")),
    "stop_background_command": ("⏹", "Остановка фоновой команды", lambda a: _a(a, "pid")),
    "list_background_commands": ("📋", "Фоновые команды", lambda a: ""),
    "search_web": ("🌐", "Поиск в сети", lambda a: one_line(_a(a, "query"), 48)),
    "read_webpage": ("📄", "Чтение страницы", lambda a: _host(_a(a, "url"))),
    "run_deep_research": ("🔬", "Исследование", lambda a: one_line(_a(a, "objective"), 52)),
    "browser_open": ("🧭", "Браузер: открываю", lambda a: _host(_a(a, "url"))),
    "browser_action": ("🖱", "Браузер", lambda a: one_line(" ".join(x for x in (_a(a, "action"), _a(a, "selector", "text", "instruction")) if x), 52)),
    "browser_inspect": ("👁", "Браузер: осмотр", lambda a: one_line(_a(a, "mode", "query", "instruction"), 40)),
    "browser_close": ("🧭", "Браузер: закрываю", lambda a: ""),
    "capture_desktop_screen": ("📸", "Снимок экрана", lambda a: ""),
    "capture_desktop_window": ("📸", "Снимок окна", lambda a: one_line(_a(a, "target"), 40)),
    "list_desktop_windows": ("🪟", "Список окон", lambda a: ""),
    "get_clipboard_image": ("📋", "Изображение из буфера", lambda a: ""),
    "view_image": ("🖼", "Просмотр изображения", lambda a: short_path(_a(a, "file_path", "path"))),
    "git_checkpoint": ("💾", "Контрольная точка", lambda a: one_line(_a(a, "message"), 48)),
    "git_rollback": ("⏪", "Откат", lambda a: one_line(_a(a, "to_checkpoint"), 40)),
    "read_git_diff": ("🔀", "Git diff", lambda a: ""),
    "run_subagent": ("🤖", "Подагент", lambda a: one_line(_a(a, "task", "role"), 52)),
    "subagents": ("🤖", "Подагенты", lambda a: one_line(_a(a, "action"), 30)),
    "ultracode": ("🚀", "Ultracode", lambda a: one_line(_a(a, "goal", "action"), 60)),
    "rooms": ("🚪", "Комната", lambda a: one_line(_a(a, "room", "task", "action"), 60)),
    "memory": ("🧠", "Память", lambda a: one_line(_a(a, "name", "query", "action"), 40)),
    "desktop_inspect": ("🖥", "Окно программы", lambda a: one_line(_a(a, "window", "action"), 40)),
    "desktop_act": ("🖱", "Действие в программе", lambda a: one_line(_a(a, "action", "window"), 40)),
    "delegate_to_leeth": ("🧠", "Leeth", lambda a: one_line(_a(a, "task"), 48)),
    "leeth_map": ("🧠", "Leeth: список", lambda a: one_line(_a(a, "criterion"), 48)),
    "leeth_scan_page": ("🧠", "Leeth: страница", lambda a: one_line(_a(a, "criterion"), 48)),
    "leeth_click_many": ("🧠", "Leeth: клики", lambda a: one_line(_a(a, "indices"), 40)),
    "leeth_fill_form": ("🧾", "Leeth: форма", lambda a: one_line(_a(a, "instruction"), 52)),
    "todo_write": ("☑️", "План задач", lambda a: ""),
    "exit_plan_mode": ("📋", "План готов", lambda a: ""),
    "ask_user_questions": ("❓", "Вопрос вам", lambda a: ""),
    "request_user_approval": ("🔐", "Запрос подтверждения", lambda a: one_line(_a(a, "message"), 48)),
    "wait_heartbeat": ("⏳", "Ожидание", lambda a: one_line(_a(a, "until", "condition_to_check"), 48)),
    "call_mcp_tool": ("🔌", "MCP", lambda a: one_line(f"{_a(a, 'server_name')}.{_a(a, 'tool_name')}".strip("."), 48)),
    "list_mcp_tools": ("🔌", "MCP: список инструментов", lambda a: _a(a, "server_name")),
    "manage_automations": ("🗓", "Автоматизации", lambda a: one_line(" ".join(x for x in (_a(a, "action"), _a(a, "name")) if x), 40)),
    "calculate": ("🧮", "Вычисление", lambda a: one_line(_a(a, "expression"), 48)),
    "create_artifact": ("🎨", "Артефакт", lambda a: short_path(_a(a, "filename"))),
    "read_skill": ("🎓", "Навык", lambda a: _a(a, "name")),
    "list_skills": ("🎓", "Навыки", lambda a: ""),
    "filter_new_items": ("🧺", "Отбор нового", lambda a: one_line(_a(a, "label"), 32)),
}


def tool_label(name: str, args: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str]:
    """(emoji, verb, detail) for a tool call -- plain text, escape before sending."""
    args = args if isinstance(args, dict) else {}
    entry = _TOOLS.get(name or "")
    if entry is None:
        return "🔧", (name or "инструмент").replace("_", " "), ""
    emoji, verb, pick = entry
    try:
        detail = str(pick(args) or "")
    except Exception:
        detail = ""
    return emoji, verb, detail


def tool_line(name: str, args: Optional[Dict[str, Any]] = None) -> str:
    emoji, verb, detail = tool_label(name, args)
    return f"{emoji} {esc(verb)}" + (f" <code>{esc(detail)}</code>" if detail else "")


def diff_counts(diff_text: str) -> Tuple[int, int]:
    added = removed = 0
    for line in (diff_text or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


# ─── Markdown -> Telegram HTML ────────────────────────────────────────────────

_FENCE_RE = re.compile(r"(`{3,}|~{3,})(?:([\w+#.\-]+)[ \t]*\n|[ \t]*\n)?(.*?)(?:\1|\Z)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_HEADER_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_HR_RE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+•]\s+(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.*)$")
_TASK_RE = re.compile(r"^\[( |x|X)\]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")
_BULLETS = ("•", "◦", "▪", "▫")
_INDENT = " "
_WIDE_TABLE = 42


def _inline(s: str) -> str:
    """Inline Markdown on already-escaped text."""
    def _url(u: str) -> str:
        return u.replace('"', "%22")

    s = re.sub(r"!\[([^\]]*)\]\((https?://[^\s)]+)\)", lambda m: f'<a href="{_url(m.group(2))}">{m.group(1) or "изображение"}</a>', s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", lambda m: f'<a href="{_url(m.group(2))}">{m.group(1)}</a>', s)
    s = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])", r"<i>\1</i>", s)
    s = re.sub(r"(?<![\w_])_(?=[^\s_])([^_\n]+?)(?<=[^\s_])_(?![\w_])", r"<i>\1</i>", s)
    s = re.sub(r"~~(?=\S)(.+?)(?<=\S)~~", r"<s>\1</s>", s)
    return s


def _plain_cell(cell: str, inline_codes: List[str]) -> str:
    cell = re.sub(r"\x00IC(\d+)\x00", lambda m: inline_codes[int(m.group(1))], cell)
    cell = re.sub(r"\*\*|__|~~", "", cell)
    cell = re.sub(r"\[([^\]]+)\]\(https?://[^\s)]+\)", r"\1", cell)
    return cell.strip()


def _render_table(rows: List[List[str]]) -> str:
    """A grid in <pre> when it fits a phone, otherwise one card per row."""
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    widths = [max(len(r[c]) for r in rows) for c in range(ncols)]
    if sum(widths) + 2 * (ncols - 1) <= _WIDE_TABLE:
        def line(r: List[str]) -> str:
            return "  ".join(r[c].ljust(widths[c]) for c in range(ncols)).rstrip()
        out = [line(rows[0]), "  ".join("─" * widths[c] for c in range(ncols))]
        out += [line(r) for r in rows[1:]]
        return "<pre>" + esc("\n".join(out)) + "</pre>"
    head = rows[0]
    cards = []
    for r in rows[1:]:
        title = r[0] or "—"
        lines = [f"<b>{esc(title)}</b>"]
        for c in range(1, ncols):
            if r[c]:
                lines.append(f"{_INDENT}• {esc(head[c] or '·')}: {esc(r[c])}")
        cards.append("\n".join(lines))
    return "\n\n".join(cards)


def format_html(text: Optional[str]) -> str:
    """Markdown -> the HTML subset Telegram understands. Always balanced, always escaped."""
    if not text:
        return ""

    blocks: List[str] = []

    def _fence(m: "re.Match[str]") -> str:
        lang, code = m.group(2) or "", m.group(3).rstrip("\n")
        attr = f' class="language-{html.escape(lang, quote=True)}"' if lang else ""
        blocks.append(f"<pre><code{attr}>{esc(code)}</code></pre>")
        return f"\x00CB{len(blocks) - 1}\x00"

    s = _FENCE_RE.sub(_fence, text.replace("\r\n", "\n"))

    inline_codes: List[str] = []

    def _icode(m: "re.Match[str]") -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    s = _INLINE_CODE_RE.sub(_icode, s)

    lines = s.split("\n")
    out: List[str] = []
    i = 0
    indent_stack: List[int] = []
    while i < len(lines):
        line = lines[i]

        # table: header row + separator row + body rows
        if _TABLE_ROW.match(line) and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            rows = []
            j = i
            while j < len(lines) and _TABLE_ROW.match(lines[j]):
                if j != i + 1:
                    rows.append([_plain_cell(c, inline_codes) for c in lines[j].strip().strip("|").split("|")])
                j += 1
            out.append(_render_table(rows))
            i = j
            indent_stack = []
            continue

        # block quote: consecutive "> " lines
        if _QUOTE_RE.match(line):
            quoted = []
            while i < len(lines) and _QUOTE_RE.match(lines[i]):
                quoted.append(_inline(esc(_QUOTE_RE.match(lines[i]).group(1))))
                i += 1
            body = "\n".join(quoted).strip()
            tag = "blockquote expandable" if len(body) > 480 or len(quoted) > 7 else "blockquote"
            out.append(f"<{tag}>{body}</blockquote>")
            indent_stack = []
            continue

        m = _HEADER_RE.match(line)
        if m:
            title = m.group(2).replace("**", "").replace("__", "")
            out.append(f"<b>{_inline(esc(title))}</b>")
            indent_stack = []
            i += 1
            continue

        if _HR_RE.match(line):
            out.append("──────────")
            indent_stack = []
            i += 1
            continue

        m = _BULLET_RE.match(line)
        ordered = None if m else _ORDERED_RE.match(line)
        if m or ordered:
            raw_indent = (m or ordered).group(1).expandtabs(4)
            width = len(raw_indent)
            while indent_stack and indent_stack[-1] > width:
                indent_stack.pop()
            if not indent_stack or indent_stack[-1] < width:
                indent_stack.append(width)
            depth = min(len(indent_stack) - 1, 3)
            if m:
                body = m.group(2)
                task = _TASK_RE.match(body)
                if task:
                    marker = "☑" if task.group(1) in "xX" else "☐"
                    body = task.group(2)
                else:
                    marker = _BULLETS[depth]
            else:
                marker, body = f"{ordered.group(2)}.", ordered.group(3)
            out.append(f"{_INDENT * depth}{marker} {_inline(esc(body))}")
            i += 1
            continue

        indent_stack = []
        out.append(_inline(esc(line)))
        i += 1

    result = "\n".join(out)
    result = re.sub(r"\n{3,}", "\n\n", result).strip("\n")
    for idx, block in enumerate(blocks):
        result = result.replace(f"\x00CB{idx}\x00", block)
    for idx, code in enumerate(inline_codes):
        result = result.replace(f"\x00IC{idx}\x00", f"<code>{esc(code)}</code>")
    return repair_html(result)


def repair_html(s: str) -> str:
    """Balance the tags: close what was left open, drop what was never opened, and
    re-nest what crossed (``<b>a<i>b</b>c</i>`` becomes ``<b>a<i>b</i></b><i>c</i>``)."""
    out: List[str] = []
    stack: List[Tuple[str, str]] = []
    pos = 0
    for m in _TAG_RE.finditer(s):
        out.append(s[pos:m.start()])
        pos = m.end()
        closing, name = m.group(1), m.group(2)
        base = name
        if base not in _TAGS:
            out.append(esc(m.group(0)))      # not ours: show it as text
            continue
        if not closing:
            stack.append((base, m.group(0)))
            out.append(m.group(0))
            continue
        depth = next((k for k in range(len(stack) - 1, -1, -1) if stack[k][0] == base), None)
        if depth is None:
            continue                          # a closer with no opener
        above = stack[depth + 1:]
        out.extend(f"</{n}>" for n, _ in reversed(above))
        out.append(f"</{base}>")
        out.extend(tag for _, tag in above)
        del stack[depth]
    out.append(s[pos:])
    out.extend(f"</{n}>" for n, _ in reversed(stack))
    return "".join(out)


def strip_html(s: str) -> str:
    """The text of an HTML message, for the plain-text retry when Telegram refuses the markup."""
    s = re.sub(r"</(?:pre|blockquote|p)>", "\n", s)
    return html.unescape(re.sub(r"<[^<>]+>", "", s)).strip()


# ─── splitting ────────────────────────────────────────────────────────────────

def _scan(stack: List[Tuple[str, str]], piece: str) -> List[Tuple[str, str]]:
    stack = list(stack)
    for m in _TAG_RE.finditer(piece):
        name = m.group(2)
        if name not in _TAGS:
            continue
        if not m.group(1):
            stack.append((name, m.group(0)))
        else:
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == name:
                    del stack[k:]
                    break
    return stack


def _cut_point(rest: str, budget: int) -> int:
    window = rest[:budget]
    floor = budget // 3
    for sep in ("\n\n", "\n", " "):
        idx = window.rfind(sep)
        if idx > floor:
            cut = idx + len(sep)
            break
    else:
        cut = budget
    head = rest[:cut]
    lt, gt = head.rfind("<"), head.rfind(">")
    if lt > gt:                                # inside a tag
        cut = lt
    else:
        m = _ENTITY_TAIL.search(head)
        if m and "&" in m.group(0):            # inside an entity
            cut = m.start()
    return cut if cut > 0 else min(budget, len(rest))


def split_html(text: str, limit: int = MAX_HTML) -> List[str]:
    """Pieces of at most ``limit`` characters, each valid HTML on its own."""
    if len(text) <= limit:
        return [text] if text else []
    limit = max(limit, 200)
    pieces: List[str] = []
    stack: List[Tuple[str, str]] = []
    rest = text
    while rest:
        reopen = "".join(tag for _, tag in stack)
        if len(reopen) + len(rest) <= limit:
            pieces.append(reopen + rest)
            break
        budget = max(50, limit - len(reopen) - _CLOSE_MARGIN)
        cut = _cut_point(rest, budget)
        piece = rest[:cut]
        new_stack = _scan(stack, piece)
        closers = "".join(f"</{n}>" for n, _ in reversed(new_stack))
        body = piece.rstrip()
        if body:
            pieces.append(reopen + body + closers)
        stack = new_stack
        rest = rest[cut:].lstrip("\n")
    return [p for p in pieces if p.strip()]


def format_markdown_for_telegram(text: Optional[str], limit: int = 3500) -> List[str]:
    """Markdown -> a list of Telegram-HTML messages, split at paragraph boundaries."""
    if not text:
        return []
    return split_html(format_html(text), min(max(limit, 200), MAX_HTML))
