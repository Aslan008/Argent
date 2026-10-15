import os
import re
from pathlib import Path
import yaml
import questionary
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from rich.markup import escape
from rich.syntax import Syntax
from rich.progress import Progress, BarColumn, TextColumn

# --- LOAD THEME ---
# Search order: an explicit ARGENT_THEME override, then the current working
# directory (project-local theme), then next to this package, then ~/.argent.
# The old code only checked the CWD, so launching Argent from any other folder
# silently dropped the user's theme.
def _find_theme_file():
    candidates = []
    env = os.environ.get("ARGENT_THEME")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.cwd() / "theme.yaml")
    candidates.append(Path(__file__).resolve().parent / "theme.yaml")
    candidates.append(Path.home() / ".argent" / "theme.yaml")
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


THEME_FILE = _find_theme_file()
default_theme_data = {
    "colors": {
        "user_prompt": "bright_white",
        "user_prompt_symbol": "#58a6ff",
        "assistant_header": "bold #58a6ff",
        "assistant_border": "#388bfd",
        "assistant_text": "white",
        "system_msg": "dim #8b949e",
        "error_msg": "bold #f85149",
        "tool_start": "#58a6ff",
        "tool_name": "bold #79c0ff",
        "tool_args": "#7ee787",
        "tool_success": "#3fb950",
        "tool_result": "#c9d1d9",
        "tool_error": "#f85149",
        "tool_end_border": "dim #30363d",
        "reasoning_label": "bold #8b949e",
        "reasoning_text": "dim #8b949e"
    },
    "settings": {
        "content_width": 110,
        "syntax_theme": "one-dark",
        "auto_save_chat": True
    }
}

theme_data = default_theme_data.copy()
if THEME_FILE is not None:
    try:
        with open(THEME_FILE, "r", encoding="utf-8") as f:
            user_theme = yaml.safe_load(f)
            if user_theme and "colors" in user_theme:
                theme_data["colors"].update(user_theme["colors"])
            if user_theme and "settings" in user_theme:
                theme_data["settings"].update(user_theme["settings"])
    except Exception:
        pass

c = theme_data["colors"]
s = theme_data["settings"]

# Create a custom Rich theme map based on yaml
custom_theme = Theme({
    "sys": c["system_msg"],
    "err": c["error_msg"],
    "tool_start": c["tool_start"],
    "tool_name": c["tool_name"],
    "tool_args": c["tool_args"],
    "tool_success": c.get("tool_success", "#3fb950"),
    "tool_result": c.get("tool_result", "#c9d1d9"),
    "tool_error": c.get("tool_error", "#f85149"),
    "assistant_header": c.get("assistant_header", "bold #58a6ff"),
    "user_symbol": c.get("user_prompt_symbol", "#58a6ff"),
    "user": c["user_prompt"]
})

import sys
# Create console with safe_box to avoid unicode box drawing errors on windows
console = Console(theme=custom_theme, safe_box=True)

# Helper function to monkey-patch or catch global print errors
def _kw(out):
    """Keyword arguments for safe_print: `out` only when one was given."""
    return {"out": out} if out is not None else {}


def safe_print(*args, out=None, **kwargs):
    """Print through the global console, or through `out` when given.

    `out` lets the TUI bridge render the very same blocks into its own width-
    bound capture console instead of swapping the module global (which would
    misroute a concurrent print from another thread).
    """
    target = out if out is not None else console
    try:
        target.print(*args, **kwargs)
    except UnicodeEncodeError:
        try:
            raw_text = " ".join(str(a) for a in args)
            enc = sys.stdout.encoding or 'utf-8'
            # Only builtins.print's own kwargs may be forwarded: passing Rich
            # options (style=, markup=, highlight=…) raises TypeError, which the
            # outer guard swallowed — losing the message this fallback exists to
            # deliver.
            passthrough = {k: v for k, v in kwargs.items()
                           if k in ("sep", "end", "file", "flush")}
            print(raw_text.encode(enc, errors='replace').decode(enc), **passthrough)
        except Exception:
            pass

# ----------------------------------------------------

def _render_code_blocks(text: str) -> list:
    """Parse markdown text, extract fenced code blocks, and return Rich renderables.
    Renders syntax with background highlighting (no box panels) for clean copy-paste."""
    elements = []
    pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    last_end = 0

    for match in pattern.finditer(text):
        before = text[last_end:match.start()]
        if before.strip():
            elements.append(Markdown(before.strip(), code_theme=s["syntax_theme"]))

        lang = match.group(1) or "text"
        code = match.group(2).rstrip('\n')

        syntax = Syntax(
            code,
            lang,
            theme=s["syntax_theme"],
            line_numbers=False,
            word_wrap=False,
            background_color="#181a1f",
            padding=(0, 1)
        )
        title_text = f"[dim]──[/dim] [bold cyan]{lang}[/bold cyan] [dim]──[/dim]"
        elements.append(Group(Text.from_markup(title_text), syntax))
        last_end = match.end()

    remainder = text[last_end:]
    if remainder.strip():
        elements.append(Markdown(remainder.strip(), code_theme=s["syntax_theme"]))

    return elements

def print_markdown(text: str, thinking: str = None, out=None):
    """Prints AI text directly without outer border panel for easy copy-paste."""
    clean_text = text.strip('\n')
    
    from rich.tree import Tree
    
    if thinking:
        reasoning_md = Markdown(thinking.strip(), code_theme=s["syntax_theme"])
        reasoning_content = Panel(
            reasoning_md,
            border_style="dim",
            style=c.get("reasoning_text", "dim white"),
            padding=(0, 1),
            expand=False
        )
        tree = Tree(f"[{c.get('reasoning_label', 'bold #607d8b')}]Анализ[/{c.get('reasoning_label', 'bold #607d8b')}]")
        tree.add(reasoning_content)
        safe_print(tree, **_kw(out))
        safe_print("", **_kw(out))
        
    if clean_text:
        elements = _render_code_blocks(clean_text)
        for el in elements:
            safe_print(el, **_kw(out))
            safe_print("", **_kw(out))

def print_committed_markdown(md: str, out=None):
    """Print one COMMITTED markdown chunk of a streaming answer.

    The stream splitter hands over complete blocks; each is printed once (it is
    never redrawn) followed by a single blank line. Shared by the REPL's live
    renderer and the TUI so a chunk looks identical in both.
    """
    md = md.strip("\n")
    if not md:
        return
    for el in _render_code_blocks(md):
        safe_print(el, **_kw(out))
    safe_print("", **_kw(out))


def print_checkpoint(sha: str, out=None):
    """The one-line notice shown when the time machine snapshots the tree."""
    safe_print(f"[dim]◇ Чекпоинт {sha} создан — откат: /rewind[/dim]", **_kw(out))


def print_banner(version: str, out=None):
    """The rule shown at the top of a fresh session."""
    target = out if out is not None else console
    target.rule(f"[bold cyan]Argent Coder[/bold cyan] [dim]v{version}[/dim]")


def print_assistant_header(out=None):
    """Prints a distinct assistant header badge before model responses."""
    header_style = c.get("assistant_header", "bold #58a6ff")
    safe_print(f"[{header_style}]✦ Argent[/{header_style}]", **_kw(out))

def print_user_prompt(text: str, out=None):
    """Prints user prompt/input with distinct styling for turns/auto-actions."""
    sym_col = c.get("user_prompt_symbol", "#58a6ff")
    usr_col = c.get("user_prompt", "bright_white")
    safe_print(f"[{sym_col}]❯[/{sym_col}] [{usr_col}]{escape(text)}[/{usr_col}]", **_kw(out))

def print_system(text: str, out=None):
    import re
    # If text contains Rich markup tags like [bold cyan]...[/], render them directly.
    # Otherwise, escape the text to prevent accidental interpretation.
    if re.search(r'\[/?[a-z]', text):
        safe_print(f"[sys]{text}[/sys]", **_kw(out))
    else:
        safe_print(f"[sys]{escape(text)}[/sys]", **_kw(out))

def print_error(text: str, out=None):
    """Print one error, with exactly one prefix and a human explanation.

    A caller that already said "Error:" used to get it twice, so a real failure
    reached the user as "Error: Error: 'utf-8' codec can't encode characters in
    position 53000-53001" — a raw Python exception wearing two labels, aimed at
    nobody. The prefix is added only when the text does not carry one, and the
    exceptions users actually hit are translated below the raw line rather than
    instead of it: the original still matters when they report it.
    """
    from error_help import explain_error

    text = str(text)
    body = text if _ERROR_PREFIX_RE.match(text) else f"Ошибка: {text}"
    safe_print(f"[{c['error_msg']}]{escape(body)}[/{c['error_msg']}]", **_kw(out))
    hint = explain_error(text)
    if hint:
        safe_print(f"[dim]{escape(hint)}[/dim]", **_kw(out))


_ERROR_PREFIX_RE = re.compile(r'^\s*(error|ошибка)\b\s*:?', re.IGNORECASE)

def _extract_thought_preview(thinking: str, max_chars: int = 120) -> str:
    """Extract a concise preview (last sentence or key conclusion) from thinking text."""
    if not thinking or not isinstance(thinking, str) or not thinking.strip():
        return ""
    lines = [ln.strip() for ln in thinking.strip().splitlines() if ln.strip()]
    if not lines:
        return ""

    tail = lines[-1]
    if len(tail) < 25 and len(lines) > 1:
        tail = lines[-2] + " " + tail
    # Clean leading markdown bullets, numbers or hashes
    tail = re.sub(r'^[#*>\-\d.]+\s*', '', tail).strip()
    if len(tail) > max_chars:
        tail = tail[:max_chars - 1] + "…"
    return tail

def print_reasoning(thinking: str, duration: float = None, out=None, mode: str = None):
    """Prints a static, finalized reasoning block. In 'summary' mode prints compact Claude Code line; in 'full' mode prints Tree.

    `mode` overrides the configured display mode (the TUI renders both variants
    of the same block so its accordion can switch without re-asking the agent)."""
    if not thinking:
        return
    from config import get_thinking_display_mode
    if (mode or get_thinking_display_mode()) == "summary":
        dur_str = f" · {duration:.1f}s" if duration and duration >= 0.1 else ""
        preview = _extract_thought_preview(thinking)
        preview_text = f" [dim italic]({escape(preview)})[/dim italic]" if preview else ""
        safe_print(f"  [dim #7d8590]◇ Анализ{dur_str}[/dim #7d8590]{preview_text}", **_kw(out))
        return

    from rich.tree import Tree
    reasoning_md = Markdown(thinking.strip(), code_theme=s["syntax_theme"])
    tree = Tree(f"[{c.get('reasoning_label', 'bold #7d8590')}]◇ Анализ[/{c.get('reasoning_label', 'bold #7d8590')}]")
    tree.add(reasoning_md)
    safe_print(tree, **_kw(out))
    safe_print("", **_kw(out)) # Spacer

def print_rethink(rethink: str, duration: float = None, out=None, mode: str = None):
    """Prints a static, finalized rethink/verification block."""
    if not rethink:
        return
    from config import get_thinking_display_mode
    if (mode or get_thinking_display_mode()) == "summary":
        dur_str = f" · {duration:.1f}s" if duration and duration >= 0.1 else ""
        preview = _extract_thought_preview(rethink)
        preview_text = f" [dim italic]({escape(preview)})[/dim italic]" if preview else ""
        safe_print(f"  [bold #d18f52]◈ Перепроверка{dur_str}[/bold #d18f52]{preview_text}", **_kw(out))
        return

    from rich.tree import Tree
    rethink_md = Markdown(rethink.strip(), code_theme=s["syntax_theme"])
    tree = Tree("[bold #d18f52]◈ Перепроверка (Rethink)[/bold #d18f52]")
    tree.add(rethink_md)
    safe_print(tree, **_kw(out))
    safe_print("", **_kw(out)) # Spacer

def create_content_panel(text: str) -> Panel:
    """Creates a content-only AI response panel for Live updates."""
    clean_text = text.strip('\n')
    if not clean_text:
        md = Text("...", style="dim")
    else:
        md = Markdown(clean_text, code_theme=s["syntax_theme"])
        
    return Panel(
        md, 
        border_style=c["assistant_border"], 
        width=min(console.size.width, s["content_width"]),
        expand=False,
        padding=(1, 2)
    )

def create_final_panel(text: str, thinking: str = None) -> list:
    """Returns a list of Rich renderables for the final response.
    No outer border panel — text is directly copyable."""
    clean_text = text.strip('\n')
    
    from rich.tree import Tree
    
    elements = []
    
    if thinking:
        reasoning_md = Markdown(thinking.strip(), code_theme=s["syntax_theme"])
        tree = Tree(f"[{c.get('reasoning_label', 'bold #607d8b')}]Анализ[/{c.get('reasoning_label', 'bold #607d8b')}]")
        tree.add(reasoning_md)
        elements.append(tree)
        elements.append("")
        
    if clean_text:
        code_elements = _render_code_blocks(clean_text)
        elements.extend(code_elements)
    
    if not elements:
        elements.append(Text("...", style="dim"))
        
    return elements

def print_reasoning_header():
    """Prints the reasoning header with the styled label."""
    from rich.text import Text
    safe_print("") # Just one spacer
    safe_print(Text.from_markup(f"[{c.get('reasoning_label', 'bold #607d8b')}]Анализ:[/{c.get('reasoning_label', 'bold #607d8b')}]"))

def create_response_panel(text: str, thinking: str = None) -> Panel:
    """Creates and returns an AI response panel (without printing it). 
    Used for Live displays."""
    clean_text = text.strip('\n')
    
    from rich.console import Group
    from rich.tree import Tree
    
    elements = []
    
    if thinking:
        reasoning_md = Markdown(thinking.strip(), code_theme=s["syntax_theme"])
        # No inner Panel during streaming to prevent nesting/blinking
        tree = Tree(f"[{c.get('reasoning_label', 'bold #607d8b')}]Анализ[/{c.get('reasoning_label', 'bold #607d8b')}]")
        tree.add(reasoning_md)
        elements.append(tree)
        elements.append("")
        
    if clean_text:
        md = Markdown(clean_text, code_theme=s["syntax_theme"])
        elements.append(md)
    
    # Placeholder if empty
    if not elements:
        elements.append(Text("...", style="dim"))
        
    return Panel(
        Group(*elements), 
        border_style=c["assistant_border"], 
        width=min(console.size.width, s["content_width"]),
        expand=False,
        padding=(1, 2)
    )

def _format_duration_short(seconds: float | None) -> str:
    """Format duration in seconds to a friendly short string (e.g. '28ms', '1.4s')."""
    if seconds is None or seconds < 0:
        return ""
    import math
    if math.isnan(seconds) or math.isinf(seconds):
        return ""
    if seconds < 1.0:
        ms = int(seconds * 1000)
        return f"{ms}ms"
    return f"{seconds:.1f}s"


def print_tool_start(name: str, args: dict = None, silent: bool = False):
    # Interactive tools have their own user-facing display
    if name in ("ask_user_questions",):
        return
    args = args or {}
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    
    from config import get_debug_mode
    is_debug = get_debug_mode()
    
    # In inline mode with a live spinner, avoid printing a redundant static line
    if silent and not is_debug:
        return

    limit = 2000 if is_debug else 150
    
    # Keep arguments somewhat truncated if massive
    if len(args_str) > limit:
        args_str = args_str[:limit-3] + "..."
    
    try:
        console.print(f"\n[tool_start]●[/tool_start] [tool_name]{name}[/tool_name][dim]([/dim][tool_args]{args_str}[/tool_args][dim])[/dim]")
    except UnicodeEncodeError:
        try:
            console.print(f"\n[tool_start]>[/] [tool_name]{name}[/]([tool_args]{args_str}[/])")
        except Exception:
            pass
    
def _format_tool_header(name: str, args: dict = None) -> str:
    """Format a crisp, human-readable action header specifying what target is being operated on."""
    args = args or {}
    import os

    # Search / Query targets (prioritize over generic path for grep/find)
    for k in ("query", "pattern", "Query", "search_term"):
        if k in args and args[k]:
            q = str(args[k]).strip()
            if len(q) > 40:
                q = q[:37] + "…"
            in_dir = f" in {os.path.basename(str(args.get('path', '')))}" if "path" in args else ""
            return f"{name}: \"{q}\"{in_dir}"

    # URL / Web targets (prioritize URL over auxiliary params like timeout, headers, etc.)
    for k in ("url", "link", "target_url", "uri", "Url"):
        if k in args and args[k]:
            u = str(args[k]).strip()
            # Modern terminals recognize full URLs and make them clickable (Ctrl+Click).
            # Preserve generous length (up to 250 chars) so full docs/repo links remain intact.
            if len(u) > 250:
                u = u[:247] + "…"
            return f"{name}: {u}"

    # File targets
    for k in ("file_path", "path", "target_file", "file"):
        if k in args and args[k]:
            raw_path = str(args[k]).replace("\\", "/")
            try:
                rel = os.path.relpath(raw_path, os.getcwd()).replace("\\", "/")
                display_path = rel if not rel.startswith("../..") and len(rel) < 45 else os.path.basename(raw_path)
            except Exception:
                display_path = raw_path.split("/")[-1]

            line_suffix = ""
            if "offset" in args or "limit" in args:
                try:
                    off_val = int(args.get("offset", 1) or 1)
                    lim_val = int(args.get("limit")) if args.get("limit") is not None else None
                    line_suffix = f":{off_val}-{off_val + lim_val}" if lim_val is not None else f":{off_val}+"
                except (ValueError, TypeError):
                    off_val = args.get("offset", 1)
                    lim_val = args.get("limit")
                    line_suffix = f":{off_val}" + (f"+{lim_val}" if lim_val is not None else "")
            elif "start_line" in args or "end_line" in args:
                s_line = args.get("start_line", 1)
                e_line = args.get("end_line", "")
                line_suffix = f":{s_line}-{e_line}" if e_line else f":{s_line}+"

            return f"{name}: {display_path}{line_suffix}"

    # Command targets
    if "command" in args and args["command"]:
        cmd = str(args["command"]).strip().replace("\n", " ")
        if len(cmd) > 55:
            cmd = cmd[:52] + "…"
        return f"{name}: {cmd}"

    # Browser targets
    if name.startswith("browser_"):
        action = name.removeprefix("browser_")
        if "url" in args and args["url"]:
            u = str(args["url"]).strip()
            if len(u) > 250:
                u = u[:247] + "…"
            return f"browser_{action}: {u}"
        if "selector" in args and args["selector"]:
            return f"browser_{action}: {args['selector']}"
        if "text" in args and args["text"]:
            return f"browser_{action}: \"{args['text']}\""
        if "expression" in args and args["expression"]:
            expr = str(args["expression"]).strip().replace("\n", " ")
            if len(expr) > 45:
                expr = expr[:42] + "…"
            return f"browser_{action}: {expr}"

    # MCP / Generic sub-tool
    if "tool_name" in args and args["tool_name"]:
        return f"{name}: {args['tool_name']}"

    # Fallback to key scalar arguments
    compact_items = [
        f"{k}={v!r}" for k, v in list(args.items())[:2]
        if not str(v).startswith(("{", "["))
    ]
    if compact_items:
        compact_str = ", ".join(compact_items)
        if len(compact_str) > 45:
            compact_str = compact_str[:42] + "…"
        return f"{name}: {compact_str}"

    return name


def _format_tool_header_markup(name: str, args: dict = None) -> str:
    """Format tool header with colored tool_name and styled target arguments."""
    header = _format_tool_header(name, args)
    prefix = f"{name}: "
    if header.startswith(prefix):
        target = header[len(prefix):]
        return f"[tool_name]{escape(name)}[/tool_name][dim]:[/dim] [bold white]{escape(target)}[/bold white]"
    return f"[tool_name]{escape(header)}[/tool_name]"


def _format_tool_summary(name: str, result: str, args: dict = None) -> str:
    """Format human-readable result summary extracting key status or outcome metrics."""
    text = str(result)
    lines = text.splitlines()
    body = [ln for ln in lines if ln.strip()]
    if not body:
        return "пусто"

    # Pytest / Test runner results
    if name in ("run_command", "start_background_command"):
        for line in reversed(lines):
            line_clean = line.strip()
            if ("passed" in line_clean or "failed" in line_clean) and any(c in line_clean for c in "=,"):
                return line_clean.strip("=").strip()[:80]
        # Git diff / commit statistics
        for line in lines:
            if "file changed" in line or "files changed" in line or "insertions(+)" in line:
                return line.strip()[:80]

    # File edit confirmations
    if name in ("write_file", "replace_in_file"):
        if text.startswith("Successfully wrote"):
            return text if len(text) <= 80 else text[:77] + "…"
        if "successfully replaced" in text.lower() or "applied" in text.lower():
            return "успешно применено"

    # Browser void probes
    if name.startswith("browser_"):
        if text in ("JS result: null", "JS result: undefined", 'JS result: "__ARGENT_UNDEFINED__"'):
            return "Действие выполнено"
        if text.startswith("JS result:"):
            val = text.removeprefix("JS result:").strip().strip('"\'')
            if len(val) <= 60:
                return val
            return val[:57] + "…"

    first = body[0].strip()
    if len(body) == 1:
        return first if len(first) <= 60 else first[:57] + "…"
    return f"{len(lines)} строк · {first[:50]}{'…' if len(first) > 50 else ''}"


def _tool_result_summary(result: str) -> str:
    """One line describing what came back, for the compact view."""
    return _format_tool_summary("", result)


def print_tool_end(name: str, result: str, duration: float = None, args: dict = None,
                   out=None, expanded: bool = None):
    """Print the finished-tool block.

    `expanded` forces the full result panel (True) or the compact two-line view
    (False); None follows the user's configuration. Errors are always expanded.
    """
    target = out if out is not None else console
    # Interactive tools — the user already saw the interaction, no need for a result panel
    if name in ("ask_user_questions",):
        return
    res_preview = str(result)

    from config import get_debug_mode, get_show_tool_results
    is_debug = get_debug_mode()

    dur_str = _format_duration_short(duration)
    dur_tag = f" · {dur_str}" if dur_str else ""

    # Errors are never collapsed: a failure the user cannot see is a failure
    # they will discover three turns later, from the model's behaviour.
    is_error = res_preview.lstrip().lower().startswith("error")
    show_full = get_show_tool_results() if expanded is None else expanded
    if not is_debug and not is_error and not show_full:
        header_markup = _format_tool_header_markup(name, args)
        summary = _format_tool_summary(name, res_preview, args)
        safe_print(
            f"  [dim #388bfd]│[/dim #388bfd] {header_markup}\n"
            f"  [dim #388bfd]│[/dim #388bfd] [dim]↳[/dim] [tool_result]{escape(summary)}[/tool_result]{dur_tag}",
            **_kw(out),
        )
        return

    lines = res_preview.splitlines()
    max_lines = 100 if is_debug else 15

    if len(lines) > max_lines:
        res_preview = "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} lines hidden in UI]"

    title_time = f" ({dur_str})" if dur_str else ""
    border_col = c.get("error_msg", "#f85149") if is_error else c.get("tool_end_border", "#30363d")
    icon = "[tool_error]✗[/tool_error]" if is_error else "[tool_success]✓[/tool_success]"
    status_label = "failed" if is_error else "finished"
    title = f"{icon} [tool_name]{name}[/tool_name] [dim]{status_label}{title_time}[/dim]"

    try:
        panel = Panel(
            f"[tool_result]{escape(res_preview)}[/tool_result]", 
            title=title, 
            title_align="left",
            border_style=border_col,
            width=min(target.size.width, s["content_width"] - 10),
            expand=False
        )
        safe_print(panel, **_kw(out))
    except UnicodeEncodeError:
        panel = Panel(
            f"[tool_result]{escape(res_preview)}[/tool_result]", 
            title=f"{icon} [tool_name]{name}[/tool_name] [dim]{status_label}{title_time}[/dim]", 
            title_align="left",
            border_style=border_col,
            width=min(target.size.width, s["content_width"] - 10),
            expand=False
        )
        safe_print(panel, **_kw(out))


def print_inline_diff(file_path: str, diff_text: str, max_lines: int = 25, out=None):
    """Render a compact syntax-highlighted diff block directly into the chat stream."""
    if not diff_text or not diff_text.strip():
        return
    raw_lines = diff_text.strip().splitlines()
    added = sum(1 for l in raw_lines if l.startswith('+') and not l.startswith('+++'))
    removed = sum(1 for l in raw_lines if l.startswith('-') and not l.startswith('---'))

    stats = f"[green]+{added}[/green], [red]-{removed}[/red]"
    import os
    basename = os.path.basename(file_path) if file_path else "file"
    title = f"[bold cyan]diff {basename}[/bold cyan] [dim]({stats})[/dim]"


    display_lines = raw_lines[:max_lines]
    if len(raw_lines) > max_lines:
        display_lines.append(f"... [{len(raw_lines) - max_lines} more lines hidden — use /diff {basename} for full view]")

    code_snippet = "\n".join(display_lines)
    syntax = Syntax(
        code_snippet,
        "diff",
        theme=s.get("syntax_theme", "one-dark"),
        line_numbers=False,
        word_wrap=False,
        background_color="#181a1f",
        padding=(0, 1)
    )
    panel = Panel(
        syntax,
        title=title,
        title_align="left",
        border_style="dim #388bfd",
        width=min((out if out is not None else console).size.width, s["content_width"] - 6),
        expand=False
    )
    safe_print("", **_kw(out))
    safe_print(panel, **_kw(out))
    safe_print("", **_kw(out))

def context_usage_markup(tokens: int, max_tokens: int, percent: float) -> str:
    """Rich markup of the one-line context meter printed after each response."""
    color = "green"
    if percent > 80:
        color = "red"
    elif percent > 60:
        color = "yellow"
    return f"[dim]Context: [{color}]{percent:.1f}%[/{color}] ({tokens}/{max_tokens} tokens)[/dim]"


def print_context_usage(tokens: int, max_tokens: int, percent: float, out=None):
    """Displays a small progress bar showing context usage."""
    safe_print(context_usage_markup(tokens, max_tokens, percent), **_kw(out))


def approval_header_markup(action: str, destructive: bool) -> str:
    """Rich markup of the line that announces a gated action awaiting a decision.

    The action is user/model-controlled text (a shell command, a path), so it is
    escaped: an unescaped `[` in it used to be parsed as Rich markup and either
    vanished or raised MarkupError in the middle of an approval prompt.
    """
    style = "bold red" if destructive else "bold yellow"
    icon = "▲ " if destructive else "◇ "
    return f"\n[{style}]{icon}Агент запрашивает разрешение: {escape(str(action))}[/{style}]"


def select_model(current_model: str) -> str:
    """Select a model for the active provider. Falls back to text input if questionary fails."""
    from config import get_context_window, set_context_window
    from providers import create_provider

    try:
        provider = create_provider()
    except Exception as e:
        print_error(f"Не удалось создать провайдера: {e}")
        return current_model

    models = provider.list_models()
    problem = getattr(getattr(provider, "_last_catalog", None), "problem", "")
    if problem:
        hint = " Укажите workspace: /key workspace <ID>." if "workspace" in problem.lower() else ""
        print_system(f"[yellow]Список моделей Anthropic недоступен ({problem}).{hint} "
                     f"Показан сохранённый или встроенный список.[/yellow]")

    if not models:
        print_error(f"Модели для {provider.name} не найдены. Проверьте настройки.")
        return current_model

    # OpenRouter exposes hundreds of paid and free models mixed together.
    # Let the user jump straight to the free tier so they don't have to hunt.
    if provider.name == "openrouter":
        free = sorted(m for m in models if provider.is_free_model(m))
        if free and len(models) > len(free):
            scope = _select_from_list(
                "Which OpenRouter models to show?",
                [f"Free only ({len(free)})", f"All models ({len(models)})"],
                f"Free only ({len(free)})",
            )
            if scope and scope.startswith("Free"):
                models = free
        # Surface free models first in the combined list.
        models = provider.sort_free_first(models)

    labels = None
    if provider.name == "claude":
        # "Claude Sonnet 5.5  (claude-sonnet-5-5)": the name people say, and the id it is saved as.
        import claude_catalog
        entries = provider.catalog().entries
        labels = {f"{claude_catalog.display_name(m, entries)}  ({m})": m for m in models}
    if labels:
        by_id = {m: label for label, m in labels.items()}
        picked = _select_from_list(
            f"Select {provider.name.upper()} model:",
            list(labels),
            by_id.get(current_model) or next(iter(labels)),
        )
        selected = labels.get(picked) if picked else None
    else:
        selected = _select_from_list(
            f"Select {provider.name.upper()} model:",
            models,
            current_model if current_model in models else models[0]
        )

    if not selected:
        print_system("[dim]Выбор модели отменён.[/dim]")
        return current_model

    current_ctx = get_context_window()

    # Anchored to what this model can actually take, when that is knowable.
    # The old fixed lists (2048/4096/8192…) had nothing to do with the model in
    # front of you: too small trims the conversation for no reason, too large
    # overflows — which is exactly what happens moving a chat from a cloud
    # model to a local one.
    from model_limits import context_choices
    detected = context_choices(selected, provider.name, current_ctx)
    if detected:
        ctx_choices = detected + ["Оставить текущее", "Своё значение…"]
        prompt = (f"Окно контекста для {selected} "
                  f"(сейчас {current_ctx}, максимум модели {detected[-1].split()[0]}):")
    else:
        if provider.name in ("zai", "openrouter", "claude", "deepseek", "xkiro", "wormsoft"):
            ctx_choices = ["8192", "16384", "32768", "65536", "131072", "200000", "500000", "1000000"]
        else:
            ctx_choices = ["2048", "4096", "8192", "16384", "32768"]
        ctx_choices += ["Оставить текущее", "Своё значение…"]
        prompt = f"Окно контекста для {selected} (сейчас {current_ctx}):"

    ctx_choice = _select_from_list(prompt, ctx_choices, "Оставить текущее")

    if ctx_choice == "Своё значение…":
        custom_val = questionary.text("Размер контекста в токенах (например 32768):").ask()
        if custom_val and custom_val.isdigit():
            set_context_window(int(custom_val))
    elif ctx_choice and ctx_choice != "Оставить текущее":
        try:
            tokens = int(ctx_choice.split()[0].replace(",", ""))
            set_context_window(tokens)
        except (ValueError, IndexError):
            pass

    # --- Step 3: Model classification ---
    from config import (
        get_model_size_category, get_model_category_override,
        set_model_category_override, is_moe_model, get_moe_specs
    )
    # Reset override so auto-detection shows the real value for the new model
    set_model_category_override(None)
    auto_category = "moe" if is_moe_model(selected) else get_model_size_category(selected)
    moe_info = ""
    specs = get_moe_specs(selected)
    if specs:
        moe_info = f" ({specs[0]:.0f}B total / {specs[1]:.0f}B active)"

    category_choices = [
        f"Auto: {auto_category}{moe_info} (рекомендуется)",
        "moe    — MoE (смешанная емкость, минифицированные схемы, factual анкор)",
        "tiny   — <3B, без native tools, минимальная история",
        "small  — 3-7B, native tools, короткая история",
        "medium — 7-13B, native tools, средняя история",
        "large  — >13B, native tools, длинная история",
        "cloud  — облачные API (Gemini, GPT, GLM)",
    ]

    cat_choice = _select_from_list(
        f"Классификация модели (авто: {auto_category}):",
        category_choices,
        category_choices[0]
    )

    if cat_choice and not cat_choice.startswith("Auto"):
        manual_cat = cat_choice.split()[0].strip()
        set_model_category_override(manual_cat)
        if manual_cat == "moe" and not is_moe_model(selected):
            print("\n[Argent Info]: Имя модели не содержит явных признаков MoE, но режим MoE активирован по вашему выбору.")
        elif manual_cat != "moe" and is_moe_model(selected):
            print(f"\n[Argent Info]: Модель распознана как MoE, но режим '{manual_cat}' активирован по вашему выбору.")

    return selected


def _select_from_list(prompt_text: str, choices: list, default: str = None) -> str | None:
    """Try questionary.select, fall back to numbered text input if it fails.

    For long lists (e.g. OpenRouter's hundreds of models) the picker enables
    type-to-filter search so the user can jump straight to a model by name
    instead of scrolling. Returns None on user cancellation (Ctrl+C / ESC)."""
    safe_default = default if default in choices else (choices[0] if choices else None)

    # 1. Native TUI Select Modal (prevents stdin freezing in Rust TUI)
    import os
    if os.environ.get("ARGENT_TUI") == "1":
        try:
            from src.bridge.repl_bridge import ReplBridge
            selected = ReplBridge.ask_active_select(prompt_text, choices, safe_default)
            return selected
        except Exception:
            pass

    try:
        # Type-to-filter pays off once a list is too long to eyeball; short
        # menus stay as plain arrow-key selects.
        if len(choices) > 12:
            return questionary.select(
                prompt_text,
                choices=choices,
                default=safe_default,
                use_search_filter=True,
                use_jk_keys=False,
                instruction="(печатайте для поиска, ↑↓ выбор, Enter подтвердить)",
            ).ask()
        else:
            return questionary.select(
                prompt_text,
                choices=choices,
                default=safe_default,
            ).ask()
    except TypeError:
        # Older questionary without search-filter kwargs: plain select.
        try:
            return questionary.select(prompt_text, choices=choices, default=safe_default).ask()
        except (KeyboardInterrupt, EOFError):
            return None
        except Exception:
            pass
    except (KeyboardInterrupt, EOFError):
        return None
    except Exception:
        pass

    # Terminal without questionary support: fallback to numbered prompt
    print_system(f"\n{prompt_text}")
    for i, choice in enumerate(choices, 1):
        marker = " ← current" if choice == default else ""
        console.print(f"  [cyan]{i}[/cyan]. {choice}{marker}")

    try:
        from prompt_toolkit import prompt as ptk_prompt
        raw = ptk_prompt(f"Enter number or name (1-{len(choices)}): ").strip()
    except (KeyboardInterrupt, EOFError):
        return None

    if not raw:
        return safe_default

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx]

    for choice in choices:
        if raw.lower() == choice.lower():
            return choice

    return default or (choices[0] if choices else None)
