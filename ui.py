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
        "assistant_border": "dodger_blue1",
        "assistant_text": "white",
        "system_msg": "dim italic white",
        "error_msg": "bold red",
        "tool_start": "cyan",
        "tool_name": "cyan",
        "tool_args": "green",
        "tool_end_border": "dim white",
        "reasoning_label": "bold #607d8b",
        "reasoning_text": "dim #607d8b"
    },
    "settings": {
        "content_width": 110,
        # Was "monokai", whose operator colour is #ff4689 — hot pink. Every
        # "=", ".", ":" and "->" in every code block and diff was rendered in
        # it, which is why the terminal read as pink overall. Sober
        # alternatives, if you prefer: "nord" (no pink at all), "github-dark".
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
    "user": c["user_prompt"]
})

import sys
# Create console with safe_box to avoid unicode box drawing errors on windows
console = Console(theme=custom_theme, safe_box=True)

# Helper function to monkey-patch or catch global print errors
def safe_print(*args, **kwargs):
    try:
        console.print(*args, **kwargs)
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

# --- CODE BLOCK TRACKING ---
_code_blocks = []

def get_code_blocks():
    return list(_code_blocks)

def clear_code_blocks():
    _code_blocks.clear()

def _render_code_blocks(text: str) -> list:
    """Parse markdown text, extract fenced code blocks, and return Rich renderables.
    Stores code blocks globally for /copy command."""
    _code_blocks.clear()
    elements = []
    pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    last_end = 0

    for i, match in enumerate(pattern.finditer(text)):
        before = text[last_end:match.start()]
        if before.strip():
            elements.append(Markdown(before.strip(), code_theme=s["syntax_theme"]))

        lang = match.group(1) or "text"
        code = match.group(2).rstrip('\n')
        _code_blocks.append({"index": i + 1, "lang": lang, "code": code})

        syntax = Syntax(code, lang, theme=s["syntax_theme"], line_numbers=True, word_wrap=False)
        title_text = f"[bold cyan][{i + 1}][/bold cyan] {lang}  [dim](copy: /copy {i + 1})[/dim]"
        panel = Panel(
            syntax,
            title=title_text,
            title_align="left",
            border_style="dim cyan",
            width=min(console.size.width, s["content_width"]),
            expand=False,
            padding=(0, 1),
        )
        elements.append(panel)
        last_end = match.end()

    remainder = text[last_end:]
    if remainder.strip():
        elements.append(Markdown(remainder.strip(), code_theme=s["syntax_theme"]))

    return elements

def print_markdown(text: str, thinking: str = None):
    """Prints AI text directly without outer border panel for easy copy-paste.
    Code blocks get their own panels with /copy support."""
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
        safe_print(tree)
        safe_print("")
        
    if clean_text:
        elements = _render_code_blocks(clean_text)
        for el in elements:
            safe_print(el)
            safe_print("")

def print_system(text: str):
    import re
    # If text contains Rich markup tags like [bold cyan]...[/], render them directly.
    # Otherwise, escape the text to prevent accidental interpretation.
    if re.search(r'\[/?[a-z]', text):
        safe_print(f"[sys]{text}[/sys]")
    else:
        safe_print(f"[sys]{escape(text)}[/sys]")

def print_error(text: str):
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
    safe_print(f"[{c['error_msg']}]{escape(body)}[/{c['error_msg']}]")
    hint = explain_error(text)
    if hint:
        safe_print(f"[dim]{escape(hint)}[/dim]")


_ERROR_PREFIX_RE = re.compile(r'^\s*(error|ошибка)\b\s*:?', re.IGNORECASE)

def print_reasoning(thinking: str):
    """Prints a static, finalized reasoning block."""
    if not thinking:
        return
    from rich.tree import Tree
    reasoning_md = Markdown(thinking.strip(), code_theme=s["syntax_theme"])
    # We remove the inner Panel to save space and fix the "matryoshka" effect.
    # The style is applied directly to the Tree node label and children.
    tree = Tree(f"[{c.get('reasoning_label', 'bold #607d8b')}]Анализ[/{c.get('reasoning_label', 'bold #607d8b')}]")
    tree.add(reasoning_md)
    safe_print(tree)
    safe_print("") # Spacer

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

def print_tool_start(name: str, args: dict):
    # Interactive tools have their own user-facing display
    if name in ("ask_user_questions",):
        return
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    
    from config import get_debug_mode
    is_debug = get_debug_mode()
    limit = 2000 if is_debug else 150
    
    # Keep arguments somewhat truncated if massive
    if len(args_str) > limit:
        args_str = args_str[:limit-3] + "..."
    
    try:
        console.print(f"\n[tool_start]> Executing:[/] [tool_name]{name}[/]([tool_args]{args_str}[/])")
    except UnicodeEncodeError:
        pass
    
def _tool_result_summary(result: str) -> str:
    """One line describing what came back, for the compact view.

    A tool result is written for the model, not for you: read_file returns the
    file, grep_search returns every match. Rendering all of it pushed the
    conversation off the screen — you scroll past your own work to find the
    answer. The interesting part is almost always the shape of the result.
    """
    text = str(result)
    lines = text.splitlines()
    body = [ln for ln in lines if ln.strip()]
    if not body:
        return "пусто"
    first = body[0].strip()
    if len(body) == 1:
        return first if len(first) <= 110 else first[:107] + "…"
    return f"{len(lines)} строк · {first[:70]}{'…' if len(first) > 70 else ''}"


def print_tool_end(name: str, result: str):
    # Interactive tools — the user already saw the interaction, no need for a result panel
    if name in ("ask_user_questions",):
        return
    res_preview = str(result)

    from config import get_debug_mode, get_show_tool_results
    is_debug = get_debug_mode()

    # Errors are never collapsed: a failure the user cannot see is a failure
    # they will discover three turns later, from the model's behaviour.
    is_error = res_preview.lstrip().lower().startswith("error")
    if not is_debug and not is_error and not get_show_tool_results():
        safe_print(f"  [dim]✓ {name} · {escape(_tool_result_summary(res_preview))}[/dim]")
        return

    lines = res_preview.splitlines()
    max_lines = 100 if is_debug else 15

    if len(lines) > max_lines:
        res_preview = "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} lines hidden in UI]"

    try:
        panel = Panel(
            f"[dim]{escape(res_preview)}[/dim]", 
            title=f"[dim]✓ Tool {name} finished[/dim]", 
            title_align="left",
            border_style=c["tool_end_border"],
            width=min(console.size.width, s["content_width"] - 10),
            expand=False
        )
        safe_print(panel)
    except UnicodeEncodeError:
        panel = Panel(
            f"[dim]{escape(res_preview)}[/dim]", 
            title=f"[dim]> Tool {name} finished[/dim]", 
            title_align="left",
            border_style=c["tool_end_border"],
            width=min(console.size.width, s["content_width"] - 10),
            expand=False
        )
        safe_print(panel)

def print_context_usage(tokens: int, max_tokens: int, percent: float):
    """Displays a small progress bar showing context usage."""
    color = "green"
    if percent > 80:
        color = "red"
    elif percent > 60:
        color = "yellow"
        
    safe_print(f"[dim]Context: [{color}]{percent:.1f}%[/{color}] ({tokens}/{max_tokens} tokens)[/dim]")

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

    selected = _select_from_list(
        f"Select {provider.name.upper()} model:",
        models,
        current_model if current_model in models else models[0]
    )

    if not selected:
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
        if provider.name in ("zai", "openrouter"):
            ctx_choices = ["8192", "16384", "32768", "65536", "131072"]
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
            val = int(ctx_choice.split()[0])
            set_context_window(val)
        except (ValueError, IndexError):
            pass

    # --- Step 3: Model classification ---
    from config import (
        get_model_size_category, get_model_category_override,
        set_model_category_override
    )
    # Reset override so auto-detection shows the real value for the new model
    set_model_category_override(None)
    auto_category = get_model_size_category(selected)

    category_choices = [
        f"Auto: {auto_category} (рекомендуется)",
        "tiny  — <3B, без native tools, минимальная история",
        "small — 3-7B, native tools, короткая история",
        "medium — 7-13B, native tools, средняя история",
        "large — >13B, native tools, длинная история",
        "cloud — облачные API (Gemini, GPT, GLM)",
    ]

    cat_choice = _select_from_list(
        f"Классификация модели (авто: {auto_category}):",
        category_choices,
        category_choices[0]
    )

    if cat_choice and not cat_choice.startswith("Auto"):
        manual_cat = cat_choice.split()[0].strip()
        set_model_category_override(manual_cat)

    return selected


def _select_from_list(prompt_text: str, choices: list, default: str = None) -> str:
    """Try questionary.select, fall back to numbered text input if it fails.

    For long lists (e.g. OpenRouter's hundreds of models) the picker enables
    type-to-filter search so the user can jump straight to a model by name
    instead of scrolling."""
    safe_default = default if default in choices else (choices[0] if choices else None)
    try:
        # Type-to-filter pays off once a list is too long to eyeball; short
        # menus stay as plain arrow-key selects.
        if len(choices) > 12:
            result = questionary.select(
                prompt_text,
                choices=choices,
                default=safe_default,
                use_search_filter=True,
                use_jk_keys=False,
                instruction="(печатайте для поиска, ↑↓ выбор, Enter подтвердить)",
            ).ask()
        else:
            result = questionary.select(
                prompt_text,
                choices=choices,
                default=safe_default,
            ).ask()
        if result:
            return result
    except TypeError:
        # Older questionary without search-filter kwargs: plain select.
        try:
            result = questionary.select(prompt_text, choices=choices, default=safe_default).ask()
            if result:
                return result
        except Exception:
            pass
    except Exception:
        pass

    print_system(f"\n{prompt_text}")
    for i, choice in enumerate(choices, 1):
        marker = " ← current" if choice == default else ""
        console.print(f"  [cyan]{i}[/cyan]. {choice}{marker}")

    try:
        from prompt_toolkit import prompt as ptk_prompt
        raw = ptk_prompt(f"Enter number or name (1-{len(choices)}): ").strip()
    except (KeyboardInterrupt, EOFError):
        return default or (choices[0] if choices else None)

    if not raw:
        return default or (choices[0] if choices else None)

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx]

    for choice in choices:
        if raw.lower() == choice.lower():
            return choice

    return default or (choices[0] if choices else None)
