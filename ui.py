import os
import re
import yaml
import questionary
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from rich.markup import escape
from rich.syntax import Syntax
import ollama
from rich.progress import Progress, BarColumn, TextColumn

# --- LOAD THEME ---
THEME_FILE = "theme.yaml"
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
        "syntax_theme": "monokai",
        "auto_save_chat": True
    }
}

theme_data = default_theme_data.copy()
if os.path.exists(THEME_FILE):
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
    safe_print(f"[{c['error_msg']}]Error: {text}[/{c['error_msg']}]")

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
    # Keep arguments somewhat truncated if massive
    if len(args_str) > 150:
        args_str = args_str[:147] + "..."
    
    try:
        console.print(f"\n[tool_start]> Executing:[/] [tool_name]{name}[/]([tool_args]{args_str}[/])")
    except UnicodeEncodeError:
        pass
    
def print_tool_end(name: str, result: str):
    # Interactive tools — the user already saw the interaction, no need for a result panel
    if name in ("ask_user_questions",):
        return
    res_preview = str(result)
    # Give it a subtle, narrow panel for tool results to distinguish from AI chat
    if len(res_preview) > 200:
        res_preview = res_preview[:197] + "..."
        
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
    """Uses questionary to display a dropdown of models for the active provider."""
    from config import get_context_window, set_context_window
    from providers import create_provider

    try:
        provider = create_provider()
    except Exception:
        return current_model

    models = provider.list_models()

    if not models:
        print_error(f"No models found for {provider.name}. Check your configuration.")
        return current_model

    try:
        selected = questionary.select(
            f"Select {provider.name.upper()} model:",
            choices=models,
            default=current_model if current_model in models else models[0]
        ).ask()

        if not selected:
            return current_model

        current_ctx = get_context_window()

        if provider.name == "zai":
            ctx_choices = [
                "8192 (Default)",
                "16384 (Large)",
                "32768 (Very Large)",
                "65536 (Maximum)",
                "131072 (Ultra - GLM-4-32B)",
                "Keep Current",
                "Custom Value..."
            ]
        else:
            ctx_choices = [
                "2048 (Fastest)",
                "4096 (Standard)",
                "8192 (Default)",
                "16384 (Large)",
                "32768 (Extreme - High VRAM)",
                "Keep Current",
                "Custom Value..."
            ]

        try:
            ctx_choice = questionary.select(
                f"Set Context Window for {selected} (Current: {current_ctx}):",
                choices=ctx_choices,
                default="Keep Current"
            ).ask()

            if ctx_choice == "Custom Value...":
                custom_val = questionary.text("Enter custom context size (e.g. 32768):").ask()
                if custom_val and custom_val.isdigit():
                    set_context_window(int(custom_val))
            elif ctx_choice != "Keep Current":
                val = int(ctx_choice.split()[0])
                set_context_window(val)
        except Exception:
            pass

        return selected

    except Exception:
        return current_model
