import os
import yaml
import questionary
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
import ollama

# --- LOAD THEME ---
THEME_FILE = "theme.yaml"
default_theme_data = {
    "colors": {
        "user_prompt": "bright_white",
        "assistant_border": "dodger_blue1",
        "assistant_text": "white",
        "system_msg": "dim italic white",
        "error_msg": "bold red",
        "tool_start": "yellow",
        "tool_name": "cyan",
        "tool_args": "green",
        "tool_end_border": "dim white"
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

def print_markdown(text: str):
    """Prints AI text inside a beautifully crafted, width-restricted panel."""
    # We strip out leading/trailing newlines to keep the panel tight
    clean_text = text.strip('\n')
    if not clean_text:
        return
        
    md = Markdown(clean_text, code_theme=s["syntax_theme"])
    panel = Panel(
        md, 
        border_style=c["assistant_border"], 
        width=min(console.size.width, s["content_width"]),
        expand=False,
        padding=(1, 2)
    )
    safe_print(panel)

def print_system(text: str):
    safe_print(f"[sys]{text}[/sys]")

def print_error(text: str):
    safe_print(f"[{c['error_msg']}]Error: {text}[/{c['error_msg']}]")

def print_tool_start(name: str, args: dict):
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    # Keep arguments somewhat truncated if massive
    if len(args_str) > 150:
        args_str = args_str[:147] + "..."
    
    try:
        console.print(f"\n[tool_start]> Executing:[/] [tool_name]{name}[/]([tool_args]{args_str}[/])")
    except UnicodeEncodeError:
        pass
    
def print_tool_end(name: str, result: str):
    res_preview = str(result)
    # Give it a subtle, narrow panel for tool results to distinguish from AI chat
    if len(res_preview) > 200:
        res_preview = res_preview[:197] + "..."
        
    try:
        panel = Panel(
            f"[dim]{res_preview}[/dim]", 
            title=f"[dim]✓ Tool {name} finished[/dim]", 
            title_align="left",
            border_style=c["tool_end_border"],
            width=min(console.size.width, s["content_width"] - 10),
            expand=False
        )
        safe_print(panel)
    except UnicodeEncodeError:
        panel = Panel(
            f"[dim]{res_preview}[/dim]", 
            title=f"[dim]> Tool {name} finished[/dim]", 
            title_align="left",
            border_style=c["tool_end_border"],
            width=min(console.size.width, s["content_width"] - 10),
            expand=False
        )
        safe_print(panel)

def select_model(current_model: str) -> str:
    """Uses questionary to display a dropdown of locally installed Ollama models."""
    try:
        models_response = ollama.list()
        
        models = []
        # Attempt to handle diff versions of the Ollama Python client response structure
        # `ollama.list()` sometimes returns an object with `models` or a dict.
        if hasattr(models_response, 'models'):
            modelsRaw = models_response.models
        elif isinstance(models_response, dict):
            modelsRaw = models_response.get('models', [])
        else:
            modelsRaw = []
            
        for m in modelsRaw:
            if isinstance(m, dict):
                models.append(m.get("model", m.get("name")))
            else:
                models.append(getattr(m, 'model', getattr(m, 'name', str(m))))
                
    except Exception as e:
        print_error(f"Could not fetch models from Ollama: {e}")
        return current_model
        
    if not models:
        print_error("No models found in Ollama. Please install one (e.g., 'ollama pull llama3.1').")
        return current_model
        
    try:
        selected = questionary.select(
            "Select the AI model to use:",
            choices=models,
            default=current_model if current_model in models else None
        ).ask()
    except Exception:
        return current_model
    
    return selected or current_model
