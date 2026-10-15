"""Live status bar renderable for Rich Live displays during model execution.

Mirrors the bottom toolbar from prompt_toolkit so the active model, tier,
thinking level, context usage, and working directory remain visible while the
model is generating or running tools.
"""

import os
from rich.text import Text
from rich.rule import Rule
from config import (
    get_thinking_display_label,
    get_model_size_category,
)


def _fmt_tokens(n: int) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class StatusBar:
    """Rich renderable for displaying agent status at the bottom of Live displays."""

    def __init__(self, agent, ui_state: dict = None):
        self.agent = agent
        self.ui_state = ui_state or {}

    def __rich_console__(self, console, options):
        try:
            mode = self.ui_state.get("mode", "CHAT")
            provider = getattr(self.agent, "provider", "?")
            model = getattr(self.agent, "model_name", "?")
            tier = get_model_size_category(model) if model else "?"
            cwd = os.path.basename(os.getcwd()) or os.getcwd()
            
            think = get_thinking_display_label()
            think_part = f" · think:{think}" if think else ""

            usage = getattr(self.agent, "_last_context_usage", None)
            ctx_part = ""
            if usage:
                try:
                    pct = int(round(usage.get("percent", 0)))
                    tokens = usage.get("tokens", 0)
                    max_tokens = usage.get("max", 0)
                    tok_str = f" {_fmt_tokens(tokens)}/{_fmt_tokens(max_tokens)}" if max_tokens else ""
                    color = "green" if pct < 50 else ("yellow" if pct < 80 else "red")
                    ctx_part = f" · ctx:[{color}]{pct}%[/{color}]{tok_str}"
                except Exception:
                    pass

            text = (
                f"[bold #58a6ff]{mode}[/bold #58a6ff] · "
                f"[cyan]{provider}:{model}[/cyan] · "
                f"[dim]tier:{tier}[/dim]"
                f"[#79c0ff]{think_part}[/#79c0ff] · "
                f"[dim]{cwd}[/dim]"
                f"{ctx_part} · "
                f"[dim #8b949e](ESC — прервать)[/dim #8b949e]"
            )
            
            yield Rule(title=Text.from_markup(text), style="dim #30363d")
        except Exception:
            yield Text("")
