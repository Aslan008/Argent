"""A room's question to the person (a `human_pause` step), put the way a person can read it.

The text of such a question is often long: the critic's remarks, a plan, a todo list, then "go on?".
A selection or an input line is one row, so the whole text used to be squeezed into its title and only
its first line was seen. Here what comes before the last line is printed into the conversation (wrapped,
scrollable, copyable) and only the question itself — the last line — is asked.
"""

from typing import Callable, List, Optional, Tuple

ACCENT = "#58a6ff"


def split_question(prompt: str) -> Tuple[str, str]:
    """(context, question): the last non-empty line is what is asked, everything above it is context."""
    lines = (prompt or "").strip().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return "", ""
    question = lines[-1].strip()
    context = "\n".join(lines[:-1]).strip()
    return context, question


def ask_human(node, state, say: Callable[[str], None], ask_text: Callable[[str], Optional[str]],
              ask_select: Callable[[str, List[str]], Optional[str]]) -> None:
    """Ask the question of `node` and put the answer where the engine reads it (`<id>.output`)."""
    try:
        from rich.markup import escape
    except Exception:                       # rich is a dependency; the fallback only keeps this importable
        def escape(text):
            return text.replace("[", "\\[")

    context, question = split_question(node.prompt or "")
    if context:
        say(f"[bold {ACCENT}]Вопрос комнаты[/bold {ACCENT}] [dim]· шаг {escape(str(node.id))}[/dim]\n"
            + escape(context))
        title = question
    else:
        title = f"[{node.id}] {question or 'Ваш ввод:'}"
    options = list(node.options or [])
    answer = ask_select(title, options) if options else ask_text(title)
    if answer:
        state.data[f"{node.id}.output"] = answer
