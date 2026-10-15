"""Identify the conversational prompt with the inflated need_tool probability."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.runtime import leeth
from src.leeth.primitives import ChoiceQuestion, NoulQuestion

CHOICES = ["read_file", "write_file", "replace_in_file", "run_command",
           "grep_search", "find_by_name", "list_directory", "browser_open",
           "browser_click", "browser_type", "browser_upload_file", "calculate",
           "ask_user_questions", "NO_TOOL", "UNFAMILIAR"]
QUESTIONS = [NoulQuestion(id="need_tool", question="Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"),
             ChoiceQuestion(id="tool", question="Какой инструмент Argent должен быть вызван в первую очередь?", choices=CHOICES)]

PROMPTS = [
    "Расскажи, как устроена команда rm -rf в операционных системах Linux",
    "Объясни мне, чем отличается async/await от Task в C#",
    "Привет! Как твои дела и чем ты можешь мне помочь?",
]

leeth.initialize()
for q in PROMPTS:
    for label, state in [
        ("unquoted(test)", f"Incoming user request in Argent session:\n{q}\nWorking directory: C:/Users/user/Desktop/Argent"),
        ("quoted(live)  ", f'Incoming user request in Argent session:\n"{q}"\nWorking directory: C:/Users/user/Desktop/Argent'),
    ]:
        res = leeth.decide(state, QUESTIONS)
        t, n = res.results["tool"], res.results["need_tool"]
        print(f"{label} | {t.choice:8s} conf={t.confidence:.3f} need={n.probability:.4f} | {q[:55]}")