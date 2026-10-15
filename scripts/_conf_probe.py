"""Measure 400M confidence on canonical fast-path queries to calibrate the dispatcher gate."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.runtime import leeth
leeth.initialize(force_device="cuda")

from src.leeth.primitives import ChoiceQuestion, NoulQuestion

ROUTING = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR",
]
Q_ROUTING = "Какой инструмент Argent должен быть вызван в первую очередь?"
Q_NOUL_TOOL = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"

QUERIES = [
    ("прочитай main.py", "read_file"),
    ("покажи содержимое config.py", "read_file"),
    ("сколько будет 2+2*22", "calculate"),
    ("запусти pytest", "run_command"),
    ("выполни git status", "run_command"),
    ("найди все упоминания TODO", "grep_search"),
    ("открой сайт example.com", "browser_open"),
    ("привет, как дела?", "NO_TOOL"),
    ("спасибо, большое!", "NO_TOOL"),
    ("настрой nginx reverse proxy", "UNFAMILIAR"),
    ("спроектируй архитектуру микросервисов", "UNFAMILIAR"),
    ("создай файл notes.md с заготовкой", "write_file"),
    ("замени функцию load в src/agent.py", "replace_in_file"),
]

print(f"{'query':<42} {'need_tool':>9} {'choice':>18} {'conf':>6} {'expected':>14}")
for q_text, expected in QUERIES:
    state = f'Incoming user request in Argent session:\n"{q_text}"\nWorking directory: C:/Users/user/Desktop/Argent'
    q1 = NoulQuestion(id="n", question=Q_NOUL_TOOL)
    q2 = ChoiceQuestion(id="c", question=Q_ROUTING, choices=list(ROUTING))
    batch = leeth.decide(state=state, questions=[q1, q2])
    nt = batch.results.get("n")
    ch = batch.results.get("c")
    print(f"{q_text:<42} {nt.value!s:>9} {ch.choice:>18} {ch.confidence:>6.2f} {expected:>14}", flush=True)