"""Decisive test: does the question-set composition change head outputs?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.runtime import leeth
from src.leeth.primitives import ChoiceQuestion, NoulQuestion, ExtractQuestion

CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR"
]
Q2 = [
    NoulQuestion(id="need_tool", question="Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"),
    ChoiceQuestion(id="tool", question="Какой инструмент Argent должен быть вызван в первую очередь?", choices=CHOICES),
]
Q3 = Q2 + [ExtractQuestion(id="param", question="Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)")]

QUERIES = ["удали папку build", "открой example.com", "сколько будет 2 + 2 * 22"]

leeth.initialize()
print(f"runtime device: {leeth.device}, ort_session: {leeth.ort_session is not None}")

for q in QUERIES:
    state = f'Incoming user request in Argent session:\n"{q}"\nWorking directory: C:/Users/user/Desktop/Argent'
    for label, questions, ml in [("2q/512", Q2, 512), ("3q/512", Q3, 512), ("2q/320", Q2, 320), ("3q/320", Q3, 320)]:
        res = leeth.decide(state, questions, max_length=ml)
        tool = res.results["tool"]
        need = res.results["need_tool"]
        print(f"  {label} | {tool.choice:12s} conf={tool.confidence:.4f} need={need.probability:.4f} | {q}")