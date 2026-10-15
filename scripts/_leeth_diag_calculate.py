"""Diagnoses heads for the failing regression test query."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.leeth import leeth
from src.leeth.dispatcher import dispatcher
from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion

TOOL_CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL",
]

QUERIES = [
    "посчитай 128 * 64",
    "сколько будет 512 / 8",
    "вычисли 17 * 23",
    "посчитай sqrt(2)",
]

for query in QUERIES:
    state = f'Incoming user request in Argent session:\n"{query}"\nWorking directory: C:/Users/user/Desktop/Argent'
    questions = [
        NoulQuestion(id="need_tool", question="Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"),
        ChoiceQuestion(id="tool", question="Какой инструмент Argent должен быть вызван в первую очередь?", choices=TOOL_CHOICES),
        ExtractQuestion(id="param", question="Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)"),
    ]
    batch = leeth.decide(state, questions)
    need = batch.results.get("need_tool")
    tool = batch.results.get("tool")
    disp = dispatcher.try_fast_dispatch(query)
    print(f"'{query}': need={need.probability:.3f} tool={tool.choice}/{tool.confidence:.3f} -> {disp.tool_name if disp else 'None'}")