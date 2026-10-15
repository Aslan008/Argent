"""Checks specific queries after the first fine-tune cycle (heads + dispatcher)."""

import json
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

CHECKS = [
    ("кратко напиши что теперь", "conversational -> expect need=0 / NO_TOOL / no dispatch"),
    ("прочитай config.py", "routine -> expect read_file dispatch"),
    ("запусти pytest -x", "routine -> expect run_command dispatch"),
    ("удали папку build", "harmful -> expect no dispatch"),
    # corrections from training batch #2 (cycle 2)
    ("покажи что в README", "corrected: expect read_file (was list_directory/0.60)"),
    ("запусти python scripts/leeth_review.py stats", "corrected: expect run_command (was read_file)"),
    ("поищи класс AgentState в коде", "corrected: expect grep_search param=AgentState (was mid-word skip)"),
    ("какие у тебя есть возможности?", "conversational -> expect NO_TOOL"),
    # corrections from training batch #6 (cycle 5)
    ("сколько будет 512 / 8", "corrected: expect calculate (tool head said NO_TOOL twice)"),
    ("посчитай sqrt(2)", "corrected: expect calculate (regressed to NO_TOOL/0.50)"),
    ("покажи содержимое папки src", "corrected: expect list_directory param=src"),
]


def main() -> None:
    print(f"runtime device: {leeth.device}")
    for query, expectation in CHECKS:
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
        print(f"\n'{query}'  ({expectation})")
        print(f"  heads: need={need.probability:.3f} tool={tool.choice}/{tool.confidence:.3f}")
        print(f"  dispatcher: {disp.tool_name if disp else 'None (delegated to LLM)'}")


if __name__ == "__main__":
    main()