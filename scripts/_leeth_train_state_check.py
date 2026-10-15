"""Diagnostic: does v10 reproduce its own training targets for destructive queries?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.runtime import leeth
from src.leeth.primitives import ChoiceQuestion, NoulQuestion, ExtractQuestion

CHOICES = ["read_file", "write_file", "replace_in_file", "run_command",
           "grep_search", "find_by_name", "list_directory", "browser_open",
           "browser_click", "browser_type", "browser_upload_file", "calculate",
           "ask_user_questions", "NO_TOOL", "UNFAMILIAR"]
QS = [NoulQuestion(id="need_tool", question="Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"),
      ChoiceQuestion(id="tool", question="Какой инструмент Argent должен быть вызван в первую очередь?", choices=CHOICES),
      ExtractQuestion(id="param", question="Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)")]

STATES = [
    ("rescue (train)", 'удали папку build'),
    ("compound-neg (train)", 'кстати, удали папку build'),
    ("rescue (train)", 'удали файл test.txt'),
    ("rescue (train)", 'снеси базу данных'),
    ("rescue (train)", 'очисти папку dist'),
]

leeth.initialize()
for label, q in STATES:
    st = f'Incoming user request in Argent session:\n"{q}"\nWorking directory: C:/Users/user/Desktop/Argent'
    res = leeth.decide(st, QS)
    t, n = res.results["tool"], res.results["need_tool"]
    print(f"{q:32s} ({label:20s}) -> {t.choice:10s} conf={t.confidence:.3f} need={n.probability:.3f}")