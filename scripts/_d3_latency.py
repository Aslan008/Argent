# -*- coding: utf-8 -*-
"""scripts/_d3_latency.py — финальный замер латентности диспетчерского пути D3.

Замеряет decide() на типичном triage-вызове (noul + choice + extract) в обоих
режимах роутера после полного прогрева: mmBERT (choice solo + legacy batch)
vs legacy (один батч на 3 вопроса).
"""
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
NEED_TOOL_Q = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"
PARAM_Q = "Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)"
CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR",
]


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def measure(router: str) -> None:
    os.environ["LEETH_ROUTER"] = router
    import importlib

    import src.leeth.runtime as rt_mod

    importlib.reload(rt_mod)
    from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion

    leeth = rt_mod.leeth
    leeth.reload(force_device="onnx")
    mode = "mmBERT(choice)+legacy(noul/extract)" if leeth.mmbert_active else "legacy-only"
    print(f"\n=== LEETH_ROUTER={router} -> {mode} ===")

    state = build_state("прочитай файл config.py и покажи содержимое")
    questions = [
        NoulQuestion(id="need_tool", question=NEED_TOOL_Q),
        ChoiceQuestion(id="tool", question=TEMPLATE_Q, choices=CHOICES),
        ExtractQuestion(id="param", question=PARAM_Q),
    ]

    # warmup (инициализация + первый прогон)
    leeth.decide(state, questions)

    lat = []
    for _ in range(12):
        t0 = time.perf_counter()
        leeth.decide(state, questions)
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"decide() noul+choice+extract: min={min(lat):.0f} мс, "
          f"median={statistics.median(lat):.0f} мс, p95={sorted(lat)[10]:.0f} мс")


def main() -> None:
    measure("mmbert")
    measure("legacy")


if __name__ == "__main__":
    main()