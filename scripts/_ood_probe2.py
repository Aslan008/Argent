# -*- coding: utf-8 -*-
"""scripts/_ood_probe2.py — роутер на «других темах»: управление, сравнение, фильтрация."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.dispatcher import dispatcher

QUERIES = [
    # --- сравнение ---
    "сравни config.py и config_old.py",
    "найди различия между двумя версиями отчёта",
    "что изменилось в коде с прошлой недели?",
    # --- фильтрация / поиск по данным ---
    "отфильтруй из лога строки с ошибками",
    "покажи только python-файлы больше 10 КБ",
    "найди все TODO в проекте",
    # --- управление системой ---
    "покажи топ-5 процессов по памяти",
    "останови зависший процесс python",
    "проверь свободное место на диске",
    "сделай бэкап папки src в zip",
    # --- подсчёт/анализ ---
    "посчитай сколько строк в каждом файле",
    "проанализируй почему тесты падают",
    # --- чистое LLM (без инструментов) ---
    "составь план рефакторинга на неделю",
    "объясни что делает этот алгоритм",
    # --- контролы ---
    "docker ps",
    "прочитай config.py",
]

infos = dispatcher.triage(QUERIES)
n_dispatch = 0
for inf in infos:
    q = inf["query"]
    if inf.get("dispatch"):
        d = inf["dispatch"]
        n_dispatch += 1
        print(f"  DISPATCH {q[:46]:<48} -> {d['tool']} (conf {d['confidence']:.3f})")
    else:
        print(f"  skip     {q[:46]:<48} reason={inf['skip_reason']}")
print(f"\nитого: {n_dispatch} диспатчей из {len(QUERIES)} — остальное уходит в LLM-флоу")