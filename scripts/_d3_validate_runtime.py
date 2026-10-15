# -*- coding: utf-8 -*-
"""scripts/_d3_validate_runtime.py — D3-валидация: батарея через реальный рантайм.

1. mmBERT-режим (LEETH_ROUTER=mmbert, по умолчанию): choice через mmBERT,
   ожидание ~66.1% (72/109) — паритет с прод-батареей E5.
2. legacy-режим (LEETH_ROUTER=legacy): ожидание ~54.1%.
3. Диспетчер end-to-end (inspect_query) на ключевых запросах в обоих режимах.
4. Латентность decide() на одном запросе.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BATTERY_PATH = Path("scripts/prod_battery.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

DISPATCH_QUERIES = [
    ("прочитай файл config.py и покажи содержимое", "read_file"),
    ("покажи git status в репозитории", "run_command"),
    ("найди все файлы с расширением .py в папке src", "find_by_name"),
    ("посчитай 15 процентов от 2400", "calculate"),
    ("привет, как твои дела?", None),
    ("напиши эссе о природе времени", None),
    ("исправь баг в функции calculate_total", None),
]


DISPATCH_CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR",
]


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def run_mode(router: str) -> None:
    os.environ["LEETH_ROUTER"] = router
    # реимпорт синглтона в свежем режиме
    import importlib

    import src.leeth.runtime as rt_mod

    importlib.reload(rt_mod)
    # диспетчер биндит синглтон при конструировании — перезагружаем и его
    import src.leeth.dispatcher as disp_mod

    importlib.reload(disp_mod)
    from src.leeth.primitives import ChoiceQuestion

    leeth = rt_mod.leeth
    ok = leeth.reload(force_device="onnx")
    mode = "mmBERT+legacy" if leeth.mmbert_active else "legacy-only"
    print(f"\n=== LEETH_ROUTER={router} → {mode} ===")

    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    correct = 0
    t0 = time.perf_counter()
    for c in battery["cases"]:
        state = build_state(c["query"])
        q = ChoiceQuestion(id="tool", question=TEMPLATE_Q, choices=DISPATCH_CHOICES)
        batch = leeth.decide(state, [q])
        res = batch.results.get("tool")
        if res is None:
            continue
        correct += int(res.choice == c["expected_tool"])
    wall = (time.perf_counter() - t0) / len(battery["cases"]) * 1000
    print(f"батарея через рантайм: {correct}/109 = {correct / 109:.1%} | {wall:.0f} мс/кейс")

    # латентность одного decide() (после прогрева)
    state = build_state("прочитай файл config.py")
    lat = []
    for _ in range(5):
        t1 = time.perf_counter()
        leeth.decide(state, [ChoiceQuestion(id="tool", question=TEMPLATE_Q, choices=DISPATCH_CHOICES)])
        lat.append((time.perf_counter() - t1) * 1000)
    print(f"латентность decide() (choice, батч-1): min={min(lat):.0f} мс, median={sorted(lat)[2]:.0f} мс")

    # диспетчер end-to-end
    from src.leeth.dispatcher import LeethDispatcher

    disp = LeethDispatcher()
    print("--- диспетчер (inspect_query) ---")
    n_ok = 0
    for query, expected in DISPATCH_QUERIES:
        info = disp.inspect_query(query)
        dispatch = info.get("dispatch")
        skip = info.get("skip_reason")
        verdict = "dispatch" if dispatch else (skip or "none")
        tool = (dispatch or {}).get("tool")
        conf = (dispatch or {}).get("confidence")
        mark = "OK " if ((dispatch is not None) == (expected is not None)) else "!! "
        n_ok += int(mark == "OK ")
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
        print(f"{mark}{query[:45]!r:48s} -> {verdict:20s} {str(tool):16s} conf={conf_s}")
    print(f"диспетчер: {n_ok}/{len(DISPATCH_QUERIES)} вердиктов корректны")


def main() -> None:
    run_mode("mmbert")
    run_mode("legacy")


if __name__ == "__main__":
    main()