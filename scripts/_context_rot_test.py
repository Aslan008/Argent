# -*- coding: utf-8 -*-
"""scripts/_context_rot_test.py — A6: длинный контекст не должен ломать mmBERT-путь.

Jev документирует context rot (точность падает с ростом state). Наш mmBERT
(ModernBERT, 8192 позиций, local_attention 128 + global каждые 3 слоя) должен
держать dept/urgency/routing на state с длинным нерелевантным префиксом.

Метод: к каждому кейсу батареи добавляется мусорный префикс (лог-вывод) длиной
0 / ~600 / ~1200 / ~2400 токенов; сравнивается точность с чистым прогоном.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.runtime import leeth
from src.leeth.runtime import MMBERT_DEPT_CLASSES, MMBERT_DEPT_QUESTION, MMBERT_URGENCY_QUESTION
from src.leeth.primitives import ChoiceQuestion, ScoreQuestion

spec = importlib.util.spec_from_file_location("mdbattery", "scripts/_multidomain_battery.py")
mdb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdb)

NOISE = (
    "2026-10-01 12:00:00 INFO  worker.pool task {i} finished in {ms}ms status=ok retries=0\n"
    "2026-10-01 12:00:01 DEBUG cache.get key=session:{i} hit=true ttl=3599\n"
    "def handler_{i}(request):\n"
    "    payload = request.json()\n"
    "    if not payload.get('id'):\n"
    "        return JsonResponse({{'error': 'missing id'}}, status=400)\n"
    "    return JsonResponse({{'ok': True, 'id': payload['id']}})\n"
)


def make_prefix(n_tokens: int) -> str:
    """Мусорный префикс ~n_tokens токенов (mmBERT-токенизатор, ~4 симв/токен RU)."""
    out = []
    n = 0
    i = 0
    while n < n_tokens:
        chunk = NOISE.format(i=i, ms=3 + (i % 17))
        out.append(chunk)
        n += len(chunk) // 4
        i += 1
    return "SESSION LOG (нерелевантный контекст):\n" + "".join(out)


def run_group(cases, build_q, prefix_tokens=0):
    ok = 0
    t0 = time.perf_counter()
    for state, gold in cases:
        full = (make_prefix(prefix_tokens) + "\n" + state) if prefix_tokens else state
        batch = leeth.decide(mdb.prod_envelope(full), [build_q()])
        res = batch.results["q"]
        pred = res.choice if hasattr(res, "choice") else res.level
        ok += pred == gold
    ms = (time.perf_counter() - t0) * 1000 / len(cases)
    return ok, len(cases), ms


def main():
    if not leeth.initialize():
        print(f"init error: {leeth._init_error}")
        sys.exit(1)
    print(f"mmbert_active={leeth.mmbert_active} du={leeth.mmbert_du_available} maxlen={leeth.mmbert_max_length}")

    dept_cases = [(st, next(g for q, g in qs if q == "department"))
                  for _, st, qs in mdb.CASES if any(q == "department" for q, _ in qs)]
    urg_cases = [(st, next(g for q, g in qs if q == "urgency"))
                 for _, st, qs in mdb.CASES if any(q == "urgency" for q, _ in qs)]
    routing = json.loads(Path("scripts/prod_battery.json").read_text(encoding="utf-8"))["cases"][:20]
    tool_classes = list(json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8")).keys())
    route_cases = [(c["query"], c["expected_tool"]) for c in routing]

    groups = {
        "dept": (dept_cases, lambda: ChoiceQuestion(id="q", question=MMBERT_DEPT_QUESTION, choices=list(MMBERT_DEPT_CLASSES))),
        "urg": (urg_cases, lambda: ScoreQuestion(id="q", question=MMBERT_URGENCY_QUESTION, levels=3, rubric=mdb.URGENCY_RUBRIC)),
        "route": (route_cases, lambda: ChoiceQuestion(id="q", question="Какой инструмент Argent должен быть вызван в первую очередь?", choices=tool_classes)),
    }

    # Точные токенные длины: префиксы калибруются РЕАЛЬНЫМ токенизатором так,
    # чтобы префикс + кейс влезали в лимит (изолируем OOD-позиции от обрезки).
    budget = leeth.mmbert_max_length
    case_len = len(leeth.mmbert_tokenizer.encode(mdb.prod_envelope(dept_cases[0][0]) + "\n[QUESTION] " + MMBERT_DEPT_QUESTION, add_special_tokens=True))
    print(f"лимит={budget}, типичный кейс={case_len} токенов -> префиксы до {budget - case_len - 10} токенов")
    prefix_steps = [0, (budget - case_len) // 4, (budget - case_len) // 2, budget - case_len - 10]

    print(f"{'группа':<7} " + " ".join(f"{'>head+tail ' + str(p) + 'т':>16}" if p else f"{'0 ток':>16}" for p in prefix_steps))
    for name, (cases, bq) in groups.items():
        cells = []
        for pt in prefix_steps:
            if pt:
                # префикс ДО кейса: make_prefix(pt) даёт ~pt реальных токенов
                ok, n, ms = run_group(cases, bq, pt)
                cells.append(f"{ok}/{n} {ms:.0f}ms")
            else:
                ok, n, ms = run_group(cases, bq, 0)
                cells.append(f"{ok}/{n} {ms:.0f}ms")
        print(f"{name:<7} " + " ".join(f"{c:>16}" for c in cells))

    # Факт-проверка: реальная токенная длина префикса
    enc = leeth.mmbert_tokenizer(make_prefix(prefix_steps[-1]), truncation=False, return_tensors="np")
    print(f"макс. префикс: {enc['input_ids'].shape[1]} токенов (лимит {budget}, кейс ~{case_len})")


if __name__ == "__main__":
    main()