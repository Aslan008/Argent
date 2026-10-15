# -*- coding: utf-8 -*-
"""scripts/_context_rot_diag.py — изоляция факторов деградации на длинном state.

Факторы:
  A. Позиции OOD: претрейн mmBERT шёл с max_length=256 — позиции 256+ не видны.
  B. Мусор внутри конверта: prod_envelope(prefix+case) кладёт логи в кавычки.
  C. Обрезка хвоста (исправлена head+tail в runtime).

Матрица: префикс ВНЕ конверта (state = prefix + envelope(case)) vs ВНУТРИ,
длины 0/100/200/400/700 РЕАЛЬНЫХ токенов (калибровка токенизатором).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.runtime import leeth
from src.leeth.runtime import MMBERT_DEPT_CLASSES, MMBERT_DEPT_QUESTION
from src.leeth.primitives import ChoiceQuestion

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


def make_prefix_real(n_tokens: int, tok) -> str:
    """Префикс с ТОЧНОЙ длиной n_tokens реальных токенов."""
    out, i = [], 0
    while True:
        chunk = NOISE.format(i=i, ms=3 + (i % 17))
        out.append(chunk)
        i += 1
        n = len(tok.encode("".join(out), add_special_tokens=False))
        if n >= n_tokens:
            break
    ids = tok.encode("".join(out), add_special_tokens=False)[:n_tokens]
    return tok.decode(ids)


def run(cases, build_state_fn, q_builder):
    ok = 0
    t0 = time.perf_counter()
    for state, gold in cases:
        batch = leeth.decide(build_state_fn(state), [q_builder()])
        res = batch.results["q"]
        ok += res.choice == gold
    return ok, len(cases), (time.perf_counter() - t0) * 1000 / len(cases)


def main():
    if not leeth.initialize():
        print(f"init error: {leeth._init_error}")
        sys.exit(1)
    tok = leeth.mmbert_tokenizer
    print(f"maxlen={leeth.mmbert_max_length}")

    dept_cases = [(st, next(g for q, g in qs if q == "department"))
                  for _, st, qs in mdb.CASES if any(q == "department" for q, _ in qs)]
    qb = lambda: ChoiceQuestion(id="q", question=MMBERT_DEPT_QUESTION, choices=list(MMBERT_DEPT_CLASSES))

    # Калибровка: длина чистого кейса
    case_len = len(tok.encode(mdb.prod_envelope(dept_cases[0][0]) + "\n[QUESTION] " + MMBERT_DEPT_QUESTION, add_special_tokens=True))
    budget = leeth.mmbert_max_length
    steps = [0, 100, 200, 400, min(700, budget - case_len - 10)]
    print(f"кейс={case_len} токенов, бюджет={budget}, шаги префикса={steps}")

    variants = {
        "вне конверта": lambda st, p: (p + "\n" + mdb.prod_envelope(st)) if p else mdb.prod_envelope(st),
        "внутри конверта": lambda st, p: mdb.prod_envelope((p + "\n" + st) if p else st),
    }

    for vname, fn in variants.items():
        print(f"\n== {vname} ==")
        print(f"{'префикс':>8} {'dept':>10} {'мс':>7}")
        for pt in steps:
            prefix = make_prefix_real(pt, tok) if pt else ""
            ok, n, ms = run(dept_cases, lambda st, p=prefix: fn(st, p), qb)
            print(f"{pt:>8} {ok}/{n:>6} {ms:>7.0f}")


if __name__ == "__main__":
    main()