# -*- coding: utf-8 -*-
"""scripts/_context_rot_order.py — позиция кейса относительно шума.

Механизм (v3): все пулы дают rot 64% — дело не в пулинге. 15 из 22 слоёв
ModernBERT — sliding_attention (окно 128): кейс-токены в КОНЦЕ state видят
только шум-соседей. Гипотеза: кейс в НАЧАЛЕ (шум после) — sliding-окно
кейс-токенов видит сам кейс.

Тест: dept-батарея, префикс 700 токенов ДО vs ПОСЛЕ кейса.
"""
import importlib.util
import random
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
    "2026-10-01 12:00:02 WARN  retry.attempt task={i} backoff=2s reason=timeout\n"
    "SELECT id, name FROM users WHERE active = 1 LIMIT 50; -- {i}\n"
)


def main():
    if not leeth.initialize():
        print(f"init error: {leeth._init_error}")
        sys.exit(1)
    tok = leeth.mmbert_tokenizer
    rng = random.Random(5)

    chunks = []
    for i in range(400):
        c = NOISE.format(i=i, ms=3 + (i % 17))
        chunks.append((c, len(tok.encode(c, add_special_tokens=False))))

    def prefix(max_tokens):
        out, total = [], 0
        while total < max_tokens:
            c, n = chunks[rng.randrange(len(chunks))]
            if total + n > max_tokens + 50:
                break
            out.append(c)
            total += n
        return "".join(out)

    dept_cases = [(st, next(g for q, g in qs if q == "department"))
                  for _, st, qs in mdb.CASES if any(q == "department" for q, _ in qs)]
    qb = lambda: ChoiceQuestion(id="q", question=MMBERT_DEPT_QUESTION, choices=list(MMBERT_DEPT_CLASSES))

    variants = {
        "кейс первый (шум после)": lambda st, p: mdb.prod_envelope(st) + "\nSESSION LOG:\n" + p,
        "шум первый (кейс в конце)": lambda st, p: p + "\n" + mdb.prod_envelope(st),
    }

    for vname, fn in variants.items():
        print(f"\n== {vname} ==")
        for pt in (0, 300, 700):
            p = prefix(pt) if pt else ""
            ok, t0 = 0, time.perf_counter()
            for st, gold in dept_cases:
                batch = leeth.decide(fn(st, p), [qb()])
                ok += batch.results["q"].choice == gold
            ms = (time.perf_counter() - t0) * 1000 / len(dept_cases)
            print(f"  префикс {pt:>4}: {ok}/14  {ms:.0f} мс/кейс")


if __name__ == "__main__":
    main()