# -*- coding: utf-8 -*-
"""scripts/_context_rot_laya.py — rot-тест Laya vs Leeth на зашумлённых кейсах.

Контекст: Leeth v3-головы дают dept rot700 64% (9/14). Неизвестно, как ведёт
себя Laya на тех же кейсах. Если Laya тоже деградирует — context rot открытая
проблема (Jev её документирует), и наш результат не хуже планки.
"""
import importlib.util
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    from laya import Router

    tok_like = None
    rng = random.Random(5)
    # префикс ~700 токенов (оценка 4 симв/токен — как в leeth-тестах)
    chunks = []
    for i in range(400):
        c = NOISE.format(i=i, ms=3 + (i % 17))
        chunks.append(c)

    def prefix(max_tokens_est):
        out, n = [], 0
        while n < max_tokens_est:
            c = chunks[rng.randrange(len(chunks))]
            out.append(c)
            n += len(c) // 4
        return "".join(out)

    dept_cases = [(st, next(g for q, g in qs if q == "department"))
                  for _, st, qs in mdb.CASES if any(q == "department" for q, _ in qs)]
    urg_cases = [(st, next(g for q, g in qs if q == "urgency"))
                 for _, st, qs in mdb.CASES if any(q == "urgency" for q, _ in qs)]

    for model_name in ("english", "multilingual", "typed-decisions"):
        try:
            router = Router(device="cpu")
        except Exception as e:
            print(f"[{model_name}] Router недоступен: {e}")
            continue
        print(f"\n### Laya {model_name} (cpu) загружен")

        for vname, mk_state in (("чистый", lambda st, p: mdb.prod_envelope(st)),
                                ("rot700 (шум до)", lambda st, p: p + "\n" + mdb.prod_envelope(st))):
            p = prefix(700) if "rot" in vname else ""
            for qname, cases, golds, build_q in (
                    ("dept", dept_cases, [g for _, g in dept_cases],
                     lambda: {"type": "choice", "instructions": mdb.Q_DEPARTMENT, "criteria": mdb.DEPT_CRITERIA}),
                    ("urgency", urg_cases, [g for _, g in urg_cases],
                     lambda: {"type": "score", "instructions": mdb.Q_URGENCY, "criteria": mdb.URGENCY_RUBRIC})):
                ok, t0 = 0, time.perf_counter()
                for (st, _), gold in zip(cases, golds):
                    state = mk_state(st, p)
                    r = router.predict(state, {"q": build_q()}, model=model_name)
                    ans = r["answers"]["q"]
                    if qname == "dept":
                        pred = ans.get("choice")
                        ok += pred == gold
                    else:
                        probs = {int(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
                        pred = max(probs, key=probs.get) + 1 if probs else None
                        ok += pred == int(gold)
                ms = (time.perf_counter() - t0) * 1000 / len(cases)
                print(f"[{model_name:>15}] [{vname:>18}] {qname:<8} {ok}/{len(cases)}  {ms:.0f} мс/кейс")
        del router
        import gc
        gc.collect()


if __name__ == "__main__":
    main()