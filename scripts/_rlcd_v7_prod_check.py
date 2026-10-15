# -*- coding: utf-8 -*-
"""scripts/_rlcd_v7_prod_check.py — проверка RLCD v7 через прод-рантайм (INT8 ONNX).

Прогоняет multidomain_val.jsonl через leeth.decide в legacy-режиме
(LEETH_ROUTER=legacy — меряем именно Leeth, не mmBERT-роутер),
считает choice_acc / noul_acc и латентность.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["LEETH_ROUTER"] = "legacy"  # до импорта runtime
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.runtime import leeth
from src.leeth.primitives import ChoiceQuestion, NoulQuestion

VAL = Path("src/leeth/data_400m/multidomain_val.jsonl")


def main() -> None:
    if not leeth.is_available:
        print("LEETH NOT AVAILABLE:", leeth._init_error)
        sys.exit(1)
    rows = [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    choice_rows = [r for r in rows if r["question_type"] == "choice" and r.get("target_choice")]
    noul_rows = [r for r in rows if r["question_type"] == "noul" and r.get("target_noul") is not None]
    print(f"cases: choice={len(choice_rows)} noul={len(noul_rows)}")

    ok_c = 0
    lat = []
    all_classes = list(json.loads(
        Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8")
    ).keys())
    for r in choice_rows:
        choices = r.get("choices") or all_classes
        q = ChoiceQuestion(id="tool", question=r["question"], choices=choices)
        t0 = time.perf_counter()
        batch = leeth.decide(r["state"], [q])
        lat.append(time.perf_counter() - t0)
        res = batch.results.get("tool")
        ok_c += int(res is not None and res.choice == r["target_choice"])

    ok_n = 0
    for r in noul_rows:
        q = NoulQuestion(id="n", question=r["question"])
        t0 = time.perf_counter()
        batch = leeth.decide(r["state"], [q])
        lat.append(time.perf_counter() - t0)
        res = batch.results.get("n")
        ok_n += int(res is not None and res.value == bool(r["target_noul"]))

    lat_ms = sorted(lat)[len(lat) // 2] * 1000 if lat else 0
    print(f"choice_acc: {ok_c}/{len(choice_rows)} = {ok_c / max(1, len(choice_rows)):.1%}")
    print(f"noul_acc:   {ok_n}/{len(noul_rows)} = {ok_n / max(1, len(noul_rows)):.1%}")
    print(f"median decide latency: {lat_ms:.1f} ms")


if __name__ == "__main__":
    main()