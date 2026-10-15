# -*- coding: utf-8 -*-
"""scripts/_flywheel_harvest.py — ревью pending-решений маховика Leeth.

Этап 1: дамп ценного подмножества (dispatch + unfamiliar + low_conf) в компактном
виде для ревью агентом. Этап 2 (после ревью): применить вердикты из verdicts.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth import feedback as fb

mode = sys.argv[1] if len(sys.argv) > 1 else "dump"

if mode == "dump":
    recs = fb.load_records()
    pend = [r for r in recs if not r.get("verdict")]
    valuable = [r for r in pend if r.get("dispatch") or r.get("skip_reason") in ("unfamiliar", "low_tool_confidence")]
    print(f"ценных pending: {len(valuable)} (из {len(pend)} pending)")
    for r in valuable:
        prop = r.get("proposal") or {}
        tool = (prop.get("tool") or {}).get("choice", "?")
        conf = (prop.get("tool") or {}).get("confidence", 0)
        d = r.get("dispatch") or {}
        dt = d.get("tool_name", "-")
        dargs = d.get("arguments") or {}
        dparam = next(iter(dargs.values()), "") if dargs else ""
        print(f"{r['id']}|{r.get('skip_reason') or 'DISPATCH'}|{tool}|{conf:.2f}|{dparam}|{r.get('query', '')[:90]}")
elif mode == "apply":
    vpath = Path("scripts/_flywheel_verdicts.json")
    spec = json.loads(vpath.read_text(encoding="utf-8"))
    partial = {k.lower(): v for k, v in spec["partial"].items()}
    acc_disp = {q.lower() for q in spec["accept_dispatch"]}
    acc_skip = {q.lower() for q in spec["accept_skip"]}
    recs = fb.load_records()
    applied = 0
    for r in recs:
        if r.get("verdict"):
            continue
        if r.get("dispatch") or r.get("skip_reason") not in ("unfamiliar", "low_tool_confidence"):
            continue  # вне ценного подмножества
        q = (r.get("query") or "").strip().lower()
        has_dispatch = bool(r.get("dispatch"))
        v = None
        if q in partial:
            p = partial[q]
            v = {"verdict": "partial", "correction_tool": p["tool"],
                 "correction_param": p["param"], "reason": "ревью: правильный инструмент"}
        elif has_dispatch and q in acc_disp:
            v = {"verdict": "accept", "reason": "ревью: диспатч корректен"}
        elif not has_dispatch and q in acc_skip:
            v = {"verdict": "accept", "reason": "ревью: skip корректен (разговорный/LLM-запрос)"}
        if v:
            res = fb.submit_verdict(r["id"], v["verdict"], correction_tool=v.get("correction_tool"),
                                    correction_param=v.get("correction_param"),
                                    source="agent", reason=v.get("reason", ""))
            applied += res
    print(f"применено вердиктов: {applied}")
    print("статистика:", json.dumps(fb.stats(), ensure_ascii=False))
elif mode == "samples":
    train, val, info = fb.build_training_samples()
    print(json.dumps(info, ensure_ascii=False))
    from collections import Counter
    cnt = Counter()
    for s in train + val:
        cnt[(s.question_type, (s.question or "")[:30])] += 1
    for k, v in cnt.most_common(10):
        print(f"  {k}: {v}")