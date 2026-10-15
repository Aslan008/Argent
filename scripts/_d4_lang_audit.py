# -*- coding: utf-8 -*-
"""scripts/_d4_lang_audit.py — языковой баланс routing-строк корпуса по классам."""
import json
import re
import sys
import collections
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path("src/leeth/data_400m/leeth_400m_train.jsonl")
P = "Incoming user request in Argent session:"
CYR = re.compile(r"[а-яё]")


def q(r):
    m = re.search(r'"(.+)"', r["state"])
    return m.group(1) if m else ""


rows = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]
rt = [r for r in rows if r["question_type"] == "choice" and r["state"].startswith(P)]
print(f"routing-choice строк в train: {len(rt)}")
stat = collections.defaultdict(lambda: {"ru": 0, "en": 0})
for r in rt:
    t = q(r)
    lang = "ru" if CYR.search(t) else "en"
    stat[r.get("target_choice")][lang] += 1
print(f"{'класс':22s} {'ru':>6} {'en':>6}")
for cls, d in sorted(stat.items(), key=lambda kv: -(kv[1]["ru"] + kv[1]["en"])):
    print(f"{cls:22s} {d['ru']:>6} {d['en']:>6}")

print("\nпримеры русских read_file:")
n = 0
for r in rt:
    if r.get("target_choice") == "read_file" and CYR.search(q(r)):
        print(" -", q(r)[:70])
        n += 1
        if n >= 6:
            break
print("\nпримеры русских run_command:")
n = 0
for r in rt:
    if r.get("target_choice") == "run_command" and CYR.search(q(r)):
        print(" -", q(r)[:70])
        n += 1
        if n >= 6:
            break