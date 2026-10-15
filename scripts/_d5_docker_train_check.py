# -*- coding: utf-8 -*-
"""scripts/_d5_docker_train_check.py — есть ли docker ps в routing-тренировке."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

routing = 0
examples = []
for line in open("src/leeth/data_400m/leeth_400m_train.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if "docker ps" in r.get("state", "").lower() and r.get("question_type") == "choice":
        q = r.get("question") or ""
        if "инструмент" in q:
            routing += 1
            if len(examples) < 3:
                examples.append((r["state"][:100].replace("\n", " | "), r.get("target_choice")))
print("routing-строк с docker ps:", routing)
for s, t in examples:
    print("  state:", s)
    print("    target:", t)