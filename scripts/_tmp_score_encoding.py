"""Inspect score-sample encoding in the corpus (target_score / score_levels)."""
import json
from pathlib import Path

n = 0
with open("src/leeth/data/leeth_train.jsonl", encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        if s["question_type"] == "score":
            print("levels=", s.get("score_levels"), "target=", s.get("target_score"),
                  "| q:", s["question"][:70])
            print("   state:", s["state"][:90].replace("\n", " "))
            n += 1
            if n >= 4:
                break
if n == 0:
    print("no score samples in train corpus")