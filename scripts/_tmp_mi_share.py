"""Measure the multi-intent family share in the regenerated corpus."""
import json
from collections import Counter

for name in ["leeth_train.jsonl", "leeth_val.jsonl"]:
    cnt = Counter()
    total = 0
    with open(f"src/leeth/data/{name}", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            total += 1
            if s["question_type"] == "score" and s["score_levels"] == 3:
                cnt["score_c" + ("1" if (s["target_score"] or 0) <= 33 else "2")] += 1
            elif s["question_type"] == "extract" and "первой задачи" in s["question"]:
                cnt["first_intent"] += 1
            elif s["question_type"] == "choice" and "И ещё" in s["state"]:
                cnt["mi_choice"] += 1
    print(f"{name}: total={total} | {dict(cnt)}")