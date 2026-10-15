"""Pick the dispatcher confidence threshold from the 400M's accuracy-vs-confidence curve."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
from src.leeth.runtime import leeth
leeth.initialize(force_device="cuda")

from src.leeth.primitives import ChoiceQuestion

val_path = ROOT / "src/leeth/data_400m/leeth_400m_val.jsonl"
samples = []
with open(val_path, encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        if s.get("question_type") == "choice" and s.get("choices"):
            samples.append(s)
samples = samples[:400]
print(f"evaluating {len(samples)} choice samples from val...", flush=True)

buckets = {}  # bucket -> [correct, total]
for i, s in enumerate(samples):
    q = ChoiceQuestion(id="c", question=s["question"], choices=list(s["choices"]))
    batch = leeth.decide(state=s["state"], questions=[q])
    res = batch.results.get("c")
    conf = res.confidence
    correct = res.choice == s["target_choice"]
    b = round(conf * 10) / 10  # 0.1 buckets
    c, t = buckets.get(b, (0, 0))
    buckets[b] = (c + correct, t + 1)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(samples)}", flush=True)

print(f"\n{'conf bucket':>12} {'acc':>7} {'n':>5}")
for b in sorted(buckets):
    c, t = buckets[b]
    print(f"{b:>12.1f} {c/t:>7.1%} {t:>5}")

# Suggest threshold: lowest bucket edge where cumulative accuracy >= 97%
print("\nSuggested gate: pick threshold where accuracy >= 97%")