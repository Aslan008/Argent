"""Phase 2 pre-migration smoke: architecture test + JSON family check."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.leeth.dataset_generator import generator  # noqa: E402

js = generator.generate_json_state_samples()
print("json samples:", len(js))
print("types:", sorted(set(s.question_type for s in js)))
print("classes used:", sorted(set(s.target_choice for s in js if s.target_choice)))
assert len(js) >= 25
assert set(s.question_type for s in js) == {"choice", "noul", "score"}
# All choice targets must exist in the trained taxonomy
import json
c2i = json.load(open(ROOT / "src/leeth/data/choice_classes.json", encoding="utf-8"))
for s in js:
    if s.target_choice:
        assert s.target_choice in c2i, f"unknown class: {s.target_choice}"
print("all choice targets exist in class_to_idx")

score_samples = [s for s in js if s.question_type == "score"]
assert all(s.score_levels == 5 for s in score_samples)
print("score_levels set on all score samples")

r = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_leeth_architecture.py", "-q", "--no-header"],
    capture_output=True, text=True, cwd=str(ROOT), timeout=240,
)
tail = (r.stdout + r.stderr).strip().splitlines()[-2:]
print("\n".join(tail))
assert r.returncode == 0, "architecture tests failed"
print("SMOKE OK")