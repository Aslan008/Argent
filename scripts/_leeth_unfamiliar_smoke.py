"""Smoke: verify the UNFAMILIAR family generates correctly + class space is wired."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from src.leeth.dataset_generator import LeethDatasetGenerator

fam = LeethDatasetGenerator(seed=1).generate_unfamiliar_samples()
choice_samples = [s for s in fam if s.question_type == "choice"]
noul_samples = [s for s in fam if s.question_type == "noul"]
print(f"unfamiliar family: {len(fam)} samples ({len(choice_samples)} choice, {len(noul_samples)} noul)")
s = choice_samples[0]
print("choice target:", s.target_choice, "| choices:", len(s.choices), "| last:", s.choices[-1])
assert s.target_choice == "UNFAMILIAR"
assert s.choices[-1] == "UNFAMILIAR" and len(s.choices) == 15
assert noul_samples[0].target_noul == 1.0

classes = json.loads((Path("src/leeth/data/choice_classes.json")).read_text(encoding="utf-8"))
assert classes.get("UNFAMILIAR") == 25, classes.get("UNFAMILIAR")
print("class_to_idx UNFAMILIAR:", classes["UNFAMILIAR"], "| total classes:", len(classes))
print("SMOKE OK")