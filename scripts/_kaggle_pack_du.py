# -*- coding: utf-8 -*-
"""scripts/_kaggle_pack_du.py — сборка датасета leeth-mmbert-du для кернела A.

Состав: fp16-база mmBERT (962 MB), multidomain train/val (dept/urgency),
battery_rows + docker + extra (routing cand11), choice_classes.json.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(".argent/kaggle/du_dataset")
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    ("src/leeth/data_400m/multidomain_train.jsonl", "multidomain_train.jsonl"),
    ("src/leeth/data_400m/multidomain_val.jsonl", "multidomain_val.jsonl"),
    (".argent/kaggle/dataset_v2/battery_rows.jsonl", "battery_rows.jsonl"),
    ("src/leeth/data_400m/docker_train.jsonl", "docker_train.jsonl"),
    ("src/leeth/data_400m/docker_val.jsonl", "docker_val.jsonl"),
    ("src/leeth/data_400m/leeth_400m_train_extra.jsonl", "leeth_400m_train_extra.jsonl"),
    ("src/leeth/data_400m/leeth_400m_val.jsonl", "leeth_400m_val.jsonl"),
    ("src/leeth/data_400m/choice_classes.json", "choice_classes.json"),
    (".argent/models/mmbert-leeth_candidate.pt", "mmbert-leeth_candidate.pt"),
    ("scripts/prod_battery.json", "prod_battery.json"),
    ("scripts/_multidomain_battery.py", "_multidomain_battery.py"),
]

for src, dst in FILES:
    p = Path(src)
    if p.exists():
        shutil.copyfile(p, OUT / dst)
        print(f"+ {dst} ({p.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"! НЕТ: {src}")

meta = {
    "title": "leeth-mmbert-du",
    "id": "hunzlee/leeth-mmbert-du",
    "licenses": [{"name": "CC0-1.0"}],
}
(OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("metadata ok")