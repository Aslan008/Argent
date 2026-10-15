# -*- coding: utf-8 -*-
"""scripts/_llm_dept_train.py — добавление LLM-кейсов в тренировочный датасет (md7-llm).

LLM-кейсы (80 шт, ChatGPT) добавляются к шаблонному датасету как доп. тренировочные
примеры department-семьи. Цель: проверить, учит ли LLM-разнообразие модель
классифицировать рукописные/живые формулировки.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _multidomain_dataset import prod_envelope
from _multidomain_battery import Q_DEPARTMENT, DEPT_CHOICES

DATA = ROOT / "src/leeth/data_400m"

cases = []
for f in ["_llm_dept_batch1.json", "_llm_dept_batch2.json"]:
    cases.extend(json.loads((ROOT / "scripts" / f).read_text(encoding="utf-8")))

# LLM-кейсы: 80% train / 20% val
n_train = int(len(cases) * 0.8)
llm_train, llm_val = cases[:n_train], cases[n_train:]

# 1. Читаем текущие train/val
train = [json.loads(line) for line in (DATA / "multidomain_train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
val = [json.loads(line) for line in (DATA / "multidomain_val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

# 2. Добавляем LLM-кейсы
for c in llm_train:
    train.append({
        "state": prod_envelope(c["text"]),
        "question_type": "choice",
        "question": Q_DEPARTMENT,
        "choices": DEPT_CHOICES,
        "target_choice": c["dept"],
    })
for c in llm_val:
    val.append({
        "state": prod_envelope(c["text"]),
        "question_type": "choice",
        "question": Q_DEPARTMENT,
        "choices": DEPT_CHOICES,
        "target_choice": c["dept"],
    })

# 3. Перезаписываем
(DATA / "multidomain_train.jsonl").write_text(
    "\n".join(json.dumps(s, ensure_ascii=False) for s in train), encoding="utf-8")
(DATA / "multidomain_val.jsonl").write_text(
    "\n".join(json.dumps(s, ensure_ascii=False) for s in val), encoding="utf-8")

print(f"train: {len(train)} (+{len(llm_train)} LLM) | val: {len(val)} (+{len(llm_val)} LLM)")