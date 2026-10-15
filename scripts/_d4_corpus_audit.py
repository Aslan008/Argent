# -*- coding: utf-8 -*-
"""scripts/_d4_corpus_audit.py — аудит корпуса перед UNFAMILIAR-финетюном.

Проверяет: (1) точные строки tool-вопроса в train/val; (2) покрытие русских
составных задач в UNFAMILIAR-строках train; (3) примеры.
"""
import json
import re
import sys
import collections
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path("src/leeth/data_400m/leeth_400m_train.jsonl")
VAL = Path("src/leeth/data_400m/leeth_400m_val.jsonl")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

tr = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]
va = [json.loads(l) for l in open(VAL, encoding="utf-8")]

# 1) точные строки tool-вопроса
tr_tool = [r for r in tr if r["question_type"] == "choice" and r["question"].startswith("Какой инструмент")]
va_tool = [r for r in va if r["question_type"] == "choice" and r["question"].startswith("Какой инструмент")]
print(f"train tool-строк: {len(tr_tool)} | val tool-строк: {len(va_tool)}")
if tr_tool:
    print("TRAIN question exact:", repr(tr_tool[0]["question"]))
if va_tool:
    print("VAL   question exact:", repr(va_tool[0]["question"]))
same = bool(tr_tool) and bool(va_tool) and tr_tool[0]["question"] == va_tool[0]["question"]
print("вопросы совпадают:", same)

# 2) routing-формат в train/val tool-строках
P = "Incoming user request in Argent session:"
tr_rt = [r for r in tr_tool if r["state"].startswith(P)]
va_rt = [r for r in va_tool if r["state"].startswith(P)]
print(f"routing-формат: train {len(tr_rt)} | val {len(va_rt)}")


def q(r):
    m = re.search(r'"(.+)"', r["state"])
    return m.group(1) if m else r["state"][:60]


# 3) покрытие составных задач в UNFAMILIAR train-строках
unf = [r for r in tr_tool if r.get("target_choice") == "UNFAMILIAR"]
print(f"\nUNFAMILIAR среди tool-строк train: {len(unf)}")
KEYS = ["исправ", "состав", "спроекти", "ревью", "проанализ", "исслед", "оптимизир",
        "задокумент", "найди и удали", "сравни", "расскажи", "что ты умеешь",
        "какие у тебя", "как ты работ", "почему", "как исправить", "запусти тесты",
        "покажи содержимое", "напиши тест", "добавь тест", "ускорь", "перепиши",
        "улучши", "рефактор", "миграц", "архитектур"]
pat = collections.Counter()
examples = {}
for r in unf:
    t = q(r).lower()
    hit = None
    for kw in KEYS:
        if kw in t:
            hit = kw
            break
    if hit:
        pat[hit] += 1
        examples.setdefault(hit, q(r))
    else:
        pat["<другое>"] += 1
        examples.setdefault("<другое>", q(r))
for k, n in pat.most_common():
    print(f"{n:5d}  {k:18s} пример: {examples[k][:70]}")

# 4) выборка routing-UNFAMILIAR (если есть)
rt_unf = [r for r in unf if r["state"].startswith(P)]
print(f"\nrouting-формат UNFAMILIAR: {len(rt_unf)}")
for r in rt_unf[:10]:
    print(" -", q(r)[:85])