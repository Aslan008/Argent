# -*- coding: utf-8 -*-
"""scripts/_e5_quant_experiment3.py — torch.ao dynamic INT8 для mmBERT.

ORT-квантизация (dynamic/static, любые комбинации) роняет батарею до 26-44%.
Последний вариант: torch.ao.quantization.quantize_dynamic (как в bench_mmbert
для латентности) + проверка ТОЧНОСТИ на батарее. Если и это не даст ≥64%,
прод-артефакт — fp32 ONNX (67.0%, 39 мс/кейс).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from torch.ao.quantization import quantize_dynamic as qdyn

BASE_DIR = Path(".argent/models/mmbert-argent")
HEADS_PATH = Path(".argent/models/mmbert-leeth.pt")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
BATTERY_PATH = Path("scripts/prod_battery.json")
OUT_PATH = Path(".argent/models/mmbert-argent/mmbert_route_torch_int8.pt")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def main() -> None:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    class_to_idx = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))

    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32).eval()
    ckpt = torch.load(Path(".argent/models/mmbert-leeth.pt"), map_location="cpu", weights_only=False)
    hidden, n_classes = ckpt["hidden"], ckpt["n_classes"]
    choice_head = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, n_classes))
    choice_head.load_state_dict({k[len("choice_head."):]: v for k, v in ckpt["heads"].items()
                                 if k.startswith("choice_head.")})

    texts = [build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q for c in battery["cases"]]
    expected = [c["expected_tool"] for c in battery["cases"]]

    def run_acc(model, tag):
        correct = 0
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in texts:
                enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt")
                out = model.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                hs = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
                pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                logits = choice_head(pooled)
                correct += int(idx_to_class[int(logits[0].argmax())] == expected[correct + len(preds)] if False else 0)
        return correct

    def run_acc(model, tag):
        correct = 0
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in texts:
                enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt")
                out = model.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                hs = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
                pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                logits = choice_head(pooled)
                correct += int(idx_to_class[int(logits[0].argmax())] == expected[correct + len(preds)] if False else 0)
        return correct

    # упрощённо: считаем accuracy инлайн
    def acc(model):
        correct = 0
        t0 = time.perf_counter()
        with torch.no_grad():
            for i, t in enumerate(texts):
                enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt")
                out = model.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                hs = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
                pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                logits = choice_head(pooled)
                correct += int(idx_to_class[int(logits[0].argmax())] == expected[i])
        wall = time.perf_counter() - t0
        return correct, wall / len(texts) * 1000

    ok0, ms0 = acc(base)
    print(f"[torch fp32] батарея: {ok0}/109 = {ok0 / 109:.1%} | {ms0:.0f} мс/кейс")

    t0 = time.perf_counter()
    qbase = qdyn(base, {torch.nn.Linear}, dtype=torch.qint8)
    print(f"quantize_dynamic(torch) за {time.perf_counter() - t0:.0f}s")
    ok1, ms1 = acc(qbase)
    print(f"[torch INT8] батарея: {ok1}/109 = {ok1 / 109:.1%} | {ms1:.0f} мс/кейс")

    torch.save({"base": qbase.base_model.state_dict(), "hidden": hidden, "n_classes": n_classes},
               OUT_PATH)
    print(f"сохранён: {OUT_PATH} ({OUT_PATH.stat().st_size / (1024 * 1024):.0f} MB)")


if __name__ == "__main__":
    main()