# -*- coding: utf-8 -*-
"""scripts/export_mmbert_prod.py — Фаза E5: экспорт mmBERT-роутера в прод (fp32 ONNX + INT8).

Зачем: прод-рантайм ходит на CPU INT8 ONNX (старая модель — leeth_int8.onnx,
см. src/leeth/export.py). Новый mmBERT-роутер (mmbert-argent + головы из
mmbert-leeth.pt) экспортируется по тому же пути: fp32 ONNX -> quantize_dynamic
-> INT8, с проверкой паритета (torch fp32 vs ONNX fp32 vs ONNX INT8) и точности
на прод-батарее (109 кейсов, CPU, батч 1 — прод-режим).

Экспортируются роутинговые головы choice + noul (score/extract — на этапе D3).
MLM-голова (256K vocab) в граф не входит — для роутинга не нужна.

Запуск: python scripts/export_mmbert_prod.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(".argent/models/mmbert-argent")
HEADS_PATH = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".argent/models/mmbert-leeth.pt"))
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
BATTERY_PATH = Path("scripts/prod_battery.json")
FP32_PATH = BASE_DIR / f"mmbert_route{SUFFIX}.onnx"
INT8_PATH = BASE_DIR / f"mmbert_route{SUFFIX}_int8.onnx"
OUT_PATH = Path(f".argent/artifacts/mmbert_export_e5{SUFFIX}.md")

TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"


def build_state(query: str) -> str:
    """Канонический прод-формат (dispatcher._build_state)."""
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


class Heads(nn.Module):
    """Точная копия Heads из port_heads.py (веса совпадают с mmbert-leeth.pt)."""

    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        self.choice_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))
        self.noul_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.score_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 10))
        self.extract_start_head = nn.Linear(hidden, 1)
        self.extract_end_head = nn.Linear(hidden, 1)


class Router(nn.Module):
    """ModernBERT-база + mean-pooling + головы choice/noul — единый ONNX-граф."""

    def __init__(self, base_model: nn.Module, heads: Heads):
        super().__init__()
        self.base = base_model
        self.choice_head = heads.choice_head
        self.noul_head = heads.noul_head

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.choice_head(pooled), self.noul_head(pooled).squeeze(-1), pooled


def main() -> None:
    import os
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    cases = battery["cases"]
    class_to_idx = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32)
    ckpt = torch.load(HEADS_PATH, map_location="cpu", weights_only=False)
    heads = Heads(ckpt["hidden"], ckpt["n_classes"])
    heads.load_state_dict(ckpt["heads"])
    router = Router(base.base_model, heads).eval()

    texts = [build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q for c in cases]
    expected = [c["expected_tool"] for c in cases]

    # ── 1) fp32 ONNX ──────────────────────────────────────────────────────────
    dummy = tokenizer("прочитай config.py", truncation=True, max_length=32, return_tensors="pt")
    torch.onnx.export(
        router,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(FP32_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["choice_logits", "noul_logits", "pooled_repr"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "choice_logits": {0: "batch"},
            "noul_logits": {0: "batch"},
            "pooled_repr": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    fp32_mb = os.path.getsize(FP32_PATH) / (1024 * 1024)
    print(f"fp32 ONNX: {FP32_PATH} ({fp32_mb:.0f} MB)")

    # ── 2) INT8 dynamic quantization ─────────────────────────────────────────
    quantize_dynamic(str(FP32_PATH), str(INT8_PATH), weight_type=QuantType.QInt8)
    int8_mb = os.path.getsize(INT8_PATH) / (1024 * 1024)
    print(f"INT8 ONNX: {INT8_PATH} ({int8_mb} MB)")

    # ── 3) Паритет: torch fp32 vs ONNX fp32 vs ONNX INT8 (8 текстов) ─────────
    sess32 = ort.InferenceSession(str(FP32_PATH), providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(str(INT8_PATH), providers=["CPUExecutionProvider"])
    diffs32, diffs8, agree8 = [], [], 0
    with torch.no_grad():
        for t in texts[:8]:
            enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt")
            ids, mask = enc["input_ids"], enc["attention_mask"]
            ref = router(ids, mask)[0]
            o32 = sess32.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
            o8 = sess8.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
            diffs32.append(float(np.max(np.abs(ref.numpy() - o32))))
            diffs8.append(float(np.max(np.abs(ref.numpy() - o8))))
            agree8 += int(int(o8[0].argmax()) == int(ref[0].argmax()))
    print(f"Паритет fp32: max|diff|={max(diffs32):.2e} | INT8: max|diff|={max(diffs8):.3f}, argmax {agree8}/8")

    # ── 4) Полная батарея через INT8 (CPU, батч 1 — прод-режим) ──────────────
    t0 = time.perf_counter()
    preds: list = []
    confs: list = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt")
        logits = sess8.run(None, {"input_ids": enc["input_ids"].numpy(),
                                  "attention_mask": enc["attention_mask"].numpy()})[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        preds.append(idx_to_class[int(logits.argmax())])
        confs.append(float(probs.max()))
    wall = time.perf_counter() - t0

    correct = 0
    per_tool: dict = {}
    corrs: list = []
    misses: list = []
    for case, pred, exp, conf in zip(cases, preds, expected, confs):
        ok = pred == exp
        correct += ok
        corrs.append(1.0 if ok else 0.0)
        bucket = per_tool.setdefault(exp, {"n": 0, "ok": 0})
        bucket["n"] += 1
        bucket["ok"] += ok
        if not ok:
            misses.append(f"- {case['query']!r}: expected **{exp}**, pred **{pred}** (conf {round(conf, 3)})")

    n = len(cases)
    acc = correct / n
    ece = 0.0
    brier = sum((c - o) ** 2 for c, o in zip(confs, corrs)) / n
    bins: list = [[] for _ in range(10)]
    for c, o in zip(confs, corrs):
        bins[min(9, int(c * 10))].append((c, o))
    for b in bins:
        if b:
            ece += len(b) / n * abs(sum(o for _, o in b) / len(b) - sum(c for c, _ in b) / len(b))

    lines = [
        "# E5: экспорт mmBERT-роутера (fp32 ONNX + INT8)",
        "",
        f"- fp32: {FP32_PATH} ({fp32_mb:.0f} MB) | INT8: {INT8_PATH} ({int8_mb:.0f} MB)",
        f"- Паритет (8 текстов): fp32 max|diff|={max(diffs32):.2e} | INT8 max|diff|={max(diffs8):.3f}, argmax {agree8}/8",
        f"- **Батарея через INT8 (CPU, батч 1): {acc:.1%}** ({correct}/{n})",
        f"- Torch fp32 GPU (E4): 67.0% | Старая модель INT8 CPU: 54.1%",
        f"- ECE: {ece:.4f} | Brier: {brier:.4f}",
        f"- Wall: {wall:.1f}s ({wall / n * 1000:.0f} мс/кейс, CPU INT8, батч 1)",
        "",
        "## Per-tool accuracy (INT8)",
        "",
        "| Expected tool | n | ok |",
        "|---|---|---|",
    ]
    for tool, b in sorted(per_tool.items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {tool} | {b['n']} | {b['ok']} ({b['ok'] / b['n']:.0%}) |")
    lines += ["", "## Misses", ""] + misses

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Батарея INT8: {acc:.1%} ({correct}/{n}) | {wall / n * 1000:.0f} мс/кейс | отчёт: {OUT_PATH}")


if __name__ == "__main__":
    main()