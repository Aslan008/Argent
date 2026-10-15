# -*- coding: utf-8 -*-
"""scripts/export_mmbert_diverse_prod.py — Экспорт разносторонне обученного mmBERT-роутера в прод (fp32 + INT8).

Интегрирует веса голов из .argent/models/leeth_diverse_candidate.pt,
обученные на Kaggle с процедурно синтезированными примерами (12 языков,
компиляторные трейсы, Accessibility Trees, процедурная математика):
- choice_head (15 инструментов Argent, с маппингом в 26-классовое пространство роутера)
- noul_head (определение необходимости инструментов)
- dept_head и urgency_head (из mmbert_dept_urgency_v4.pt)

Выходы ONNX графа:
['choice_logits', 'noul_logits', 'dept_logits', 'urgency_logits', 'pooled_repr']

Запуск:
python scripts/export_mmbert_diverse_prod.py [--promote]
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

BASE_DIR = Path(".argent/models/mmbert-argent")
CANDIDATE_PATH = Path("src/leeth/data/leeth_diverse_checkpoint.pt")
DU_PATH = Path(".argent/models/mmbert_dept_urgency_v4.pt")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
BATTERY_PATH = Path("scripts/prod_battery.json")

FP32_OUT = BASE_DIR / "mmbert_route_diverse.onnx"
INT8_OUT = BASE_DIR / "mmbert_route_diverse_int8.onnx"
REPORT_PATH = Path(".argent/artifacts/mmbert_export_diverse.md")

TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


class DeptHead(nn.Module):
    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UrgencyHead(nn.Module):
    def __init__(self, hidden: int, n_levels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, n_levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiverseRouter(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        choice_head: nn.Module,
        noul_head: nn.Module,
        dept_head: nn.Module,
        urgency_head: nn.Module,
        du_pool: str = "mean",
    ):
        super().__init__()
        self.base = base_model
        self.choice_head = choice_head
        self.noul_head = noul_head
        self.dept_head = dept_head.net
        self.urgency_head = urgency_head.net
        self.du_pool = du_pool

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        mx = hs.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4).max(dim=1).values
        du_in = pooled if self.du_pool == "mean" else torch.cat([pooled, mx], dim=-1)

        choice_logits = self.choice_head(pooled)
        noul_logits = self.noul_head(pooled).squeeze(-1)
        dept_logits = self.dept_head(du_in)
        urg_logits = self.urgency_head(du_in)

        return choice_logits, noul_logits, dept_logits, urg_logits, pooled


def main():
    parser = argparse.ArgumentParser(description="Export diverse mmBERT router to ONNX and INT8")
    parser.add_argument("--promote", action="store_true", help="Promote the exported model to production")
    args = parser.parse_args()

    print(f"Loading base model from {BASE_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32)

    hidden = base.config.hidden_size

    # 1. Load candidate checkpoint
    print(f"Loading candidate checkpoint from {CANDIDATE_PATH}...")
    cand = torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)
    cand_sd = cand["state_dict"]
    cand_classes = cand["choice_classes"]
    metrics = cand.get("metrics", {})
    print(f"Candidate metrics: {metrics}")

    # 2. Map 15 candidate classes into the 26 routing classes of choice_classes.json
    all_classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    n_classes = len(all_classes)  # Complete routing vocabulary size (including read_background_command)

    choice_linear = nn.Linear(hidden, n_classes)
    with torch.no_grad():
        # Initialize unused classes with very low logit bias
        choice_linear.weight.zero_()
        choice_linear.bias.fill_(-100.0)

        # Copy trained weights for active Argent tool classes
        for cand_idx, cls_name in enumerate(cand_classes):
            if cls_name in all_classes:
                target_idx = all_classes[cls_name]
                if target_idx < n_classes:
                    choice_linear.weight[target_idx] = cand_sd["choice_head.weight"][cand_idx]
                    choice_linear.bias[target_idx] = cand_sd["choice_head.bias"][cand_idx]
                    print(f"Mapped class '{cls_name}': cand_idx {cand_idx} -> target_idx {target_idx}")

    # 3. Noul head
    noul_linear = nn.Linear(hidden, 1)
    with torch.no_grad():
        noul_linear.weight.copy_(cand_sd["noul_head.weight"])
        noul_linear.bias.copy_(cand_sd["noul_head.bias"])

    # 4. Dept and Urgency heads from DU checkpoint
    print(f"Loading DU heads from {DU_PATH}...")
    du = torch.load(DU_PATH, map_location="cpu", weights_only=False)
    dept_head = DeptHead(du["hidden"], len(du["dept_classes"]))
    dept_head.load_state_dict(du["dept_head"])
    urg_head = UrgencyHead(du["hidden"], du["urgency_levels"])
    urg_head.load_state_dict(du["urgency_head"])

    router = DiverseRouter(
        base_model=base.base_model,
        choice_head=choice_linear,
        noul_head=noul_linear,
        dept_head=dept_head,
        urgency_head=urg_head,
        du_pool=du.get("pool", "mean"),
    ).eval()

    # 5. Export to FP32 ONNX
    print(f"Exporting to FP32 ONNX: {FP32_OUT}...")
    dummy = tokenizer("прочитай config.py", truncation=True, max_length=32, return_tensors="pt")
    torch.onnx.export(
        router,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(FP32_OUT),
        input_names=["input_ids", "attention_mask"],
        output_names=["choice_logits", "noul_logits", "dept_logits", "urgency_logits", "pooled_repr"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "choice_logits": {0: "batch"},
            "noul_logits": {0: "batch"},
            "dept_logits": {0: "batch"},
            "urgency_logits": {0: "batch"},
            "pooled_repr": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    fp32_mb = FP32_OUT.stat().st_size / (1024 * 1024)
    print(f"FP32 ONNX exported: {fp32_mb:.1f} MB")

    # 6. Quantize to INT8
    print(f"Quantizing to INT8: {INT8_OUT}...")
    quantize_dynamic(str(FP32_OUT), str(INT8_OUT), weight_type=QuantType.QInt8)
    int8_mb = INT8_OUT.stat().st_size / (1024 * 1024)
    print(f"INT8 ONNX exported: {int8_mb:.1f} MB")

    # 7. Check parity on sample inputs
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    cases = battery["cases"]
    idx_to_class = {v: k for k, v in all_classes.items() if v < n_classes}

    sess32 = ort.InferenceSession(str(FP32_OUT), providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(str(INT8_OUT), providers=["CPUExecutionProvider"])

    texts = [build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q for c in cases]
    expected = [c["expected_tool"] for c in cases]

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

    print(f"Parity fp32 max|diff|={max(diffs32):.2e} | INT8 max|diff|={max(diffs8):.3f}, argmax agree {agree8}/8")

    # 8. Evaluate full prod battery
    def run_eval(sess, name: str):
        t0 = time.perf_counter()
        preds, confs = [], []
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt")
            logits = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0][0]
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            best_idx = int(logits.argmax())
            preds.append(idx_to_class.get(best_idx, f"unknown_{best_idx}"))
            confs.append(float(probs.max()))
        wall = time.perf_counter() - t0
        correct = sum(p == exp for p, exp in zip(preds, expected))
        acc = correct / len(cases)
        ms_per_case = (wall / len(cases)) * 1000.0
        print(f"[{name}] Battery Accuracy: {acc:.1%} ({correct}/{len(cases)}) | {ms_per_case:.1f} ms/case")
        return acc, ms_per_case, preds, confs

    acc32, ms32, preds32, confs32 = run_eval(sess32, "FP32")
    acc8, ms8, preds8, confs8 = run_eval(sess8, "INT8")

    # 9. Promotion
    report_lines = [
        "# Экспорт mmBERT-Diverse Роутера (fp32 + INT8)",
        "",
        f"- **FP32 ONNX**: `{FP32_OUT}` ({fp32_mb:.1f} MB)",
        f"- **INT8 ONNX**: `{INT8_OUT}` ({int8_mb:.1f} MB)",
        f"- **Паритет**: fp32 max|diff| = {max(diffs32):.2e}, INT8 argmax agree = {agree8}/8",
        f"- **Обучение (Kaggle GPU)**: Choice Val Acc = {metrics.get('choice_val_accuracy', 0):.1%}, Noul Val Acc = {metrics.get('noul_val_accuracy', 0):.1%}",
        f"- **Прод-батарея (109 кейсов)**:",
        f"  - FP32: {acc32:.1%} ({ms32:.1f} мс/кейс)",
        f"  - INT8: {acc8:.1%} ({ms8:.1f} мс/кейс)",
        "",
    ]

    if args.promote:
        prod_fp32 = BASE_DIR / "mmbert_route_prod.onnx"
        prod_int8 = BASE_DIR / "mmbert_route_prod_int8.onnx"
        backup_fp32 = BASE_DIR / "mmbert_route_prod_backup_pre_diverse.onnx"
        backup_int8 = BASE_DIR / "mmbert_route_prod_int8_backup_pre_diverse.onnx"

        if prod_fp32.exists():
            shutil.copy2(prod_fp32, backup_fp32)
            print(f"Backup created: {backup_fp32}")
        if prod_int8.exists():
            shutil.copy2(prod_int8, backup_int8)
            print(f"Backup created: {backup_int8}")

        shutil.copy2(FP32_OUT, prod_fp32)
        shutil.copy2(INT8_OUT, prod_int8)
        print(f"PROMOTED to production: {prod_fp32} and {prod_int8}")
        report_lines.append(f"**ПРОМОУШЕН ВЫПОЛНЕН**: боевые модели `{prod_fp32.name}` и `{prod_int8.name}` обновлены.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
