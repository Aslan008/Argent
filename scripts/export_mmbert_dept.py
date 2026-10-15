# -*- coding: utf-8 -*-
"""scripts/export_mmbert_dept.py — Вариант А: экспорт mmBERT-роутера с dept/urgency головами.

Расширяет прод-граф mmbert_route_prod.onnx (choice/noul) двумя головами из
mmbert_dept_urgency.pt: dept_logits (5 классов) + urgency_logits (3 уровня).
База и choice/noul головы НЕ меняются — routing-батарея 109 обязана остаться
91/109 (регресс-гейт). dept/urgency батареи — гейты промоушена.

Выходы ONNX: choice_logits, noul_logits, dept_logits, urgency_logits.
Старый рантайм (ожидает 2 выхода) не ломается: он читает outs[0]/outs[1].

Запуск: python scripts/export_mmbert_dept.py [--promote]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else Path(".argent/models/mmbert-argent")
# ВАЖНО: база должна совпадать с той, на которой обучались головы (fp16 у Kaggle-кернела),
# иначе фичи обучения и экспорта расходятся (v5: fp16-обучение vs fp32-экспорт = 55% вместо 89%).
HEADS_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else Path(".argent/models/mmbert-leeth.pt")
DU_PATH = Path(".argent/models/mmbert_dept_urgency_v4.pt")  # v4: Kaggle-кернел leeth-du-kernel (dept 94.7%, urg 100% val)
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
ROUTING_BATTERY = Path("scripts/prod_battery.json")
FP32_PATH = BASE_DIR / "mmbert_route_du.onnx"
INT8_PATH = BASE_DIR / "mmbert_route_du_int8.onnx"
OUT_PATH = Path(".argent/artifacts/mmbert_export_du.md")

TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
Q_DEPARTMENT = "В какой отдел нужно направить это обращение?"
Q_URGENCY = "Насколько срочно нужно решить проблему клиента?"


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


class Heads(nn.Module):
    """Точная копия Heads из port_heads.py (choice/noul — routing)."""

    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        self.choice_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))
        self.noul_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))


class DeptHead(nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class UrgencyHead(nn.Module):
    def __init__(self, hidden, n_levels):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, n_levels))

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    """ModernBERT-база + пулы + 4 головы — единый ONNX-граф.

    choice/noul — mean-pool (как в E5). dept/urgency — mm-пул v3:
    concat(mean, max) — max-pool устойчив к размыванию шума (context-rot).
    """

    def __init__(self, base_model, heads, dept_head, urg_head, du_pool="mm"):
        super().__init__()
        self.base = base_model
        self.choice_head = heads.choice_head
        self.noul_head = heads.noul_head
        self.dept_head = dept_head.net
        self.urgency_head = urg_head.net
        # Пул dept/urgency берётся из DU-чекпоинта: v4 (Kaggle) обучен на mean,
        # старый v3-чекпоинт — на mm (concat(mean, max)).
        self.du_pool = du_pool

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        mx = hs.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4).max(dim=1).values
        du_in = pooled if self.du_pool == "mean" else torch.cat([pooled, mx], dim=-1)
        return (self.choice_head(pooled), self.noul_head(pooled).squeeze(-1),
                self.dept_head(du_in), self.urgency_head(du_in), pooled)


def run_battery(sess, tokenizer, cases, texts, idx_to_class, head_kind):
    """Батарея батч-1 (прод-режим). head_kind: choice|dept|urgency."""
    out_idx = {"choice": 0, "dept": 2, "urgency": 3}[head_kind]
    preds, confs = [], []
    t0 = time.perf_counter()
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt")
        logits = sess.run(None, {"input_ids": enc["input_ids"].numpy(),
                                 "attention_mask": enc["attention_mask"].numpy()})[out_idx][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        preds.append(int(logits.argmax()))
        confs.append(float(probs.max()))
    wall = time.perf_counter() - t0
    correct = 0
    corrs = []
    misses = []
    for i, (case, pred, conf) in enumerate(zip(cases, preds, confs)):
        if head_kind == "choice":
            exp = case["expected_tool"]
            ok = idx_to_class[pred] == exp
        elif head_kind == "dept":
            exp = case["gold"]
            ok = DEPT_CLASSES[pred] == exp
        else:
            exp = case["gold"]
            ok = pred + 1 == int(exp)
        correct += ok
        corrs.append(1.0 if ok else 0.0)
        if not ok:
            label = idx_to_class[pred] if head_kind == "choice" else (DEPT_CLASSES[pred] if head_kind == "dept" else pred + 1)
            q = case.get("query") or case.get("state") or ""
            misses.append(f"- {q[:70]!r}: expected **{exp}**, pred **{label}** (conf {conf:.3f})")
    n = len(cases)
    acc = correct / n
    ece = 0.0
    brier = sum((c - o) ** 2 for c, o in zip(confs, corrs)) / n
    bins = [[] for _ in range(10)]
    for c, o in zip(confs, corrs):
        bins[min(9, int(c * 10))].append((c, o))
    for b in bins:
        if b:
            ece += len(b) / n * abs(sum(o for _, o in b) / len(b) - sum(c for c, _ in b) / len(b))
    return acc, ece, brier, wall, misses


DEPT_CLASSES = ["billing", "technical", "sales", "account", "other"]


def main() -> None:
    promote = "--promote" in sys.argv
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # ── Батареи ──────────────────────────────────────────────────────────────
    routing = json.loads(ROUTING_BATTERY.read_text(encoding="utf-8"))["cases"]
    spec_du = Path("scripts/_multidomain_battery.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("mdbattery", spec_du)
    mdb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mdb)
    dept_cases = [{"state": mdb.prod_envelope(st), "gold": next(g for q, g in qs if q == "department")}
                  for _, st, qs in mdb.CASES if any(q == "department" for q, _ in qs)]
    urg_cases = [{"state": mdb.prod_envelope(st), "gold": next(g for q, g in qs if q == "urgency")}
                 for _, st, qs in mdb.CASES if any(q == "urgency" for q, _ in qs)]

    class_to_idx = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32)
    ckpt = torch.load(HEADS_PATH, map_location="cpu", weights_only=False)
    heads = Heads(ckpt["hidden"], ckpt["n_classes"])
    # Чекпоинт несёт все 5 голов (port_heads.py); грузим только choice/noul —
    # strict=False, но проверяем, что routing-ключи покрыты полностью.
    missing, unexpected = heads.load_state_dict(ckpt["heads"], strict=False)
    assert not missing, f"routing-головы недозагружены: {missing}"
    allowed_unexpected = {k for k in unexpected if k.startswith(("score_head.", "extract_start_head.", "extract_end_head."))}
    assert allowed_unexpected == set(unexpected), f"неожиданные ключи: {unexpected}"
    du = torch.load(DU_PATH, map_location="cpu", weights_only=False)
    dept_head = DeptHead(du["hidden"], len(du["dept_classes"]))
    dept_head.load_state_dict(du["dept_head"])
    urg_head = UrgencyHead(du["hidden"], du["urgency_levels"])
    urg_head.load_state_dict(du["urgency_head"])
    router = Router(base.base_model, heads, dept_head, urg_head, du_pool=du.get("pool", "mm")).eval()

    routing_texts = [build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q for c in routing]
    dept_texts = [c["state"] + "\n[QUESTION] " + Q_DEPARTMENT for c in dept_cases]
    urg_texts = [c["state"] + "\n[QUESTION] " + Q_URGENCY for c in urg_cases]

    # ── 1) fp32 ONNX ─────────────────────────────────────────────────────────
    dummy = tokenizer("прочитай config.py", truncation=True, max_length=32, return_tensors="pt")
    torch.onnx.export(
        router,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(FP32_PATH),
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
    fp32_mb = os.path.getsize(FP32_PATH) / (1024 * 1024)
    print(f"fp32 ONNX: {FP32_PATH} ({fp32_mb:.0f} MB)")

    # ── 2) INT8 ──────────────────────────────────────────────────────────────
    quantize_dynamic(str(FP32_PATH), str(INT8_PATH), weight_type=QuantType.QInt8)
    int8_mb = os.path.getsize(INT8_PATH) / (1024 * 1024)
    print(f"INT8 ONNX: {INT8_PATH} ({int8_mb:.0f} MB)")

    # ── 3) Паритет fp32: новые выходы vs torch ───────────────────────────────
    sess32 = ort.InferenceSession(str(FP32_PATH), providers=["CPUExecutionProvider"])
    diffs = []
    diffs_pooled = []
    with torch.no_grad():
        for t in dept_texts[:4] + urg_texts[:4]:
            enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt")
            ref = router(enc["input_ids"], enc["attention_mask"])
            o = sess32.run(None, {"input_ids": enc["input_ids"].numpy(),
                                  "attention_mask": enc["attention_mask"].numpy()})
            diffs.append(float(np.max(np.abs(ref[2].detach().numpy() - o[2]))))
            diffs.append(float(np.max(np.abs(ref[3].detach().numpy() - o[3]))))
            diffs_pooled.append(float(np.max(np.abs(ref[4].detach().numpy() - o[4]))))
    print(f"Паритет fp32 dept/urgency: max|diff|={max(diffs):.2e} | pooled: max|diff|={max(diffs_pooled):.2e}")

    # ── 4) Батареи через fp32 (ПРОД-режим: прод-файл fp32, INT8 — фолбэк) ────
    # Исторический факт: INT8-квантование mmBERT никогда не брало routing-гейт
    # (cand10: INT8 45/109 vs fp32 91/109) — прод всегда ходит на fp32 ONNX.
    r_acc, r_ece, r_brier, r_wall, r_miss = run_battery(sess32, tokenizer, routing, routing_texts, idx_to_class, "choice")
    d_acc, d_ece, d_brier, d_wall, d_miss = run_battery(sess32, tokenizer, dept_cases, dept_texts, idx_to_class, "dept")
    u_acc, u_ece, u_brier, u_wall, u_miss = run_battery(sess32, tokenizer, urg_cases, urg_texts, idx_to_class, "urgency")

    ROUTE_GATE = 91 / 109  # cand10 — регресс-гейт
    DEPT_GATE, URG_GATE = 0.70, 0.615
    gates_ok = r_acc >= ROUTE_GATE and d_acc >= DEPT_GATE and u_acc >= URG_GATE
    print(f"\nROUTING: {r_acc:.1%} ({round(r_acc * 109)}/109, гейт >= {ROUTE_GATE:.1%}) ECE={r_ece:.4f} {r_wall / 109 * 1000:.0f} мс/кейс")
    print(f"DEPT:    {d_acc:.1%} (гейт >= {DEPT_GATE:.0%}) ECE={d_ece:.4f}")
    print(f"URGENCY: {u_acc:.1%} (гейт >= {URG_GATE:.1%}) ECE={u_ece:.4f}")
    print(f"ГЕЙТЫ: {'OK' if gates_ok else 'FAIL'}")

    lines = [
        "# Экспорт mmBERT-роутера с dept/urgency головами (вариант А)",
        "",
        f"- fp32: {FP32_PATH} ({fp32_mb:.0f} MB) | INT8: {INT8_PATH} ({int8_mb:.0f} MB)",
        f"- Паритет fp32 dept/urgency: max|diff|={max(diffs):.2e}",
        "",
        "| Батарея | Точность | ECE | Brier | мс/кейс |",
        "|---|---|---|---|---|",
        f"| Routing 109 | {r_acc:.1%} ({round(r_acc * 109)}/109) | {r_ece:.4f} | {r_brier:.4f} | {r_wall / 109 * 1000:.0f} |",
        f"| Department 14 | {d_acc:.1%} | {d_ece:.4f} | {d_brier:.4f} | {d_wall / 14 * 1000:.0f} |",
        f"| Urgency 13 | {u_acc:.1%} | {u_ece:.4f} | {u_brier:.4f} | {u_wall / 13 * 1000:.0f} |",
        "",
        f"- Гейты: routing >= {ROUTE_GATE:.1%} {'OK' if r_acc >= ROUTE_GATE else 'FAIL'}, "
        f"dept >= {DEPT_GATE:.0%} {'OK' if d_acc >= DEPT_GATE else 'FAIL'}, "
        f"urgency >= {URG_GATE:.1%} {'OK' if u_acc >= URG_GATE else 'FAIL'}",
        "",
        "## Промахи routing", "",
    ] + r_miss + ["", "## Промахи dept", ""] + d_miss + ["", "## Промахи urgency", ""] + u_miss

    if promote and gates_ok:
        prod = BASE_DIR / "mmbert_route_prod.onnx"
        backup = BASE_DIR / "mmbert_route_prod_backup_pre_du.onnx"
        if prod.exists():
            backup.write_bytes(prod.read_bytes())
        INT8_PROD = BASE_DIR / "mmbert_route_prod_int8.onnx"
        backup8 = BASE_DIR / "mmbert_route_prod_int8_backup_pre_du.onnx"
        if INT8_PROD.exists():
            backup8.write_bytes(INT8_PROD.read_bytes())
        import shutil
        shutil.copyfile(FP32_PATH, prod)
        shutil.copyfile(INT8_PATH, INT8_PROD)
        lines += ["", f"## ПРОМОУШЕН: {prod.name} + INT8 заменены (бэкапы: {backup.name}, {backup8.name})"]
        print(f"ПРОМОУШЕН: {prod} + {INT8_PROD}")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт: {OUT_PATH}")


if __name__ == "__main__":
    main()