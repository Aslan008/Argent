# -*- coding: utf-8 -*-
"""scripts/_d4_eval_candidate.py — оценка кандидата choice-головы до промоушена.

1. Torch fp32 (base + candidate heads): батарея raw-argmax + UNFAMILIAR recall.
2. Экспорт кандидата во ВРЕМЕННЫЙ fp32 ONNX -> INT8 dynamic.
3. Батарея через временный INT8 (прод-режим, батч-1).
4. Батч-инвариантность: batch=2 vs соло (D3-фикс Pad не должен потеряться).
5. Гейт-анализ 0.65 (need_p/conf/ALLOWED) — сравнение с текущим прод-артефактом.

Промоушен НЕ выполняется — только отчёт.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

BASE_DIR = Path(".argent/models/mmbert-argent")
HEADS_PATH = Path(".argent/models/mmbert-leeth_candidate.pt")
PROD_INT8 = BASE_DIR / "mmbert_route_prod_int8.onnx"
CAND_FP32 = BASE_DIR / "_d4_cand_fp32.onnx"
CAND_INT8 = BASE_DIR / "_d4_cand_int8.onnx"
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
BATTERY_PATH = Path("scripts/prod_battery.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
NEED_TOOL_Q = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"
DISPATCH_CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR",
]
ALLOWED_FAST_TOOLS = {"read_file", "run_command", "grep_search", "find_by_name",
                      "list_directory", "calculate", "search_files"}
THR = 0.65


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


class Heads(torch.nn.Module):
    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        self.choice_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, n_classes))
        self.noul_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(), torch.nn.Linear(hidden // 2, 1))
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(), torch.nn.Linear(hidden // 2, 10))
        self.extract_start_head = torch.nn.Linear(hidden, 1)
        self.extract_end_head = torch.nn.Linear(hidden, 1)


class Router(torch.nn.Module):
    def __init__(self, base_model, heads):
        super().__init__()
        self.base = base_model
        self.choice_head = heads.choice_head
        self.noul_head = heads.noul_head

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.choice_head(pooled), self.noul_head(pooled).squeeze(-1)


def gate_verdict(need_p, norm, best):
    if need_p < THR:
        return "need_tool_false"
    if best == "UNFAMILIAR":
        return "unfamiliar"
    if best == "NO_TOOL":
        return "no_tool_choice"
    if norm[best] < THR:
        return "low_tool_confidence"
    if best not in ALLOWED_FAST_TOOLS:
        return "tool_not_allowed"
    return "dispatch"


def analyze(sess, tok, c2i, cases, label):
    """Батарея через сессию: raw acc + гейт-анализ."""
    idx_to_class = {v: k for k, v in c2i.items()}
    raw_ok, rows = 0, []
    t0 = time.perf_counter()
    for c in cases:
        state = build_state(c["query"])
        t1 = state + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t1, truncation=True, max_length=256, return_tensors="np")
        outs = sess.run(None, {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                               "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})
        logits = outs[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        cand = {k: float(probs[c2i[k]]) for k in DISPATCH_CHOICES if k in c2i}
        total = sum(cand.values()) or 1.0
        norm = {k: v / total for k, v in cand.items()}
        best = max(norm, key=norm.get)
        t2 = state + "\n[QUESTION] " + NEED_TOOL_Q
        enc2 = tok(t2, truncation=True, max_length=256, return_tensors="np")
        outs2 = sess.run(None, {"input_ids": np.asarray(enc2["input_ids"], dtype=np.int64),
                                "attention_mask": np.asarray(enc2["attention_mask"], dtype=np.int64)})
        need_p = 1.0 / (1.0 + np.exp(-float(outs2[1][0].item())))
        raw_ok += int(idx_to_class[int(logits.argmax())] == c["expected_tool"])
        rows.append({"query": c["query"], "expected": c["expected_tool"], "best": best,
                     "conf": norm[best], "need_p": need_p,
                     "verdict": gate_verdict(need_p, norm, best)})
    wall = (time.perf_counter() - t0) / len(cases) * 1000

    disp = [r for r in rows if r["verdict"] == "dispatch"]
    ok = sum(1 for r in disp if r["best"] == r["expected"])
    false_cases = [r for r in disp if r["best"] != r["expected"]]
    unfam_exp = [r for r in rows if r["expected"] == "UNFAMILIAR"]
    unfam_hit = sum(1 for r in unfam_exp if r["best"] == "UNFAMILIAR")
    print(f"\n=== {label} ===")
    print(f"raw argmax: {raw_ok}/{len(cases)} = {raw_ok / len(cases):.1%} | {wall:.0f} мс/кейс")
    print(f"гейты 0.65: dispatch={len(disp)} ok={ok} false={len(disp) - ok}")
    for r in false_cases:
        print(f"  FALSE: {r['query'][:60]!r} expected={r['expected']} best={r['best']} conf={r['conf']:.2f}")
    print(f"UNFAMILIAR recall (raw): {unfam_hit}/{len(unfam_exp)}")
    from collections import Counter
    verdicts = Counter(r["verdict"] for r in rows)
    print("вердикты:", dict(verdicts))
    per_tool = {}
    for r in rows:
        b = per_tool.setdefault(r["expected"], {"n": 0, "ok": 0})
        b["n"] += 1
        b["ok"] += int(r["best"] == r["expected"])
    print("per-tool (худшие):")
    for tool, b in sorted(per_tool.items(), key=lambda kv: kv[1]["ok"] / kv[1]["n"])[:6]:
        print(f"  {tool:18s} {b['ok']}/{b['n']} ({b['ok'] / b['n']:.0%})")
    misses = [r for r in rows if r["best"] != r["expected"]]
    print(f"промахи raw ({len(misses)}):")
    for r in misses[:12]:
        print(f"  - {r['query'][:55]!r}: exp={r['expected']} pred={r['best']} ({r['conf']:.2f})")
    return {"raw_acc": raw_ok / len(cases), "dispatch": len(disp), "ok": ok,
            "false": len(disp) - ok, "unfam_recall": f"{unfam_hit}/{len(unfam_exp)}"}


def main() -> None:
    import onnxruntime as ort
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from onnxruntime.quantization import QuantType, quantize_dynamic

    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    cases = battery["cases"]
    c2i = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))

    # ── 1) torch fp32 reference (candidate) ───────────────────────────────────
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), torch_dtype=torch.float32)
    base.eval()
    ckpt = torch.load(HEADS_PATH, map_location="cpu", weights_only=False)
    heads = Heads(ckpt["hidden"], ckpt["n_classes"])
    heads.load_state_dict(ckpt["heads"])
    heads.eval()
    router = Router(base.base_model, heads).eval()

    texts = [build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q for c in cases]
    idx_to_class = {v: k for k, v in c2i.items()}
    torch_ok = 0
    with torch.no_grad():
        for t, c in zip(texts, cases):
            enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
            logits = router(enc["input_ids"], enc["attention_mask"])[0][0]
            torch_ok += int(idx_to_class[int(logits.argmax())] == c["expected_tool"])
    print(f"torch fp32 (candidate): {torch_ok}/{len(cases)} = {torch_ok / len(cases):.1%}")

    # ── 2) временный экспорт ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    dummy = tok("прочитай config.py", truncation=True, max_length=32, return_tensors="pt")
    torch.onnx.export(
        router, (dummy["input_ids"], dummy["attention_mask"]), str(CAND_FP32),
        input_names=["input_ids", "attention_mask"],
        output_names=["choice_logits", "noul_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "choice_logits": {0: "batch"}, "noul_logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"fp32 ONNX: {CAND_FP32} ({CAND_FP32.stat().st_size / 1e6:.0f} MB, {time.perf_counter() - t0:.0f}s)")
    quantize_dynamic(str(CAND_FP32), str(CAND_INT8), weight_type=QuantType.QInt8)
    print(f"INT8 ONNX: {CAND_INT8} ({CAND_INT8.stat().st_size / 1e6:.0f} MB)")

    # ── 3) батч-инвариантность кандидата ──────────────────────────────────────
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess_c = ort.InferenceSession(str(CAND_INT8), so, providers=["CPUExecutionProvider"])
    test_texts = texts[:10]
    max_diff, agree = 0.0, 0
    encs = [tok(t, truncation=True, max_length=256, return_tensors="np") for t in test_texts]
    solo_logits = []
    for enc in encs:
        o = sess_c.run(None, {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                              "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})[0][0]
        solo_logits.append(o)
    pair = tok([test_texts[0], test_texts[1]], truncation=True, max_length=256,
               padding=True, return_tensors="np")
    batch_logits = sess_c.run(None, {"input_ids": np.asarray(pair["input_ids"], dtype=np.int64),
                                     "attention_mask": np.asarray(pair["attention_mask"], dtype=np.int64)})[0]
    for i in range(2):
        d = float(np.max(np.abs(batch_logits[i] - solo_logits[i])))
        max_diff = max(max_diff, d)
        agree += int(int(batch_logits[i].argmax()) == int(solo_logits[i].argmax()))
    print(f"батч-инвариантность кандидата: max|diff|={max_diff:.4f}, argmax {agree}/2 "
          f"{'OK' if max_diff < 0.5 else '!!! БАТЧ-БАГ (как до D3-фикса)'}")

    # ── 4) батареи: прод vs кандидат ──────────────────────────────────────────
    sess_p = ort.InferenceSession(str(PROD_INT8), so, providers=["CPUExecutionProvider"])
    prod_m = analyze(sess_p, tok, c2i, cases, "ПРОД (текущий артефакт)")
    cand_m = analyze(sess_c, tok, c2i, cases, "КАНДИДАТ (финетюн UNFAMILIAR)")
    print("\n=== ИТОГ ===")
    for k in ("raw_acc", "dispatch", "ok", "false", "unfam_recall"):
        print(f"{k:14s}: прод={prod_m[k]}  кандидат={cand_m[k]}")


if __name__ == "__main__":
    main()