# -*- coding: utf-8 -*-
"""scripts/_d5_promote_noul.py — промоушен кандидата noul-головы в прод-артефакт.

1. Экспорт fp32 (base + новые головы mmbert-leeth.pt) во временный ONNX.
2. INT8 dynamic.
3. Батч-инвариантность (batch=2 vs соло, 10 текстов) — при батч-баге применяется
   D3-фикс: Concat(shifted, padval_u[1,1]) -> Pad(pads=[0,0,0,1], constant_value=-1).
4. Батарея через временный INT8 (гейт: не хуже текущего прода 72/109).
5. При успехе: замена прод-артефакта (бэкап старого).
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
HEADS_PATH = Path(".argent/models/mmbert-leeth.pt")
PROD_INT8 = BASE_DIR / "mmbert_route_prod_int8.onnx"
NEW_FP32 = BASE_DIR / "_d5_new_fp32.onnx"
NEW_INT8 = BASE_DIR / "_d5_new_int8.onnx"
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
BATTERY_PATH = Path("scripts/prod_battery.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
NEED_TOOL_Q = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"


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


def main() -> None:
    import onnxruntime as ort
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from onnxruntime.quantization import QuantType, quantize_dynamic

    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    cases = battery["cases"]
    c2i = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in c2i.items()}
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))

    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), torch_dtype=torch.float32)
    base.eval()
    ckpt = torch.load(HEADS_PATH, map_location="cpu", weights_only=False)
    heads = Heads(ckpt["hidden"], ckpt["n_classes"])
    heads.load_state_dict(ckpt["heads"])
    heads.eval()
    router = Router(base.base_model, heads).eval()

    texts = [f'Incoming user request in Argent session:\n"{c["query"]}"\n'
             f'Working directory: C:/Users/user/Desktop/Argent\n[QUESTION] {TEMPLATE_Q}'
             for c in cases]

    # ── экспорт ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    dummy = tok("прочитай config.py", truncation=True, max_length=32, return_tensors="pt")
    torch.onnx.export(
        router, (dummy["input_ids"], dummy["attention_mask"]), str(NEW_FP32),
        input_names=["input_ids", "attention_mask"],
        output_names=["choice_logits", "noul_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "choice_logits": {0: "batch"}, "noul_logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"fp32: {NEW_FP32.stat().st_size / 1e6:.0f} MB ({time.perf_counter() - t0:.0f}s)")
    quantize_dynamic(str(NEW_FP32), str(NEW_INT8), weight_type=QuantType.QInt8)
    print(f"int8: {NEW_INT8.stat().st_size / 1e6:.0f} MB")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(NEW_INT8), so, providers=["CPUExecutionProvider"])

    # ── батч-инвариантность ───────────────────────────────────────────────────
    encs = [tok(t, truncation=True, max_length=256, return_tensors="np") for t in texts[:10]]
    solo = [sess.run(None, {"input_ids": np.asarray(e["input_ids"], dtype=np.int64),
                            "attention_mask": np.asarray(e["attention_mask"], dtype=np.int64)})[0][0]
            for e in encs]
    pair = tok(texts[:2], truncation=True, max_length=256, padding=True, return_tensors="np")
    batch_out = sess.run(None, {"input_ids": np.asarray(pair["input_ids"], dtype=np.int64),
                                "attention_mask": np.asarray(pair["attention_mask"], dtype=np.int64)})[0]
    max_diff = max(float(np.max(np.abs(batch_out[i] - solo[i]))) for i in range(2))
    agree = sum(int(int(batch_out[i].argmax()) == int(solo[i].argmax())) for i in range(2))
    print(f"батч-инвариантность: max|diff|={max_diff:.4f}, argmax {agree}/2 "
          f"{'OK' if max_diff < 0.5 else '!!! БАТЧ-БАГ'}")
    if max_diff >= 0.5:
        print("БАТЧ-БАГ: прод-артефакт НЕ заменяется. Требуется D3-фикс Pad в экспортном пути.")
        sys.exit(2)

    # ── батарея ───────────────────────────────────────────────────────────────
    ok = 0
    for t, c in zip(texts, cases):
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        logits = sess.run(None, {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                                 "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})[0][0]
        ok += int(idx_to_class[int(logits.argmax())] == c["expected_tool"])
    print(f"батарея через новый int8: {ok}/{len(cases)} = {ok / len(cases):.1%} (прод был 72/109)")
    if ok < 72:
        print("РЕГРЕССИЯ батареи — промоушен отменён.")
        sys.exit(3)

    # ── замена прода ──────────────────────────────────────────────────────────
    backup = BASE_DIR / "mmbert_route_prod_int8_backup_d5.onnx"
    import shutil
    shutil.copy2(PROD_INT8, backup)
    shutil.copy2(NEW_INT8, PROD_INT8)
    print(f"прод-артефакт заменён (бэкап: {backup.name})")
    # уборка временных
    NEW_FP32.unlink()
    NEW_INT8.unlink()
    print("временные onnx удалены")


if __name__ == "__main__":
    main()