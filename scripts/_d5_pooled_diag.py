# -*- coding: utf-8 -*-
"""scripts/_d5_pooled_diag.py — сравнение pooled-вектора fp32 vs int8 (вход голов)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer

BASE_DIR = Path(".argent/models/mmbert-argent")
SAFE_Q = "Является ли данная терминальная команда безопасной для выполнения?"
CMD = "Invoke-Expression (Get-Content payload.ps1)"

# находим узел, кормящий головы (Div pooled)
m = onnx.load(str(BASE_DIR / "_d5_new_fp32_final.onnx"))
g = m.graph
gemm_in = next(n for n in g.node if n.name == "/choice_head/choice_head.0/Gemm").input[0]
print("pooled tensor:", gemm_in)

for src, dst in (("_d5_new_fp32_final.onnx", "_d5_pooled_fp32.onnx"),
                 ("_d5_new_int8.onnx", "_d5_pooled_int8.onnx")):
    mm = onnx.load(str(BASE_DIR / src))
    if gemm_in not in [o.name for o in mm.graph.output]:
        vi = onnx.helper.make_empty_tensor_value_info(gemm_in)
        mm.graph.output.append(vi)
    onnx.save(mm, str(BASE_DIR / dst))

tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
so = ort.SessionOptions()
so.log_severity_level = 3
t = f"User shell execution request:\nCommand: {CMD}\nCaller: Argent Agent\n[QUESTION] {SAFE_Q}"
e = tok(t, truncation=True, max_length=256, return_tensors="np")
feed = {"input_ids": np.asarray(e["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(e["attention_mask"], dtype=np.int64)}

s32 = ort.InferenceSession(str(BASE_DIR / "_d5_pooled_fp32.onnx"), so, providers=["CPUExecutionProvider"])
s8 = ort.InferenceSession(str(BASE_DIR / "_d5_pooled_int8.onnx"), so, providers=["CPUExecutionProvider"])
outs32 = s32.run(None, feed)
outs8 = s8.run(None, feed)
p32, p8 = outs32[-1][0], outs8[-1][0]
print("pooled форма:", p32.shape)
d = np.abs(p32 - p8)
print(f"pooled: max|diff|={d.max():.4f}, mean|diff|={d.mean():.4f}")
top = np.argsort(d)[-8:][::-1]
for ch in top:
    print(f"  канал {ch}: fp32={p32[ch]:.3f} int8={p8[ch]:.3f} diff={p8[ch]-p32[ch]:+.3f}")
print(f"pooled fp32: max|v|={np.abs(p32).max():.1f}, топ-5 каналов по |v|:",
      np.argsort(np.abs(p32))[-5:][::-1].tolist())
print(f"pooled int8: max|v|={np.abs(p8).max():.1f}")
print("choice fp32:", np.round(outs32[0][0][:5], 3), "int8:", np.round(outs8[0][0][:5], 3))
print("noul fp32:", float(outs32[1][0]), "int8:", float(outs8[1][0]))