# -*- coding: utf-8 -*-
"""scripts/_export_rollback_memopt.py — экспорт ONNX с оптимизацией памяти.

Тот же пайплайн, что _export_prod_onnx.py, но:
1) del checkpoint + gc.collect() после load_state_dict (минус ~1.5 ГБ RAM);
2) квантование отдельным процессом (не держит torch-модель).
"""

import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from src.leeth.model import LeethConfig, LeethModel
from src.leeth import model as _leeth_model

sys.modules["model_leeth"] = _leeth_model  # алиас для Kaggle-чекпоинтов

MODEL_DIR = Path("src/leeth/data_400m")
CKPT = MODEL_DIR / "leeth_best.pt"
OUT_FP32 = MODEL_DIR / "leeth.onnx"

# 1) Загрузка чекпоинта
checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
config = checkpoint["config"]
if isinstance(config, dict):
    config = LeethConfig(**config)
model = LeethModel(config)
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.eval()
del checkpoint  # освобождаем 1.45 ГБ до экспорта
gc.collect()
print("model loaded, checkpoint freed", flush=True)

# 2) ONNX-экспорт (FP32)
dummy_ids = torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long)
dummy_mask = torch.ones((1, 32), dtype=torch.long)
torch.onnx.export(
    model,
    (dummy_ids, dummy_mask),
    str(OUT_FP32),
    opset_version=17,
    input_names=["input_ids", "attention_mask"],
    output_names=[
        "hidden_states", "pooled_repr", "choice_logits", "noul_logits",
        "noul_probs", "score_logits", "start_logits", "end_logits", "mlm_logits",
    ],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "hidden_states": {0: "batch_size", 1: "seq_len"},
        "pooled_repr": {0: "batch_size"},
        "choice_logits": {0: "batch_size"},
        "noul_logits": {0: "batch_size"},
        "noul_probs": {0: "batch_size"},
        "score_logits": {0: "batch_size", 1: "levels"},
        "start_logits": {0: "batch_size", 1: "seq_len"},
        "end_logits": {0: "batch_size", 1: "seq_len"},
        "mlm_logits": {0: "batch_size", 1: "seq_len"},
    },
    dynamo=False,
    do_constant_folding=True,
)
del model
gc.collect()
print("FP32 EXPORTED:", OUT_FP32, OUT_FP32.stat().st_size, flush=True)