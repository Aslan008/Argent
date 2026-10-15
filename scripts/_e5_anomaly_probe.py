# -*- coding: utf-8 -*-
"""scripts/_e5_anomaly_probe.py — диагностика взрыва активаций на слое 12 mmBERT.

INT8-квантизация роняет батарею 67% -> 26-44% у всех вариантов. Причина:
max|activation| растёт с ~75 (layer 11) до ~6714 (layer 12) — шкала активаций
вычисляется по выбросу. Скрипт локализует источник выброса (attn vs mlp) и
проверяет веса/нормы слоёв 11-12.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE = Path(".argent/models/mmbert-argent")
BATTERY = Path("scripts/prod_battery.json")
TQ = "Какой инструмент Argent должен быть вызван в первую очередь?"


def build_state(q: str) -> str:
    n = q[:1].lower() + q[1:] if q[:1].isalpha() else q
    return f'Incoming user request in Argent session:\n"{n}"\nWorking directory: C:/Users/user/Desktop/Argent'


def main() -> None:
    base = AutoModelForMaskedLM.from_pretrained(str(BASE), dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(str(BASE))
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    enc = tok(build_state(battery["cases"][0]["query"]) + "\n[QUESTION] " + TQ,
              return_tensors="pt")
    bm = base.base_model
    with torch.no_grad():
        h = bm.embeddings(enc["input_ids"])
        seq_len = enc["input_ids"].shape[1]
        pos_ids = torch.arange(seq_len).unsqueeze(0)
        pos = bm.rotary_emb(h, position_ids=pos_ids, layer_type="full_attention")
        for i in range(13):
            layer = bm.layers[i]
            attn_out = layer.attn(layer.attn_norm(h), position_embeddings=pos,
                                  attention_mask=enc["attention_mask"].bool())
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h2 = h + attn_out
            mlp_out = layer.mlp(layer.mlp_norm(h2))
            h = h2 + mlp_out
            print(f"layer {i}: attn_out max={attn_out.abs().max():.1f}, "
                  f"mlp_out max={mlp_out.abs().max():.1f}, h max={h.abs().max():.1f}")
            if i >= 11:
                break
    for i in (11, 12):
        l = bm.layers[i]
        print(f"--- layer {i}: mlp.Wi max={l.mlp.Wi.weight.abs().max():.3f}, "
              f"mlp.Wo max={l.mlp.Wo.weight.abs().max():.3f}, "
              f"attn.Wqkv max={l.attn.Wqkv.weight.abs().max():.3f}")
        print(f"    attn_norm weight max={l.attn_norm.weight.abs().max():.3f}, "
              f"mlp_norm weight max={l.mlp_norm.weight.abs().max():.3f}")


if __name__ == "__main__":
    main()