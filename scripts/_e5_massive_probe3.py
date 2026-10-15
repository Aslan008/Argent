# -*- coding: utf-8 -*-
"""scripts/_e5_massive_probe3.py — каскад massive-каналов: слои 11..21.

Для каждого слоя (с учётом уже сделанных выносов) находим канал act с
max|значение| на ВСЕХ токенах всех 20 текстов + среднее значение этого канала
на токенах, где он выстреливает (|act| > 100). Результат — SPECS для exp9.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE = Path(".argent/models/mmbert-argent")
BATTERY = Path("scripts/prod_battery.json")
TQ = "Какой инструмент Argent должен быть вызван в первую очередь?"
THRESH = 100.0


def build_state(q: str) -> str:
    n = q[:1].lower() + q[1:] if q[:1].isalpha() else q
    return f'Incoming user request in Argent session:\n"{n}"\nWorking directory: C:/Users/user/Desktop/Argent'


def main() -> None:
    base = AutoModelForMaskedLM.from_pretrained(str(BASE), dtype=torch.float32).eval()
    bm = base.base_model
    tok = AutoTokenizer.from_pretrained(str(BASE))
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    texts = [build_state(c["query"]) + "\n[QUESTION] " + TQ for c in battery["cases"][:20]]

    specs = []
    with torch.no_grad():
        for layer_i in range(11, 22):
            # применяем уже найденные выносы (обнуляем канал act в Wo)
            for li, ch, _ in specs:
                bm.layers[li].mlp.Wo.weight[:, ch] = 0.0
            vals = []
            for t in texts:
                enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
                h = bm.embeddings(enc["input_ids"])
                pos_ids = torch.arange(enc["input_ids"].shape[1]).unsqueeze(0)
                pos = bm.rotary_emb(h, position_ids=pos_ids, layer_type="full_attention")
                for i in range(layer_i):
                    layer = bm.layers[i]
                    attn_out = layer.attn(layer.attn_norm(h), position_embeddings=pos,
                                          attention_mask=enc["attention_mask"].bool())
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h2 = h + attn_out
                    h = h2 + layer.mlp(layer.mlp_norm(h2))
                layer = bm.layers[layer_i]
                attn_out = layer.attn(layer.attn_norm(h), position_embeddings=pos,
                                      attention_mask=enc["attention_mask"].bool())
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h2 = h + attn_out
                hn = layer.mlp_norm(h2)
                ff = layer.mlp.Wi(hn)
                inp, gate = ff.chunk(2, dim=-1)
                act = layer.mlp.act(inp) * gate
                # канал с max|act| по всем токенам
                am = int(act.abs().argmax())
                tok_i, ch = am // act.shape[-1], am % act.shape[-1]
                val = float(act[0, tok_i, ch])
                # среднее по токенам, где канал выстреливает
                mask = act[0, :, ch].abs() > 100
                mean_v = float(act[0, mask, ch].mean()) if mask.any() else 0.0
                n_fire = int(mask.sum())
                specs.append((layer_i, ch, mean_v))
                print(f"layer {layer_i}: канал {ch}, max {val:.1f} (токен {tok_i}), "
                      f"выстрелов {n_fire}/{act.shape[1]}, mean по ним {mean_v:.1f}")
                if abs(val) < 100:
                    print(f"  (выброс исчерпан на слое {layer_i}: max {val:.1f})")
                    specs.pop()
                    break
            else:
                continue
            break

    print()
    print("SPECS = [")
    for li, ch, mv in specs:
        print(f"    ({li}, {ch}, {mv:.1f}),")
    print("]")


if __name__ == "__main__":
    main()