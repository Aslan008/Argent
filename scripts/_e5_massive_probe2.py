# -*- coding: utf-8 -*-
"""scripts/_e5_massive_probe2.py — massive-активация слоя 13: какой канал?

После выноса канала 924 слоя 11 выброс сместился: слои 13-21 держат ~2900
(было ~6700). Ищем канал-источник в слое 12 (его MLP рождает выброс для 13+).
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
    bm = base.base_model
    tok = AutoTokenizer.from_pretrained(str(BASE))
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    texts = [build_state(c["query"]) + "\n[QUESTION] " + TQ for c in battery["cases"][:5]]

    # вынос канала 924 слоя 11 (как в exp8)
    l11 = bm.layers[11]
    with torch.no_grad():
        l11.mlp.Wi.weight[2076, :] = 0.0
        l11.mlp.Wo.weight[:, 924] = 0.0

    with torch.no_grad():
        enc = tok(texts[0], truncation=True, max_length=256, return_tensors="pt")
        h = bm.embeddings(enc["input_ids"])
        pos_ids = torch.arange(enc["input_ids"].shape[1]).unsqueeze(0)
        pos = bm.rotary_emb(h, position_ids=pos_ids, layer_type="full_attention")
        for i in range(12):
            layer = bm.layers[i]
            attn_out = layer.attn(layer.attn_norm(h), position_embeddings=pos,
                                  attention_mask=enc["attention_mask"].bool())
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h2 = h + attn_out
            h = h2 + layer.mlp(layer.mlp_norm(h2))
        # слой 12 подробно: где max в act?
        layer = bm.layers[12]
        attn_out = layer.attn(layer.attn_norm(h), position_embeddings=pos,
                              attention_mask=enc["attention_mask"].bool())
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h2 = h + attn_out
        hn = layer.mlp_norm(h2)
        ff = layer.mlp.Wi(hn)
        inp, gate = ff.chunk(2, dim=-1)
        act = layer.mlp.act(inp) * gate
        am = int(act.abs().argmax())
        tok_i, ch = am // act.shape[-1], am % act.shape[-1]
        print(f"act max слоя 12: token {tok_i} ({tok.decode(enc['input_ids'][0, tok_i:tok_i+1])!r}), "
              f"channel {ch}, value {act[0, tok_i, ch].item():.1f}")
        top = act[0, 0].abs().topk(5)
        print("top-5 каналов act на <bos>:", [(int(i), round(float(v), 1)) for v, i in zip(top.values, top.indices)])
        # по остальным токенам
        top_r = act[0, 1:].abs().max(dim=0).values.topk(3)
        print("top-3 каналов по остальным токенам:", [(int(i), round(float(v), 1)) for v, i in zip(top_r.values, top_r.indices)])


if __name__ == "__main__":
    main()