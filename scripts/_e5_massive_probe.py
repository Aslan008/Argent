# -*- coding: utf-8 -*-
"""scripts/_e5_massive_probe.py — massive-активация: константный ли это байас?

Гипотеза: act[<bos>, канал 924] слоя 11 почти константен по текстам ->
компонент Wo[:,924]*act924 на bos ведёт себя как константный вектор-байас.
Если так, его можно вынести из MatMul в bias (или в константный Add) и
INT8-квантизация перестанет видеть выброс в активациях.
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


def build_state(q: str) -> str:
    n = q[:1].lower() + q[1:] if q[:1].isalpha() else q
    return f'Incoming user request in Argent session:\n"{n}"\nWorking directory: C:/Users/user/Desktop/Argent'


def main() -> None:
    base = AutoModelForMaskedLM.from_pretrained(str(BASE), dtype=torch.float32).eval()
    bm = base.base_model
    tok = AutoTokenizer.from_pretrained(str(BASE))
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    texts = [build_state(c["query"]) + "\n[QUESTION] " + TQ for c in battery["cases"][:20]]

    vals = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
            h = bm.embeddings(enc["input_ids"])
            pos_ids = torch.arange(enc["input_ids"].shape[1]).unsqueeze(0)
            pos = bm.rotary_emb(h, position_ids=pos_ids, layer_type="full_attention")
            for i in range(11):
                layer = bm.layers[i]
                attn_out = layer.attn(layer.attn_norm(h), position_embeddings=pos,
                                      attention_mask=enc["attention_mask"].bool())
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h2 = h + attn_out
                h = h2 + layer.mlp(layer.mlp_norm(h2))
            layer = bm.layers[11]
            attn_out = layer.attn(layer.attn_norm(h), position_embeddings=pos,
                                  attention_mask=enc["attention_mask"].bool())
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h2 = h + attn_out
            hn = layer.mlp_norm(h2)
            ff = layer.mlp.Wi(hn)
            inp, gate = ff.chunk(2, dim=-1)
            act = layer.mlp.act(inp) * gate
            vals.append(act[0, 0, 924].item())

    vals = np.array(vals)
    print(f"act[<bos>, ch924] по 20 текстам: min={vals.min():.1f} max={vals.max():.1f} "
          f"mean={vals.mean():.1f} std={vals.std():.1f} (CV={vals.std() / vals.mean():.1%})")


if __name__ == "__main__":
    main()