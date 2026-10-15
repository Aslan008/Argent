# -*- coding: utf-8 -*-
"""scripts/_e5_after_bias_probe.py — активации ПОСЛЕ выноса канала 924 слоя 11.

Эксперимент 8: bias fp32 = 67.0% (паритет), но bias int8 = 56.0% — лучше, чем
40-44% без выноса, но всё ещё плохо. Проверяем: остались ли другие massive-
активации (слои 12-21)? Если да — повторяем вынос для них.
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

    # применяем ту же правку в torch: обнуляем Wi колонку 2076 и Wo строку 924 слоя 11
    l11 = bm.layers[11]
    with torch.no_grad():
        # torch Wi: (2304, 768) — строка 2076 = gate-канал 924; Wo: (768, 1152) — строка 924
        l11.mlp.Wi.weight[2076, :] = 0.0
        l11.mlp.Wo.weight[:, 924] = 0.0

    maxes = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
            out = base.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                  output_hidden_states=True)
            maxes.append([h.abs().max().item() for h in out.hidden_states])
    m = torch.tensor(maxes).mean(0)
    print("max|activation| по слоям (после выноса канала 924 слоя 11):")
    for i, v in enumerate(m.tolist()):
        flag = "  <-- выброс" if v > 200 else ""
        print(f"  layer {i}: {v:.1f}{flag}")


if __name__ == "__main__":
    main()