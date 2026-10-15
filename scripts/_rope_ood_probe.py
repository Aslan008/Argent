# -*- coding: utf-8 -*-
"""scripts/_rope_ood_probe.py — проверка OOD-позиций mmBERT (претрейн был 256).

Метод: длинные тексты (код проекта, 1000+ токенов), MLM-лосс маскированных
токенов по диапазонам позиций: 0-128, 128-256, 256-512, 512-1024.
Если лосс растёт с позицией — RoPE-позиции 256+ OOD, нужен MLM-прогон
на длинных последовательностях.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from pathlib import Path as P

BASE_DIR = P(".argent/models/mmbert-argent")


def main():
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    model = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Длинные тексты: код проекта
    texts = []
    for f in Path("src/leeth").glob("*.py"):
        t = f.read_text(encoding="utf-8", errors="replace")
        if len(t) > 4000:
            texts.append(t[:6000])
        if len(texts) >= 8:
            break
    print(f"текстов: {len(texts)}")

    ranges = [(0, 128), (128, 256), (256, 512), (512, 1024)]
    losses = {r: [] for r in ranges}
    with torch.no_grad():
        for t in texts:
            enc = tok(t, truncation=True, max_length=1024, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            n = int(mask.sum())
            if n < 600:
                continue
            # маскируем 15% токенов детерминированно
            gen = torch.Generator().manual_seed(7)
            prob = torch.rand(ids.shape, generator=gen).to(device)
            special = (ids == tok.cls_token_id) | (ids == tok.sep_token_id) | (ids == tok.pad_token_id)
            to_mask = (prob < 0.15) & ~special
            labels = ids.clone()
            labels[~to_mask] = -100
            ids_m = ids.masked_fill(to_mask, tok.mask_token_id)
            out = model(input_ids=ids_m, attention_mask=mask, labels=labels)
            # per-token лосс
            logits = out.logits
            ce = torch.nn.functional.cross_entropy(
                logits[0], labels[0], reduction="none", ignore_index=-100)
            pos = torch.arange(n, device=device)
            masked_pos = pos[labels[0] != -100]
            tok_losses = ce[:n]
            for r in ranges:
                sel = masked_pos[(masked_pos >= r[0]) & (masked_pos < r[1])]
                if len(sel):
                    losses[r].append(tok_losses[sel].mean().item())

    print(f"\n{'позиции':<12} {'MLM loss':>9} {'n_masked':>9}")
    for r in ranges:
        if losses[r]:
            import numpy as np
            print(f"{r[0]}-{r[1]:<8} {np.mean(losses[r]):>9.3f} {len(losses[r]):>9}")


if __name__ == "__main__":
    main()