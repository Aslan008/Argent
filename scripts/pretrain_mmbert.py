# -*- coding: utf-8 -*-
"""scripts/pretrain_mmbert.py — Фаза E2: continue-pretrain mmBERT (ДОЛГО ⏳⏳, дни).

Зачем: mmBERT знает общий язык, но не знает наш домен (Argent, тулинг,
скиллы, браузер, безопасность). Continue-pretrain на MLM по нашему корпусу
адаптирует базу под домен ДО обучения голов.

Корпус: код проекта (collect_code_corpus) + наши датасеты (state/question
тексты из train jsonl) — модель видит и код, и наши форматы состояний.

БЕЗ --run: dry-run. С --run: претрейн (ДНИ на RTX 4060).

Запуск (по команде): python scripts/pretrain_mmbert.py --run --epochs 2
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_DIR = Path(".argent/models/mmbert-base")
OUT_DIR = Path(".argent/models/mmbert-argent")
CORPUS_SOURCES = [
    "src/leeth/data_400m/leeth_400m_train.jsonl",
    "src/leeth/data/leeth_train.jsonl",
    "src/leeth/data/leeth_feedback.jsonl",
]


def collect_texts(max_texts: int = 60000) -> list:
    """Корпус: код проекта + наши state/question тексты."""
    from src.leeth.pretrain import collect_code_corpus
    texts = collect_code_corpus()
    for src in CORPUS_SOURCES:
        path = Path(src)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                state = rec.get("state")
                if state and len(state) > 40:
                    texts.append(state)
                if len(texts) >= max_texts:
                    break
        if len(texts) >= max_texts:
            break
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="запустить претрейн (ДОЛГО, дни)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="градиентная аккумуляция: эффективный батч = batch-size * grad-accum")
    parser.add_argument("--freeze-embeddings", action="store_true",
                        help="заморозить input-эмбеддинги и MLM-декодер (экономия ~4.5 GB VRAM; тело модели адаптируется, пространство эмбеддингов стабильно для E3)")
    parser.add_argument("--max-texts", type=int, default=60000)
    args = parser.parse_args()

    texts = collect_texts(args.max_texts)

    if not args.run:
        print("\n=== DRY-RUN (долгая операция НЕ запущена) ===")
        print(f"База:        {MODEL_DIR} (сначала: python scripts/bench_mmbert.py --download)")
        print(f"Корпус:      {len(texts)} текстов")
        print(f"Обучение:    MLM continue-pretrain, {args.epochs} эпох, lr={args.lr}, seq<={args.max_length}")
        print(f"Выход:       {OUT_DIR}")
        print(f"Оценка времени: ДНИ на RTX 4060 (не ускоряется) ⏳⏳")
        print("\nДля запуска: python scripts/pretrain_mmbert.py --run --epochs 2")
        return

    # ── Долгая операция ⏳⏳ (дни) ────────────────────────────────────────────
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForMaskedLM.from_pretrained(str(MODEL_DIR), dtype=torch.float32)
    model.to(device)
    if args.freeze_embeddings:
        for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
            if emb is not None:
                emb.weight.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in params)
    print(f"Параметров: {n_total / 1e6:.0f}M, обучаемых: {n_train / 1e6:.0f}M "
          f"({'эмбеддинги+декодер заморожены' if args.freeze_embeddings else 'вся модель'})")

    class TextDataset(Dataset):
        """RAM-lean: храним токенизацию numpy-массивами (int32) вместо питоновских
        списков (~7x меньше памяти на 109K текстов; иначе commit-OOM на WDDM)."""

        def __init__(self, texts, tokenizer, max_length):
            import numpy as np
            self.enc = []
            for t in texts:
                e = tokenizer(t, truncation=True, max_length=max_length,
                              return_special_tokens_mask=True)
                self.enc.append({k: np.asarray(v, dtype=np.int32)
                                 for k, v in e.items()})

        def __len__(self):
            return len(self.enc)

        def __getitem__(self, idx):
            return {k: v.tolist() for k, v in self.enc[idx].items()}

    dataset = TextDataset(texts, tokenizer, args.max_length)
    texts.clear()  # строки корпуса больше не нужны — освободить RAM до тренировки
    import gc
    gc.collect()
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        total, steps = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(**batch)
                loss = out.loss / args.grad_accum
            if not torch.isnan(loss) and loss.item() > 0:
                scaler.scale(loss).backward()
                total += loss.item() * args.grad_accum
                steps += 1
                if steps % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        if steps % args.grad_accum != 0:  # хвост аккумуляции в конце эпохи
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        print(f"epoch {epoch}/{args.epochs}: mlm_loss={total / max(1, steps):.4f} "
              f"steps={steps} ({time.perf_counter() - t0:.0f}s)")
        if epoch == 1:  # страховка: чекпоинт после первой эпохи (финальный save ниже)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(OUT_DIR))
            tokenizer.save_pretrained(str(OUT_DIR))
            print(f"checkpoint after epoch 1 -> {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"Saved -> {OUT_DIR}")
    print("СЛЕДУЮЩИЙ ШАГ (E3): python scripts/port_heads.py — перенос голов.")


if __name__ == "__main__":
    main()