# -*- coding: utf-8 -*-
"""scripts/finetune_rlcd.py — Фаза D2: RLCD-обучение на контрастных парах (ДОЛГО ⏳).

Зачем: контрастное обучение раздвигает похожие классы (верный/похожий неверный
инструмент; безопасная/маскированная опасная команда) — то, чего CE-лосс не
делает. Смешанный лосс (supervised + contrastive) сохраняет базовую точность.

Датасет: src/leeth/data/contrastive_pairs.jsonl (Фаза C).
БЕЗ --run: dry-run. С --run: обучение.

Запуск (по команде): python scripts/finetune_rlcd.py --run
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PAIRS_PATH = Path("src/leeth/data/contrastive_pairs.jsonl")
MODEL_DIR = Path("src/leeth/data_400m")
OUT_PATH = Path("src/leeth/data/leeth_rlcd_best.pt")


def load_pairs() -> list:
    pairs = []
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def clean_sample(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="запустить обучение (долгая операция)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--contrastive-weight", type=float, default=0.4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--max-pairs", type=int, default=2000, help="лимит пар за прогон")
    args = parser.parse_args()

    pairs = load_pairs()

    if not args.run:
        print("\n=== DRY-RUN (долгая операция НЕ запущена) ===")
        print(f"Пары:        {len(pairs)} из {PAIRS_PATH} (лимит {args.max_pairs})")
        print(f"Обучение:    supervised + contrastive (weight={args.contrastive_weight}, margin={args.margin})")
        print(f"Эпохи:       {args.epochs}, lr={args.lr}, batch={args.batch_size}")
        print(f"Выход:       {OUT_PATH}")
        print(f"Оценка времени: ~{args.epochs * 0.5:.1f} ч на RTX 4060")
        print("\nДля запуска: python scripts/finetune_rlcd.py --run")
        return

    # ── Долгая операция ⏳ ────────────────────────────────────────────────────
    import torch
    import torch.nn.functional as F
    from src.leeth.model import LeethConfig, LeethModel
    from src.leeth.tokenizer import LeethTokenizer
    from src.leeth.dataset_generator import LeethSample
    from src.leeth.contrastive import (
        noul_contrastive_loss, choice_contrastive_loss, combined_loss,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = LeethTokenizer.load(MODEL_DIR)
    with open(MODEL_DIR / "choice_classes.json", "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    # Чекпоинт пикклится со старым модулем 'model_leeth' (классы переехали в
    # src.leeth.model) — алиасим, иначе pickle падает с ModuleNotFoundError.
    if "model_leeth" not in sys.modules:
        from src.leeth import model as _leeth_model

        sys.modules["model_leeth"] = _leeth_model
    checkpoint = torch.load(MODEL_DIR / "leeth_best.pt", map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if isinstance(cfg, dict):
        cfg = LeethConfig(**cfg)
    model = LeethModel(cfg)
    # Чекпоинт старее DynamicOptionHead (голова дообучалась отдельно, kernel B):
    # грузим strict=False — RLCD тренирует только choice/noul головы. Если
    # расхождение шире головы — падаем сразу с понятной ошибкой, а не учим
    # модель на половине весов.
    res = model.load_state_dict(checkpoint["state_dict"], strict=False)
    missing = [k for k in res.missing_keys if not k.startswith("dynamic_opt_head.")]
    if missing or res.unexpected_keys:
        raise RuntimeError(
            f"Неожиданное несоответствие чекпоинта: missing={missing[:5]} "
            f"unexpected={list(res.unexpected_keys)[:5]}"
        )
    model.to(device)

    def encode(sample_rec: dict) -> tuple:
        sample = LeethSample(**clean_sample(sample_rec))
        text = sample.state + "\n[QUESTION] " + sample.question
        ids = tokenizer.encode(text, max_length=320, truncation=True, add_special_tokens=True)
        return ids, sample

    rng = random.Random(2024)
    train_pairs = pairs[: args.max_pairs]
    print(f"Training on {len(train_pairs)} pairs, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        sup_sum, cont_sum, steps = 0.0, 0.0, 0
        order = list(range(len(train_pairs)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch = [train_pairs[i] for i in order[start:start + args.batch_size]]
            pos_ids, pos_samples, neg_ids, neg_samples = [], [], [], []
            for pair in batch:
                p_ids, p_sample = encode(pair["positive"])
                n_ids, n_sample = encode(pair["negative"])
                pos_ids.append(p_ids); pos_samples.append(p_sample)
                neg_ids.append(n_ids); neg_samples.append(n_sample)

            def pad_batch(id_lists):
                max_len = max(len(x) for x in id_lists)
                pad = tokenizer.pad_token_id
                ids_t = torch.tensor([x + [pad] * (max_len - len(x)) for x in id_lists], dtype=torch.long, device=device)
                mask_t = torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in id_lists], dtype=torch.long, device=device)
                return ids_t, mask_t

            pos_ids_t, pos_mask_t = pad_batch(pos_ids)
            neg_ids_t, neg_mask_t = pad_batch(neg_ids)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pos_out = model(pos_ids_t, attention_mask=pos_mask_t)
                neg_out = model(neg_ids_t, attention_mask=neg_mask_t)

                # Supervised: стандартный multi-task лосс на обеих половинах пары.
                def targets_for(samples):
                    choice_t, noul_t = [], []
                    for s in samples:
                        if s.question_type == "choice" and s.target_choice:
                            choice_t.append(class_to_idx.get(s.target_choice, -100))
                            noul_t.append(-1.0)
                        elif s.question_type == "noul" and s.target_noul is not None:
                            choice_t.append(-100)
                            noul_t.append(float(s.target_noul))
                        else:
                            choice_t.append(-100); noul_t.append(-1.0)
                    return (
                        torch.tensor(choice_t, dtype=torch.long, device=device),
                        torch.tensor(noul_t, dtype=torch.float, device=device),
                    )

                pos_choice, pos_noul = targets_for(pos_samples)
                neg_choice, neg_noul = targets_for(neg_samples)

                sup = torch.tensor(0.0, device=device)
                cmask = pos_choice != -100
                if cmask.any():
                    # CE ТОЛЬКО на positive: у negative choice-пары «метка» —
                    # намеренно неверный инструмент; CE к ней учил бы модель
                    # неправильному роутингу. Раздвигает пары контрастивный лосс.
                    sup = sup + F.cross_entropy(pos_out["choice_logits"][cmask], pos_choice[cmask], label_smoothing=0.05)
                nmask = pos_noul >= 0.0
                if nmask.any():
                    sup = sup + F.binary_cross_entropy_with_logits(pos_out["noul_logits"][nmask], pos_noul[nmask])
                    sup = sup + F.binary_cross_entropy_with_logits(neg_out["noul_logits"][nmask], neg_noul[nmask])

                # Contrastive: маржа между positive и negative.
                cont = torch.tensor(0.0, device=device)
                if nmask.any():
                    cont = cont + noul_contrastive_loss(
                        pos_out["noul_probs"][nmask], neg_out["noul_logits"][nmask].sigmoid(), margin=args.margin
                    )
                if cmask.any():
                    cont = cont + choice_contrastive_loss(
                        pos_out["choice_logits"][cmask], pos_choice[cmask],
                        neg_out["choice_logits"][cmask], neg_choice[cmask], margin=1.0,
                    )

                loss = combined_loss(sup, cont, contrastive_weight=args.contrastive_weight)

            if not torch.isnan(loss) and loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                sup_sum += sup.item()
                cont_sum += cont.item()
                steps += 1
        print(f"epoch {epoch}/{args.epochs}: sup={sup_sum / max(1, steps):.4f} "
              f"cont={cont_sum / max(1, steps):.4f} steps={steps} "
              f"({time.perf_counter() - t0:.0f}s)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Не сохраняем dynamic_opt_head: в чекпоинте-источнике её не было, здесь она
    # случайная (RLCD её не тренирует) — иначе будущий экспорт затрёт обученную голову.
    out_sd = {k: v for k, v in model.state_dict().items() if not k.startswith("dynamic_opt_head.")}
    torch.save({"state_dict": out_sd, "config": cfg}, OUT_PATH)
    print(f"Saved -> {OUT_PATH}")
    print("СЛЕДУЮЩИЙ ШАГ (D3): валидация на прод-батарее + safety-кейсы, затем export.")


if __name__ == "__main__":
    main()