# -*- coding: utf-8 -*-
"""scripts/finetune_ranking.py — Фаза F3/G2: обучение ranking-головы (ДОЛГО ⏳).

Зачем: ranking-голова (bi-encoder) учится сопоставлять запрос и описание
элемента (скилл или браузерный элемент). После обучения route() начинает
возвращать маршруты вместо None — до этого routers честно делегируют.

Датасеты:
  - src/leeth/data/skill_pairs.jsonl   (Фаза F2)
  - src/leeth/data/browser_pairs.jsonl (Фаза G1)

БЕЗ --run: dry-run. С --run: обучение + сохранение ranking_head.pt.

Запуск (по команде): python scripts/finetune_ranking.py --run [--domain skills|browser|both]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SKILL_PAIRS = Path("src/leeth/data/skill_pairs.jsonl")
BROWSER_PAIRS = Path("src/leeth/data/browser_pairs.jsonl")
MODEL_DIR = Path("src/leeth/data_400m")
OUT_PATH = Path("src/leeth/data/ranking_head.pt")


def build_rows(domain: str) -> list:
    """Единый формат строк: (query_text, positive_text, [distractor_texts])."""
    rows = []
    if domain in ("skills", "both") and SKILL_PAIRS.exists():
        with open(SKILL_PAIRS, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("positive") is None:
                    continue  # negative-only строки не учат скорингу
                items = [f"Skill {rec['positive']}"] + [f"Skill {d}" for d in rec["distractors"]]
                rows.append({"query": rec["query"], "items": items,
                             "target": 0, "split": rec.get("split", "train")})
    if domain in ("browser", "both") and BROWSER_PAIRS.exists():
        with open(BROWSER_PAIRS, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                items = [f"{e['role']}: {e['text']}" for e in rec["elements"]]
                rows.append({"query": rec["goal"], "items": items,
                             "target": rec["target_index"], "split": rec.get("split", "train")})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="запустить обучение (долгая операция)")
    parser.add_argument("--domain", choices=["skills", "browser", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=256)
    args = parser.parse_args()

    rows = build_rows(args.domain)
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")

    if not args.run:
        print("\n=== DRY-RUN (долгая операция НЕ запущена) ===")
        print(f"Домен:       {args.domain}")
        print(f"Строки:      {len(rows)} (train={n_train}, val={n_val})")
        print(f"Обучение:    InfoNCE на bi-encoder (backbone frozen), {args.epochs} эпох, lr={args.lr}")
        print(f"Выход:       {OUT_PATH}")
        print(f"Оценка времени: ~{args.epochs * 0.7:.1f} ч на RTX 4060")
        print("\nДля запуска: python scripts/finetune_ranking.py --run")
        return

    # ── Долгая операция ⏳ ────────────────────────────────────────────────────
    import torch
    import torch.nn.functional as F
    from src.leeth.model import LeethConfig, LeethModel
    from src.leeth.tokenizer import LeethTokenizer
    from src.leeth.ranking import LeethRankingModel
    from src.leeth.contrastive import info_nce_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = LeethTokenizer.load(MODEL_DIR)
    checkpoint = torch.load(MODEL_DIR / "leeth_best.pt", map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if isinstance(cfg, dict):
        from src.leeth.model import LeethConfig
        cfg = LeethConfig(**cfg)
    backbone = LeethModel(cfg)
    backbone.load_state_dict(checkpoint["state_dict"])
    backbone.to(device)

    ranking = LeethRankingModel(backbone, embed_dim=args.embed_dim, trainable_backbone=False)
    ranking.backbone.eval()   # замороженная база: dropout выключен, эмбеддинги стабильные
    ranking.rank_query_proj.train()
    ranking.rank_item_proj.train()
    ranking.to(device)

    def encode(text: str, max_length: int = 96) -> list:
        return tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=True)

    def pad_to(id_lists, max_len):
        pad = tokenizer.pad_token_id
        return torch.tensor([x + [pad] * (max_len - len(x)) for x in id_lists], dtype=torch.long, device=device), \
               torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in id_lists], dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(
        list(ranking.rank_query_proj.parameters()) + list(ranking.rank_item_proj.parameters()),
        lr=args.lr, weight_decay=0.01,
    )

    rng = random.Random(31337)
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        loss_sum, acc_sum, steps = 0.0, 0.0, 0
        order = [r for r in rows if r["split"] == "train"]
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            optimizer.zero_grad()
            # InfoNCE по батчу: query кодируется один раз, items — все кандидаты.
            q_texts = [r["query"] for r in batch]
            q_ids = [encode(t) for t in q_texts]
            max_q = max(len(x) for x in q_ids)
            pad = tokenizer.pad_token_id
            q_ids_t = torch.tensor([x + [pad] * (max_q - len(x)) for x in q_ids], dtype=torch.long, device=device)
            q_mask_t = torch.tensor([[1] * len(x) + [0] * (max_q - len(x)) for x in q_ids], dtype=torch.long, device=device)
            # ОБУЧЕНИЕ: grad-enabled кодирование (encode_query под no_grad —
            # только для инференса; через него градиенты не текут).
            q_emb = ranking.encode_query_trainable(q_ids_t, q_mask_t)  # [B, D]

            # Все элементы батча в один плоский список: [B * n_items, D].
            item_texts = []
            for r in batch:
                items = r.get("items") or r.get("distractors") or []
                if r.get("positive"):
                    items = [r["positive"]] + list(r["distractors"])
                item_texts.extend(items)
            i_ids = [encode(t, 64) for t in item_texts]
            max_i = max(len(x) for x in i_ids)
            i_ids_t = torch.tensor([x + [pad] * (max_i - len(x)) for x in i_ids], dtype=torch.long, device=device)
            i_mask_t = torch.tensor([[1] * len(x) + [0] * (max_i - len(x)) for x in i_ids], dtype=torch.long, device=device)
            i_emb = ranking.encode_item_trainable(i_ids_t, i_mask_t)  # [N, D]

            # InfoNCE на каждую строку батча: positive vs его дистракторы.
            loss = q_emb.new_zeros(())
            n_pairs = 0
            offset = 0
            for bi, r in enumerate(batch):
                items = r.get("items") or ([r["positive"]] + list(r["distractors"]) if r.get("positive") else [])
                n = len(items)
                if n < 2:
                    offset += n
                    continue
                q_i = q_emb[bi]
                pos_i = i_emb[offset + r["target"]]
                negs = [i_emb[offset + k] for k in range(n) if k != r["target"]]
                loss = loss + info_nce_loss(q_i.unsqueeze(0), pos_i.unsqueeze(0), [g.unsqueeze(0) for g in negs])
                offset += n
                n_pairs += 1
            if n_pairs == 0:
                continue
            loss = loss / n_pairs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranking.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item()
            steps += 1
        print(f"epoch {epoch}/{args.epochs}: loss={loss_sum / max(1, steps):.4f} ({time.perf_counter() - t0:.0f}s)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"query_proj": ranking.rank_query_proj.state_dict(),
                "item_proj": ranking.rank_item_proj.state_dict(),
                "embed_dim": args.embed_dim}, OUT_PATH)
    print(f"Saved -> {OUT_PATH}")
    print("СЛЕДУЮЩИЙ ШАГ (F4/G3): rebuild_index + валидация top-1 на val-сплите.")


if __name__ == "__main__":
    main()