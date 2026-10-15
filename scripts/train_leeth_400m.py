"""
scripts/train_leeth_400m.py — Train a ~400M Leeth model from scratch on the
diverse universal dataset (DiverseUniversalGenerator + original synthetic mix).

Hardware-fit notes (RTX 4060 Laptop 8GB, measured 23.09.2026):
- bf16 autocast + gradient checkpointing + bitsandbytes AdamW8bit
- micro-batch 8 x grad accum 4 (effective 32), max_length 256 (P95=97 tokens)
- trains into src/leeth/data_400m/ — production v24 in src/leeth/data/ untouched
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.leeth.model import LeethConfig, LeethModel
from src.leeth.tokenizer import LeethTokenizer
from src.leeth.train import LeethTorchDataset, collate_leeth_batch, compute_calibrated_loss
from src.leeth.dataset_generator import LeethSample
from src.leeth.diverse_generator import DiverseUniversalGenerator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=120_000)
    p.add_argument("--val-samples", type=int, default=3_000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--out", type=str, default="src/leeth/data_400m")
    p.add_argument("--val-every", type=int, default=400, help="optimizer steps between validations")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=300, help="micro-steps between last-checkpoint saves")
    p.add_argument("--val-subset", type=int, default=1500)
    p.add_argument("--limit-train", type=int, default=0, help="smoke test: cap total train rows")
    return p.parse_args()


def build_dataset(args, out_dir: Path):
    """Diverse universal samples + original synthetic train set, shuffled."""
    train_path = out_dir / "leeth_400m_train.jsonl"
    val_path = out_dir / "leeth_400m_val.jsonl"
    if not train_path.exists():
        t0 = time.perf_counter()
        n = DiverseUniversalGenerator(seed=args.seed).generate_and_save(train_path, args.samples)
        print(f"generated {n} diverse train samples in {time.perf_counter()-t0:.1f}s", flush=True)
    if not val_path.exists():
        DiverseUniversalGenerator(seed=args.seed + 500).generate_and_save(val_path, args.val_samples)

    def load_jsonl(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(LeethSample(**json.loads(line)))
        return rows

    train_rows = load_jsonl(train_path)
    orig_path = ROOT / "src/leeth/data/leeth_train.jsonl"
    if orig_path.exists():
        train_rows += load_jsonl(orig_path)  # keep original task distribution in the mix
    val_rows = load_jsonl(val_path)
    random.Random(args.seed).shuffle(train_rows)
    if args.limit_train > 0:
        train_rows = train_rows[: args.limit_train]
    print(f"dataset: {len(train_rows)} train / {len(val_rows)} val", flush=True)
    return train_rows, val_rows


def make_optimizer(model, lr):
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
        print("optimizer: bitsandbytes AdamW8bit", flush=True)
    except Exception as e:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
        print(f"optimizer: torch AdamW (bnb unavailable: {str(e)[:60]})", flush=True)
    return opt


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def evaluate(model, val_ds, device, subset, max_length):
    model.eval()
    torch.cuda.empty_cache()
    subset_ds = torch.utils.data.Subset(val_ds, range(min(subset, len(val_ds))))
    loader = DataLoader(subset_ds, batch_size=16, shuffle=False, collate_fn=collate_leeth_batch)
    totals, counts = {}, {}
    n_batches = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            for k in ("choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"):
                batch[k] = batch[k].to(device)
            outputs = model(ids, attention_mask=mask)
            _, m = compute_calibrated_loss(outputs, batch)
            n_b = batch["input_ids"].size(0)
            for k, v in m.items():
                if k.endswith("_sum") or k.endswith("_count") or k in ("choice_correct",):
                    counts[k] = counts.get(k, 0.0) + v
                elif k not in ("total_loss",):
                    totals[k] = totals.get(k, 0.0) + v
            totals["total_loss"] = totals.get("total_loss", 0.0) + m["total_loss"] * n_b
            counts["total_n"] = counts.get("total_n", 0) + n_b
            counts["noul_n"] = counts.get("noul_n", 0) + int(m.get("noul_count", 0))
            counts["span_n"] = counts.get("span_n", 0) + int(m.get("span_count", 0))
    model.train()
    val_loss = totals["total_loss"] / max(1, counts.get("total_n", 1))
    acc = counts.get("choice_correct", 0) / max(1, counts.get("choice_count", 1))
    em = counts.get("span_em_sum", 0) / max(1, counts.get("span_n", 1))
    brier = counts.get("brier_sum", 0) / max(1, counts.get("noul_n", 1))
    return {"val_loss": val_loss, "choice_acc": acc, "span_em": em, "noul_brier": brier}


def main():
    args = parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    tok = LeethTokenizer.load(ROOT / "src/leeth/data")
    with open(ROOT / "src/leeth/data/choice_classes.json", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    print(f"tokenizer vocab={tok.vocab_size}, classes={len(class_to_idx)}", flush=True)

    train_rows, val_rows = build_dataset(args, out_dir)
    train_ds = LeethTorchDataset(train_rows, tok, max_length=args.max_length, class_to_idx=class_to_idx)
    val_ds = LeethTorchDataset(val_rows, tok, max_length=args.max_length, class_to_idx=class_to_idx)

    config = LeethConfig(
        vocab_size=tok.vocab_size,
        hidden_size=1024,
        num_layers=28,
        num_heads=16,
        intermediate_size=4096,
        max_position_embeddings=1024,
        num_choices=len(class_to_idx),
        dropout=0.1,
    )
    model = LeethModel(config).to(device)
    model.gradient_checkpointing = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,} ({n_params/1e6:.0f}M)", flush=True)

    opt = make_optimizer(model, args.lr)
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.accum))
    total_opt_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, args.warmup, total_opt_steps))
    print(f"total optimizer steps: {total_opt_steps} ({steps_per_epoch}/epoch)", flush=True)

    # resume
    start_step = 0
    last_path = out_dir / "leeth_400m_last.pt"
    if last_path.exists():
        # Load on CPU: avoids holding the 2GB checkpoint on GPU alongside the model
        # (with Argent's production model on the same GPU this spills to shared memory)
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        del ck
        torch.cuda.empty_cache()
        print(f"resumed from step {start_step}", flush=True)

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_leeth_batch, num_workers=0, pin_memory=True,
        drop_last=True,
    )

    model.train()
    best = float("inf")
    micro = start_step * args.accum
    t_epoch = time.perf_counter()
    losses = []
    step_t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)

    def save_last(step):
        torch.save({
            "state_dict": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "step": step, "config": config.__dict__,
            "class_to_idx": class_to_idx, "tokenizer_dir": "src/leeth/data",
        }, last_path)

    while start_step < total_opt_steps:
        micro_target = (start_step + 1) * args.accum
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            for k in ("choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"):
                batch[k] = batch[k].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(ids, attention_mask=mask)
                loss, m = compute_calibrated_loss(outputs, batch)
            (loss / args.accum).backward()
            losses.append(loss.item())
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                start_step += 1

                if start_step % 25 == 0:
                    dt = time.perf_counter() - step_t0
                    step_t0 = time.perf_counter()
                    mem = torch.cuda.max_memory_allocated() / 2**30
                    avg = sum(losses[-100:]) / len(losses[-100:])
                    eta_h = (total_opt_steps - start_step) * dt / 25 / 3600
                    print(
                        f"step {start_step}/{total_opt_steps} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} "
                        f"{dt/25:.2f}s/step vram={mem:.2f}GB eta={eta_h:.1f}h",
                        flush=True,
                    )
                    torch.cuda.reset_peak_memory_stats()
                if start_step % args.save_every == 0:
                    save_last(start_step)
                if start_step % args.val_every == 0:
                    metrics = evaluate(model, val_ds, device, args.val_subset, args.max_length)
                    score = metrics["val_loss"]
                    print(f"  VAL @{start_step}: {json.dumps(metrics)}", flush=True)
                    if score < best:
                        best = score
                        torch.save({
                            "config": config.__dict__, "state_dict": model.state_dict(),
                            "class_to_idx": class_to_idx, "tokenizer_dir": "src/leeth/data",
                            "val_metrics": metrics, "step": start_step,
                            "params": n_params,
                        }, out_dir / "leeth_400m_best.pt")
                        print(f"  saved BEST (val_loss={score:.4f})", flush=True)
                if start_step >= total_opt_steps:
                    break
        # epoch boundary
        print(f"epoch done at step {start_step}, elapsed {(time.perf_counter()-t_epoch)/60:.1f} min", flush=True)
        t_epoch = time.perf_counter()
        # reshuffle via new loader epoch happens automatically (shuffle=True)

    save_last(start_step)
    metrics = evaluate(model, val_ds, device, args.val_subset, args.max_length)
    torch.save({
        "config": config.__dict__, "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx, "tokenizer_dir": "src/leeth/data",
        "val_metrics": metrics, "step": start_step, "params": n_params,
        "meta": {"samples": len(train_rows), "epochs": args.epochs, "lr": args.lr,
                 "batch_eff": args.batch_size * args.accum, "max_length": args.max_length},
    }, out_dir / "leeth_400m_final.pt")
    print(f"TRAINING DONE. final metrics: {json.dumps(metrics)}", flush=True)


if __name__ == "__main__":
    main()