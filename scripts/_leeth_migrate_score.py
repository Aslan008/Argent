"""
scripts/_leeth_migrate_score.py — Phase 2 migration: ordinal score head + JSON states.

Loads leeth_best.pt, swaps in the fresh 10-level ordinal score head (the rest
of the network — backbone, choice, noul, extract heads and all active-learning
gains — is preserved), regenerates the synthetic dataset (new 1-5 scale score
texts + JSON-state family), trains on a score-boosted mix at low LR, gates on
non-score val losses, saves v6 + atomic best, re-exports ONNX, reloads runtime.

Run:  python scripts/_leeth_migrate_score.py   (background; ~10-20 min on CPU)
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.leeth.dataset_generator import LeethDatasetGenerator, LeethSample  # noqa: E402
from src.leeth.finetune import _eval_loss, _next_version  # noqa: E402
from src.leeth.model import LeethModel  # noqa: E402
from src.leeth.tokenizer import LeethTokenizer  # noqa: E402
from src.leeth.train import LeethTorchDataset, collate_leeth_batch, compute_calibrated_loss  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("leeth_migrate")

DATA_DIR = Path("src/leeth/data")
BEST_PATH = DATA_DIR / "leeth_best.pt"
CLASSES_PATH = DATA_DIR / "choice_classes.json"
SYNTH_TRAIN_PATH = DATA_DIR / "leeth_train.jsonl"
SYNTH_VAL_PATH = DATA_DIR / "leeth_val.jsonl"

MAX_LENGTH = 320


def _load_jsonl(path: Path) -> list:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(LeethSample(**json.loads(line)))
    return samples


def _count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _type_loss(model, samples, qtype, tokenizer, class_to_idx, device):
    subset = [s for s in samples if s.question_type == qtype]
    if not subset:
        return float("nan")
    dataset = LeethTorchDataset(subset, tokenizer, max_length=MAX_LENGTH, class_to_idx=class_to_idx)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_leeth_batch)
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            loss, _ = compute_calibrated_loss(outputs, batch)
            total += loss.item()
            batches += 1
    return total / max(1, batches)


def main():
    started = time.time()
    device = "cpu"  # CPU on purpose: CUDA VRAM is held by the main LLM
    summary = {"device": device}

    tokenizer = LeethTokenizer.load(DATA_DIR)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    checkpoint = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config.max_score_levels = 10  # extend the stored config for the new head

    model = LeethModel(config)
    # strict=False does NOT tolerate shape mismatches — drop the old scalar
    # score head entirely so it stays freshly initialized for ordinal training.
    old_sd = checkpoint["state_dict"]
    filtered = {k: v for k, v in old_sd.items() if not k.startswith("score_head")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    summary["missing_keys"] = sorted(missing)
    summary["unexpected_keys"] = sorted(unexpected)
    logger.info("Loaded checkpoint: missing=%s unexpected=%s", sorted(missing), sorted(unexpected))
    model.to(device)

    # ── Regenerate datasets: new 1-5 score texts + JSON-state family ─────────
    train_count = _count_lines(SYNTH_TRAIN_PATH)
    val_count = _count_lines(SYNTH_VAL_PATH)
    logger.info("Regenerating datasets: train=%d val=%d (same seeds/sizes)", train_count, val_count)
    LeethDatasetGenerator(seed=42).generate_and_save(SYNTH_TRAIN_PATH, target_count=train_count)
    LeethDatasetGenerator(seed=9999).generate_and_save(SYNTH_VAL_PATH, target_count=val_count)
    train_samples = _load_jsonl(SYNTH_TRAIN_PATH)
    val_samples = _load_jsonl(SYNTH_VAL_PATH)
    score_train = [s for s in train_samples if s.question_type == "score"]
    other_train = [s for s in train_samples if s.question_type != "score"]
    summary["train_total"] = len(train_samples)
    summary["train_score"] = len(score_train)
    logger.info("Train samples: %d total, %d score", len(train_samples), len(score_train))

    # ── BEFORE metrics (score head is fresh/random) ──────────────────────────
    before = {qt: _type_loss(model, val_samples, qt, tokenizer, class_to_idx, device)
              for qt in ("choice", "noul", "score", "extract")}
    logger.info("Before migration: %s", json.dumps({k: round(v, 4) for k, v in before.items()}))
    summary["before"] = before

    # ── Train: score-boosted mix, low LR, few epochs ─────────────────────────
    rng = random.Random(7)
    mix = score_train * 3 + rng.sample(other_train, min(len(other_train), 1000))
    rng.shuffle(mix)
    dataset = LeethTorchDataset(mix, tokenizer, max_length=MAX_LENGTH, class_to_idx=class_to_idx)
    # batch_size 4 (like finetune): batch 16 x seq 320 attention matrices for
    # backward need several GB RAM and got the process OOM-killed silently.
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_leeth_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, foreach=False)

    epochs = 2
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss, nb = 0.0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            for k in ["choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"]:
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss, _ = compute_calibrated_loss(outputs, batch)
            if torch.isnan(loss) or loss.item() <= 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
            if nb % 50 == 0:
                logger.info("Epoch %d/%d batch %d/%d | loss=%.4f", epoch, epochs, nb, len(loader), epoch_loss / max(1, nb))
                sys.stdout.flush()
                sys.stderr.flush()
        logger.info("Migration epoch %d/%d | loss=%.4f | batches=%d", epoch, epochs, epoch_loss / max(1, nb), nb)
        sys.stdout.flush()
        sys.stderr.flush()

    # ── AFTER + gate ─────────────────────────────────────────────────────────
    after = {qt: _type_loss(model, val_samples, qt, tokenizer, class_to_idx, device)
             for qt in ("choice", "noul", "score", "extract")}
    logger.info("After migration: %s", json.dumps({k: round(v, 4) for k, v in after.items()}))
    summary["after"] = after

    gate_ok = True
    for qt in ("choice", "noul", "extract"):
        b, a = before[qt], after[qt]
        if b == b and a == a:  # skip NaN
            if a > b * 1.05:
                gate_ok = False
                logger.warning("GATE FAIL: %s val loss %.4f -> %.4f (> +5%%)", qt, b, a)
    score_dropped = (before["score"] == before["score"]) and (after["score"] < before["score"] * 0.9)
    summary["gate_ok"] = gate_ok
    summary["score_improved"] = bool(score_dropped)
    accepted = gate_ok and score_dropped
    summary["accepted"] = accepted

    version = _next_version()
    payload = {
        "config": config,
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "val_loss": after["score"],
        "brier_score": checkpoint.get("brier_score"),
        "finetune_meta": {
            "ts": time.time(),
            "migration": "ordinal_score_head",
            "epochs": epochs,
            "mix_size": len(mix),
            "before": before,
            "after": after,
        },
    }

    if accepted:
        versioned = DATA_DIR / f"leeth_best_v{version}.pt"
        torch.save(payload, versioned)
        tmp = BEST_PATH.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, BEST_PATH)
        summary["checkpoint_version"] = str(versioned)
        logger.info("ACCEPTED: promoted to %s", BEST_PATH)
    else:
        candidate = DATA_DIR / f"leeth_candidate_v{version}.pt"
        torch.save(payload, candidate)
        summary["checkpoint_version"] = str(candidate)
        logger.info("REJECTED: candidate kept at %s", candidate)
        # Restore the original checkpoint state for the runtime (old ONNX still matches it).

    # ── ONNX re-export (atomic) + runtime reload ─────────────────────────────
    # On REJECT the on-disk leeth_best.pt is unchanged and the existing ONNX
    # already matches it — re-exporting here would export the STALE checkpoint.
    if accepted:
        try:
            from src.leeth.export import export_to_onnx, quantize_onnx_model

            def _atomic_replace(src: Path, dst: Path, attempts: int = 5) -> None:
                for attempt in range(attempts):
                    try:
                        os.replace(src, dst)
                        return
                    except OSError:
                        if attempt == attempts - 1:
                            raise
                        time.sleep(1.0)

            stamp = int(time.time())
            tmp_onnx = export_to_onnx(
                str(DATA_DIR), weights_filename="leeth_best.pt",
                output_filename=f"leeth.onnx.tmp{stamp}",
            )
            _atomic_replace(tmp_onnx, DATA_DIR / "leeth.onnx")
            tmp_int8 = quantize_onnx_model(DATA_DIR / "leeth.onnx", output_filename=f"leeth_int8.onnx.tmp{stamp}")
            _atomic_replace(tmp_int8, DATA_DIR / "leeth_int8.onnx")
            summary["onnx_reexported"] = True
        except Exception as export_err:
            summary["onnx_reexported"] = False
            summary["onnx_error"] = str(export_err)
            logger.error("ONNX re-export failed: %s", export_err)

    try:
        from src.leeth.runtime import leeth
        leeth.reload()
        summary["runtime_reloaded"] = True
    except Exception as reload_err:
        summary["runtime_reloaded"] = False
        summary["reload_error"] = str(reload_err)
        logger.error("Runtime reload failed: %s", reload_err)

    summary["elapsed_s"] = round(time.time() - started, 1)
    out = ROOT / ".argent" / "artifacts" / "leeth_migration_score.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Summary written to %s", out)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()