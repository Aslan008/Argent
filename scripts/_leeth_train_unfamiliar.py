"""
scripts/_leeth_train_unfamiliar.py — Phase 3: teach the choice head an explicit
UNFAMILIAR class (delegate-to-LLM) without touching the backbone.

Loads leeth_best.pt (v6 graft), regenerates the synthetic datasets with the new
UNFAMILIAR family, freezes everything except choice_head + noul_head, trains on
a choice/noul mix (unfamiliar + routing replay + conversational negatives),
gates on in-dist routing val CE (must not degrade) and UNFAMILIAR val accuracy,
promotes versioned + atomic best, re-exports ONNX + INT8, verifies.

Run:  python scripts/_leeth_train_unfamiliar.py   (background; GPU if free)
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
from src.leeth.finetune import _next_version  # noqa: E402
from src.leeth.model import LeethModel  # noqa: E402
from src.leeth.tokenizer import LeethTokenizer  # noqa: E402
from src.leeth.train import LeethTorchDataset, collate_leeth_batch, compute_calibrated_loss  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("leeth_unfamiliar")

DATA_DIR = Path("src/leeth/data")
BEST_PATH = DATA_DIR / "leeth_best.pt"
CLASSES_PATH = DATA_DIR / "choice_classes.json"
SYNTH_TRAIN_PATH = DATA_DIR / "leeth_train.jsonl"
SYNTH_VAL_PATH = DATA_DIR / "leeth_val.jsonl"

MAX_LENGTH = 320
ROUTING_QUESTION = "Какой инструмент Argent должен быть вызван в первую очередь?"


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


def _to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


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
            batch = _to_device(batch, device)
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            loss, _ = compute_calibrated_loss(outputs, batch)
            total += loss.item()
            batches += 1
    return total / max(1, batches)


def _choice_accuracy(model, samples, tokenizer, class_to_idx, device, only_unfamiliar=None):
    """Accuracy of the choice head on choice samples; optionally filter by family."""
    subset = []
    for s in samples:
        if s.question_type != "choice" or not s.target_choice:
            continue
        is_unfam = s.target_choice == "UNFAMILIAR"
        if only_unfamiliar is True and not is_unfam:
            continue
        if only_unfamiliar is False and is_unfam:
            continue
        subset.append(s)
    if not subset:
        return float("nan"), 0
    dataset = LeethTorchDataset(subset, tokenizer, max_length=MAX_LENGTH, class_to_idx=class_to_idx)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_leeth_batch)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            preds = torch.argmax(outputs["choice_logits"], dim=-1).cpu()
            mask = batch["choice_target"].cpu() != -100
            correct += (preds[mask] == batch["choice_target"].cpu()[mask]).sum().item()
            total += int(mask.sum().item())
    return correct / max(1, total), total


def main():
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"device": device, "mode": "train_unfamiliar_class"}

    tokenizer = LeethTokenizer.load(DATA_DIR)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    assert "UNFAMILIAR" in class_to_idx, "UNFAMILIAR missing from choice_classes.json"
    summary["unfamiliar_idx"] = class_to_idx["UNFAMILIAR"]
    summary["num_classes"] = len(class_to_idx)

    checkpoint = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = LeethModel(config)
    model.load_state_dict(checkpoint["state_dict"])  # strict: v6 is consistent
    model.to(device)

    # ── Freeze everything except choice_head + noul_head ─────────────────────
    trainable = []
    for name, p in model.named_parameters():
        if name.startswith("choice_head") or name.startswith("noul_head") or name == "temperature":
            p.requires_grad = True
            trainable.append(name)
        else:
            p.requires_grad = False
    summary["trainable_modules"] = sorted({n.split(".")[0] for n in trainable})
    logger.info("Trainable: %s (backbone + score/extract heads frozen)", summary["trainable_modules"])

    # ── Regenerate datasets with the UNFAMILIAR family ───────────────────────
    train_count = _count_lines(SYNTH_TRAIN_PATH)
    val_count = _count_lines(SYNTH_VAL_PATH)
    logger.info("Regenerating datasets: train=%d val=%d (same seeds/sizes)", train_count, val_count)
    LeethDatasetGenerator(seed=42).generate_and_save(SYNTH_TRAIN_PATH, target_count=train_count)
    LeethDatasetGenerator(seed=9999).generate_and_save(SYNTH_VAL_PATH, target_count=val_count)
    train_samples = _load_jsonl(SYNTH_TRAIN_PATH)
    val_samples = _load_jsonl(SYNTH_VAL_PATH)
    summary["train_total"] = len(train_samples)
    summary["train_unfamiliar"] = sum(1 for s in train_samples if s.target_choice == "UNFAMILIAR")
    logger.info("Train: %d samples, %d UNFAMILIAR", len(train_samples), summary["train_unfamiliar"])

    # ── BEFORE metrics ───────────────────────────────────────────────────────
    routing_val = [s for s in val_samples
                   if s.question_type == "choice" and s.target_choice and s.target_choice != "UNFAMILIAR"]
    before_routing_ce = _type_loss(model, routing_val, "choice", tokenizer, class_to_idx, device)
    before_noul_ce = _type_loss(model, val_samples, "noul", tokenizer, class_to_idx, device)
    before_unfam_acc, _ = _choice_accuracy(model, val_samples, tokenizer, class_to_idx, device, only_unfamiliar=True)
    before = {"routing_ce": before_routing_ce, "noul_ce": before_noul_ce, "unfamiliar_acc": before_unfam_acc}
    summary["before"] = before
    logger.info("Before: %s", json.dumps({k: round(v, 4) for k, v in before.items()}))

    # ── Train on choice + noul samples (score/extract heads frozen -> excluded) ──
    # Oversample the UNFAMILIAR family x3: it is a small fraction of the mix
    # and the fresh class needs more gradient mass than established ones.
    unfam_states = {s.state for s in train_samples
                    if s.question_type == "choice" and s.target_choice == "UNFAMILIAR"}
    base_mix = [s for s in train_samples if s.question_type in ("choice", "noul")]
    unfamiliar_mix = [s for s in base_mix if s.state in unfam_states]
    mix = base_mix + unfamiliar_mix * 2
    random.Random(11).shuffle(mix)
    logger.info("Mix: %d samples (%d unfamiliar x3)", len(mix), len(unfamiliar_mix))
    dataset = LeethTorchDataset(mix, tokenizer, max_length=MAX_LENGTH, class_to_idx=class_to_idx)
    # Backbone is frozen -> no activation storage -> batch 16 is safe.
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_leeth_batch)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01, foreach=False
    )

    epochs = 3
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
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
            if nb % 50 == 0:
                logger.info("Epoch %d/%d batch %d/%d | loss=%.4f", epoch, epochs, nb, len(loader), epoch_loss / max(1, nb))
                sys.stdout.flush()
                sys.stderr.flush()
        logger.info("Unfamiliar epoch %d/%d | loss=%.4f | batches=%d", epoch, epochs, epoch_loss / max(1, nb), nb)
        sys.stdout.flush()
        sys.stderr.flush()

    # ── AFTER + gate ─────────────────────────────────────────────────────────
    after_routing_ce = _type_loss(model, routing_val, "choice", tokenizer, class_to_idx, device)
    after_noul_ce = _type_loss(model, val_samples, "noul", tokenizer, class_to_idx, device)
    after_unfam_acc, unfam_n = _choice_accuracy(model, val_samples, tokenizer, class_to_idx, device, only_unfamiliar=True)
    after = {"routing_ce": after_routing_ce, "noul_ce": after_noul_ce, "unfam_acc": after_unfam_acc, "unfam_n": unfam_n}
    logger.info("After: %s", json.dumps({k: round(v, 4) for k, v in after.items() if isinstance(v, float)}))
    summary["after"] = after

    gate_ok = True
    if after_routing_ce > before_routing_ce * 1.05:
        gate_ok = False
        logger.warning("GATE FAIL: routing CE %.4f -> %.4f (> +5%%)", before_routing_ce, after_routing_ce)
    if after_noul_ce > before_noul_ce * 1.05:
        gate_ok = False
        logger.warning("GATE FAIL: noul CE %.4f -> %.4f (> +5%%)", before_noul_ce, after_noul_ce)
    if not (after_unfam_acc == after_unfam_acc) or after_unfam_acc < 0.85:
        gate_ok = False
        logger.warning("GATE FAIL: UNFAMILIAR accuracy %.3f < 0.85", after_unfam_acc)
    accepted = gate_ok
    summary["gate_ok"] = gate_ok
    summary["accepted"] = accepted
    logger.info("Gate: ok=%s", gate_ok)

    payload = {
        "config": config,
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "val_loss": after_routing_ce,
        "brier_score": checkpoint.get("brier_score"),
        "finetune_meta": {
            "ts": time.time(),
            "migration": "unfamiliar_class",
            "epochs": epochs,
            "mix_size": len(mix),
            "before": before,
            "after": after,
        },
    }

    version = _next_version()
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

    # ── ONNX re-export (atomic) — only when accepted ─────────────────────────
    if accepted:
        try:
            from src.leeth.export import export_to_onnx, quantize_onnx_model, verify_exported_model

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
            metrics = verify_exported_model(str(DATA_DIR))
            summary["verify"] = metrics
            logger.info("Verify: %s", json.dumps(metrics))
        except Exception as export_err:
            summary["onnx_reexported"] = False
            summary["onnx_error"] = str(export_err)
            logger.error("ONNX re-export failed: %s", export_err)

    summary["elapsed_s"] = round(time.time() - started, 1)
    out = ROOT / ".argent" / "artifacts" / "leeth_train_unfamiliar.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Summary written to %s", out)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()