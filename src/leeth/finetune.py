"""
src/leeth/finetune.py — Active-Learning Fine-Tuning for Leeth from Feedback.

Consumes labeled decisions from src/leeth/feedback.py (verdicts recorded by
the agent/human reviewer), converts them to LeethSample objects, mixes them
with a replay buffer of synthetic samples (anti-forgetting), fine-tunes the
current leeth_best.pt checkpoint with a small learning rate, and promotes the
new weights ONLY if validation improves:

Gate (both must hold):
  1. feedback_val_loss_new  <= feedback_val_loss_before
  2. synthetic_val_loss_new <= synthetic_val_loss_before * 1.10

Every accepted run produces a versioned checkpoint (leeth_best_vN.pt), updates
leeth_best.pt atomically, re-exports ONNX + INT8, and reloads the runtime.
Rejected candidates are kept as leeth_candidate_vN.pt for inspection.

Run:  python -m src.leeth.finetune
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.leeth import feedback
from src.leeth.dataset_generator import LeethSample
from src.leeth.model import LeethConfig, LeethModel, config_from_checkpoint
from src.leeth.tokenizer import LeethTokenizer
from src.leeth.train import LeethTorchDataset, collate_leeth_batch, compute_calibrated_loss
from logger import get_logger

logger = get_logger("leeth_finetune")

DATA_DIR = Path("src/leeth/data")
BEST_PATH = DATA_DIR / "leeth_best.pt"
CLASSES_PATH = DATA_DIR / "choice_classes.json"
SYNTH_VAL_PATH = DATA_DIR / "leeth_val.jsonl"
SYNTH_TRAIN_PATH = DATA_DIR / "leeth_train.jsonl"


def _load_jsonl_samples(path: Path, limit: Optional[int] = None) -> List[LeethSample]:
    samples: List[LeethSample] = []
    if not path.exists():
        return samples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(LeethSample(**json.loads(line)))
            except Exception:
                continue
            if limit and len(samples) >= limit:
                break
    return samples


def _eval_loss(model: LeethModel, samples: List[LeethSample], tokenizer, class_to_idx: Dict[str, int],
               device: str, max_length: int = 320, batch_size: int = 8) -> float:
    if not samples:
        return float("nan")
    dataset = LeethTorchDataset(samples, tokenizer, max_length=max_length, class_to_idx=class_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_leeth_batch)
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            for k in ("choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"):
                batch[k] = batch[k].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss, _ = compute_calibrated_loss(outputs, batch)
            total += loss.item()
            batches += 1
    return total / max(1, batches)


def _eval_calibration(model: LeethModel, samples: List[LeethSample], tokenizer, class_to_idx: Dict[str, int],
                      device: str, max_length: int = 320, batch_size: int = 8) -> Tuple[float, float, int]:
    """ECE/Brier for the choice head on labeled val samples (top-prob vs correctness).

    Calibration is tracked alongside val loss so a fine-tune that improves loss
    while degrading calibration is visible in the gate report.
    """
    choice_samples = [s for s in samples if s.question_type == "choice" and s.target_choice]
    if not choice_samples:
        return float("nan"), float("nan"), 0
    dataset = LeethTorchDataset(choice_samples, tokenizer, max_length=max_length, class_to_idx=class_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_leeth_batch)
    model.eval()
    confs: List[float] = []
    corr: List[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            for k in ("choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"):
                batch[k] = batch[k].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["choice_logits"], dim=-1)
            top_probs, top_idx = probs.max(dim=-1)
            targets = batch["choice_target"].tolist()
            for tp, ti, tg in zip(top_probs.tolist(), top_idx.tolist(), targets):
                confs.append(float(tp))
                corr.append(1.0 if int(ti) == int(tg) else 0.0)
    if not confs:
        return float("nan"), float("nan"), 0
    n_bins = 10
    bins: List[List[tuple]] = [[] for _ in range(n_bins)]
    for c, o in zip(confs, corr):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, o))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_c = sum(c for c, _ in b) / len(b)
        acc = sum(o for _, o in b) / len(b)
        ece += len(b) / len(confs) * abs(acc - avg_c)
    brier = sum((c - o) ** 2 for c, o in zip(confs, corr)) / len(confs)
    return ece, brier, len(confs)


def _weight_fingerprint(model: LeethModel) -> float:
    """Cheap proof-of-change: sum of the choice head weights."""
    for name, param in model.named_parameters():
        if "choice_head" in name:
            return float(param.detach().float().abs().sum().item())
    return -1.0


def _next_version(data_path: Optional[Path] = None) -> int:
    root = data_path if data_path is not None else DATA_DIR
    existing = [p for p in root.glob("leeth_best_v*.pt")]
    return len(existing) + 1


def finetune_from_feedback(
    epochs: int = 10,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    replay_ratio: float = 0.3,
    replay_max: int = 2000,
    min_samples: int = 20,
    max_length: int = 320,
    correction_boost: int = 3,
    device: Optional[str] = None,
    extra_replay: Optional[List[LeethSample]] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Fine-tunes Leeth on labeled feedback + synthetic replay, with a val gate.

    extra_replay: optional curriculum injection (e.g. a balanced new-skill
    subset) appended to the random replay buffer.
    data_dir: which model directory to fine-tune (checkpoint, tokenizer,
    classes, replay corpus, ONNX export). Defaults to the legacy
    src/leeth/data; PRODUCTION fine-tunes must pass src/leeth/data_400m —
    otherwise the wrong (stale) checkpoint is trained and re-exported.
    """
    started = time.time()
    data_path = Path(data_dir) if data_dir else DATA_DIR
    best_path = data_path / "leeth_best.pt"
    classes_path = data_path / "choice_classes.json"
    synth_train_path = data_path / "leeth_train.jsonl"
    if not synth_train_path.exists():
        synth_train_path = data_path / "leeth_400m_train.jsonl"
    synth_val_path = data_path / "leeth_val.jsonl"
    if not synth_val_path.exists():
        synth_val_path = data_path / "leeth_400m_val.jsonl"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_fb, val_fb, fb_info = feedback.build_training_samples(val_ratio=0.2, correction_boost=correction_boost)
    summary: Dict[str, Any] = {"feedback_info": fb_info, "device": device}
    if len(train_fb) < min_samples:
        summary["status"] = "not_enough_samples"
        summary["message"] = (
            f"Only {len(train_fb)} feedback training samples (need >= {min_samples}). "
            f"Labeled records: {fb_info.get('labeled_records', 0)}. Keep using Argent with review mode ON."
        )
        logger.info(summary["message"])
        return summary

    # Replay buffer against catastrophic forgetting.
    replay: List[LeethSample] = []
    if replay_ratio > 0 and synth_train_path.exists():
        rng = random.Random(1234)
        pool = _load_jsonl_samples(synth_train_path)
        k = min(len(pool), int(len(train_fb) * replay_ratio) + 1, replay_max)
        replay = rng.sample(pool, k) if pool else []
    if extra_replay:
        replay = replay + list(extra_replay)
    logger.info("Feedback samples: train=%d val=%d | replay=%d (extra=%d)",
                len(train_fb), len(val_fb), len(replay), len(extra_replay or []))

    tokenizer = LeethTokenizer.load(data_path)
    with open(classes_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    config = config_from_checkpoint(checkpoint["config"])
    model = LeethModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    synth_val = _load_jsonl_samples(synth_val_path, limit=400)
    fingerprint_before = _weight_fingerprint(model)

    val_fb_before = _eval_loss(model, val_fb, tokenizer, class_to_idx, device, max_length) if val_fb else float("nan")
    synth_before = _eval_loss(model, synth_val, tokenizer, class_to_idx, device, max_length)
    ece_before, brier_before, n_choice_val = (
        _eval_calibration(model, val_fb, tokenizer, class_to_idx, device, max_length) if val_fb else (float("nan"), float("nan"), 0)
    )
    logger.info("Before fine-tune: feedback_val=%.4f synth_val=%.4f choice_ece=%.4f choice_brier=%.4f",
                val_fb_before, synth_before, ece_before, brier_before)

    train_ds = LeethTorchDataset(train_fb + replay, tokenizer, max_length=max_length, class_to_idx=class_to_idx)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_leeth_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    model.train()
    step_losses: List[float] = []
    for epoch in range(1, epochs + 1):
        epoch_loss, num_batches = 0.0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            for k in ("choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"):
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss, _ = compute_calibrated_loss(outputs, batch)
            if torch.isnan(loss) or loss.item() <= 0:
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            num_batches += 1
        logger.info("Fine-tune epoch %d/%d | loss=%.4f | batches=%d", epoch, epochs, epoch_loss / max(1, num_batches), num_batches)
        step_losses.append(epoch_loss / max(1, num_batches))

    val_fb_after = _eval_loss(model, val_fb, tokenizer, class_to_idx, device, max_length) if val_fb else float("nan")
    synth_after = _eval_loss(model, synth_val, tokenizer, class_to_idx, device, max_length)
    ece_after, brier_after, _ = (
        _eval_calibration(model, val_fb, tokenizer, class_to_idx, device, max_length) if val_fb else (float("nan"), float("nan"), 0)
    )
    fingerprint_after = _weight_fingerprint(model)
    logger.info("After fine-tune: feedback_val=%.4f synth_val=%.4f choice_ece=%.4f choice_brier=%.4f",
                val_fb_after, synth_after, ece_after, brier_after)

    gate_feedback = (val_fb_after <= val_fb_before) if val_fb else True
    gate_synth = synth_after <= synth_before * 1.10
    accepted = bool(gate_feedback and gate_synth)
    version = _next_version(data_path)

    save_payload = {
        "config": config,
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "val_loss": synth_after,
        "brier_score": checkpoint.get("brier_score"),
        "finetune_meta": {
            "ts": time.time(),
            "feedback_train": len(train_fb),
            "feedback_val": len(val_fb),
            "replay": len(replay),
            "epochs": epochs,
            "lr": learning_rate,
            "feedback_val_before": val_fb_before,
            "feedback_val_after": val_fb_after,
            "synth_val_before": synth_before,
            "synth_val_after": synth_after,
            "choice_ece_before": ece_before,
            "choice_ece_after": ece_after,
            "choice_brier_before": brier_before,
            "choice_brier_after": brier_after,
            "n_choice_val": n_choice_val,
            "weight_fingerprint_before": fingerprint_before,
            "weight_fingerprint_after": fingerprint_after,
        },
    }

    if accepted:
        versioned = data_path / f"leeth_best_v{version}.pt"
        torch.save(save_payload, versioned)
        tmp = best_path.with_suffix(".pt.tmp")
        torch.save(save_payload, tmp)
        os.replace(tmp, best_path)
        summary["checkpoint_version"] = str(versioned)
        logger.info("ACCEPTED: promoted to %s (versioned copy saved).", best_path)
    else:
        candidate = data_path / f"leeth_candidate_v{version}.pt"
        torch.save(save_payload, candidate)
        summary["checkpoint_version"] = str(candidate)
        logger.info("REJECTED by val gate (feedback gate=%s, synth gate=%s); candidate kept at %s", gate_feedback, gate_synth, candidate)

    # Re-export ONNX so the INT8 runtime serves the new weights.
    try:
        from src.leeth.export import export_to_onnx, quantize_onnx_model

        def _atomic_replace(src: Path, dst: Path, attempts: int = 5) -> None:
            # Windows: the target may be briefly locked (AV scan / another
            # process mapping it) — retry a few times before giving up.
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
            str(data_path), weights_filename="leeth_best.pt",
            output_filename=f"leeth.onnx.tmp{stamp}",
        )
        _atomic_replace(tmp_onnx, data_path / "leeth.onnx")
        tmp_int8 = quantize_onnx_model(data_path / "leeth.onnx", output_filename=f"leeth_int8.onnx.tmp{stamp}")
        _atomic_replace(tmp_int8, data_path / "leeth_int8.onnx")
        summary["onnx_reexported"] = True
    except Exception as export_err:
        summary["onnx_reexported"] = False
        summary["onnx_error"] = str(export_err)
        stale = data_path / f"leeth_int8.onnx.stale-{int(time.time())}"
        try:
            (data_path / "leeth_int8.onnx").rename(stale)
            logger.warning("Stale INT8 ONNX moved to %s; runtime will use fresh PyTorch weights.", stale)
        except Exception:
            pass

    # Hot-reload the runtime with the new weights.
    try:
        from src.leeth.runtime import leeth
        leeth.reload()
        summary["runtime_reloaded"] = True
    except Exception as reload_err:
        summary["runtime_reloaded"] = False
        summary["reload_error"] = str(reload_err)

    summary.update({
        "status": "accepted" if accepted else "rejected",
        "feedback_val_before": val_fb_before,
        "feedback_val_after": val_fb_after,
        "synth_val_before": synth_before,
        "synth_val_after": synth_after,
        "choice_ece_before": ece_before,
        "choice_ece_after": ece_after,
        "choice_brier_before": brier_before,
        "choice_brier_after": brier_after,
        "weights_changed": abs(fingerprint_after - fingerprint_before) > 1e-6,
        "epoch_losses": step_losses,
        "elapsed_s": round(time.time() - started, 1),
    })
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = finetune_from_feedback()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))