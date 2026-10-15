"""
src/leeth/train.py — Multi-Task Calibrated Training Engine for Leeth.

Implements:
- Multi-task loss combining Choice (CE), Noul (BCE + Brier Score Calibration), Score (SmoothL1), and Extract (Span CE).
- Hardware acceleration via CUDA / AMP FP16 on NVIDIA RTX 4060 GPU.
- Automatic span target alignment from substring characters to BPE token positions.
- Validation checkpointing and evaluation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.leeth.dataset_generator import LeethDatasetGenerator, LeethSample
from src.leeth.model import LeethConfig, LeethModel
from src.leeth.tokenizer import LeethTokenizer, SPACE_CHAR
from logger import get_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = get_logger("leeth_train")

class LeethTorchDataset(Dataset):
    """Encodes LeethSample items into tensor batches with tokenized span alignment."""

    def __init__(
        self,
        samples: List[LeethSample],
        tokenizer: LeethTokenizer,
        max_length: int = 512,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_to_idx = class_to_idx or {}
        self.encoded_items = [self._process_sample(s) for s in samples]

    def __len__(self) -> int:
        return len(self.encoded_items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encoded_items[idx]

    def _process_sample(self, sample: LeethSample) -> Dict[str, torch.Tensor]:
        # Construct structured input: [CLS] [STATE] <state> [SEP] [QUESTION] [<TYPE>] <question> [SEP]
        type_token = f"[{sample.question_type.upper()}]"
        text = f"[STATE] {sample.state} [SEP] [QUESTION] {type_token} {sample.question}"

        input_ids = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        attention_mask = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in input_ids]

        # Targets initialization (defaults to ignore index / dummy values)
        choice_target = -100
        noul_target = -1.0
        score_target = -1.0
        score_levels = 0
        start_target = 0
        end_target = 0

        q_type = sample.question_type
        if q_type == "choice" and sample.target_choice:
            choice_target = self.class_to_idx.get(sample.target_choice, 0)
        elif q_type == "noul" and sample.target_noul is not None:
            noul_target = float(sample.target_noul)
        elif q_type == "score" and sample.target_score is not None:
            score_target = float(sample.target_score)
            score_levels = int(sample.score_levels or 5)
        elif q_type == "extract" and sample.target_span_text:
            target_str = sample.target_span_text.strip()
            # 1. Direct sublist match (checking both with and without leading whitespace variant)
            span_tokens = self.tokenizer.tokenize(target_str)
            matched = False
            if span_tokens:
                first_tok = span_tokens[0]
                alt_first = first_tok[1:] if first_tok.startswith(SPACE_CHAR) else (SPACE_CHAR + first_tok)
                candidate_lists = [
                    [self.tokenizer.vocab.get(t, self.tokenizer.unk_token_id) for t in span_tokens],
                    [self.tokenizer.vocab.get(alt_first, self.tokenizer.unk_token_id)] + [self.tokenizer.vocab.get(t, self.tokenizer.unk_token_id) for t in span_tokens[1:]]
                ]
                for span_ids in candidate_lists:
                    if not matched and span_ids:
                        for i in range(len(input_ids) - len(span_ids) + 1):
                            if input_ids[i : i + len(span_ids)] == span_ids:
                                start_target = i
                                end_target = i + len(span_ids) - 1
                                matched = True
                                break

            # 2. Fallback: character-to-token offset alignment
            if not matched and target_str and target_str in text:
                # The token stream decodes to "[CLS]" + text (+ trailing [SEP]):
                # account for the [CLS] prefix when mapping char offsets to tokens.
                cls_offset = len(self.tokenizer.cls_token) if self.tokenizer.cls_token else 0
                char_start = text.find(target_str) + cls_offset
                char_end = char_start + len(target_str)
                cur_pos = 0
                s_tok = None
                e_tok = None
                for idx, tid in enumerate(input_ids):
                    raw_tok = self.tokenizer.inv_vocab.get(tid, "")
                    if raw_tok in self.tokenizer.special_tokens:
                        tok_str = raw_tok
                    else:
                        tok_str = raw_tok.replace(getattr(self.tokenizer, "SPACE_CHAR", SPACE_CHAR), " ")
                    tok_len = len(tok_str)
                    next_pos = cur_pos + tok_len
                    if s_tok is None and next_pos > char_start:
                        s_tok = idx
                    if s_tok is not None and cur_pos < char_end:
                        e_tok = idx
                    cur_pos = next_pos
                if s_tok is not None and e_tok is not None and s_tok <= e_tok:
                    start_target = s_tok
                    end_target = e_tok

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "question_type": q_type,
            "choice_target": torch.tensor(choice_target, dtype=torch.long),
            "noul_target": torch.tensor(noul_target, dtype=torch.float),
            "score_target": torch.tensor(score_target, dtype=torch.float),
            "score_levels": torch.tensor(score_levels, dtype=torch.long),
            "start_target": torch.tensor(start_target, dtype=torch.long),
            "end_target": torch.tensor(end_target, dtype=torch.long),
        }


def collate_leeth_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "question_type": [item["question_type"] for item in batch],
        "choice_target": torch.stack([item["choice_target"] for item in batch]),
        "noul_target": torch.stack([item["noul_target"] for item in batch]),
        "score_target": torch.stack([item["score_target"] for item in batch]),
        "score_levels": torch.stack([item["score_levels"] for item in batch]),
        "start_target": torch.stack([item["start_target"] for item in batch]),
        "end_target": torch.stack([item["end_target"] for item in batch]),
    }


def compute_calibrated_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    brier_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates multi-task loss with Brier Score probability calibration and per-head metrics."""
    loss_metrics = {}
    total_loss = torch.tensor(0.0, device=outputs["choice_logits"].device)

    # 1. Choice Loss & Accuracy
    choice_mask = batch["choice_target"] != -100
    if choice_mask.any():
        loss_choice = F.cross_entropy(
            outputs["choice_logits"][choice_mask],
            batch["choice_target"][choice_mask],
            label_smoothing=0.05,
        )
        total_loss = total_loss + loss_choice
        loss_metrics["choice_loss"] = loss_choice.item()
        preds = torch.argmax(outputs["choice_logits"][choice_mask], dim=-1)
        loss_metrics["choice_correct"] = (preds == batch["choice_target"][choice_mask]).sum().item()
        loss_metrics["choice_count"] = choice_mask.sum().item()

    # 2. Noul Loss & Brier Score Calibration
    noul_mask = batch["noul_target"] >= 0.0
    if noul_mask.any():
        logits = outputs["noul_logits"][noul_mask]
        preds = outputs["noul_probs"][noul_mask]
        targets = batch["noul_target"][noul_mask]
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        brier_sum = torch.sum((preds - targets) ** 2)
        noul_count = noul_mask.sum().item()
        brier_loss = brier_sum / max(1, noul_count)

        calibrated_noul = bce_loss + (brier_weight * brier_loss)
        total_loss = total_loss + calibrated_noul
        loss_metrics["noul_loss"] = bce_loss.item()
        loss_metrics["brier_sum"] = brier_sum.item()
        loss_metrics["brier_score"] = brier_loss.item()
        loss_metrics["noul_count"] = noul_count

    # 3. Score Loss (ordinal cross-entropy over rubric levels) & MAE
    score_mask = batch["score_target"] >= 0.0
    if score_mask.any():
        levels = batch["score_levels"][score_mask]
        targets100 = batch["score_target"][score_mask]
        level_targets = torch.clamp(torch.ceil(targets100 / 100.0 * levels.to(targets100.dtype)), min=1, max=10).long() - 1
        logits = outputs["score_logits"][score_mask]
        arange = torch.arange(logits.size(-1), device=logits.device).unsqueeze(0)
        valid = arange < levels.unsqueeze(1)
        # dtype-safe fill: -1e9 overflows fp16 under autocast (Half max ~65504)
        masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        # NOTE: no label smoothing here — it would spread mass over the
        # masked out-of-range levels and produce a huge loss leak.
        loss_score = F.cross_entropy(masked_logits, level_targets)
        total_loss = total_loss + loss_score
        loss_metrics["score_loss"] = loss_score.item()
        with torch.no_grad():
            probs = torch.softmax(masked_logits, dim=-1)
            expected_level = (probs * torch.arange(1, logits.size(-1) + 1, device=logits.device, dtype=probs.dtype)).sum(-1)
            preds100 = expected_level / levels.to(probs.dtype) * 100.0
            mae_sum = torch.sum(torch.abs(preds100 - targets100))
        loss_metrics["score_mae_sum"] = mae_sum.item()
        loss_metrics["score_count"] = score_mask.sum().item()

    # 4. Extract Span Loss & Exact Match (EM)
    extract_mask = torch.tensor([qt == "extract" for qt in batch["question_type"]], device=total_loss.device)
    if extract_mask.any():
        loss_start = F.cross_entropy(
            outputs["start_logits"][extract_mask],
            batch["start_target"][extract_mask],
            label_smoothing=0.02,
        )
        loss_end = F.cross_entropy(
            outputs["end_logits"][extract_mask],
            batch["end_target"][extract_mask],
            label_smoothing=0.02,
        )
        span_loss = loss_start + loss_end
        total_loss = total_loss + span_loss
        loss_metrics["span_loss"] = span_loss.item()

        pred_start = torch.argmax(outputs["start_logits"][extract_mask], dim=-1)
        pred_end = torch.argmax(outputs["end_logits"][extract_mask], dim=-1)
        em = (pred_start == batch["start_target"][extract_mask]) & (pred_end == batch["end_target"][extract_mask])
        loss_metrics["span_em_sum"] = em.sum().item()
        loss_metrics["span_count"] = extract_mask.sum().item()

    loss_metrics["total_loss"] = total_loss.item()
    return total_loss, loss_metrics


def train_leeth(
    output_dir: str = "src/leeth/data",
    num_samples: int = 5000,
    val_samples: int = 500,
    batch_size: int = 16,
    epochs: int = 5,
    learning_rate: float = 3e-4,
    device: Optional[str] = None,
) -> Tuple[LeethModel, LeethTokenizer]:
    """Trains the full Leeth model from scratch and saves the trained weights."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training Leeth model on device: %s", device)

    # 1. Generate Separate High-Entropy Train and Val Datasets
    logger.info("Synthesizing %d training samples and %d validation samples...", num_samples, val_samples)
    train_generator = LeethDatasetGenerator(seed=42)
    train_path = out_path / "leeth_train.jsonl"
    train_generator.generate_and_save(train_path, target_count=num_samples)
    train_data = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            train_data.append(LeethSample(**json.loads(line)))

    val_generator = LeethDatasetGenerator(seed=9999)
    val_path = out_path / "leeth_val.jsonl"
    val_generator.generate_and_save(val_path, target_count=val_samples)
    val_data = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            val_data.append(LeethSample(**json.loads(line)))

    # Collect choice universe
    all_choices = set()
    for s in train_data + val_data:
        if s.target_choice:
            all_choices.add(s.target_choice)
        if s.choices:
            all_choices.update(s.choices)
    class_to_idx = {c: i for i, c in enumerate(sorted(all_choices))}
    logger.info("Total choice categories: %d", len(class_to_idx))

    # Save class mapping
    with open(out_path / "choice_classes.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    # 2. Train Tokenizer on Dataset Text
    corpus = [s.state + " " + s.question for s in train_data]
    logger.info("Training LeethTokenizer on %d corpus documents...", len(corpus))
    tokenizer = LeethTokenizer()
    tokenizer.train_from_corpus(corpus, target_vocab_size=4096, min_freq=1)
    tokenizer.save(out_path)
    logger.info("Tokenizer trained. Vocabulary size: %d", tokenizer.vocab_size)

    # 3. Datasets & Loaders
    train_ds = LeethTorchDataset(train_data, tokenizer, max_length=320, class_to_idx=class_to_idx)
    val_ds = LeethTorchDataset(val_data, tokenizer, max_length=320, class_to_idx=class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_leeth_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_leeth_batch)

    # 4. Initialize Leeth Model Architecture (Leeth-Medium ~23.6M params)
    config = LeethConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        intermediate_size=1536,
        max_position_embeddings=1024,
        num_choices=max(len(class_to_idx), 32),
        dropout=0.1,
    )
    model = LeethModel(config).to(device)

    # Load pretrained backbone weights if available
    pretrained_weights = out_path / "leeth_pretrained.pt"
    if pretrained_weights.exists():
        try:
            pt_ckpt = torch.load(pretrained_weights, map_location=device, weights_only=False)
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in pt_ckpt["state_dict"].items()
                if k in model_dict and model_dict[k].shape == v.shape and not k.startswith("choice_head") and not k.startswith("noul_head") and not k.startswith("score_head") and not k.startswith("extract_")
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info("Successfully transferred %d pretrained backbone tensors from %s", len(pretrained_dict), pretrained_weights)
        except Exception as e:
            logger.warning("Could not load pretrained weights, continuing from scratch: %s", e)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Initialized LeethModel: %.2fM parameters", total_params / 1e6)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val_loss = float("inf")
    best_weights_path = out_path / "leeth_best.pt"

    # 6. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        num_batches = 0
        t0 = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            for k in ["choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"]:
                batch[k] = batch[k].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss, metrics = compute_calibrated_loss(outputs, batch)

            if not torch.isnan(loss) and loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_epoch_loss += loss.item()
                num_batches += 1

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(train_loader):
                logger.info(
                    "Epoch %d/%d | Batch %d/%d (%.1f%%) | Loss: %.4f",
                    epoch,
                    epochs,
                    batch_idx + 1,
                    len(train_loader),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    total_epoch_loss / max(1, num_batches),
                )

        avg_train_loss = total_epoch_loss / max(1, num_batches)
        elapsed = time.perf_counter() - t0

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_choice_corr = 0
        val_choice_tot = 0
        val_brier_sum = 0.0
        val_noul_tot = 0
        val_score_mae = 0.0
        val_score_tot = 0
        val_span_em = 0
        val_span_tot = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                for k in ["choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"]:
                    batch[k] = batch[k].to(device)

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss, metrics = compute_calibrated_loss(outputs, batch)

                val_loss += loss.item()
                val_batches += 1
                val_choice_corr += metrics.get("choice_correct", 0)
                val_choice_tot += metrics.get("choice_count", 0)
                val_brier_sum += metrics.get("brier_sum", 0.0)
                val_noul_tot += metrics.get("noul_count", 0)
                val_score_mae += metrics.get("score_mae_sum", 0.0)
                val_score_tot += metrics.get("score_count", 0)
                val_span_em += metrics.get("span_em_sum", 0)
                val_span_tot += metrics.get("span_count", 0)

        avg_val_loss = val_loss / max(1, val_batches)
        choice_acc = (100.0 * val_choice_corr / max(1, val_choice_tot)) if val_choice_tot > 0 else 0.0
        avg_brier = (val_brier_sum / max(1, val_noul_tot)) if val_noul_tot > 0 else 0.0
        avg_score_mae = (val_score_mae / max(1, val_score_tot)) if val_score_tot > 0 else 0.0
        span_em_acc = (100.0 * val_span_em / max(1, val_span_tot)) if val_span_tot > 0 else 0.0

        logger.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.4f | Val Loss: %.4f | Choice Acc: %.1f%% | Noul Brier: %.4f | Score MAE: %.2f | Span EM: %.1f%%",
            epoch,
            epochs,
            elapsed,
            avg_train_loss,
            avg_val_loss,
            choice_acc,
            avg_brier,
            avg_score_mae,
            span_em_acc,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "config": config,
                    "state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "val_loss": avg_val_loss,
                    "brier_score": avg_brier,
                },
                best_weights_path,
            )
            logger.info("Saved new best model checkpoint to %s (Val Loss: %.4f)", best_weights_path, avg_val_loss)

    logger.info("Training complete. Best checkpoint saved to %s", best_weights_path)
    return model, tokenizer


if __name__ == "__main__":
    train_leeth(num_samples=6000, val_samples=600, epochs=3, batch_size=16, learning_rate=3e-4)
