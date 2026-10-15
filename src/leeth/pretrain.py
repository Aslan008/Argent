"""
src/leeth/pretrain.py — Self-Supervised Masked Language Modeling (MLM) Pretraining for Leeth.

Pretrains the bidirectional Leeth Transformer backbone on real codebases:
- Ingests Argent codebase (.py) and polyglot snippets (C#, C++, Rust, TypeScript, Shell).
- Implements dynamic 15% token masking ([MASK], random, unchanged).
- Optimizes Cross-Entropy over masked tokens using PyTorch AMP on NVIDIA RTX 4060 GPU.
- Saves pretrained backbone checkpoint to initialize decision heads.
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.leeth.model import LeethConfig, LeethModel
from src.leeth.tokenizer import LeethTokenizer
from logger import get_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = get_logger("leeth_pretrain")

class LeethMLMDataset(Dataset):
    """Dataset for dynamic Masked Language Modeling."""

    def __init__(self, tokenized_sequences: List[List[int]], tokenizer: LeethTokenizer, mlm_prob: float = 0.15):
        self.sequences = tokenized_sequences
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.mask_token_id = tokenizer.vocab.get("[MASK]", 4)
        self.vocab_size = tokenizer.vocab_size

        # Special tokens that must never be masked
        self.special_ids = {
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            self.mask_token_id,
        }

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = list(self.sequences[idx])
        labels = [-100] * len(tokens)

        for i in range(len(tokens)):
            tid = tokens[i]
            if tid in self.special_ids:
                continue

            if random.random() < self.mlm_prob:
                labels[i] = tid
                prob = random.random()
                if prob < 0.8:
                    tokens[i] = self.mask_token_id
                elif prob < 0.9:
                    tokens[i] = random.randint(len(self.tokenizer.special_tokens), self.vocab_size - 1)
                # Remaining 10% keeps original token

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, attention_mask, labels


def collate_mlm_batch(batch, pad_token_id: int):
    max_len = max(item[0].size(0) for item in batch)
    padded_inputs = []
    padded_masks = []
    padded_labels = []

    for input_ids, attention_mask, labels in batch:
        pad_size = max_len - input_ids.size(0)
        padded_inputs.append(F.pad(input_ids, (0, pad_size), value=pad_token_id))
        padded_masks.append(F.pad(attention_mask, (0, pad_size), value=0))
        padded_labels.append(F.pad(labels, (0, pad_size), value=-100))

    return (
        torch.stack(padded_inputs),
        torch.stack(padded_masks),
        torch.stack(padded_labels),
    )


def collect_code_corpus(root_dir: str = ".") -> List[str]:
    """Collects real code lines and snippets from the project and reference sources."""
    corpus = []
    root_path = Path(root_dir)

    # Ingest key Argent Python sources
    extensions = {".py", ".cs", ".cpp", ".hpp", ".rs", ".ts", ".js", ".json"}
    ignore_dirs = {".git", ".system_generated", "brain", "__pycache__", "venv",
                   "venv_pyqt6", "node_modules", ".argent", ".pytest_cache", ".hypothesis"}

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in extensions:
                fpath = Path(dirpath) / fname
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                    # Break into coherent paragraphs/chunks of up to 40 lines
                    lines = text.splitlines()
                    for i in range(0, len(lines), 20):
                        chunk = "\n".join(lines[i : i + 20]).strip()
                        if len(chunk) > 40:
                            corpus.append(chunk)
                except Exception:
                    pass

    logger.info("Collected %d code chunks from workspace for MLM pretraining.", len(corpus))
    return corpus


def pretrain_leeth_mlm(
    data_dir: str = "src/leeth/data",
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    max_seq_len: int = 128,
    device: Optional[str] = None,
) -> Path:
    """Runs Self-Supervised Masked Language Modeling pretraining."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting MLM pretraining on device: %s", device)

    data_path = Path(data_dir)
    tokenizer = LeethTokenizer.load(data_path)

    # 1. Collect and tokenize corpus
    corpus = collect_code_corpus()
    if not corpus:
        corpus = ["using System; public class Test { void Run() {} }"] * 100

    tokenized_seqs = []
    for text in corpus:
        ids = tokenizer.encode(text, max_length=max_seq_len, truncation=True, add_special_tokens=True)
        if len(ids) > 6:
            tokenized_seqs.append(ids)

    logger.info("Pretraining with %d tokenized sequences (seq_len <= %d).", len(tokenized_seqs), max_seq_len)

    dataset = LeethMLMDataset(tokenized_seqs, tokenizer, mlm_prob=0.15)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_mlm_batch(b, tokenizer.pad_token_id),
    )

    # 2. Model Architecture
    config = LeethConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        intermediate_size=1536,
        max_position_embeddings=1024,
        num_choices=32,
        dropout=0.1,
    )
    model = LeethModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # 3. Pretraining Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.perf_counter()

        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model.forward_mlm(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1), ignore_index=-100)

            if not torch.isnan(loss) and loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        elapsed = time.perf_counter() - t0
        logger.info("MLM Pretrain Epoch %d/%d (%.1fs) | Loss: %.4f", epoch, epochs, elapsed, avg_loss)

    out_file = data_path / "leeth_pretrained.pt"
    torch.save({"config": config, "state_dict": model.state_dict()}, out_file)
    logger.info("Saved pretrained backbone checkpoint to %s", out_file)
    return out_file


if __name__ == "__main__":
    pretrain_leeth_mlm(epochs=3, batch_size=64)
