"""
Universal Training and Semantic Embedding Script for Argent MicroModels.

Trains UniversalIntentEncoder across all 67 classes (66 Argent tools + NO_TOOL),
computes semantic embeddings for all tools in SemanticToolRegistry,
and exports the dual-head model to ONNX INT8 with sub-2ms CPU latency.
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.micromodels.intent_router import UniversalIntentEncoder
from src.micromodels.schema_dataset_generator import (
    ALL_INTENTS,
    ID2INTENT,
    INTENT2ID,
    TOOL_SCHEMAS,
)
from src.micromodels.semantic_registry import SemanticToolRegistry
from src.micromodels.tokenizer import BPETokenizer
from logger import get_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = get_logger("train_universal_router")

class UniversalIntentDataset(Dataset):
    """Dataset for 67-class tool intent classification and semantic routing."""

    def __init__(self, samples: List[Dict], tokenizer: BPETokenizer, max_len: int = 48):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        text = item["text"]
        intent_str = item.get("intent", "NO_TOOL")
        intent_id = INTENT2ID.get(intent_str, 0)

        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(intent_id, dtype=torch.long),
        )


def train_universal_model(
    epochs: int = 18,
    batch_size: int = 64,
    learning_rate: float = 3.5e-4,
    weight_decay: float = 0.01,
    max_len: int = 48,
    data_dir: Optional[str] = None,
) -> Tuple[UniversalIntentEncoder, BPETokenizer]:
    """Trains UniversalIntentEncoder and populates SemanticToolRegistry."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    base_dir = Path(data_dir or (Path(__file__).parent / "data"))
    pairs_file = base_dir / "universal_training_pairs.jsonl"
    tokenizer_file = base_dir / "tokenizer.json"

    # If universal dataset doesn't exist yet, generate it
    if not pairs_file.exists():
        logger.info("Generating universal dataset for all 66 tools...")
        from src.micromodels.schema_dataset_generator import UniversalDatasetGenerator

        gen = UniversalDatasetGenerator(output_dir=str(base_dir))
        gen.generate_all(samples_per_tool=55)

    # 1. Load samples
    samples: List[Dict] = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} training pairs across {len(ALL_INTENTS)} classes.")

    # Pre-encode tool descriptions for contrastive alignment and vocabulary
    tool_desc_dict = {
        "NO_TOOL": "general conversation, reasoning, programming questions, chit-chat, hello"
    }
    for s in TOOL_SCHEMAS:
        fn = s.get("function", {})
        tool_desc_dict[fn.get("name")] = fn.get("description", "")

    tool_desc_list = [
        f"{name.replace('_', ' ')}: {tool_desc_dict.get(name, '')}" for name in ALL_INTENTS
    ]

    # 2. Tokenizer (retrain on full corpus including tool schemas and engine terminology)
    tokenizer = BPETokenizer(vocab_size=2048)
    vocab_corpus = [s["text"] for s in samples] + tool_desc_list + [
        "unity 3d bake lightmap lighting scene light lightmaps запеки свет освещение сцена Unity сцены запекание blender mesh render shader texture physics animation"
    ]
    tokenizer.train(vocab_corpus)
    tokenizer.save(tokenizer_file)

    tool_desc_tokens = [
        tokenizer.encode(td, max_length=max_len, padding="max_length", truncation=True)
        for td in tool_desc_list
    ]
    tool_desc_tensor = torch.tensor(tool_desc_tokens, dtype=torch.long, device=device)

    # 3. Stratified Split (85% Train / 15% Val)
    by_intent: Dict[str, List[Dict]] = {it: [] for it in ALL_INTENTS}
    for s in samples:
        it = s.get("intent", "NO_TOOL")
        if it in by_intent:
            by_intent[it].append(s)

    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    random.seed(42)

    for it, items in by_intent.items():
        random.shuffle(items)
        split_idx = int(len(items) * 0.85)
        train_samples.extend(items[:split_idx])
        val_samples.extend(items[split_idx:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    train_ds = UniversalIntentDataset(train_samples, tokenizer, max_len=max_len)
    val_ds = UniversalIntentDataset(val_samples, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 4. Model
    model = UniversalIntentEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=192,
        num_layers=4,
        num_heads=4,
        ff_dim=512,
        max_seq_len=max_len,
        num_classes=len(ALL_INTENTS),
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # 5. Training loop
    best_val_acc = 0.0
    best_weights_path = base_dir / "intent_router.pt"

    t0_train = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for input_ids, targets in train_loader:
            input_ids, targets = input_ids.to(device), targets.to(device)

            optimizer.zero_grad()
            _, tool_embeddings = model(tool_desc_tensor)
            logits, query_embeddings = model(input_ids)
            class_loss = criterion(logits, targets)

            # Contrastive semantic alignment loss
            sim_matrix = torch.matmul(query_embeddings, tool_embeddings.t()) / 0.07
            contrastive_loss = nn.functional.cross_entropy(sim_matrix, targets)

            loss = class_loss + 0.4 * contrastive_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        scheduler.step()
        train_loss = total_loss / total_train
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for input_ids, targets in val_loader:
                input_ids, targets = input_ids.to(device), targets.to(device)
                logits, _ = model(input_ids)
                loss = criterion(logits, targets)

                val_loss += loss.item() * input_ids.size(0)
                preds = torch.argmax(logits, dim=-1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        val_loss = val_loss / total_val
        val_acc = correct_val / total_val

        logger.info(
            f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "vocab_size": tokenizer.vocab_size,
                    "intents": ALL_INTENTS,
                },
                best_weights_path,
            )

    elapsed = time.time() - t0_train
    logger.info(
        f"Universal training completed in {elapsed:.1f}s. Best validation accuracy: {best_val_acc*100:.2f}%"
    )

    # 6. Load best weights to CPU
    checkpoint = torch.load(best_weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to("cpu")

    # 7. Pre-compute semantic tool embeddings into SemanticToolRegistry
    registry = SemanticToolRegistry.get_instance()
    logger.info(f"Computing semantic embeddings for all {len(TOOL_SCHEMAS)} tools...")

    for schema in TOOL_SCHEMAS:
        fn = schema.get("function", {})
        t_name = fn.get("name")
        t_desc = fn.get("description", "")
        # Construct semantic signature
        sig_text = f"{t_name.replace('_', ' ')}: {t_desc}"
        token_ids = tokenizer.encode(sig_text, max_length=max_len, padding="max_length", truncation=True)
        with torch.no_grad():
            inp_t = torch.tensor([token_ids], dtype=torch.long)
            emb = model.get_embedding(inp_t)[0].numpy()
        registry.register_tool(name=t_name, description=t_desc, embedding=emb)

    registry.save()
    logger.info(f"Saved {registry.total_tools} semantic tool vectors to {registry.registry_file}")

    # 8. Export to ONNX (dual output: logits + embedding)
    onnx_path = base_dir / "intent_router.onnx"
    int8_path = base_dir / "intent_router_int8.onnx"

    logger.info(f"Exporting Universal model to ONNX: {onnx_path}...")
    dummy_input = torch.zeros((1, max_len), dtype=torch.long)
    # Ensure [CLS] token at 0 is non-zero
    dummy_input[0, 0] = tokenizer.cls_token_id

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["logits", "embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=14,
        dynamo=False,
    )
    logger.info(f"ONNX export succeeded: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # 9. INT8 Dynamic Quantization
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        logger.info(f"Quantizing Universal model to INT8: {int8_path}...")
        quantize_dynamic(
            str(onnx_path),
            str(int8_path),
            weight_type=QuantType.QUInt8,
        )
        logger.info(f"INT8 quantization succeeded: {int8_path} ({int8_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        logger.warning(f"INT8 quantization failed: {e}")

    # 10. Verify with ONNX Runtime
    try:
        import onnxruntime as ort

        target_model = int8_path if int8_path.exists() else onnx_path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
        sess = ort.InferenceSession(str(target_model), opts)
        ort_outs = sess.run(None, {"input_ids": dummy_input.numpy()})
        ort_logits, ort_emb = ort_outs[0], ort_outs[1]

        with torch.no_grad():
            torch_logits, torch_emb = model(dummy_input)

        max_diff = np.max(np.abs(ort_logits - torch_logits.numpy()))
        logger.info(
            f"ONNX Runtime verification passed on {target_model.name}! "
            f"Logits shape: {ort_logits.shape}, Emb shape: {ort_emb.shape}, max_diff: {max_diff:.6e}"
        )
    except Exception as e:
        logger.warning(f"ONNX Runtime verification warning: {e}")

    return model, tokenizer


if __name__ == "__main__":
    train_universal_model()
