"""
src/leeth/export.py — ONNX Export and Quantization for Leeth System One Model.

Provides:
- export_to_onnx(): Exports trained PyTorch weights to standard ONNX format.
- quantize_onnx_model(): Quantizes ONNX model to INT8 dynamic quantization for CPU.
- verify_exported_model(): Validates max numerical divergence between PyTorch and ONNX.
"""

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.leeth.model import LeethConfig, LeethModel, config_from_checkpoint
from logger import get_logger

logger = get_logger("leeth_export")

def _load_checkpoint(weights_path: Path) -> dict:
    """torch.load с алиасом model_leeth -> src.leeth.model.

    Чекпоинты, файнтюненные вне src/ (Kaggle kernel), ссылаются на модуль
    'model_leeth' — без алиаса pickle падает с ModuleNotFoundError.
    """
    import sys

    if "model_leeth" not in sys.modules:
        from src.leeth import model as _leeth_model

        sys.modules["model_leeth"] = _leeth_model
    return torch.load(weights_path, map_location="cpu", weights_only=False)


def export_to_onnx(
    model_dir: str = "src/leeth/data",
    weights_filename: str = "leeth_best.pt",
    output_filename: str = "leeth.onnx",
    opset_version: int = 17,
) -> Path:
    """Exports trained PyTorch Leeth checkpoint to ONNX format."""
    model_dir_path = Path(model_dir)
    weights_path = model_dir_path / weights_filename
    output_path = model_dir_path / output_filename

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights checkpoint not found: {weights_path}")

    checkpoint = _load_checkpoint(weights_path)
    config = config_from_checkpoint(checkpoint["config"])
    model = LeethModel(config)
    # Checkpoints predating DynamicOptionHead lack its weights — load leniently
    # (the head stays randomly initialized and is NOT exported: the legacy ONNX
    # graph keeps the 4-primitive surface; dynamic scoring exports separately).
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    dummy_ids = torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long)
    dummy_mask = torch.ones((1, 32), dtype=torch.long)

    output_names = [
        "hidden_states",
        "pooled_repr",
        "choice_logits",
        "noul_logits",
        "noul_probs",
        "score_logits",
        "start_logits",
        "end_logits",
        "mlm_logits",
    ]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "hidden_states": {0: "batch_size", 1: "seq_len"},
        "pooled_repr": {0: "batch_size"},
        "choice_logits": {0: "batch_size"},
        "noul_logits": {0: "batch_size"},
        "noul_probs": {0: "batch_size"},
        "score_logits": {0: "batch_size", 1: "levels"},
        "start_logits": {0: "batch_size", 1: "seq_len"},
        "end_logits": {0: "batch_size", 1: "seq_len"},
        "mlm_logits": {0: "batch_size", 1: "seq_len"},
    }

    logger.info("Exporting Leeth PyTorch model to ONNX: %s", output_path)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Successfully exported ONNX model (%0.2f MB) to %s", size_mb, output_path)
    return output_path


def quantize_onnx_model(
    onnx_path: Path,
    output_filename: str = "leeth_int8.onnx",
) -> Path:
    """Quantizes an ONNX model to INT8 using dynamic quantization for high CPU speed."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = onnx_path.parent / output_filename
    logger.info("Quantizing ONNX model to INT8: %s", int8_path)

    quantize_dynamic(
        str(onnx_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
    )

    size_mb = os.path.getsize(int8_path) / (1024 * 1024)
    logger.info("Successfully quantized model to INT8 (%0.2f MB) at %s", size_mb, int8_path)
    return int8_path


def verify_exported_model(
    model_dir: str = "src/leeth/data",
    seq_len: int = 48,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Verifies that PyTorch, ONNX, and ONNX INT8 models produce aligned outputs.

    seed: pins the random probe inputs. Production callers omit it (random
    probes cover more of the input space); TESTS pass a seed — an unseeded
    probe makes the max-logit-drift assertion flaky (the drift depends on
    which token ids land in the probe, 4.2-9.4 observed run-to-run).
    """
    import onnxruntime as ort

    model_dir_path = Path(model_dir)
    weights_path = model_dir_path / "leeth_best.pt"
    onnx_path = model_dir_path / "leeth.onnx"
    int8_path = model_dir_path / "leeth_int8.onnx"

    checkpoint = _load_checkpoint(weights_path)
    config = config_from_checkpoint(checkpoint["config"])
    model = LeethModel(config)
    # Lenient load: checkpoints predating DynamicOptionHead lack its weights
    # (see export_leeth_to_onnx above).
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        dummy_ids = torch.randint(0, config.vocab_size, (1, seq_len), dtype=torch.long, generator=generator)
    else:
        dummy_ids = torch.randint(0, config.vocab_size, (1, seq_len), dtype=torch.long)
    dummy_mask = torch.ones((1, seq_len), dtype=torch.long)

    with torch.no_grad():
        pt_outputs = model(dummy_ids, attention_mask=dummy_mask)

    ort_inputs = {
        "input_ids": dummy_ids.numpy(),
        "attention_mask": dummy_mask.numpy(),
    }

    # Verify standard ONNX
    sess_fp32 = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_fp32_outputs = sess_fp32.run(None, ort_inputs)

    diff_choice = float(np.max(np.abs(pt_outputs["choice_logits"].numpy() - ort_fp32_outputs[2])))
    diff_noul = float(np.max(np.abs(pt_outputs["noul_probs"].numpy() - ort_fp32_outputs[4])))
    diff_score = float(np.max(np.abs(pt_outputs["score_logits"].numpy() - ort_fp32_outputs[5])))

    # Verify INT8 ONNX
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    ort_int8_outputs = sess_int8.run(None, ort_inputs)

    diff_int8_choice = float(np.max(np.abs(pt_outputs["choice_logits"].numpy() - ort_int8_outputs[2])))
    diff_int8_noul = float(np.max(np.abs(pt_outputs["noul_probs"].numpy() - ort_int8_outputs[4])))
    diff_int8_score = float(np.max(np.abs(pt_outputs["score_logits"].numpy() - ort_int8_outputs[5])))

    metrics = {
        "fp32_max_choice_diff": diff_choice,
        "fp32_max_noul_diff": diff_noul,
        "fp32_max_score_diff": diff_score,
        "int8_max_choice_diff": diff_int8_choice,
        "int8_max_noul_diff": diff_int8_noul,
        "int8_max_score_diff": diff_int8_score,
    }
    # Free the ~1.4 GB PyTorch model + ONNX sessions BEFORE returning —
    # under pytest the runtime singleton may hold another 1.4 GB, and
    # Windows access-violates when peak RSS approaches the commit limit.
    del model, sess_fp32, sess_int8
    gc.collect()
    logger.info("Verification metrics: %s", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Export Leeth model to ONNX & INT8")
    parser.add_argument("--model-dir", default="src/leeth/data", help="Model directory")
    args = parser.parse_args()

    onnx_file = export_to_onnx(model_dir=args.model_dir)
    int8_file = quantize_onnx_model(onnx_file)
    metrics = verify_exported_model(model_dir=args.model_dir)
    print("Export, quantization, and verification complete:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
