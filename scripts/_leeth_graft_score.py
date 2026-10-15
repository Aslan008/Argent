"""Graft the trained 10-level score head (candidate v6) onto the untouched v5 backbone.

Why: the full migration run trained the score head perfectly (CE 1.60 -> 0.0094)
but drifted choice (+9.6% val CE) -> gate REJECTED. The drift came from training
the whole network. Grafting the head onto the pristine v5 backbone keeps
choice/noul/extract bitwise identical to v5 while inheriting the trained head.

Gate: non-score val losses must match v5's BEFORE metrics (same weights), score
must drop well below v5's random-head 1.60. On pass: promote as v7 + leeth_best.pt,
re-export ONNX + INT8, verify. On fail: leave everything untouched.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.leeth.dataset_generator import LeethDatasetGenerator  # noqa: E402
from src.leeth.finetune import _next_version  # noqa: E402
from src.leeth.model import LeethModel  # noqa: E402
from src.leeth.tokenizer import LeethTokenizer  # noqa: E402
from src.leeth.train import LeethTorchDataset, collate_leeth_batch, compute_calibrated_loss  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("leeth_graft")

DATA_DIR = Path("src/leeth/data")
BEST_PATH = DATA_DIR / "leeth_best.pt"
CAND_PATH = DATA_DIR / "leeth_candidate_v6.pt"
CLASSES_PATH = DATA_DIR / "choice_classes.json"
SYNTH_VAL_PATH = DATA_DIR / "leeth_val.jsonl"
MAX_LENGTH = 320

# BEFORE metrics from the migration run (same val set, v5 weights + random head).
BEFORE = {"choice": 0.3904613141830151, "noul": 0.0018969223605363606,
          "score": 1.602574692832099, "extract": 330.1793490323153}


def _load_jsonl(path: Path) -> list:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(__import__("src.leeth.dataset_generator", fromlist=["LeethSample"]).LeethSample(**json.loads(line)))
    return samples


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


def _atomic_replace(src: Path, dst: Path, attempts: int = 5) -> None:
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(1.0)


def main():
    started = time.time()
    device = "cpu"
    summary = {"device": device, "mode": "graft_score_head"}

    tokenizer = LeethTokenizer.load(DATA_DIR)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    best_ck = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
    cand_ck = torch.load(CAND_PATH, map_location="cpu", weights_only=False)

    # ── Hybrid state: v5 everything + v6 score head ──────────────────────────
    hybrid_sd = dict(best_ck["state_dict"])
    grafted = [k for k in cand_ck["state_dict"] if k.startswith("score_head")]
    for k in grafted:
        hybrid_sd[k] = cand_ck["state_dict"][k]
    summary["grafted_keys"] = grafted

    model = LeethModel(best_ck["config"])
    model.load_state_dict(hybrid_sd)  # strict: config 10 levels + 10-out head = consistent
    model.to(device)
    logger.info("Hybrid model built: v5 backbone + v6 score head (%d keys grafted)", len(grafted))

    # ── Val set (same deterministic regeneration as the migration) ───────────
    val_count = sum(1 for line in open(SYNTH_VAL_PATH, encoding="utf-8") if line.strip())
    LeethDatasetGenerator(seed=9999).generate_and_save(SYNTH_VAL_PATH, target_count=val_count)
    val_samples = _load_jsonl(SYNTH_VAL_PATH)
    logger.info("Val samples: %d", len(val_samples))

    after = {qt: _type_loss(model, val_samples, qt, tokenizer, class_to_idx, device)
             for qt in ("choice", "noul", "score", "extract")}
    logger.info("Hybrid metrics: %s", json.dumps({k: round(v, 4) for k, v in after.items()}))
    summary["after"] = after

    # ── Gate ─────────────────────────────────────────────────────────────────
    gate_ok = True
    for qt in ("choice", "noul", "extract"):
        b, a = BEFORE[qt], after[qt]
        if a > b * 1.02:  # same weights -> must be ~identical; 2% tolerance for fp noise
            gate_ok = False
            logger.warning("GATE FAIL: %s %.4f -> %.4f", qt, b, a)
    score_dropped = after["score"] < BEFORE["score"] * 0.9
    accepted = gate_ok and score_dropped
    summary["gate_ok"] = gate_ok
    summary["score_improved"] = bool(score_dropped)
    summary["accepted"] = accepted
    logger.info("Gate: ok=%s score_improved=%s accepted=%s", gate_ok, score_dropped, accepted)

    if not accepted:
        summary["action"] = "kept_v5_untouched"
        out = ROOT / ".argent" / "artifacts" / "leeth_graft_score.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        logger.info("REJECTED — v5 stays active. Summary: %s", out)
        return

    # ── Promote: versioned + atomic best ─────────────────────────────────────
    version = _next_version()
    payload = {
        "config": best_ck["config"],
        "state_dict": hybrid_sd,
        "class_to_idx": class_to_idx,
        "val_loss": after["score"],
        "brier_score": best_ck.get("brier_score"),
        "graft_meta": {
            "ts": time.time(),
            "mode": "graft_score_head_v6_onto_v5",
            "grafted_keys": grafted,
            "before": BEFORE,
            "after": after,
        },
    }
    versioned = DATA_DIR / f"leeth_best_v{version}.pt"
    torch.save(payload, versioned)
    tmp = BEST_PATH.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    _atomic_replace(tmp, BEST_PATH)
    summary["checkpoint_version"] = str(versioned)
    logger.info("ACCEPTED: promoted to %s (v%d)", BEST_PATH, version)

    # ── ONNX re-export (atomic) — hybrid checkpoint is consistent ────────────
    from src.leeth.export import export_to_onnx, quantize_onnx_model, verify_exported_model

    stamp = int(time.time())
    tmp_onnx = export_to_onnx(str(DATA_DIR), weights_filename="leeth_best.pt",
                              output_filename=f"leeth.onnx.tmp{stamp}")
    _atomic_replace(tmp_onnx, DATA_DIR / "leeth.onnx")
    tmp_int8 = quantize_onnx_model(DATA_DIR / "leeth.onnx", output_filename=f"leeth_int8.onnx.tmp{stamp}")
    _atomic_replace(tmp_int8, DATA_DIR / "leeth_int8.onnx")
    summary["onnx_reexported"] = True
    logger.info("ONNX + INT8 re-exported")

    metrics = verify_exported_model(str(DATA_DIR))
    summary["verify"] = metrics
    logger.info("Verify: %s", json.dumps(metrics))

    summary["elapsed_s"] = round(time.time() - started, 1)
    out = ROOT / ".argent" / "artifacts" / "leeth_graft_score.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Summary written to %s", out)


if __name__ == "__main__":
    main()