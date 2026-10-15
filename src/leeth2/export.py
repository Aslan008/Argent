"""ONNX export of the Leeth 2 graph (CPU/GPU inference without PyTorch) and a parity check against PyTorch.

    python -m src.leeth2.export <model_dir> <out.onnx> [--fp16]

Inputs  : input_ids [B,L], attention_mask [B,L], opt_index [B,L], rd_pos [B], state_mask [B,L], n_opts [B], opt_slots [K]
Outputs : choice_logits [B,K], noul_logit [B], score_logits [B,10], start_logits [B,L], end_logits [B,L]
``opt_slots`` is arange(1, K+1) — its length is the number of option slots, a dynamic axis, so one graph serves
0..255 options. Logits are raw: temperature calibration (config) is applied by the caller.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .model import Leeth2Model

INPUTS = ["input_ids", "attention_mask", "opt_index", "rd_pos", "state_mask", "n_opts", "opt_slots"]
OUTPUTS = ["choice_logits", "noul_logit", "score_logits", "start_logits", "end_logits"]


class OnnxWrapper(torch.nn.Module):
    def __init__(self, model: Leeth2Model):
        super().__init__()
        self.m = model

    def forward(self, input_ids, attention_mask, opt_index, rd_pos, state_mask, n_opts, opt_slots):
        o = self.m(input_ids, attention_mask, opt_index, rd_pos, state_mask.bool(), n_opts, slots=opt_slots)
        return o["choice_logits"], o["noul_logit"], o["score_logits"], o["start_logits"], o["end_logits"]


def example_inputs(vocab: int, n_markers: int, B: int = 2, L: int = 40, K: int = 4) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(10, vocab - 1, (B, L), generator=g)
    ids[:, 0] = 2
    rd = torch.tensor([L - 8] * B)
    ids[torch.arange(B), rd] = vocab + 1  # [RD]
    opt = torch.zeros((B, L), dtype=torch.long)
    for k in range(K):  # options live in the last 6 positions: K tags spread over them
        opt[:, L - 6 + (k % 6)] = k + 1
    state = torch.zeros((B, L), dtype=torch.bool)
    state[:, 1 : L - 10] = True
    state[torch.arange(B), rd] = True
    return {
        "input_ids": ids, "attention_mask": torch.ones((B, L), dtype=torch.long), "opt_index": opt, "rd_pos": rd, "state_mask": state,
        "n_opts": torch.tensor([K] * B), "opt_slots": torch.arange(1, K + 1),
    }


def export(model: Leeth2Model, out_path: str | Path, opset: int = 17, half: bool = False) -> Path:
    model = model.eval().float()
    wrapper = OnnxWrapper(model).eval()
    ex = example_inputs(model.cfg.vocab_size, model.cfg.n_markers)
    out = Path(out_path)
    if out.exists():
        out.unlink()  # overwriting a big existing file fails on Windows (Errno 22)
    torch.onnx.export(
        wrapper, tuple(ex[k] for k in INPUTS), str(out), input_names=INPUTS, output_names=OUTPUTS, opset_version=opset, dynamo=False,
        dynamic_axes={
            "input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"}, "opt_index": {0: "B", 1: "L"}, "rd_pos": {0: "B"}, "state_mask": {0: "B", 1: "L"},
            "n_opts": {0: "B"}, "opt_slots": {0: "K"}, "choice_logits": {0: "B", 1: "K"}, "noul_logit": {0: "B"}, "score_logits": {0: "B"}, "start_logits": {0: "B", 1: "L"}, "end_logits": {0: "B", 1: "L"},
        },
    )
    return out


def parity(model: Leeth2Model, onnx_path: str | Path, trials: int = 3) -> Dict[str, float]:
    """Max |torch - onnxruntime| on random inputs of several shapes (batch, length and option count all vary)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = model.eval().float()
    worst: Dict[str, float] = {k: 0.0 for k in OUTPUTS}
    for B, L, K in [(2, 40, 4), (1, 64, 7), (3, 33, 2)][:trials]:
        ex = example_inputs(model.cfg.vocab_size, model.cfg.n_markers, B, L, K)
        with torch.no_grad():
            ref = OnnxWrapper(model)(*[ex[k] for k in INPUTS])
        got = sess.run(OUTPUTS, {k: ex[k].numpy() for k in INPUTS})
        for name, a, b in zip(OUTPUTS, ref, got):
            a = a.numpy()
            mask = np.abs(a) < 1e8  # masked slots carry -1e9; compare the live part
            worst[name] = max(worst[name], float(np.max(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0)
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    m = Leeth2Model.from_pretrained(a.model_dir)
    p = export(m, a.out, a.opset)
    print("exported", p, f"{p.stat().st_size / 1e6:.0f} MB")
    print("parity (max abs diff):", parity(m, p))


if __name__ == "__main__":
    main()
