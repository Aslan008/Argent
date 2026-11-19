"""Reusable diagnostic tool for inspecting activation geometry and logit breakdown."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.leeth2.runtime import Leeth2Runtime
from src.leeth2.data.synth.argent_context import PROD_OPTIONS
from src.leeth2.data.argent import TOOL_QUESTION
from src.leeth.primitives import ChoiceQuestion
from src.leeth2.encoding import collate


def inspect_choice_geometry(model_dir: str, queries: Sequence[str], device: str = "cuda") -> None:
    rt = Leeth2Runtime.load(model_dir, device=device)
    model = rt.model
    enc = rt.encoder

    for q in queries:
        state = f'Incoming user request in Argent session:\n"{q}"\nWorking directory: C:/Users/user/Desktop/Argent'
        sample = rt._to_sample(state, ChoiceQuestion("c", TOOL_QUESTION, choices=PROD_OPTIONS))
        encoded = enc.encode(sample)
        raw_b = collate([encoded], pad_id=enc.pad)
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw_b.items()}
        
        with torch.no_grad():
            hidden = model.encode_states(b["input_ids"], b["attention_mask"])
            B = hidden.size(0)
            h_rd = hidden[torch.arange(B, device=hidden.device), b["rd_pos"]]
            m = b["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
            qv_raw = torch.cat([h_rd, mean], dim=-1)
            
            slots = torch.arange(1, len(PROD_OPTIONS) + 1, device=hidden.device)
            opt = model.pool_options(hidden, b["opt_index"], slots)
            
            qv = model.choice_head.q(qv_raw).unsqueeze(1)
            ov = model.choice_head.o(opt)
            
            mlp_term = model.choice_head.mix(ov + qv).squeeze(-1)
            dot_term = (ov * qv).sum(-1) / model.choice_head.scale
            total = mlp_term + dot_term
            
            perm = encoded.perm
            shown_to_orig = {s: o for s, o in enumerate(perm)}
            
            print(f"\nQuery: {q!r}")
            for shown_idx in range(len(PROD_OPTIONS)):
                orig_idx = shown_to_orig[shown_idx]
                name = PROD_OPTIONS[orig_idx]
                print(f"  {name:22s}: total={total[0, shown_idx]:+.3f} (mlp={mlp_term[0, shown_idx]:+.3f}, dot={dot_term[0, shown_idx]:+.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", default=".argent/models/leeth2/final")
    parser.add_argument("--queries", nargs="+", default=["проверь статус репозитория", "запусти линтер на проекте"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    inspect_choice_geometry(args.model_dir, args.queries, args.device)
