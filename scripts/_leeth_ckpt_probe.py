"""Read-only probe: inspect score-head shape + config in every leeth checkpoint."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("src/leeth/data")
out = []
for p in sorted(DATA_DIR.glob("leeth_*.pt")):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    cfg = ck.get("config")
    msl = getattr(cfg, "max_score_levels", "N/A")
    sd = ck.get("state_dict", {})
    sh3 = sd.get("score_head.3.weight")
    sh_keys = sorted(k for k in sd if k.startswith("score_head"))
    out.append(
        {
            "file": p.name,
            "max_score_levels": msl,
            "score_head_keys": sh_keys,
            "score_head.3.weight": list(sh3.shape) if sh3 is not None else None,
            "val_loss": ck.get("val_loss"),
        }
    )
print(json.dumps(out, indent=2, default=str))