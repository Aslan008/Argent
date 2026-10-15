"""Minimal memory probe: 363M fwd/bwd at B=8x256 with checkpointing + 8bit Adam."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from src.leeth.model import LeethConfig, LeethModel

def report(tag):
    alloc = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    free = torch.cuda.mem_get_info()[0] / 2**30
    print(f"  {tag}: allocated={alloc:.2f}GB reserved={reserved:.2f}GB free(device)={free:.2f}GB", flush=True)

torch.cuda.empty_cache()
report("start")

cfg = LeethConfig(vocab_size=3560, hidden_size=1024, num_layers=28, num_heads=16,
                  intermediate_size=4096, dropout=0.0)
model = LeethModel(cfg).cuda()
model.gradient_checkpointing = True
model.train()
report("model on gpu")

try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    print("  opt: AdamW8bit", flush=True)
except Exception as e:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(f"  opt: torch AdamW ({str(e)[:50]})", flush=True)

ids = torch.randint(0, 3000, (8, 256), device="cuda")
mask = torch.ones_like(ids)
for attempt in range(3):
    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, attention_mask=mask)
            loss = out["choice_logits"].float().pow(2).mean() + out["noul_probs"].float().pow(2).mean()
        report(f"attempt {attempt} after fwd")
        loss.backward()
        report(f"attempt {attempt} after bwd")
        opt.step(); opt.zero_grad(set_to_none=True)
        report(f"attempt {attempt} after step")
    except RuntimeError as e:
        print(f"  attempt {attempt} FAILED: {str(e)[:120]}", flush=True)
        torch.cuda.empty_cache()
print("DONE", flush=True)