"""Focused probe: does 363M Leeth TRAINING fit on RTX 4060 8GB? Minimal, fast."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from src.leeth.model import LeethConfig, LeethModel

torch.cuda.empty_cache()
cfg = LeethConfig(vocab_size=3560, hidden_size=1024, num_layers=28, num_heads=16,
                  intermediate_size=4096, dropout=0.0)
model = LeethModel(cfg).cuda()
n = sum(p.numel() for p in model.parameters())
print(f"params: {n:,} ({n/1e6:.0f}M)  fp32 weights: {n*4/2**30:.2f} GB", flush=True)

def train_step(B, autocast):
    ids = torch.randint(0, 3000, (B, 512), device="cuda")
    mask = torch.ones_like(ids)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        # warmup 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = model(ids, attention_mask=mask)
            loss = out["choice_logits"].float().pow(2).mean() + out["noul_probs"].float().pow(2).mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                out = model(ids, attention_mask=mask)
                loss = out["choice_logits"].float().pow(2).mean() + out["noul_probs"].float().pow(2).mean()
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f"  train B={B} autocast={autocast}: step {ts[1]*1000:7.0f} ms  peak VRAM {mem:.2f} GB  -> OK", flush=True)
    except RuntimeError as e:
        print(f"  train B={B} autocast={autocast}: OOM ({str(e)[:70]})", flush=True)
    finally:
        del opt
        torch.cuda.empty_cache()

train_step(4, False)   # чистый fp32
train_step(4, True)    # bf16 autocast
train_step(8, True)
train_step(16, True)

# bf16 inference quick check (deployment path)
model_b = model.to(torch.bfloat16)
ids = torch.randint(0, 3000, (2, 512), device="cuda"); mask = torch.ones_like(ids)
with torch.inference_mode():
    model_b(ids, attention_mask=mask)
    torch.cuda.synchronize()
    ts = []
    for _ in range(8):
        t0 = time.perf_counter(); model_b(ids, attention_mask=mask); torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    ts.sort()
    print(f"  infer bf16 B=2 S=512: P50 {ts[4]:.1f} ms  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GB", flush=True)

del model_b
torch.cuda.empty_cache()
print("DONE", flush=True)