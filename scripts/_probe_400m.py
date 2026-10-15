"""Empirical probe v2: ~400M Leeth on this machine. inference_mode + fixed VRAM print."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from src.leeth.model import LeethConfig, LeethModel

def count(m): return sum(p.numel() for p in m.parameters())

def bench(model, B, S, runs=8, warmup=3, backward=False, opt=None, autocast=False):
    ids = torch.randint(0, 3000, (B, S), device="cuda")
    mask = torch.ones_like(ids)
    ctx = torch.inference_mode() if not backward else torch.autocast("cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        with ctx:
            out = model(ids, attention_mask=mask)
            if backward:
                loss = out["choice_logits"].float().pow(2).mean() + out["noul_probs"].float().pow(2).mean()
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with ctx:
            out = model(ids, attention_mask=mask)
            if backward:
                loss = out["choice_logits"].float().pow(2).mean() + out["noul_probs"].float().pow(2).mean()
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts)//2], torch.cuda.max_memory_allocated() / 2**30

def safe_bench(model, B, S, **kw):
    try:
        ms, mem = bench(model, B, S, **kw)
        tag = "train" if kw.get("backward") else "infer"
        print(f"  {tag} B={B:>3} S={S}: P50 {ms:7.1f} ms  peak VRAM {mem:.2f} GB", flush=True)
    except RuntimeError as e:
        print(f"  {'train' if kw.get('backward') else 'infer'} B={B:>3} S={S}: OOM ({str(e)[:60]})", flush=True)
        torch.cuda.empty_cache()

def probe(name, cfg):
    print(f"\n=== {name} ===", flush=True)
    torch.cuda.empty_cache()
    model = LeethModel(cfg).cuda()
    n = count(model)
    print(f"params: {n:,} ({n/1e6:.0f}M)  fp32 weights: {n*4/2**30:.2f} GB", flush=True)
    for B, S in [(2,512), (32,512), (255,512)]:
        safe_bench(model, B, S)
    # bf16 weights inference (runtime INT8/FP16 path analog)
    model_b = model.to(torch.bfloat16)
    for B, S in [(2,512), (32,512), (255,512)]:
        safe_bench(model_b, B, S)
    model_b = model_b.float()
    # training: bf16 autocast + Adam fp32
    for B in (4, 8, 16):
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        torch.cuda.empty_cache()
        safe_bench(model, B, 512, runs=5, warmup=2, backward=True, opt=opt)
        del opt; torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()

base = dict(vocab_size=3560, hidden_size=1024, num_layers=28, num_heads=16, intermediate_size=4096, dropout=0.0)
probe("A: h=1024 L=28 vocab=3560", LeethConfig(**base))
probe("B: h=1024 L=28 vocab=16384", LeethConfig(**{**base, "vocab_size": 16384}))

# CPU fallback (ONNX analog) — config A, inference_mode
torch.cuda.empty_cache()
m = LeethModel(LeethConfig(**base)).eval()
ids = torch.randint(0, 3000, (2, 512)); mask = torch.ones_like(ids)
with torch.inference_mode():
    m(ids, attention_mask=mask)
    t0 = time.perf_counter(); m(ids, attention_mask=mask)
print(f"\nCPU fp32 (2x512): {(time.perf_counter()-t0)*1000:.0f} ms — ориентир для ONNX-фолбэка")