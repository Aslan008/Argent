"""Diagnose which head produces loss ~300 in the 400M training."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch.utils.data import DataLoader
from src.leeth.model import LeethConfig, LeethModel
from src.leeth.tokenizer import LeethTokenizer
from src.leeth.train import LeethTorchDataset, collate_leeth_batch, compute_calibrated_loss

tok = LeethTokenizer.load(ROOT / "src/leeth/data")
c2i = json.load(open(ROOT / "src/leeth/data/choice_classes.json", encoding="utf-8"))
rows = []
with open(ROOT / "src/leeth/data_400m_smoke/leeth_400m_train.jsonl", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))
print("sample types:", {t: sum(1 for r in rows if r["question_type"] == t) for t in ("choice", "noul", "score", "extract")})

ds = LeethTorchDataset([type("S", (), r) for r in rows[:400]], tok, max_length=256, class_to_idx=c2i)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_leeth_batch)

cfg = LeethConfig(vocab_size=3560, hidden_size=1024, num_layers=28, num_heads=16,
                  intermediate_size=4096, max_position_embeddings=1024,
                  num_choices=len(c2i), dropout=0.1)
model = LeethModel(cfg).cuda()
model.gradient_checkpointing = True
model.train()

it = iter(loader)
for step in range(6):
    batch = next(it)
    ids = batch["input_ids"].cuda(); mask = batch["attention_mask"].cuda()
    for k in ("choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"):
        batch[k] = batch[k].cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids, attention_mask=mask)
        loss, m = compute_calibrated_loss(out, batch)
    print(f"step {step}: total={loss.item():.2f} temp={model.temperature.item():.4f} "
          f"choice={m.get('choice_loss', 0):.3f} noul={m.get('noul_loss', 0):.3f} "
          f"score={m.get('score_loss', 0):.3f} span={m.get('span_loss', 0):.3f} "
          f"noul_count={m.get('noul_count', 0)} span_count={m.get('span_count', 0)}")
    loss.backward()
    # temperature gradient
    print(f"  temp grad: {model.temperature.grad.item() if model.temperature.grad is not None else None}")
    model.zero_grad(set_to_none=True)
print("DONE")