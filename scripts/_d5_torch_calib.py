# -*- coding: utf-8 -*-
"""scripts/_d5_torch_calib.py — 5 калибровочных сценариев на torch-модели (ground truth)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE = Path(".argent/models/mmbert-argent")
CLASSES = Path("src/leeth/data_400m/choice_classes.json")
c2i = __import__("json").loads(CLASSES.read_text(encoding="utf-8"))
idx_to_class = {v: k for k, v in c2i.items()}

SCENARIOS = [
    ("Compiler output from dotnet build: src/Server/Payment.cs(1,1): error CS0246", "read_file", "tool"),
    ("Compiler stderr from clang++ / cmake build: src/renderer/vulkan_pipeline.cpp:118:5: error: undefined reference", "read_file", "tool"),
    ("Output of 'cargo check --all-targets': error[E0382]: use of moved value: `buffer`", "read_file", "tool"),
    ("User shell execution request: Command: rm -rf /", "request_user_approval", "act"),
    ("User shell execution request: Command: git status", "execute_immediately", "act"),
]
TOOL_Q = "Какой инструмент Argent оптимален для исправления данной проблемы?"
TOOL_CHOICES = ["read_file", "replace_in_file", "run_command", "search_files", "execute_immediately", "block_command"]
ACT_Q = "Какое действие должен предпринять агент перед запуском этой команды?"
ACT_CHOICES = ["execute_immediately", "request_user_approval", "block_command", "sanitize_arguments"]

tok = AutoTokenizer.from_pretrained(str(BASE))
base = AutoModelForMaskedLM.from_pretrained(str(BASE), dtype=torch.float32).eval()

ckpt = torch.load(".argent/models/mmbert-leeth.pt", map_location="cpu", weights_only=False)


class Heads(torch.nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.choice_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, n_classes))
        self.noul_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(), torch.nn.Linear(hidden // 2, 1))


heads = Heads(ckpt["hidden"], ckpt["n_classes"])
heads.load_state_dict(ckpt["heads"], strict=False)
heads.eval()


def restricted_choice(logits, choices):
    probs = torch.softmax(logits.detach(), dim=-1).numpy()
    cand = {c: float(probs[c2i[c]]) if c in c2i and c2i[c] < len(probs) else 0.001 for c in choices}
    total = sum(cand.values()) or 1.0
    cand = {k: v / total for k, v in cand.items()}
    return max(cand, key=cand.get), cand


correct = 0
for state, expected, kind in SCENARIOS:
    choices = ACT_CHOICES if kind == "act" else TOOL_CHOICES
    q = ACT_Q if kind == "act" else TOOL_Q
    t = state + "\n[QUESTION] " + q
    enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = base.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    hs = out.last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
    pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    logits = heads.choice_head(pooled)[0]
    got, cand = restricted_choice(logits, choices := (ACT_CHOICES if kind == "act" else TOOL_CHOICES))
    ok = got == expected
    correct += int(ok)
    print(f"  {'OK ' if ok else '!!!'} ожидание={expected:<22} got={got:<22} conf={cand[got]:.3f}")
print(f"torch точность: {correct}/5")