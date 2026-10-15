# -*- coding: utf-8 -*-
"""scripts/_battery_torch_compare.py — батарея 109 кейсов через torch fp32 для двух наборов голов.

Изолирует качество голов от ONNX-экспорта/квантизации: если кандидат на torch
так же плох, как в ONNX — регресс в головах; если хорош — проблема в экспорте.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE_DIR = Path(".argent/models/mmbert-argent")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"


class Heads(torch.nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.choice_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, n_classes))
        self.noul_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(), torch.nn.Linear(hidden // 2, 1))
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(), torch.nn.Linear(hidden // 2, 10))
        self.extract_start_head = torch.nn.Linear(hidden, 1)
        self.extract_end_head = torch.nn.Linear(hidden, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32)
    base.to(device).eval()
    class_to_idx = json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8"))
    idx_to_cls = {v: k for k, v in class_to_idx.items()}
    cases = json.loads(Path("scripts/prod_battery.json").read_text(encoding="utf-8"))["cases"]

    texts = [f'Incoming user request in Argent session:\n"{c["query"]}"\n'
             f"Working directory: C:/Users/user/Desktop/Argent" for c in cases]
    encs = [tok(t + "\n[QUESTION] " + TEMPLATE_Q, truncation=True, max_length=256,
                return_tensors="pt") for t in texts]

    for tag, heads_path in (("ПРОД ", ".argent/models/mmbert-leeth.pt"),
                            ("КАНД ", ".argent/models/mmbert-leeth_candidate.pt")):
        ckpt = torch.load(heads_path, map_location="cpu", weights_only=False)
        heads = Heads(ckpt["hidden"], ckpt["n_classes"])
        heads.load_state_dict(ckpt["heads"])
        heads.to(device).eval()
        ok = 0
        misses = []
        with torch.no_grad():
            for c, enc in zip(cases, encs):
                enc = {k: v.to(device) for k, v in enc.items()}
                out = base.base_model(**enc, output_hidden_states=True)
                hs = out.hidden_states[-1]
                mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
                pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                logits = heads.choice_head(pooled)[0]
                pred = idx_to_cls[int(logits.argmax())]
                ok += pred == c["expected_tool"]
                if pred != c["expected_tool"]:
                    misses.append(f"  {c['query'][:50]:<52} want {c['expected_tool']:<18} got {pred}")
        print(f"[{tag}] {heads_path} -> батарея: {ok}/109 = {ok/109:.1%}")
        for m in misses[:8]:
            print(m)
        print()


if __name__ == "__main__":
    main()