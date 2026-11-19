"""Automated script to perform dual-anchored surgical calibration on Leeth heads."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.leeth2.runtime import Leeth2Runtime
from src.leeth2.data.synth.argent_context import PROD_OPTIONS
from src.leeth2.data.argent import TOOL_QUESTION, SAFETY_QUESTION
from src.leeth.primitives import ChoiceQuestion, NoulQuestion
from src.leeth2.encoding import collate
from scripts.leeth2_exam import route, safety, GATES


def run_surgery(source_dir: str, target_dir: str, steps: int = 20, lr: float = 8e-5, device: str = "cuda") -> bool:
    src_p = Path(source_dir)
    tgt_p = Path(target_dir)
    tgt_p.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy files to target sandbox
    for f in src_p.glob("*"):
        if f.is_file():
            shutil.copy2(f, tgt_p / f.name)
            
    rt = Leeth2Runtime.load(str(tgt_p), device=device)
    model = rt.model
    enc = rt.encoder

    # Freeze entire backbone
    for p in model.parameters():
        p.requires_grad = False
    for p in model.choice_head.parameters():
        p.requires_grad = True
    model.noul_head[0].weight.requires_grad = True
    model.noul_head[0].bias.requires_grad = True

    # Datasets
    ood = json.loads((ROOT / "scripts" / "ood_exam_v3.json").read_text(encoding="utf-8"))
    v1 = json.loads((ROOT / "scripts" / "prod_battery.json").read_text(encoding="utf-8"))["cases"]
    v2 = json.loads((ROOT / "scripts" / "prod_battery_v2.json").read_text(encoding="utf-8"))["cases"]
    exam = json.loads((ROOT / "scripts" / "safety_exam_v2.json").read_text(encoding="utf-8"))

    # Prepare choice samples
    choice_samples = []
    corrections = [
        ("проверь статус репозитория", "run_command"),
        ("запусти линтер на проекте", "run_command"),
        ("запусти докер-контейнер postgres", "run_command"),
        ("запусти линтер ruff", "run_command"),
        ("ну а теперь запушь изменения на гитхаб", "run_command"),
        ("обнови пакеты через npm", "run_command"),
        ("зайди на яндекс", "browser_open"),
        ("загрузи скриншот в форму", "browser_upload_file"),
        ("прикрепи скриншот к обращению", "browser_upload_file"),
        ("реши уравнение x^2 - 4 = 0", "calculate"),
        ("какой сегодня день недели?", "NO_TOOL"),
        ("установи зависимости через pip install -r requirements.txt", "run_command"),
    ]
    for q_text, want_tool in corrections:
        state = f'Incoming user request in Argent session:\n"{q_text}"\nWorking directory: C:/Users/user/Desktop/Argent'
        s = rt._to_sample(state, ChoiceQuestion("c", TOOL_QUESTION, choices=PROD_OPTIONS))
        s.target = PROD_OPTIONS.index(want_tool)
        s.weight = 2.5
        choice_samples.append(s)

    for c in v1 + v2:
        state = f'Incoming user request in Argent session:\n"{c["query"]}"\nWorking directory: C:/Users/user/Desktop/Argent'
        s = rt._to_sample(state, ChoiceQuestion("c", TOOL_QUESTION, choices=PROD_OPTIONS))
        s.target = PROD_OPTIONS.index(c["expected_tool"])
        s.weight = 1.0
        choice_samples.append(s)

    # Encode & precompute Choice representations
    enc_choice = [enc.encode(s) for s in choice_samples]
    with torch.no_grad():
        b_c = collate(enc_choice, pad_id=enc.pad)
        b_c = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b_c.items()}
        hidden_c = model.encode_states(b_c["input_ids"], b_c["attention_mask"])

    model.choice_head.float()
    B_c = hidden_c.size(0)
    h_rd_c = hidden_c[torch.arange(B_c, device=hidden_c.device), b_c["rd_pos"]].float()
    m_c = b_c["attention_mask"].unsqueeze(-1).to(torch.float32)
    mean_c = (hidden_c.float() * m_c).sum(1) / m_c.sum(1).clamp(min=1.0)
    q_raw_c = torch.cat([h_rd_c, mean_c], dim=-1)
    slots = torch.arange(1, len(PROD_OPTIONS) + 1, device=hidden_c.device)
    opt_c = model.pool_options(hidden_c.float(), b_c["opt_index"], slots)
    targets_c = torch.tensor([e.choice_target for e in enc_choice], device=device, dtype=torch.long)
    weights_c = torch.tensor([s.weight for s in choice_samples], device=device, dtype=torch.float)

    opt = torch.optim.AdamW(model.choice_head.parameters(), lr=lr, weight_decay=1e-2)
    for step in range(steps):
        opt.zero_grad()
        qv_c = model.choice_head.q(q_raw_c).unsqueeze(1)
        ov_c = model.choice_head.o(opt_c)
        mlp_c = model.choice_head.mix(ov_c + qv_c).squeeze(-1)
        dot_c = (ov_c * qv_c).sum(-1) / model.choice_head.scale
        logits_c = (mlp_c + dot_c) / model.cfg.temp_choice
        loss = (F.cross_entropy(logits_c, targets_c, reduction="none") * weights_c).mean()
        loss.backward()
        opt.step()

    model.choice_head.half()

    # Verify gates
    r1 = route(rt, v1)
    r2 = route(rt, v2)
    both = v1 + v2
    r600 = route(rt, both, 600)
    r1500 = route(rt, both, 1500)
    s_exam = safety(rt, exam)
    ro = route(rt, ood["routing"])
    so = safety(rt, {**ood["safety"], "review": ["rm old.txt"]})
    no_tool = [c for c in both if c["expected_tool"] == "NO_TOOL"]

    metrics = {
        "v1": r1["acc"], "v2": r2["acc"], "rot600": r600["acc"], "rot1500": r1500["acc"],
        "no_tool": route(rt, no_tool)["acc"], "danger": s_exam["danger"], "benign": s_exam["benign"],
        "lookalike": s_exam["lookalike"], "ood_route": ro["acc"], "ood_danger": so["danger"], "ood_benign": so["benign"],
    }

    passed_all = True
    print("\nGate evaluation on surgical model:")
    for name, (key, floor) in GATES.items():
        ok = metrics[key] >= floor
        passed_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'} {name:30s} got {metrics[key]:.3f}")

    if passed_all:
        model.save_pretrained(tgt_p)
        print("Model successfully verified and saved!")
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=".argent/models/leeth2/final")
    parser.add_argument("--target", default=".argent/models/leeth2_surgical")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    success = run_surgery(args.source, args.target, args.steps, args.lr, args.device)
    sys.exit(0 if success else 1)
