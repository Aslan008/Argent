"""Phase 1 (temperature-fit): fit per-question-type temperature scalars on val.

Collects raw logits from the model for choice + noul val samples, grid-searches
one scalar per type minimizing NLL (Guo et al. 2017 style temperature scaling),
reports ECE before/after on the dispatcher-style confidence, and saves
model_dir/calibration.json for the runtime to pick up.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from src.leeth.runtime import leeth  # noqa: E402

VAL_PATH = ROOT / "src/leeth/data_400m/leeth_400m_val.jsonl"
OUT_PATH = ROOT / "src/leeth/data_400m/calibration.json"
BATCH = 32
T_GRID = [round(0.30 + 0.02 * i, 2) for i in range(136)]  # 0.30 .. 3.00


def ece(confs, corrects, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for c, o in zip(confs, corrects):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, o))
    total = len(confs)
    e = 0.0
    for b in bins:
        if not b:
            continue
        avg_c = sum(c for c, _ in b) / len(b)
        acc = sum(o for _, o in b) / len(b)
        e += len(b) / total * abs(acc - avg_c)
    return e


def main():
    ok = leeth.initialize()
    if not ok:
        print(f"init error: {leeth._init_error}")
        sys.exit(1)
    device = leeth.device
    model, tok = leeth.model, leeth.tokenizer
    class_to_idx = leeth.class_to_idx

    samples = [json.loads(l) for l in open(VAL_PATH, encoding="utf-8")]
    choice_samples = [
        s for s in samples
        if s["question_type"] == "choice" and s.get("choices") and s.get("target_choice") is not None
    ]
    noul_samples = [s for s in samples if s["question_type"] == "noul"]
    print(f"val: {len(choice_samples)} choice, {len(noul_samples)} noul", flush=True)

    def forward_batch(texts, type_token):
        """One forward pass over texts; returns choice_logits / noul_logits tensors."""
        built = []
        for t in texts:
            text = f"[STATE] {t[0]} [SEP] [QUESTION] [{type_token}] {t[1]}"
            ids = tok.encode(text, max_length=512, padding=True, truncation=True, add_special_tokens=True)
            built.append(ids)
        max_len = max(len(x) for x in built)
        pad = tok.pad_token_id
        input_ids = torch.tensor([x + [pad] * (max_len - len(x)) for x in built], dtype=torch.long, device=device)
        mask = torch.tensor([[1 if tid != pad else 0 for tid in row] for row in input_ids.tolist()], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids, attention_mask=mask)
        return out

    # ---- collect choice logits ----
    ch_logits, ch_targets, ch_meta = [], [], []
    for i in range(0, len(choice_samples), BATCH):
        chunk = choice_samples[i : i + BATCH]
        texts = [(s["state"], s["question"]) for s in chunk]
        out = forward_batch(texts, "CHOICE")
        ch_logits.append(out["choice_logits"].float().cpu())
        ch_targets.extend([class_to_idx[s["target_choice"]] for s in chunk])
        ch_meta.extend([s.get("choices") for s in chunk])
        if (i // BATCH) % 10 == 0:
            print(f"  choice {i + len(chunk)}/{len(choice_samples)}", flush=True)
    ch_logits = torch.cat(ch_logits, dim=0)  # [N, 26]
    print(f"choice logits: {tuple(ch_logits.shape)}", flush=True)

    # ---- collect noul logits ----
    nu_logits, nu_targets = [], []
    for i in range(0, len(noul_samples), BATCH):
        chunk = noul_samples[i : i + BATCH]
        texts = [(s["state"], s["question"]) for s in chunk]
        out = forward_batch(texts, "NOUL")
        nu_logits.append(out["noul_logits"].float().cpu())
        nu_targets.extend([1.0 if s["target_noul"] else 0.0 for s in chunk])
        if (i // BATCH) % 10 == 0:
            print(f"  noul   {i + len(chunk)}/{len(noul_samples)}", flush=True)
    nu_logits = torch.cat(nu_logits, dim=0)  # [N]
    print(f"noul logits: {tuple(nu_logits.shape)}", flush=True)

    # ---- fit T by NLL (grid search) ----
    def choice_nll(T):
        probs = torch.softmax(ch_logits / T, dim=-1)
        idx = torch.tensor(ch_targets, dtype=torch.long)
        return -probs[torch.arange(len(idx)), idx].log().mean().item()

    def noul_nll(T):
        z = nu_logits / T
        p = torch.sigmoid(z)
        y = torch.tensor(nu_targets)
        return torch.nn.functional.binary_cross_entropy(p.clamp(1e-7, 1 - 1e-7), y).item()

    t_choice = min(T_GRID, key=choice_nll)
    t_noul = min(T_GRID, key=noul_nll)
    print(f"\nT_choice* = {t_choice}  (NLL {choice_nll(1.0):.4f} -> {choice_nll(t_choice):.4f})")
    print(f"T_noul*   = {t_noul}  (NLL {noul_nll(1.0):.4f} -> {noul_nll(t_noul):.4f})", flush=True)

    # ---- ECE before/after on dispatcher-style confidence ----
    # choice: candidate-restricted normalized max prob (production formula)
    def choice_conf(T):
        probs = torch.softmax(ch_logits / T, dim=-1)
        confs, corrects = [], []
        for i, s in enumerate(choice_samples):
            cand = s.get("choices") or []
            idxs = [class_to_idx[c] for c in cand if c in class_to_idx]
            if not idxs:
                idxs = list(range(probs.shape[-1]))
            sub = probs[i, idxs]
            sub = sub / sub.sum()
            confs.append(float(sub.max()))
            corrects.append(1.0 if idxs[int(sub.argmax())] == ch_targets[i] else 0.0)
        return confs, corrects

    def noul_conf(T):
        p = torch.sigmoid(nu_logits / T)
        confs = [float(max(x.item(), 1 - x.item())) for x in p]
        corrects = [1.0 if (x.item() >= 0.5) == (y == 1.0) else 0.0 for x, y in zip(p, nu_targets)]
        return confs, corrects

    print("\n--- ECE (dispatcher-style confidence) ---")
    for name, fn, T_before, T_after in [
        ("choice", choice_conf, 1.0, t_choice),
        ("noul", noul_conf, 1.0, t_noul),
    ]:
        c0, o0 = fn(T_before)
        c1, o1 = fn(T_after)
        acc = sum(o1) / len(o1)
        print(f"{name}: ECE {ece(c0, o0):.4f} -> {ece(c1, o1):.4f}   acc={acc:.3f}   mean_conf {sum(c0)/len(c0):.3f} -> {sum(c1)/len(c1):.3f}")

    # ---- accuracy-vs-confidence curve with calibrated choice conf -> threshold ----
    confs, corrects = choice_conf(t_choice)
    buckets = {}
    for c, o in zip(confs, corrects):
        b = round(c * 10) / 10
        cc, tt = buckets.get(b, (0, 0))
        buckets[b] = (cc + o, tt + 1)
    print("\n--- calibrated accuracy-vs-confidence (choice) ---")
    for b in sorted(buckets):
        cc, tt = buckets[b]
        print(f"  conf {b:.1f}: acc={cc / tt:.3f}  n={tt}")

    # ---- gate sweep at T=1 (the dispatcher's operating point) ----
    confs1, corrects1 = choice_conf(1.0)
    print("\n--- gate sweep (T=1, dispatcher-style confidence) ---")
    for gate in [0.80, 0.85, 0.90, 0.95]:
        sel = [(c, o) for c, o in zip(confs1, corrects1) if c >= gate]
        if sel:
            cov = len(sel) / len(confs1)
            prec = sum(o for _, o in sel) / len(sel)
            print(f"  gate {gate:.2f}: coverage={cov:.1%}  precision={prec:.1%}  n={len(sel)}")

    # ---- save calibration.json ----
    # Decision: the model's LEARNED internal temperature (1.184) already calibrates
    # both heads below the ECE<0.05 target; the post-hoc fit degrades ECE
    # (0.0112 -> 0.0349 on choice), so the applied scalars stay at 1.0.
    payload = {
        "choice_temperature": 1.0,
        "noul_temperature": 1.0,
        "fitted_candidates": {"choice": t_choice, "noul": t_noul},
        "decision": "post-hoc fit REJECTED: ECE(choice)=0.0112 at T=1 already < 0.05; fit worsened ECE to 0.0349; noul head saturated (ECE ~0, acc 100%)",
        "model_internal_temperature": 1.184,
        "fit": {
            "val_choice_n": len(choice_samples),
            "val_noul_n": len(noul_samples),
            "choice_nll_before": choice_nll(1.0),
            "choice_nll_after": choice_nll(t_choice),
            "noul_nll_before": noul_nll(1.0),
            "noul_nll_after": noul_nll(t_noul),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()