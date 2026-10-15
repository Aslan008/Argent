# -*- coding: utf-8 -*-
"""leeth_choice_kernel.py — ночной прогон на Kaggle: choice-голова + docker + контраст.

Рецепт D4 (_d4_finetune_choice.py) + docker-строки + контрастная маржа (D1).
База заморожена, тренируется только choice_head. Гейты до сохранения:
  - val choice CE: деградация > +5% -> FAIL
  - val routing acc: падение > 1 п.п. -> FAIL
  - val UNFAMILIAR acc: < 0.99 -> FAIL
  - docker val acc: < 0.9 -> FAIL (новый)
Выход: /kaggle/working/mmbert-leeth_candidate.pt + report.json
"""
import json
import os
import random
import sys
import time
from pathlib import Path

IN_DIR = Path("/kaggle/input/leeth-dataset")
LOCAL = Path(".argent/models/mmbert-argent")
SMOKE = os.environ.get("SMOKE") == "1"

if IN_DIR.exists():
    BASE_DIR = IN_DIR
    HEADS_PATH = IN_DIR / "mmbert-leeth.pt"
    DATA = IN_DIR
    OUT = Path("/kaggle/working")
else:  # локальный смоук-тест
    BASE_DIR = LOCAL
    HEADS_PATH = Path(".argent/models/mmbert-leeth.pt")
    DATA = Path("src/leeth/data_400m")
    OUT = Path(".argent/artifacts")

TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
P = "Incoming user request in Argent session:"
EPOCHS = 1 if SMOKE else 2
LR = 5e-5
BATCH = 8 if SMOKE else 32
DOCKER_OVERSAMPLE = 4
MAX_LEN = 256
MARGIN = 1.0
CONTRASTIVE_W = 0.4


def log(msg):
    print(msg, flush=True)


def load_rows(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    t0 = time.time()
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    # transformers >= 4.48 нужна для ModernBERT
    import transformers
    from packaging.version import Version
    if Version(transformers.__version__) < Version("4.48"):
        import subprocess
        log(f"transformers {transformers.__version__} < 4.48 — обновляю...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                        "transformers>=4.48,<5"], check=True)
        import importlib
        importlib.reload(transformers)
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    log(f"transformers {transformers.__version__}, torch {torch.__version__}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "-", "smoke": SMOKE}

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), torch_dtype=torch.float32)
    base.to(device).eval()
    for p in base.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(HEADS_PATH, map_location="cpu", weights_only=False)
    hidden, n_classes = ckpt["hidden"], ckpt["n_classes"]

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

    heads = Heads(hidden, n_classes)
    heads.load_state_dict(ckpt["heads"])
    heads.to(device)
    for name, p in heads.named_parameters():
        p.requires_grad_(name.startswith("choice_head"))

    class_to_idx = json.loads((DATA / "choice_classes.json").read_text(encoding="utf-8"))
    unfam_idx = class_to_idx["UNFAMILIAR"]

    # ── данные ────────────────────────────────────────────────────────────────
    def is_choice(r):
        return r.get("question_type") == "choice" and r.get("target_choice")

    corpus = load_rows(DATA / "leeth_400m_train.jsonl")
    extra = load_rows(DATA / "leeth_400m_train_extra.jsonl")
    docker = load_rows(DATA / "docker_train.jsonl")
    corpus_choice = [r for r in corpus if is_choice(r)]
    extra_choice = [r for r in extra if is_choice(r)]
    mix = corpus_choice + extra_choice + docker * DOCKER_OVERSAMPLE
    random.Random(7).shuffle(mix)
    if SMOKE:
        mix = docker[:16] + mix[:48]  # смоук обязан покрыть контрастный путь
    summary["data"] = {"corpus_choice": len(corpus_choice), "extra": len(extra_choice),
                       "docker": len(docker), "docker_x": DOCKER_OVERSAMPLE, "mix": len(mix)}
    log(f"mix: {len(mix)} (corpus {len(corpus_choice)} + extra {len(extra_choice)} + docker {len(docker)}x{DOCKER_OVERSAMPLE})")

    def collate(batch):
        texts = [r["state"] + "\n[QUESTION] " + r["question"] for r in batch]
        targets = [class_to_idx.get(r["target_choice"], -100) for r in batch]
        confs = [class_to_idx.get(r.get("confuser"), -100) for r in batch]
        enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        return enc, torch.tensor(targets, dtype=torch.long), torch.tensor(confs, dtype=torch.long)

    def pooled_forward(enc):
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = base.base_model(**enc, output_hidden_states=True)
        hs = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return heads.choice_head(pooled)

    def margin_loss(pos, neg, margin=MARGIN):
        if pos.numel() == 0:
            return pos.new_zeros(())
        return F.relu(margin - (pos - neg)).mean()

    def eval_choice(rows_subset):
        loader_eval = DataLoader(rows_subset, batch_size=64, shuffle=False, collate_fn=collate)
        heads.eval()
        total, nb, correct, unfam_ok, unfam_n, count = 0.0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for enc, targets, _conf in loader_eval:
                logits = pooled_forward(enc)
                targets = targets.to(device)
                m = targets != -100
                if not m.any():
                    continue
                total += F.cross_entropy(logits[m], targets[m], label_smoothing=0.05).item()
                nb += 1
                preds = logits[m].argmax(-1)
                correct += (preds == targets[m]).sum().item()
                count += m.sum().item()
                um = targets == unfam_idx
                unfam_ok += (preds[um] == targets[um]).sum().item()
                unfam_n += um.sum().item()
        heads.train()
        return total / max(1, nb), correct / max(1, count), (unfam_ok / unfam_n if unfam_n else float("nan")), count

    val_all = [r for r in load_rows(DATA / "leeth_400m_val.jsonl") if is_choice(r)]
    val_routing = [r for r in val_all if r["question"] == TEMPLATE_Q and r["state"].startswith(P)]
    docker_val = load_rows(DATA / "docker_val.jsonl")
    summary["val"] = {"all": len(val_all), "routing": len(val_routing), "docker": len(docker_val)}

    def eval_metrics(tag):
        a = eval_choice(val_all)
        rt = eval_choice(val_routing)
        d = eval_choice(docker_val) if docker_val else (float("nan"), float("nan"), float("nan"), 0)
        summary[tag] = {"val_ce": a[0], "val_acc": a[1], "unfam_acc": a[2],
                        "routing_acc": rt[1], "docker_acc": d[1]}
        log(f"{tag}: val_ce={a[0]:.4f} val_acc={a[1]:.4f} unfam={a[2]:.4f} "
            f"routing={rt[1]:.4f} docker={d[1]:.4f}")
        return summary[tag]

    before = eval_metrics("before")

    loader = DataLoader(mix, batch_size=BATCH, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad],
                                  lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    heads.train()
    for epoch in range(1, EPOCHS + 1):
        te = time.time()
        total, steps, correct, count = 0.0, 0, 0, 0
        for enc, targets, confs in loader:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = pooled_forward(enc)
                targets = targets.to(device)
                confs = confs.to(device)
                m = targets != -100
                ce = F.cross_entropy(logits[m], targets[m], label_smoothing=0.05)
                dm = m & (confs >= 0) & (confs != targets)  # docker-строки с конфузером
                contrast = margin_loss(logits[dm].gather(1, targets[dm].unsqueeze(1)).squeeze(1),
                                       logits[dm].gather(1, confs[dm].unsqueeze(1)).squeeze(1),
                                       margin=MARGIN) if dm.any() else logits.new_zeros(())
                loss = ce + CONTRASTIVE_W * contrast
            if not torch.isnan(loss) and loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in heads.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total += loss.item()
                steps += 1
                correct += (logits[m].argmax(-1) == targets[m]).sum().item()
                count += m.sum().item()
            if steps % 100 == 0 and steps > 0:
                log(f"  ep{epoch} step {steps}/{len(loader)} loss={total/steps:.4f} "
                    f"acc={correct/max(1,count):.4f} | {time.time()-te:.0f}s")
        log(f"epoch {epoch}/{EPOCHS}: loss={total/max(1,steps):.4f} train_acc={correct/max(1,count):.4f} ({time.time()-te:.0f}s)")
        if not SMOKE:
            torch.save({"heads": heads.state_dict(), "hidden": hidden, "n_classes": n_classes},
                       OUT / f"ckpt_epoch{epoch}.pt")

    after = eval_metrics("after")

    gate_ok = True
    reasons = []
    if after["val_ce"] > before["val_ce"] * 1.05:
        gate_ok = False; reasons.append(f"val CE +{100*(after['val_ce']/before['val_ce']-1):.1f}% > +5%")
    if after["routing_acc"] < before["routing_acc"] - 0.01:
        gate_ok = False; reasons.append(f"routing acc -{100*(before['routing_acc']-after['routing_acc']):.1f} п.п.")
    if not (after["unfam_acc"] == after["unfam_acc"]) or after["unfam_acc"] < 0.99:
        gate_ok = False; reasons.append(f"UNFAMILIAR acc {after['unfam_acc']} < 0.99")
    if not (after["docker_acc"] == after["docker_acc"]) or after["docker_acc"] < 0.9:
        gate_ok = False; reasons.append(f"docker acc {after['docker_acc']} < 0.9")
    summary["gate_ok"] = gate_ok
    summary["gate_reasons"] = reasons
    summary["elapsed_s"] = round(time.time() - t0, 1)
    log("Gate: " + ("OK" if gate_ok else "FAIL: " + "; ".join(reasons)))

    torch.save({"heads": heads.state_dict(), "hidden": hidden, "n_classes": n_classes},
               OUT / "mmbert-leeth_candidate.pt")
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DONE in {summary['elapsed_s']}s | gate={'OK' if gate_ok else 'FAIL'}")


if __name__ == "__main__":
    main()