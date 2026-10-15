# -*- coding: utf-8 -*-
"""scripts/_kaggle_train_choice_v2.py — обучение choice-головы на предрасчитанных фичах.

Фичи = pooled-выход замороженной базы, детерминированы (база в eval, dropout=0),
см. _kaggle_precompute_feats.py. Математически идентично рецепту D4, но в ~50 раз
быстрее: не нужен ни base-форвард, ни transformers, ни интернет.

Лосс: CE(label_smoothing=0.05) + 0.4 * margin(correct, confuser) на docker-строках.
Гейты: val CE +5% | routing acc -1пп | UNFAMILIAR >= 0.99 | docker >= 0.85.
Выход: mmbert-leeth_candidate.pt + отчёт.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IN_KAGGLE = Path("/kaggle/input/leeth-dataset")
IN_LOCAL = Path(".argent/kaggle/dataset_v2")
if not IN_KAGGLE.exists() and Path("/kaggle/input").exists():
    # новый KGAT-CLI монтирует датасеты в /kaggle/input/datasets/<user>/<slug>/
    _hits = list(Path("/kaggle/input").rglob("mmbert-leeth.pt"))
    if _hits:
        IN_KAGGLE = _hits[0].parent
IN = IN_KAGGLE if IN_KAGGLE.exists() else IN_LOCAL
SMOKE = os.environ.get("SMOKE") == "1"

EPOCHS = 3
LR = 5e-5
BATCH = 32
DOCKER_OVERSAMPLE = 12
FEEDBACK_OVERSAMPLE = int(os.environ.get("FEEDBACK_OVERSAMPLE", "0"))  # фидбек: только partial-коррекции, малый оверсэмплинг
BATTERY_OVERSAMPLE = int(os.environ.get("BATTERY_OVERSAMPLE", "8"))  # таргетированные строки по промахам батареи
MARGIN = 1.0
CONTRASTIVE_W = 0.4
SEED = 7
NOISE_SIGMA = float(os.environ.get("NOISE_SIGMA", "0"))  # аугментация фичей для int8-робастности


def log(m):
    print(m, flush=True)


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    summary = {"device": device, "seed": SEED}

    ckpt = torch.load(IN / "mmbert-leeth.pt", map_location="cpu", weights_only=False)
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
    choice_params = [p for n, p in heads.named_parameters() if n.startswith("choice_head")]
    for p in heads.parameters():
        p.requires_grad_(False)
    for p in choice_params:
        p.requires_grad_(True)

    class_to_idx = json.loads((IN / "choice_classes.json").read_text(encoding="utf-8"))
    unfam_idx = class_to_idx["UNFAMILIAR"]

    def load_npz(name):
        z = np.load(IN / f"feat_{name}.npz")
        return (torch.from_numpy(z["feats"].astype(np.float32)),
                torch.from_numpy(z["targets"].astype(np.int64)),
                torch.from_numpy(z["confs"].astype(np.int64)))

    Xc, yc, cc = load_npz("corpus")
    Xe, ye, ce_ = load_npz("extra")
    Xd, yd, cd = load_npz("docker")
    fb_path = IN / "feat_feedback.npz"
    if fb_path.exists():
        zf = np.load(fb_path)
        Xf = torch.from_numpy(zf["feats"].astype(np.float32))
        yf = torch.from_numpy(zf["targets"].astype(np.int64))
        cf = torch.from_numpy(zf["confs"].astype(np.int64))
    else:
        Xf = yf = cf = None
    bat_path = IN / "feat_battery.npz"
    if bat_path.exists():
        zb = np.load(bat_path)
        Xb = torch.from_numpy(zb["feats"].astype(np.float32))
        yb = torch.from_numpy(zb["targets"].astype(np.int64))
        cb = torch.from_numpy(zb["confs"].astype(np.int64))
    else:
        Xb = yb = cb = None
    Xv, yv, _ = load_npz("val")
    Xdv, ydv, cdv = load_npz("docker_val")
    routing_mask = torch.from_numpy(np.load(IN / "feat_val.npz")["routing_mask"])
    summary["data"] = {"corpus": len(Xc), "extra": len(Xe), "docker": len(Xd),
                       "docker_x": DOCKER_OVERSAMPLE, "val": len(Xv),
                       "val_routing": int(routing_mask.sum()), "docker_val": len(Xdv),
                       "feedback": 0 if Xf is None else len(Xf),
                       "feedback_x": FEEDBACK_OVERSAMPLE if Xf is not None else 0,
                       "battery": 0 if Xb is None else len(Xb),
                       "battery_x": BATTERY_OVERSAMPLE if Xb is not None else 0}
    log(f"data: {summary['data']}")

    parts_x = [Xc, Xe] + [Xd] * DOCKER_OVERSAMPLE
    parts_y = [yc, ye] + [yd] * DOCKER_OVERSAMPLE
    parts_c = [torch.full((len(Xc),), -100, dtype=torch.int64),
               torch.full((len(Xe),), -100, dtype=torch.int64)] + [cd] * DOCKER_OVERSAMPLE
    if Xf is not None and FEEDBACK_OVERSAMPLE > 0:
        parts_x += [Xf] * FEEDBACK_OVERSAMPLE
        parts_y += [yf] * FEEDBACK_OVERSAMPLE
        parts_c += [cf] * FEEDBACK_OVERSAMPLE
    if Xb is not None and BATTERY_OVERSAMPLE > 0:
        parts_x += [Xb] * BATTERY_OVERSAMPLE
        parts_y += [yb] * BATTERY_OVERSAMPLE
        parts_c += [cb] * BATTERY_OVERSAMPLE
    X = torch.cat(parts_x)
    y = torch.cat(parts_y)
    c = torch.cat(parts_c)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))
    X, y, c = X[perm], y[perm], c[perm]
    if SMOKE:
        dmask = (c >= 0)
        keep = torch.cat([torch.where(dmask)[0][:24], torch.where(~dmask)[0][:40]])
        X, y, c = X[keep], y[keep], c[keep]
    X, y, c = X.to(device), y.to(device), c.to(device)
    log(f"mix: {len(X)} rows")

    Xv, yv = Xv.to(device), yv.to(device)
    Xdv, ydv, cdv = Xdv.to(device), ydv.to(device), cdv.to(device)
    rv = routing_mask.to(device)

    def eval_subset(Xs, ys):
        heads.eval()
        with torch.no_grad():
            logits = heads.choice_head(Xs)
        m = ys != -100
        ce = F.cross_entropy(logits[m], ys[m], label_smoothing=0.05).item()
        acc = (logits[m].argmax(-1) == ys[m]).float().mean().item()
        um = ys == unfam_idx
        unfam = (logits[um].argmax(-1) == unfam_idx).float().mean().item() if um.any() else float("nan")
        heads.train()
        return ce, acc, unfam

    def metrics(tag):
        a = eval_subset(Xv, yv)
        r = eval_subset(Xv[rv], yv[rv])
        d = eval_subset(Xdv, ydv)
        summary[tag] = {"val_ce": a[0], "val_acc": a[1], "unfam_acc": a[2],
                        "routing_acc": r[1], "docker_acc": d[1]}
        log(f"{tag}: val_ce={a[0]:.4f} val_acc={a[1]:.4f} unfam={a[2]:.4f} "
            f"routing={r[1]:.4f} docker={d[1]:.4f}")
        return summary[tag]

    before = metrics("before")

    opt = torch.optim.AdamW(choice_params, lr=LR, weight_decay=0.01)
    heads.train()
    for epoch in range(1, EPOCHS + 1):
        te = time.time()
        perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED + epoch))
        total, steps, correct, count = 0.0, 0, 0, 0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb, cb = X[idx], y[idx], c[idx]
            if NOISE_SIGMA > 0:
                xb = xb + NOISE_SIGMA * torch.randn_like(xb)
            m = yb != -100
            logits = heads.choice_head(xb)
            ce = F.cross_entropy(logits[m], yb[m], label_smoothing=0.05)
            dm = m & (cb >= 0) & (cb != yb)
            if dm.any():
                pos = logits[dm].gather(1, yb[dm].unsqueeze(1)).squeeze(1)
                negl = logits[dm].gather(1, cb[dm].unsqueeze(1)).squeeze(1)
                contrast = F.relu(MARGIN - (pos - negl)).mean()
            else:
                contrast = logits.new_zeros(())
            loss = ce + CONTRASTIVE_W * contrast
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(choice_params, max_norm=1.0)
            opt.step()
            total += loss.item()
            steps += 1
            correct += (logits[m].argmax(-1) == yb[m]).sum().item()
            count += m.sum().item()
        log(f"epoch {epoch}/{EPOCHS}: loss={total / max(1, steps):.4f} "
            f"train_acc={correct / max(1, count):.4f} ({time.time() - te:.0f}s)")

    after = metrics("after")

    gates, reasons = True, []
    if after["val_ce"] > before["val_ce"] * 1.05:
        gates = False
        reasons.append(f"val CE +{100 * (after['val_ce'] / before['val_ce'] - 1):.1f}% > +5%")
    if after["routing_acc"] < before["routing_acc"] - 0.01:
        gates = False
        reasons.append(f"routing acc -{100 * (before['routing_acc'] - after['routing_acc']):.1f} п.п.")
    if after["unfam_acc"] < 0.99:
        gates = False
        reasons.append(f"UNFAMILIAR acc {after['unfam_acc']:.4f} < 0.99")
    if after["docker_acc"] < 0.80:
        gates = False
        reasons.append(f"docker acc {after['docker_acc']:.4f} < 0.80")
    summary["gate_ok"] = gates
    summary["gate_reasons"] = reasons
    summary["elapsed_s"] = round(time.time() - t0, 1)
    log("Gate: " + ("OK" if gates else "FAIL: " + "; ".join(reasons)))

    out_dir = Path("/kaggle/working") if IN_KAGGLE.exists() else Path(".argent/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"heads": heads.state_dict(), "hidden": hidden, "n_classes": n_classes},
               out_dir / "mmbert-leeth_candidate.pt")
    rep_dir = Path("/kaggle/working") if IN_KAGGLE.exists() else Path(".argent/artifacts")
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "leeth_choice_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DONE in {summary['elapsed_s']}s | gate={'OK' if gates else 'FAIL'}")


if __name__ == "__main__":
    main()