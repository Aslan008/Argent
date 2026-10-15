# -*- coding: utf-8 -*-
"""Content-question head of Leeth 2 (the `noul_content` experiment): features -> fit -> candidate directory.

The head is a gated residual mixture on top of the installed yes/no read-out (see Leeth2.forward, flag `noul_content`):
    z = (1-g)*z_old + g*(a*z_old + b + delta(q)),   g = sigmoid(gate(q)),   q = [h_RD, mean-pool]
It is trained on frozen-encoder features (CPU, ~50 pairs/s), against an anchor of old-distribution noul rows with true
labels, safety5 / ordinary generators, and a gate that must stay shut on them -- so commands and safety behave as installed.

VERDICT (2026-10-14, report `reports/leeth_2_1_triage_and_plan_2026-10-13.md`, section 10): NOT installed. Safety is
unchanged to float rounding, but on topics unseen in training the head misses more positives with confidence than the
installed head does (bench3: sure-no on positives 2.5% -> 17%). The tool-level example cut (`src/leeth/bulk.py`) gives the
recall gain without touching the model. Kept for A/B (`LEETH2_MODEL_DIR=<out>/final`) and for round 5, where the
content questions must enter the encoder training instead of a head on frozen features.

    python scripts/leeth_content_head.py extract --model .argent/models/leeth2/final --out work/content_feats --mix-dir <kaggle mix dir>
    python scripts/leeth_content_head.py fit     --feats work/content_feats --model .argent/models/leeth2/final --out work/content_head.pt
    python scripts/leeth_content_head.py build   --weights work/content_head.pt --model .argent/models/leeth2/final --out .argent/models/leeth2_content_cand/final
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

T_OLD_DEFAULT = None  # read from the model's leeth2_config.json (temp_noul)


# ---------------------------------------------------------------- features
def _qfeats(rt, samples, batch: int = 24):
    """Frozen-encoder features of the noul head for every sample: q = [read-out hidden, mean-pool] (2H) and the old logit."""
    import torch

    from src.leeth2.encoding import collate

    model, enc = rt.model, rt.encoder
    model.eval()
    encoded = [enc.encode(s, rng=None, want_offsets=False) for s in samples]
    order = sorted(range(len(samples)), key=lambda i: len(encoded[i]))
    out = np.zeros((len(samples), 2 * model.cfg.hidden_size), dtype=np.float32)
    logit = np.zeros(len(samples), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(order), batch):
            idx = order[a: a + batch]
            b = collate([encoded[k] for k in idx], pad_id=enc.pad)
            ids, mask, rd = b["input_ids"], b["attention_mask"], b["rd_pos"]
            hidden = model.encode_states(ids, mask)
            B = hidden.size(0)
            h_rd = hidden[torch.arange(B), rd]
            m = mask.unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
            q = torch.cat([h_rd, mean], dim=-1).float()
            lg = model.noul_head(q.to(next(model.noul_head.parameters()).dtype)).squeeze(-1).float()
            for r, k in enumerate(idx):
                out[k] = q[r].numpy()
                logit[k] = float(lg[r])
    return out, logit


def _save(path: Path, rt, samples) -> None:
    X, L = _qfeats(rt, samples)
    y = np.array([float(s.target) for s in samples], dtype=np.float32)
    np.savez_compressed(path, X=X, L=L, y=y)
    print(f"{path.name}: {X.shape}", flush=True)


def _noul_rows(path: Path, n: int, seed: int, skip: int = 0):
    from src.leeth2.schema import Sample

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"ptype":"noul"' in line or '"ptype": "noul"' in line:
                d = json.loads(line)
                if d.get("ptype") == "noul":
                    rows.append(d)
    random.Random(seed).shuffle(rows)
    rows = rows[skip: skip + n]
    return [Sample(d["task"], d.get("family", "x"), d.get("lang", "mix"), "noul", d["state"], d["question"], None,
                   float(d["target"]), meta={"task0": d["task"]}) for d in rows]


def cmd_extract(a) -> None:
    from src.leeth2.data.synth import content as C
    from src.leeth2.data.synth import generate
    from src.leeth2.data.synth import safety5 as S5
    from src.leeth2.runtime import Leeth2Runtime

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rt = Leeth2Runtime.load(Path(a.model))
    _save(out / "content_train.npz", rt, C.build("train", seed=11))
    _save(out / "content_held.npz", rt, C.build("held", seed=12))
    mix = Path(a.mix_dir)
    _save(out / "old_dev.npz", rt, _noul_rows(mix / "dev.jsonl", 3000, 1))
    _save(out / "old_test.npz", rt, _noul_rows(mix / "test.jsonl", 4000, 2))
    _save(out / "old_anchor.npz", rt, _noul_rows(mix / "test.jsonl", 9000, 2, skip=4000))  # disjoint from old_test
    ss = [s for s in generate(S5.gen_safety5, 3500, seed=20261014) if s.ptype == "noul"][:6000]
    ss += [s for s in generate(S5.gen_ordinary, 4000, seed=20261015) if s.ptype == "noul"][:3000]
    _save(out / "safety_anchor.npz", rt, ss)
    held = [s for s in generate(S5.gen_safety5_held, 700, seed=777001) if s.ptype == "noul"][:1500]
    _save(out / "safety_held.npz", rt, held)


# ---------------------------------------------------------------- fit
def _auc(z, pos) -> float:
    P, N = z[pos], z[~pos]
    return float(((P[:, None] > N[None, :]) + 0.5 * (P[:, None] == N[None, :])).mean())


def _load(feats: Path, name: str):
    d = np.load(feats / f"{name}.npz")
    return d["X"], d["L"], d["y"]


def cmd_fit(a) -> None:
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    feats = Path(a.feats)
    Xc, _, yc = _load(feats, "content_train")
    Xh, _, yh = _load(feats, "content_held")
    anchors = [_load(feats, n) for n in ("old_dev", "old_anchor")]
    Xs, Ls, ys = _load(feats, "safety_anchor")
    k = int(len(Xs) * 0.85)
    XA = np.concatenate([anchors[0][0], anchors[1][0], Xs[:k]])
    LA = np.concatenate([anchors[0][1], anchors[1][1], Ls[:k]])
    yA = np.concatenate([anchors[0][2], anchors[1][2], ys[:k]])

    model_dir = Path(a.model)
    cfg = json.loads((model_dir / "leeth2_config.json").read_text(encoding="utf-8"))
    t_old = float(cfg["temp_noul"])
    sd = load_file(str(model_dir / "model.safetensors"))
    W = {k.replace("noul_head.", ""): v.float() for k, v in sd.items() if k.startswith("noul_head.")}
    dim = Xc.shape[1]
    T = torch.tensor

    class Res(torch.nn.Module):
        """Delta has a zero last layer: the content expert starts as the installed head, the gate starts shut."""

        def __init__(self, delta_hidden=256, gate_hidden=64):
            super().__init__()
            self.old = torch.nn.Sequential(torch.nn.Linear(dim, W["0.weight"].shape[0]), torch.nn.GELU(),
                                           torch.nn.Dropout(0.0), torch.nn.Linear(W["0.weight"].shape[0], 1))
            self.old.load_state_dict({"0.weight": W["0.weight"], "0.bias": W["0.bias"],
                                      "3.weight": W["3.weight"], "3.bias": W["3.bias"]})
            self.old.requires_grad_(False)
            self.delta = torch.nn.Sequential(torch.nn.Linear(dim, delta_hidden), torch.nn.GELU(), torch.nn.Linear(delta_hidden, 1))
            torch.nn.init.zeros_(self.delta[2].weight)
            torch.nn.init.zeros_(self.delta[2].bias)
            self.a = torch.nn.Parameter(torch.tensor(1.0))
            self.b = torch.nn.Parameter(torch.tensor(0.0))
            self.gate = torch.nn.Sequential(torch.nn.Linear(dim, gate_hidden), torch.nn.GELU(), torch.nn.Linear(gate_hidden, 1))
            torch.nn.init.constant_(self.gate[2].bias, -4.0)

        def forward(self, x, parts=False):
            zo = self.old(x).squeeze(-1)
            zc = self.a * zo + self.b + self.delta(x).squeeze(-1)
            g = torch.sigmoid(self.gate(x).squeeze(-1))
            z = (1 - g) * zo + g * zc
            return (z, g, zc, zo) if parts else z

    torch.manual_seed(a.seed)
    m = Res()
    params = [p for n, p in m.named_parameters() if not n.startswith("old.")]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    Xc_t, yc_t, XA_t, yA_t = T(Xc), T(yc), T(XA), T(yA)
    bs = 128
    clamp = lambda g: g.clamp(1e-6, 1 - 1e-6)  # noqa: E731
    for _ in range(a.steps):
        ic = torch.randint(0, len(Xc_t), (bs,))
        ia = torch.randint(0, len(XA_t), (bs,))
        zc, gc, zcc, zoc = m(Xc_t[ic], parts=True)
        za, ga, _, _ = m(XA_t[ia], parts=True)
        loss = (F.binary_cross_entropy_with_logits(zc, yc_t[ic]) + F.binary_cross_entropy_with_logits(za, yA_t[ia])
                + F.binary_cross_entropy(clamp(gc), torch.ones_like(gc))     # the gate opens on content questions...
                + F.binary_cross_entropy(clamp(ga), torch.zeros_like(ga))    # ...and stays shut on everything else
                + a.lam_delta * 0.1 * ((zcc - zoc) ** 2).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        zh = m(T(Xh)).numpy()
    ts = np.arange(0.3, 2.01, 0.025)
    Tc = float(min(ts, key=lambda t: float(F.binary_cross_entropy_with_logits(T(zh / t), T(yh)).item())))
    s = t_old / Tc  # the runtime divides by temp_noul: fold the content temperature into the expert
    st = {k: v.clone() for k, v in m.state_dict().items() if not k.startswith("old.")}
    out = {"content_delta.0.weight": st["delta.0.weight"], "content_delta.0.bias": st["delta.0.bias"],
           "content_delta.2.weight": st["delta.2.weight"] * s, "content_delta.2.bias": st["delta.2.bias"] * s,
           "content_scale": (st["a"] * s).reshape(()), "content_bias": (st["b"] * s).reshape(()),
           "content_gate.0.weight": st["gate.0.weight"], "content_gate.0.bias": st["gate.0.bias"],
           "content_gate.2.weight": st["gate.2.weight"], "content_gate.2.bias": st["gate.2.bias"]}
    torch.save(out, a.out)
    pos = yh > 0.5
    p = 1 / (1 + np.exp(-zh / Tc))
    print(f"T_content={Tc:.3f} fold={s:.3f}; held: recall={float((p[pos] >= .7).mean()):.3f} "
          f"fp={float((p[~pos] >= .7).mean()):.3f} auc={_auc(zh, pos):.3f}; saved {a.out}")


# ---------------------------------------------------------------- build
def cmd_build(a) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    src, dst = Path(a.model), Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != "model.safetensors":
            shutil.copy2(f, dst / f.name)
    sd = load_file(str(src / "model.safetensors"))
    for k, v in torch.load(a.weights).items():
        sd[k] = v.to(torch.float16).contiguous() if v.dim() > 0 else v.to(torch.float32).reshape(()).contiguous()
    save_file(sd, str(dst / "model.safetensors"))
    cfg = json.loads((dst / "leeth2_config.json").read_text(encoding="utf-8"))
    cfg["noul_content"] = True
    (dst / "leeth2_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"candidate: {dst} (temp_noul {cfg['temp_noul']} unchanged); A/B with LEETH2_MODEL_DIR={dst}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--mix-dir", required=True, help="dir with dev.jsonl / test.jsonl of the mixed corpus")
    f = sub.add_parser("fit")
    f.add_argument("--feats", required=True)
    f.add_argument("--model", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--steps", type=int, default=3000)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--lam-delta", type=float, default=1.0)
    f.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("build")
    b.add_argument("--weights", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.modules.setdefault("_wmi", None)  # platform.uname() goes through WMI, which hangs here
    {"extract": cmd_extract, "fit": cmd_fit, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
