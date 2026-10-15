"""Grow a trained Leeth 2 checkpoint into a larger one WITHOUT changing a single output.

Why: the model was fine-tuned for days on Kaggle; throwing those weights away to get more capacity would throw the days away.
Instead the new capacity is added in the two places where a residual transformer lets it be inert at first:

  * **more layers** -- blocks of three layers (the global/local attention pattern repeats every third layer, so the layers
    after a block keep their type) are inserted as copies of the three layers before them, with the two output projections
    of every copy (``attn.Wo``, ``mlp.Wo``) set to zero.  A pre-norm residual layer is ``x + attn(norm(x))`` then
    ``x + mlp(norm(x))``; with zero output projections it is the identity, so the grown network computes exactly what the
    small one did.
  * **a wider GLU** -- each layer's MLP gets extra hidden units whose output weights are zero, which add nothing to the sum.

Training then moves the zero weights away from zero (their gradients are not zero), and the new capacity starts to work.
``verify_equivalent`` proves the claim on a real checkpoint (max difference of every head's output).

    python -m leeth2.grow --src run/final --out grown --blocks 5,11,17,20 --mlp-mult 2
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

GROUP = 3                                       # ModernBERT: every third layer attends globally
_LAYER_KEY = re.compile(r"^backbone\.layers\.(\d+)\.(.+)$")


def plan_blocks(n_layers: int, after: Sequence[int]) -> List[int]:
    """Validate the insertion points: a block goes after layer ``a`` (a % 3 == 2) and copies layers a-2..a (never layer 0)."""
    out = sorted(set(int(a) for a in after))
    for a in out:
        if a % GROUP != GROUP - 1:
            raise ValueError(f"a block must follow a layer whose index is 2 mod 3 (global, local, local); got {a}")
        if not (GROUP - 1 <= a < n_layers) or a - (GROUP - 1) < 1:
            raise ValueError(f"layer {a} cannot be the end of a copied block (layer 0 has no attention norm, so it is never copied)")
    return out


def layer_order(n_layers: int, after: Sequence[int]) -> List[Tuple[int, bool]]:
    """New layer list as (source index, is_copy): the originals in order, a copied block after each insertion point."""
    ins = set(plan_blocks(n_layers, after))
    order: List[Tuple[int, bool]] = []
    for i in range(n_layers):
        order.append((i, False))
        if i in ins:
            order.extend((j, True) for j in range(i - (GROUP - 1), i + 1))
    return order


def grow_state(sd: Dict[str, torch.Tensor], n_layers: int, intermediate: int, after: Sequence[int], mlp_mult: float = 1.0,
               seed: int = 0, new_unit_std: Optional[float] = None) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """New state dict (backbone layers renumbered, MLPs widened) plus the new (layer count, intermediate size)."""
    gen = torch.Generator().manual_seed(seed)
    new_inter = int(round(intermediate * mlp_mult)) if mlp_mult != 1.0 else intermediate
    if new_inter < intermediate:
        raise ValueError("the MLP can only grow")
    order = layer_order(n_layers, after)
    out: Dict[str, torch.Tensor] = {k: v for k, v in sd.items() if not _LAYER_KEY.match(k)}

    by_layer: Dict[int, Dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        m = _LAYER_KEY.match(k)
        if m:
            by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = v
    if sorted(by_layer) != list(range(n_layers)):
        raise ValueError(f"checkpoint has layers {sorted(by_layer)}, expected 0..{n_layers - 1}")

    for new_i, (src, is_copy) in enumerate(order):
        for name, w in by_layer[src].items():
            w = w.clone()
            if is_copy and name in ("attn.Wo.weight", "mlp.Wo.weight"):
                w = torch.zeros_like(w)                              # the copy starts as the identity
            if new_inter != intermediate:
                if name == "mlp.Wi.weight":                          # [input half ; gate half] x hidden
                    h = w.shape[1]
                    extra = new_inter - intermediate
                    std = new_unit_std if new_unit_std is not None else float(w.float().std())
                    a, b = w[:intermediate], w[intermediate:]
                    na = (torch.randn(extra, h, generator=gen) * std).to(w.dtype)
                    nb = (torch.randn(extra, h, generator=gen) * std).to(w.dtype)
                    w = torch.cat([a, na, b, nb], dim=0)
                elif name == "mlp.Wo.weight":                        # hidden x intermediate: new columns are zero
                    w = torch.cat([w, torch.zeros(w.shape[0], new_inter - intermediate, dtype=w.dtype)], dim=1)
            out[f"backbone.layers.{new_i}.{name}"] = w
    return out, len(order), new_inter


def grow_config(backbone: Dict, n_layers: int, intermediate: int) -> Dict:
    cfg = copy.deepcopy(backbone)
    cfg["num_hidden_layers"] = n_layers
    cfg["intermediate_size"] = intermediate
    every = int(cfg.get("global_attn_every_n_layers", GROUP) or GROUP)
    if every != GROUP:
        raise ValueError(f"this growth keeps blocks of {GROUP} layers but the backbone has global_attn_every_n_layers={every}")
    if "layer_types" in cfg and cfg["layer_types"] is not None:
        cfg["layer_types"] = ["full_attention" if i % every == 0 else "sliding_attention" for i in range(n_layers)]
    return cfg


def grow_checkpoint(src: Path, out: Path, after: Sequence[int], mlp_mult: float = 1.0, seed: int = 0) -> Dict[str, float]:
    """Read ``src`` (a Leeth 2 ``final`` dir), write the grown model to ``out``. Returns parameter counts."""
    from safetensors.torch import load_file, save_file

    cfg = json.loads((src / "leeth2_config.json").read_text(encoding="utf-8"))
    bb = cfg["backbone"]
    sd = load_file(str(src / "model.safetensors"))
    new_sd, n_layers, inter = grow_state(sd, int(bb["num_hidden_layers"]), int(bb["intermediate_size"]), after, mlp_mult, seed)
    cfg["backbone"] = grow_config(bb, n_layers, inter)
    cfg["version"] = f"{cfg.get('version', 'leeth2')}+grown"
    cfg["grown"] = {"new_layers": [i for i, (_, copy_) in enumerate(layer_order(int(bb["num_hidden_layers"]), after)) if copy_],
                    "from_layers": int(bb["num_hidden_layers"]), "from_intermediate": int(bb["intermediate_size"])}
    out.mkdir(parents=True, exist_ok=True)
    (out / "leeth2_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in new_sd.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in src.iterdir():                                           # tokenizer files travel with the model
        if f.is_file() and f.name not in ("model.safetensors", "leeth2_config.json"):
            (out / f.name).write_bytes(f.read_bytes())
    total = sum(v.numel() for v in new_sd.values())
    return {"layers": n_layers, "intermediate": inter, "params_M": round(total / 1e6, 1), "src_params_M": round(sum(v.numel() for v in sd.values()) / 1e6, 1)}


def is_grown(path: Path) -> bool:
    """Was this model grown?  (``path``: a Leeth 2 ``final`` dir, or a run dir holding one.)  A grown start model has to be built from its
    own config: the trainer's default builds the original network, whose layers the grown weights no longer line up with."""
    for cand in (Path(path), Path(path) / "final"):
        f = cand / "leeth2_config.json"
        if f.exists():
            return bool(json.loads(f.read_text(encoding="utf-8")).get("grown"))
    return False


@torch.no_grad()
def verify_equivalent(src: Path, grown: Path, n: int = 6, seq: int = 96, device: str = "cpu") -> Dict[str, float]:
    """Max |difference| of every head output between the small and the grown model on random token batches."""
    from .model import Leeth2Model

    a = Leeth2Model.from_pretrained(src, device=device).eval()
    b = Leeth2Model.from_pretrained(grown, device=device).eval()
    g = torch.Generator().manual_seed(1)
    V = a.cfg.vocab_size
    ids = torch.randint(5, min(V, 5000), (n, seq), generator=g)
    ids[:, 10:12] = V + 1                                            # some marker tokens
    mask = torch.ones(n, seq, dtype=torch.long)
    mask[n // 2:, seq - 20:] = 0
    opt = torch.zeros(n, seq, dtype=torch.long)
    opt[:, 20:30], opt[:, 40:50] = 1, 2
    kw = dict(input_ids=ids.to(device), attention_mask=mask.to(device), opt_index=opt.to(device), rd_pos=torch.full((n,), 5, device=device),
              state_mask=(mask.bool() & (opt == 0)).to(device), n_opts=torch.full((n,), 2, device=device), k_max=2)
    ra, rb = a(**kw), b(**kw)
    return {k: float((ra[k].float() - rb[k].float()).abs().max()) for k in ra if torch.isfinite(ra[k]).all()}


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="5,11,17,20", help="insert a copied 3-layer block after each of these layers")
    ap.add_argument("--mlp-mult", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    info = grow_checkpoint(Path(a.src), Path(a.out), [int(x) for x in a.blocks.split(",") if x], a.mlp_mult, a.seed)
    print(json.dumps(info))
    if a.verify:
        print(json.dumps(verify_equivalent(Path(a.src), Path(a.out))))


if __name__ == "__main__":
    main()
