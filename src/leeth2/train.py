"""DDP trainer (Kaggle 2x T4, fp16) — also runs on one CPU process for tests.

    torchrun --nproc_per_node=2 -m leeth2.train --config train_config.json

What it does that a bare loop would not:
  * embeddings frozen, layer-wise LR decay, no weight decay on norms/biases/markers
  * length-bucketed batches under a padded-token budget (T4 has 15 GB, no bf16)
  * EMA of the trainable weights — evaluation and the final model use the EMA
  * measures its own speed early and shrinks the schedule to fit the wall-clock budget, so a
    12-hour session always ends with a finished (decayed) learning rate, never a cut-off one
  * resumable: ``state.pt`` holds weights, optimiser, scaler, EMA and the data position
  * OOM on one rank is survivable: the step is dropped on every rank, not hung
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from . import calibrate, evaluate
from .datafile import JsonlReader, exact_token_lengths, plan_batches, read_all
from .encoding import Leeth2Encoder, collate
from .losses import compute_losses
from .model import Leeth2Model


@dataclass
class TrainConfig:
    base: str = "jhu-clsp/mmBERT-base"
    data_dir: str = "/kaggle/input/leeth2-corpus"
    out_dir: str = "/kaggle/working/run"
    train_file: str = "train.jsonl"
    dev_file: str = "dev.jsonl"
    calib_file: str = "calib.jsonl"
    max_len: int = 512
    calib_max_len: int = 0  # > 0: the temperatures are fitted with THIS window (the run-time one, 4096) instead of max_len: long option lists are squeezed less
    tokens_per_batch: int = 10000
    max_batch: int = 256
    grad_accum: int = 2
    lr: float = 6e-5
    head_lr_mult: float = 8.0
    llrd: float = 0.92
    new_lr_mult: float = 1.0  # a grown model: the layers that were inserted (and start as identities) learn this many times faster
    weight_decay: float = 0.01
    warmup_frac: float = 0.04
    min_lr_frac: float = 0.05
    epochs: float = 1.0
    max_steps: int = 0  # 0 = derive from epochs
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    eval_every: int = 1500
    save_every: int = 1500
    log_every: int = 25
    time_budget_s: float = 10.5 * 3600
    seed: int = 1234
    amp: bool = True
    grad_ckpt: bool = False
    workers: int = 1
    len_workers: int = 4  # processes used once at start-up to measure every sample's true token length
    attn: str = "sdpa"
    brier: float = 0.25
    dev_cap: int = 6000
    resume: bool = True
    resume_from: str = ""
    arch_from: str = ""  # build the network from this saved model's config instead of the hub base (a GROWN model: leeth2.grow); weights still come from init_from
    init_from: str = ""  # start a NEW run (fresh optimiser, schedule, data position) from a finished model: a run dir with state.pt (EMA weights) or a final/ dir
    lam_choice: float = 1.0
    lam_noul: float = 1.0
    lam_score: float = 0.7
    lam_extract: float = 1.0
    lam_brier_choice: float = 0.0  # >0: add the multi-class Brier score of the option distribution (see losses.choice_brier)


# ----------------------------------------------------------------------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world() -> int:
    return dist.get_world_size() if is_dist() else 1


def log(msg: str, out: Optional[Path] = None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def jlog(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, reader: JsonlReader, batches: List[List[int]], encoder: Leeth2Encoder, seed: int):
        self.reader, self.batches, self.encoder, self.seed = reader, batches, encoder, seed

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, b: int) -> Dict[str, Any]:
        rng = random.Random(self.seed * 1_000_003 + b)
        enc = [self.encoder.encode(self.reader.get(i), rng=rng) for i in self.batches[b]]
        return collate(enc, pad_id=self.encoder.pad)


class EMA:
    def __init__(self, params: Dict[str, torch.nn.Parameter], decay: float):
        self.decay = decay
        self.shadow = {k: p.detach().clone().float() for k, p in params.items()}
        self.params = params

    @torch.no_grad()
    def update(self, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        names = list(self.shadow)
        sh = [self.shadow[n] for n in names]
        cur = [self.params[n].detach().float() for n in names]
        torch._foreach_mul_(sh, d)
        torch._foreach_add_(sh, cur, alpha=1.0 - d)

    @contextlib.contextmanager
    def swapped(self):
        """Temporarily load the EMA weights into the live parameters."""
        backup = {k: p.detach().clone() for k, p in self.params.items()}
        with torch.no_grad():
            for k, p in self.params.items():
                p.copy_(self.shadow[k].to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for k, p in self.params.items():
                    p.copy_(backup[k])


def build_param_groups(model: Leeth2Model, cfg: TrainConfig) -> List[Dict[str, Any]]:
    bb = model.backbone
    n_layers = len(bb.layers)
    groups: Dict[tuple, Dict[str, Any]] = {}

    def add(name: str, p: torch.nn.Parameter, lr: float) -> None:
        nodecay = p.ndim < 2 or name.endswith(".bias") or "norm" in name or name.startswith("marker_embed")
        key = (round(lr, 12), nodecay)
        g = groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": 0.0 if nodecay else cfg.weight_decay, "base_lr": lr})
        g["params"].append(p)

    new_layers = set((model.cfg.grown or {}).get("new_layers", []))
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone.layers."):
            li = int(name.split(".")[2])
            lr = cfg.lr * (cfg.llrd ** (n_layers - 1 - li))
            if li in new_layers:
                lr *= cfg.new_lr_mult
        elif name.startswith("backbone.embeddings"):
            lr = cfg.lr * (cfg.llrd**n_layers)
        elif name.startswith("backbone."):
            lr = cfg.lr  # final norm
        else:
            lr = cfg.lr * cfg.head_lr_mult
        add(name, p, lr)
    return list(groups.values())


def lr_factor(step: int, total: int, warmup: int, min_frac: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * prog))


def atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def sync_flag(flag: bool, device: torch.device) -> bool:
    """True on every rank if it is True on any rank."""
    if not is_dist():
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


def load_init(model: Leeth2Model, path: Path) -> str:
    """Weights of a finished run as the starting point of a fine-tune; returns a one-line description.

    ``state.pt`` is preferred: it holds the EMA weights in full precision, while ``final/model.safetensors`` was saved in half.
    """
    state = path / "state.pt"
    if state.exists():
        st = torch.load(state, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        params = dict(model.named_parameters())
        n = 0
        for k, v in st["ema"].items():
            if k in params:
                params[k].data.copy_(v.to(device=params[k].device, dtype=params[k].dtype))
                n += 1
        return f"{state} (EMA weights of {n} tensors, step {int(st.get('step', 0))})"
    for cand in (path / "final", path):
        if (cand / "model.safetensors").exists():
            from safetensors.torch import load_file

            want = model.state_dict()
            sd = {k: v.to(want[k].dtype) for k, v in load_file(str(cand / "model.safetensors")).items() if k in want}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                raise RuntimeError(f"init_from {cand}: {len(missing)} tensors missing, e.g. {missing[:3]}")
            return f"{cand / 'model.safetensors'} ({len(sd)} tensors)"
    raise FileNotFoundError(f"init_from {path}: neither state.pt nor final/model.safetensors")


# ----------------------------------------------------------------------------------------
def run(cfg: TrainConfig) -> None:
    t_start = time.time()
    if "RANK" in os.environ and not is_dist():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(hours=3))
    r, w = rank(), world()
    local = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    torch.manual_seed(cfg.seed + r)
    np.random.seed(cfg.seed + r)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(cfg.data_dir)
    if r == 0:
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")
        log(f"world={w} device={device} cuda={use_cuda} base={cfg.base}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base)
    if cfg.arch_from:
        model = Leeth2Model.from_arch(cfg.arch_from, attn_implementation=cfg.attn)
    else:
        model = Leeth2Model.from_base(cfg.base, tokenizer=tokenizer, attn_implementation=cfg.attn)
    if cfg.grad_ckpt:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if cfg.init_from and not (cfg.resume and (out / "state.pt").exists()):
        what = load_init(model, Path(cfg.init_from))
        if r == 0:
            log(f"fine-tuning from {what}")
    model.to(device)
    encoder = Leeth2Encoder(tokenizer, model.cfg.vocab_size, max_len=cfg.max_len)
    counts = model.count_parameters()
    if r == 0:
        log(f"parameters: total={counts['total']/1e6:.1f}M trainable={counts['trainable']/1e6:.1f}M")

    train_reader = JsonlReader(data / cfg.train_file)
    n_train = len(train_reader)
    len_cache = out / f"token_lens_{cfg.max_len}.npy"
    if r == 0 and not len_cache.exists():
        t_len = time.time()
        exact_token_lengths(data / cfg.train_file, cfg.base, model.cfg.vocab_size, cfg.max_len, cfg.len_workers, cache=len_cache)
        log(f"measured the true token length of {n_train} samples in {time.time() - t_len:.0f}s")
    if is_dist():
        dist.barrier()
    token_lens = np.load(len_cache)
    plan_kw = dict(caps=train_reader.caps, token_lens=token_lens)
    plan = plan_batches(train_reader.lens, cfg.tokens_per_batch, cfg.max_batch, cfg.seed, 0, w, cfg.max_len, **plan_kw)
    my_batches = plan[r::w]
    micro_per_epoch = len(my_batches)
    steps_per_epoch = micro_per_epoch // cfg.grad_accum
    total_steps = cfg.max_steps or int(steps_per_epoch * cfg.epochs)
    if r == 0:
        log(f"train examples={n_train} batches/epoch/rank={micro_per_epoch} steps/epoch={steps_per_epoch} planned steps={total_steps}")

    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    groups = build_param_groups(model, cfg)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6, fused=use_cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and use_cuda)
    ema = EMA(trainable, cfg.ema_decay)
    ddp = model
    if w > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if use_cuda else None, find_unused_parameters=False, gradient_as_bucket_view=True)

    step = 0
    state_path = out / "state.pt"
    resume_path = Path(cfg.resume_from) / "state.pt" if cfg.resume_from else state_path
    if cfg.resume and resume_path.exists():
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        ema.shadow = {k: v.to(device) for k, v in st["ema"].items()}
        step = int(st["step"])
        # keep the saved schedule (it may have been shrunk to fit a time budget) unless the caller
        # explicitly asked for a different length
        if not cfg.max_steps:
            total_steps = int(st.get("total_steps", total_steps))
        if r == 0:
            log(f"resumed from {resume_path} at step {step}/{total_steps}")

    ds = BatchDataset(train_reader, my_batches, encoder, cfg.seed)
    skip = (step * cfg.grad_accum) % max(1, micro_per_epoch)
    ds.batches = my_batches[skip:] + my_batches[:skip]
    def make_loader() -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            ds, batch_size=None, shuffle=False, num_workers=cfg.workers, prefetch_factor=4 if cfg.workers else None, pin_memory=use_cuda
        )

    epoch_i = int((step * cfg.grad_accum) // max(1, micro_per_epoch))
    loader = make_loader()
    it = iter(loader)

    dev: List[Any] = []
    if r == 0 and (data / cfg.dev_file).exists():
        dev = read_all(data / cfg.dev_file)
        if len(dev) > cfg.dev_cap:
            rng = random.Random(7)
            dev = rng.sample(dev, cfg.dev_cap)
        log(f"dev samples: {len(dev)}")

    lam = {"choice": cfg.lam_choice, "noul": cfg.lam_noul, "score": cfg.lam_score, "extract": cfg.lam_extract, "choice_brier": cfg.lam_brier_choice}
    warmup = max(10, int(cfg.warmup_frac * total_steps))
    log_path = out / "train_log.jsonl"
    step_times: List[float] = []
    tok_seen = 0
    run_stats: Dict[str, float] = {}
    shrunk = False
    last_eval_step = step
    stop = False

    def do_eval(tag: str) -> Dict[str, float]:
        if r != 0 or not dev:
            return {}
        t0 = time.time()
        with ema.swapped():
            preds = evaluate.predict(model, encoder, dev, device=str(device), amp=cfg.amp and use_cuda, calibrated=False)
        rep = evaluate.report(dev, preds)
        h = evaluate.headline(rep)
        (out / f"eval_{tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        log(f"EVAL {tag}: " + " ".join(f"{k}={v:.4f}" for k, v in h.items()) + f" ({time.time()-t0:.0f}s)")
        jlog(log_path, {"eval": tag, "step": step, **h})
        return h

    def save_state() -> None:
        if r != 0:
            return
        atomic_save(
            {"model": {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable or k.startswith("marker_embed")},
             "opt": opt.state_dict(), "scaler": scaler.state_dict(), "ema": {k: v.cpu() for k, v in ema.shadow.items()},
             "step": step, "total_steps": total_steps},
            state_path,
        )

    if is_dist():
        dist.barrier()
    log(f"training: step {step} -> {total_steps}, warmup {warmup}")
    model.train()
    t_loop = time.time()
    while step < total_steps and not stop:
        t0 = time.time()
        f = lr_factor(step, total_steps, warmup, cfg.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        opt.zero_grad(set_to_none=True)
        step_stats: Dict[str, float] = {}
        # An OOM on one rank is deliberately NOT caught: swallowing it would leave the other rank
        # waiting in a gradient all-reduce forever. The process dies, torchrun tears the group down and
        # the launcher restarts from state.pt with a smaller token budget. A non-finite loss needs no
        # special case either: its gradients reach every rank through the all-reduce and GradScaler
        # skips the step everywhere.
        for micro in range(cfg.grad_accum):
            try:
                batch = next(it)
            except StopIteration:  # next epoch: a fresh shuffle, not the same order again
                epoch_i += 1
                ds.batches = plan_batches(train_reader.lens, cfg.tokens_per_batch, cfg.max_batch, cfg.seed, epoch_i, w, cfg.max_len, **plan_kw)[r::w]
                loader = make_loader()
                it = iter(loader)
                batch = next(it)
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            sync = ddp.no_sync() if (w > 1 and micro < cfg.grad_accum - 1) else contextlib.nullcontext()
            with sync:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.amp and use_cuda):
                    res = ddp(**batch)
                loss, stats = compute_losses(res, batch, lam, cfg.brier)
                scaler.scale(loss / cfg.grad_accum).backward()
            for k, v in stats.items():
                step_stats[k] = step_stats.get(k, 0.0) + (v / cfg.grad_accum if k.startswith("loss_") else v)
            tok_seen += int(batch["attention_mask"].sum())
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(list(trainable.values()), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        ema.update(step)
        step += 1
        dt = time.time() - t0
        step_times.append(dt)

        # shrink the schedule once, early, so the session ends on a decayed LR
        if not shrunk and step == max(60, warmup + 20):
            reserve = min(900.0, 0.08 * cfg.time_budget_s)  # evals, calibration and export after the last step
            remaining = cfg.time_budget_s - (time.time() - t_start) - reserve
            per = float(np.median(step_times[-40:]))
            fit = max(step + 20, step + int(remaining / per))
            fit = int(sync_max(fit, device, lo=True))
            if fit < total_steps:
                log(f"schedule shrunk {total_steps} -> {fit} steps to fit the time budget ({per:.2f}s/step)")
                total_steps = fit
            shrunk = True

        if step % cfg.log_every == 0 and r == 0:
            tail = step_times[-cfg.log_every :]
            tps = tok_seen / max(1e-6, time.time() - t_loop) * w
            mem = torch.cuda.max_memory_allocated(device) / 2**30 if use_cuda else 0.0
            msg = {"step": step, "lr": opt.param_groups[0]["lr"], "gn": float(gn), "s_step": float(np.mean(tail)), "tok_s": tps, "mem_gb": mem, **{k: round(v, 4) for k, v in step_stats.items() if k.startswith("loss_")}}
            jlog(log_path, msg)
            log(f"step {step}/{total_steps} " + " ".join(f"{k}={v:.4g}" for k, v in msg.items() if k != "step"))

        elapsed = time.time() - t_start
        out_of_time = sync_flag(elapsed > cfg.time_budget_s, device)
        if step % cfg.eval_every == 0 or out_of_time:
            do_eval(f"step{step}")
            last_eval_step = step
            model.train()
        if step % cfg.save_every == 0 or out_of_time:
            save_state()
        if out_of_time:
            log("time budget exhausted — stopping")
            stop = True
        if is_dist() and step % cfg.eval_every == 0:
            dist.barrier()

    # ---------------------------------------------------------------- finalisation
    save_state()
    if is_dist():
        dist.barrier()
    if r == 0:
        finalize(cfg, model, ema, encoder, data, out, device, step, time.time() - t_start)
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def sync_max(x: int, device: torch.device, lo: bool = False) -> int:
    if not is_dist():
        return x
    t = torch.tensor([float(x)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN if lo else dist.ReduceOp.MAX)
    return int(t.item())


def finalize(cfg: TrainConfig, model: Leeth2Model, ema: EMA, encoder: Leeth2Encoder, data: Path, out: Path, device: torch.device, step: int, secs: float) -> None:
    """EMA weights -> temperature calibration on held-out data -> report -> fp16 export."""
    use_amp = cfg.amp and device.type == "cuda"
    report: Dict[str, Any] = {"steps": step, "seconds": round(secs), "base": cfg.base}
    with ema.swapped():
        if (data / cfg.calib_file).exists():
            calib = read_all(data / cfg.calib_file)
            cal_enc = encoder
            if cfg.calib_max_len > 0:
                cal_enc = Leeth2Encoder(encoder.tok, encoder.V, max_len=cfg.calib_max_len, hard_max_len=encoder.hard_max_len)
            preds = evaluate.predict(model, cal_enc, calib, device=str(device), amp=use_amp, calibrated=False)
            temps = calibrate.fit_temperatures(calib, preds)
            calibrate.apply_temperatures(model, temps)
            report["temperatures"] = temps
            log(f"temperatures: {temps}")
        for name in ("dev", "test"):
            fp = data / f"{name}.jsonl"
            if fp.exists():
                ss = read_all(fp)
                rep = evaluate.report(ss, evaluate.predict(model, encoder, ss, device=str(device), amp=use_amp, calibrated=True))
                report[name] = evaluate.headline(rep)
                (out / f"final_{name}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
                log(f"FINAL {name}: {report[name]}")
        model.save_pretrained(out / "final", half=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    encoder.tok.save_pretrained(out / "final")
    log("done")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    a = ap.parse_args(argv)
    cfg = TrainConfig()
    if a.config:
        for k, v in json.loads(Path(a.config).read_text(encoding="utf-8")).items():
            setattr(cfg, k, v)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(json.loads(v)) if not isinstance(cur, str) else v)
    run(cfg)


if __name__ == "__main__":
    main()
