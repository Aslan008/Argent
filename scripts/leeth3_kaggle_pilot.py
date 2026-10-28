"""Leeth 3 data pilot on Kaggle: does the teacher corpus fix what it was written for, and does more of it still help?

    python scripts/leeth3_kaggle_pilot.py data [--push]          stage the dataset hunzlee/leeth3-pilot (code + teacher rows)
    python scripts/leeth3_kaggle_pilot.py kernel 50 [--push]     write (and push) hunzlee/leeth3-pilot-50 (half of the teacher rows)
    python scripts/leeth3_kaggle_pilot.py kernel 100 [--push]    ... hunzlee/leeth3-pilot-100 (all of them)
    python scripts/leeth3_kaggle_pilot.py kernel 100 --holdout --name v2 [--hours 2]   all but the probe holdout (probes.py)
    python scripts/leeth3_kaggle_pilot.py kernel 25 --holdout --name curve25 --hours 4    learning curve (plan 5.6): 25 / 50 / 100

A probe, not a release (plan section 2.3 still holds for releases): both kernels start from the installed s70 (output ``mix/s70`` of
``hunzlee/leeth21-soup``) and train on the teacher rows (src/leeth3 records + their expansion, as ``to_samples`` writes them) mixed
with a replay of the round-3 Leeth 2.1 corpus (``hunzlee/leeth21-data3``), so what s70 knew is not dropped. The halves are split by
teacher group (twins and paraphrases stay together). The weights come back to the PC and are measured there on the frozen exams -
the exams never leave the machine.

Long records above ~7.5k tokens are left out: the Leeth 2 encoder's window is 8192 (Leeth 3 windows are phase F3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leeth3.expand import approx_tokens  
from src.leeth3.probes import in_probe_holdout  
from src.leeth3.records import parse_record, to_samples  

WORK = ROOT / ".argent" / "kaggle" / "leeth3"
OWNER = "hunzlee"
DATASET = "leeth3-pilot"
WINDOWS = (1024, 2048, 4096, 8192)
MAX_TOKENS = 7500
ENV_NOTE = "kaggle CLI lives in the system Python (not .venv)"


def _quarter(key: str) -> int:
    """0-3 by teacher group. Quarter 0 lies inside half 0 (``quarter % 2 == half``), so the curve's 25 % is part of its 50 %."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 4


def _half(key: str) -> int:
    return _quarter(key) % 2


def source_key(rec) -> str:
    """Expanded records carry ``src:<group or id>`` of the record they came from; the halves must agree with it."""
    if rec.group.startswith("src:"):
        return rec.group[4:]
    return rec.group_key or rec.id


def teacher_rows(data: Path) -> tuple:
    files = [data / "accepted" / "records.jsonl", data / "expanded" / "catalog.jsonl", data / "expanded" / "long.jsonl"]
    rows, stats, skipped = [], Counter(), Counter()
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = parse_record(line)
            for s in to_samples(rec):
                d = json.loads(s.to_json())
                est = approx_tokens(s.state) + approx_tokens(s.question) + sum(approx_tokens(o) + 2 for o in (s.options or []))
                if est > MAX_TOKENS:
                    skipped[f.stem] += 1
                    continue
                meta = d.setdefault("meta", {})
                if est > 600:                                    # the trainer's base window is 640: long rows carry their own
                    meta["max_len"] = next(w for w in WINDOWS if w >= est * 1.1) if est * 1.1 <= WINDOWS[-1] else WINDOWS[-1]
                meta["half"] = _half(source_key(rec))
                meta["quarter"] = _quarter(source_key(rec))
                meta["probe"] = int(in_probe_holdout(source_key(rec)))   # kept out of training when the kernel says so
                meta["source"] = f.stem
                rows.append(json.dumps(d, ensure_ascii=False, separators=(",", ":")))
                stats[(f.stem, rec.domain, meta["half"], meta["probe"])] += 1
    return rows, stats, skipped


def stage_data(data: Path) -> Path:
    d = WORK / "ds_pilot"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    rows, stats, skipped = teacher_rows(data)
    manifest = {"rows": len(rows), "skipped_too_long": dict(skipped),
                "by_source_domain_half_probe": {"/".join(map(str, k)): v for k, v in sorted(stats.items())},
                "expanded_manifest": json.loads((data / "expanded" / "manifest.json").read_text(encoding="utf-8"))}
    with zipfile.ZipFile(d / "leeth3_pilot.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("teacher.jsonl", "\n".join(rows) + "\n")
        z.writestr("teacher_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
        for p in sorted((ROOT / "src" / "leeth2").rglob("*.py")):
            if "__pycache__" not in p.parts:
                z.write(p, "leeth2/" + p.relative_to(ROOT / "src" / "leeth2").as_posix())
    (d / "dataset-metadata.json").write_text(json.dumps({"title": DATASET, "id": f"{OWNER}/{DATASET}", "licenses": [{"name": "CC0-1.0"}]}),
                                             encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "expanded_manifest"}, ensure_ascii=False, indent=1))
    return d


KERNEL = r'''"""Kaggle GPU kernel: Leeth 3 data pilot (written by scripts/leeth3_kaggle_pilot.py) - s70 + teacher rows (HALF) + replay."""
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

FRACTION = __FRACTION__              # 0.25: teacher rows of quarter 0 only; 0.5: of half 0 (quarters 0 and 2); 1.0: all
# Leave the probe holdout out (half of the groups of B0215 and later, src/leeth3/probes.py): scripts/leeth3_probe.py --holdout-only
# then measures this model on records it never saw, like the first pilots on B0215-B0310.
HOLDOUT = __HOLDOUT__
UPSAMPLE = 3                         # each teacher row this many times per epoch (the teacher corpus is ~10k rows)
REPLAY = 80000                       # rows of the Leeth 2.1 round-3 corpus, so what s70 knew is kept
# Leave the old command-safety rows out of the replay (the dispatcher's safety / risk / act questions). Pilot 1: benign commands of
# unseen programs stayed at 0.25 from 50 % to 100 % of the teacher rows; the old rows label "review" commands 0.35-0.45 and know
# no user words - the hypothesis is that they pull the teacher's policy back.
DROP_COMMAND_SAFETY = __DROP__
COMMAND_SAFETY_QUESTIONS = {"Является ли данная терминальная команда безопасной для выполнения?",
                            "Оцените уровень риска выполнения данной команды от 0 (полностью безопасно) до 100 (катастрофический риск)",
                            "Какое действие должен предпринять агент перед запуском этой команды?"}
TRAIN_S = int(__HOURS__ * 3600)
DATA_KERNEL = "leeth21-data3"
INIT_KERNEL = "leeth21-soup"
INIT_SUB = "s70"
RUN = dict(base="/kaggle/working/base", max_len=640, tokens_per_batch=8500, grad_accum=2, lr=1e-5, warmup_frac=0.03, epochs=1.0, max_steps=0,
           eval_every=1000, save_every=1000, dev_cap=3000, ema_decay=0.998, calib_max_len=4096, lam_brier_choice=0.3)
MAX_RESTARTS = 3
T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lines_of(path):
    # File lines, not str.splitlines(): that also cuts at U+2028 / U+0085, which the replay corpus has inside JSON strings (run 1 died on it)
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def bootstrap_code():
    cands = [p.parent for p in Path("/kaggle/input").rglob("encoding.py") if (p.parent / "schema.py").exists() and "leeth3-pilot" in str(p)]
    assert cands, "the leeth3-pilot dataset (code + teacher rows) is not attached"
    dst = Path("/kaggle/working/pkg/leeth2")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(cands[0], dst)
    sys.path.insert(0, "/kaggle/working/pkg")
    log(f"code from {cands[0]}")


def find_base():
    for p in Path("/kaggle/input").rglob("train.jsonl"):
        if DATA_KERNEL in str(p) and (p.parent / "dev.jsonl").exists():
            return p.parent
    raise SystemExit(f"{DATA_KERNEL}: no train.jsonl + dev.jsonl under /kaggle/input")


def init_model():
    for p in Path("/kaggle/input").rglob("model.safetensors"):
        if INIT_KERNEL in str(p) and p.parent.name == INIT_SUB:
            return p.parent
    raise SystemExit(f"no {INIT_SUB}/model.safetensors of {INIT_KERNEL} under /kaggle/input")


def build_mix():
    from leeth2.datafile import build_index

    base = find_base()
    teacher_file = next(p for p in Path("/kaggle/input").rglob("teacher.jsonl") if "leeth3-pilot" in str(p))
    def kept(meta):
        if FRACTION >= 1.0:
            return True
        return meta["quarter"] == 0 if FRACTION <= 0.25 else meta["half"] == 0
    teacher = [l for l in lines_of(teacher_file) if kept(json.loads(l)["meta"])]
    if HOLDOUT:
        teacher = [l for l in teacher if not json.loads(l)["meta"].get("probe")]
    pool = lines_of(base / "train.jsonl")
    dropped = {}
    if DROP_COMMAND_SAFETY:
        keep = []
        for l in pool:
            d = json.loads(l)
            if d.get("question") in COMMAND_SAFETY_QUESTIONS:
                dropped[d.get("task")] = dropped.get(d.get("task"), 0) + 1
            else:
                keep.append(l)
        pool = keep
        log(f"command-safety rows left out of the replay: {sum(dropped.values())} {dropped}")
    replay = random.Random(33).sample(pool, REPLAY)
    train = replay + teacher * UPSAMPLE
    random.Random(52).shuffle(train)
    mix = Path("/kaggle/working/mix")
    shutil.rmtree(mix, ignore_errors=True)
    mix.mkdir(parents=True)
    (mix / "train.jsonl").write_text("\n".join(train) + "\n", encoding="utf-8")
    for role in ("dev", "calib", "test"):
        shutil.copy2(base / f"{role}.jsonl", mix / f"{role}.jsonl")
    build_index(mix / "train.jsonl")
    man = {"fraction": FRACTION, "holdout": HOLDOUT, "hours": TRAIN_S / 3600, "drop_command_safety": DROP_COMMAND_SAFETY, "replay_dropped": sum(dropped.values()), "teacher_rows": len(teacher), "upsample": UPSAMPLE, "replay": len(replay), "train": len(train)}
    (mix / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    log(f"mix {man}")
    return mix


def train(corpus, init):
    out = Path("/kaggle/working/run")
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(RUN, data_dir=str(corpus), out_dir=str(out), init_from=str(init))
    env = {**os.environ, "PYTHONPATH": "/kaggle/working/pkg", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "NCCL_P2P_DISABLE": "1"}
    n_gpu = max(1, subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU "))
    for attempt in range(MAX_RESTARTS + 1):
        cfg["time_budget_s"] = float(TRAIN_S)
        (out / "launch_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={n_gpu}", "-m", "leeth2.train", "--config", str(out / "launch_config.json")]
        log(f"train #{attempt}: tokens_per_batch={cfg['tokens_per_batch']} grad_ckpt={cfg.get('grad_ckpt', False)}")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail = (tail + [line])[-80:]
        if proc.wait() == 0:
            break
        oom = "out of memory" in "".join(tail).lower()
        log(f"training exited non-zero (oom={oom})")
        if attempt == MAX_RESTARTS:
            raise SystemExit("giving up")
        if oom:                      # an 8k-token row alone can be the batch: trade speed for memory first, then shrink the batches
            if not cfg.get("grad_ckpt"):
                cfg["grad_ckpt"] = True
            else:
                cfg["tokens_per_batch"] = int(cfg["tokens_per_batch"] * 0.78)
        time.sleep(5)
    for stale in [out / "state.pt", *out.glob("eval_step*.json")]:
        if stale.exists():
            stale.unlink()
    return out / "final"


def main():
    bootstrap_code()
    from huggingface_hub import snapshot_download

    snapshot_download("jhu-clsp/mmBERT-base", ignore_patterns=["*.onnx", "*.msgpack", "*.h5", "*.ot", "*.gguf", "onnx/*"], local_dir="/kaggle/working/base")
    mix = build_mix()
    init = init_model()
    log(f"start weights {init}; budget {TRAIN_S / 3600:.1f} h")
    final = train(mix, init)
    log(f"final model {final}")
    for d in ("/kaggle/working/base", "/kaggle/working/pkg", "/kaggle/working/mix"):
        shutil.rmtree(d, ignore_errors=True)
    log(f"done in {(time.time() - T0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
'''


def make_kernel(pct: int, no_safety_replay: bool = False, name: str = "", hours: float = 2.0, holdout: bool = False) -> Path:
    """``name`` overrides the kernel suffix (leeth3-pilot-<name>); the first pilots were "50", "100" and "100-nosafe"."""
    name = name or (f"{pct}-nosafe" if no_safety_replay else str(pct))
    d = WORK / f"kernel_pilot_{name}"
    d.mkdir(parents=True, exist_ok=True)
    code = f"leeth3_pilot_{name.replace('-', '_')}.py"
    src = (KERNEL.replace("__FRACTION__", str(pct / 100)).replace("__DROP__", str(no_safety_replay))
           .replace("__HOLDOUT__", str(holdout)).replace("__HOURS__", repr(float(hours))))
    compile(src, code, "exec")
    (d / code).write_text(src, encoding="utf-8")
    meta = {"id": f"{OWNER}/leeth3-pilot-{name}", "title": f"leeth3-pilot-{name}", "code_file": code, "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{OWNER}/{DATASET}"], "competition_sources": [],
            "kernel_sources": [f"{OWNER}/leeth21-data3", f"{OWNER}/leeth21-soup"], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def kaggle(*args: str, cwd: Path = None) -> str:
    r = subprocess.run(["kaggle", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = "\n".join(l for l in (r.stdout + "\n" + r.stderr).splitlines() if "Warning" not in l and "warnings.warn" not in l).strip()
    print(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("data", "kernel"))
    ap.add_argument("pct", nargs="?", type=int, choices=(25, 50, 100))
    ap.add_argument("--data", type=Path, default=ROOT / "leeth3_data")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-safety-replay", action="store_true", help="leave the old command-safety rows out of the replay")
    ap.add_argument("--holdout", action="store_true", help="leave the probe holdout (half the groups of B0215+) out of training")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--name", default="", help="kernel suffix: leeth3-pilot-<name>")
    a = ap.parse_args(argv)
    if a.what == "data":
        d = stage_data(a.data)
        if a.push:
            out = kaggle("datasets", "create", "-p", str(d), "-r", "zip")
            if "already" in out.lower() or "error" in out.lower():
                kaggle("datasets", "version", "-p", str(d), "-m", "teacher corpus update", "--dir-mode", "zip")
        return 0
    if a.pct is None:
        ap.error("kernel needs 25, 50 or 100")
    d = make_kernel(a.pct, a.no_safety_replay, a.name, a.hours, a.holdout)
    print("written", d)
    if a.push:
        kaggle("kernels", "push", "-p", str(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
