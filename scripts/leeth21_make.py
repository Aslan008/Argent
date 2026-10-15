"""Write (and optionally push) the Leeth 2.1 training kernels from .argent/kaggle/leeth2/kernel_leeth21/leeth21_train.py.

    python scripts/leeth21_make.py a --brier 0.5 --hours 3.0 [--push]     hunzlee/leeth21-train-a
    python scripts/leeth21_make.py b --brier 0.0 --hours 3.0 [--push]     hunzlee/leeth21-train-b
    python scripts/leeth21_make.py c1 --brier 0.5 --hours 2.0 --lr 1e-5 --data leeth21-data2 --init leeth21-train-b --init-sub final [--push]

A kernel trains on the output of ``leeth21-data`` starting from ``leeth2-interp-g2`` / soup0.5 (the installed model).  ``--push`` first checks
that everything the kernel mounts exists and is finished, so GPU hours are never spent on a kernel that stops at its first line.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".argent" / "kaggle" / "leeth2"
OWNER = "hunzlee"


def make(name: str, brier: float, hours: float, init: str = "leeth2-interp-g2", init_sub: str = "soup0.5", data: str = "leeth21-data", lr: float = 1.5e-5) -> Path:
    src = (WORK / "kernel_leeth21" / "leeth21_train.py").read_text(encoding="utf-8")

    def sub(pattern: str, repl: str) -> None:
        nonlocal src
        new, n = re.subn(pattern, repl, src, count=1, flags=re.M)
        assert n == 1, pattern
        src = new

    sub(r"^LAM_BRIER = .*$", f"LAM_BRIER = {brier}")
    sub(r"^LR = .*$", f"LR = {lr}")
    sub(r"^TRAIN_S = .*$", f"TRAIN_S = int({hours} * 3600)")
    sub(r'^DATA_KERNEL = .*$', f'DATA_KERNEL = "{data}"')
    sub(r'^INIT_KERNEL = .*$', f'INIT_KERNEL = "{init}"')
    sub(r'^INIT_SUB = .*$', f'INIT_SUB = "{init_sub}"')
    d = WORK / f"kernel_leeth21_{name}"
    d.mkdir(exist_ok=True)
    code = f"leeth21_train_{name}.py"
    (d / code).write_text(src, encoding="utf-8")
    meta = {"id": f"{OWNER}/leeth21-train-{name}", "title": f"leeth21-train-{name}", "code_file": code, "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{OWNER}/leeth2-code"], "competition_sources": [],
            "kernel_sources": [f"{OWNER}/{data}", f"{OWNER}/{init}"], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    compile(src, code, "exec")
    return d


def preflight(meta: dict) -> list:
    problems = []
    for kind, slugs, ok in (("kernels", meta["kernel_sources"], "COMPLETE"), ("datasets", meta["dataset_sources"], "ready")):
        for slug in slugs:
            r = subprocess.run(["kaggle", kind, "status", slug], capture_output=True, text=True)
            lines = [l.strip() for l in (r.stdout + "\n" + r.stderr).splitlines() if l.strip() and "warn" not in l.lower()]
            if not any(ok in l for l in lines):
                problems.append(f"{kind[:-1]} {slug}: {' | '.join(lines)[:200] or '(no answer)'}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--brier", type=float, default=0.0)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--init", default="leeth2-interp-g2")
    ap.add_argument("--init-sub", default="soup0.5")
    ap.add_argument("--data", default="leeth21-data")
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    d = make(a.name, a.brier, a.hours, a.init, a.init_sub, a.data, a.lr)
    print("written", d)
    if a.push:
        problems = preflight(json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8")))
        if problems:
            sys.exit("not pushed -- the kernel would not start:\n  " + "\n  ".join(problems))
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
