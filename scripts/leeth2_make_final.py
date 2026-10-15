"""Write (and optionally push) the final-training kernels, from the template .argent/kaggle/leeth2/kernel_final/final_v1.py.

    python scripts/leeth2_make_final.py data --data-only --push                       # CPU kernel: builds the mixed corpus once (no GPU hours)
    python scripts/leeth2_make_final.py base  --init leeth2-merge-v2 --hours 2.6 --data-kernel leeth2-final-data --push
    python scripts/leeth2_make_final.py grown --init leeth2-grow-v1  --hours 5.0 --data-kernel leeth2-final-data --push

Without ``--data-kernel`` the training kernel builds the corpus itself (about half an hour of the GPU session); with it, the corpus is the
output of the data kernel, so two training kernels (the plain and the grown model) train on exactly the same rows and the GPU time is
spent on training only.

``--grow`` grows the start model inside the kernel (leeth2.grow); leave it off when ``--init`` is already a grown model (e.g. the
output of the capacity A/B): the kernel notices (leeth2.grow.is_grown) and builds the network from that model's own config, with a gentler
learning-rate boost for the inserted layers (1.5x instead of 3x, they are no longer fresh identities).
The kernel is named hunzlee/leeth2-final-<name>; with --push it first checks that everything it mounts exists and is finished.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".argent" / "kaggle" / "leeth2"
OWNER = "hunzlee"


def make(name: str, init: str, hours: float, grow: bool, init_sub: str = "run", data_kernel: Optional[str] = None, data_only: bool = False,
         targeted: str = "leeth2-targeted-v4", play_more_scale: float = 0.45, replay_rows: int = 190000) -> Path:
    src = (WORK / "kernel_final" / "final_v1.py").read_text(encoding="utf-8")
    mode = "data" if data_only else ("train" if data_kernel else "all")

    def sub(pattern: str, repl: str) -> None:
        nonlocal src
        new, n = re.subn(pattern, repl, src, count=1, flags=re.M)
        assert n == 1, pattern
        src = new

    sub(r'^INIT_KERNEL = .*$', f'INIT_KERNEL = "{init}"')
    sub(r'^INIT_SUB = .*$', f'INIT_SUB = "{init_sub}"')
    sub(r'^GROW = .*$', f"GROW = {grow}")
    sub(r'^MODE = .*$', f'MODE = "{mode}"')
    sub(r'^DATA_KERNEL = .*$', f'DATA_KERNEL = "{data_kernel or f"leeth2-final-{name}"}"')
    sub(r'^TRAIN_S = .*$', f"TRAIN_S = int({hours} * 3600)")
    sub(r'^PLAY_MORE_SCALE = [0-9.]+', f"PLAY_MORE_SCALE = {play_more_scale}")
    sub(r'^REPLAY_ROWS = [0-9]+', f"REPLAY_ROWS = {replay_rows}")
    d = WORK / f"kernel_final_{name}"
    d.mkdir(exist_ok=True)
    code = f"final_{name}.py"
    (d / code).write_text(src, encoding="utf-8")
    datasets = [f"{OWNER}/leeth2-code"] + ([] if mode == "train" else [f"{OWNER}/{targeted}"])
    kernels = ([] if mode == "train" else [f"{OWNER}/leeth2-corpus-v1"]) + ([] if mode == "data" else [f"{OWNER}/{init}"]) + ([f"{OWNER}/{data_kernel}"] if mode == "train" else [])
    meta = {"id": f"{OWNER}/leeth2-final-{name}", "title": f"leeth2-final-{name}", "code_file": code, "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": mode != "data", "enable_internet": True, "dataset_sources": datasets, "competition_sources": [], "kernel_sources": kernels, "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    compile(src, code, "exec")
    return d


def preflight(meta: dict) -> List[str]:
    """What the kernel will mount must exist and be finished/ready; otherwise hours of GPU time would be spent on a kernel that stops at its first line."""
    problems = []
    for kind, slugs, ok in (("kernels", meta["kernel_sources"], "COMPLETE"), ("datasets", meta["dataset_sources"], "ready")):
        for slug in slugs:
            r = subprocess.run(["kaggle", kind, "status", slug], capture_output=True, text=True)
            noise = lambda l: "warn" in l.lower()            # the CLI's RequestsDependencyWarning arrives on stderr, and stdout may lack its final newline
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip()] + [l.strip() for l in r.stderr.splitlines() if l.strip() and not noise(l)]
            if not any(ok in l for l in lines):
                problems.append(f"{kind[:-1]} {slug}: {' | '.join(lines)[:200] or '(no answer)'}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--init", default="leeth2-merge-v2", help="kernel slug (without owner) whose run/final is the start model")
    ap.add_argument("--init-sub", default="run")
    ap.add_argument("--hours", type=float, default=2.6)
    ap.add_argument("--grow", action="store_true")
    ap.add_argument("--data-kernel", default=None, help="train on the corpus that this (finished) data kernel produced instead of building it")
    ap.add_argument("--data-only", action="store_true", help="build only the corpus (a CPU kernel)")
    ap.add_argument("--targeted", default="leeth2-targeted-v4", help="the routing/safety dataset the corpus is mixed from")
    ap.add_argument("--play-more-scale", type=float, default=0.45, help="episodes of the eight newer game environments, relative to the base counts of play.GENERATORS_MORE")
    ap.add_argument("--replay-rows", type=int, default=190000, help="rows of the general corpus replayed next to the routing / games / sites data")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    d = make(a.name, a.init, a.hours, a.grow, a.init_sub, a.data_kernel, a.data_only, a.targeted, a.play_more_scale, a.replay_rows)
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
