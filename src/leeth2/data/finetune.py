"""Corpus for the Leeth 2 fine-tune: targeted Argent data + a replay slice of the original corpus.

    python -m leeth2.data.finetune build --out OUT [--scale 1.0] [--exams DIR]
    python -m leeth2.data.finetune mix --targeted OUT --base CORPUS_V1 --out MIX [--replay 450000]

`build` runs anywhere (no network, a minute): it writes train/dev/calib/test.jsonl from data/synth/argent_router.py.
`mix` runs on the training machine, where the 2.75M-row v1 corpus is: it keeps a seeded random slice of it (so the model
does not forget what it already knows), drops the v1 rows the new data replaces, and unions the evaluation splits.

The hand-written exams are the whole point of measuring this: any generated sample whose request or command equals one
of theirs is dropped, so none of them can leak into training however the grammar happens to combine its parts.
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..schema import Sample
from .common import stable_hash
from . import synth as SY
from .argent import SAFETY_BENIGN, SAFETY_DANGEROUS, SAFETY_QUESTION, TOOL_QUESTION
from .synth import argent_router as AR
from .synth import argent_context as CTX
from .synth import argent_hardening as HARD

ROLES = ("train", "dev", "calib", "test")
EVAL_CALLS = {"dev": 200, "calib": 400, "test": 800}   # generator calls per generator and role
# v1 rows these generators supersede (same question, new and much more varied data): left out of the replay slice
SUPERSEDED_QUESTIONS = {SAFETY_QUESTION, AR.RISK_QUESTION, AR.ACT_QUESTION}
# v3 also regenerates routing (16 options, session context), need-tool and argument rows: the v1 template versions leave the replay slice
SUPERSEDED_V3 = SUPERSEDED_QUESTIONS | {TOOL_QUESTION, AR.NEED_TOOL_QUESTION, AR.ARG_QUESTION}
V3_V2_SHARE = 0.45       # v2 generators kept at this share of their v2 size next to the v3 ones (nothing the v2 learnt is dropped)
_REQUEST = re.compile(r'Incoming user request in Argent session:\n"(.*?)"\n(?:Working directory:|Active background processes:)', re.S)
_COMMAND = re.compile(r"Command: (.*?)\nCaller:", re.S)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def exam_texts(exam_dir: Path) -> Set[str]:
    """Every request / command the exams contain, normalised. Missing files are fine (Kaggle has no scripts/)."""
    out: Set[str] = {norm(c) for c in list(SAFETY_DANGEROUS) + list(SAFETY_BENIGN)}
    for name in ("prod_battery.json", "prod_battery_v2.json"):
        p = exam_dir / name
        if p.exists():
            out |= {norm(c["query"]) for c in json.loads(p.read_text(encoding="utf-8")).get("cases", [])}
    p = exam_dir / "ood_exam_v3.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        out |= {norm(c["query"]) for c in d.get("routing", [])}
        for items in d.get("safety", {}).values():
            out |= {norm(c) for c in items}
    p = exam_dir / "real_exam_v4.json"
    if p.exists():
        out |= {norm(q) for q in json.loads(p.read_text(encoding="utf-8")).get("tools", {})}
    for p in (exam_dir / "real_queries.json", exam_dir.parent / ".argent" / "real_queries.json"):   # the owner's real requests (kept out of the repo): evaluation only
        if p.exists():
            out |= {norm(q) for q in json.loads(p.read_text(encoding="utf-8"))}
    p = exam_dir / "safety_exam_v2.json"
    if p.exists():
        for key, items in json.loads(p.read_text(encoding="utf-8")).items():
            if key != "meta":
                out |= {norm(c) for c in items}
    return out


def leaks(state: str, blocked: Set[str]) -> bool:
    for rx in (_REQUEST, _COMMAND):
        m = rx.search(state)
        if m and norm(m.group(1)) in blocked:
            return True
    return False


def generators(v3: bool, v4: bool = False):
    """(name, function, training calls). v3 keeps a share of the v2 grammar and adds the production-state generators; v4 adds the clusters of false dispatches."""
    if not v3:
        return list(AR.GENERATORS)
    out = [(n, f, int(c * V3_V2_SHARE)) for n, f, c in AR.GENERATORS] + list(CTX.GENERATORS)
    if not v4:
        return out
    # v4: two-step requests are labelled the way the owner's batteries label them (argent_router.compound), in both route grammars
    owner = {AR.gen_route, CTX.gen_route16}
    return [(n, functools.partial(f, owner=True) if f in owner else f, c) for n, f, c in out] + list(HARD.GENERATORS)


def generate_role(role: str, seed: int, scale: float, blocked: Set[str], v3: bool = False, v4: bool = False) -> List[Sample]:
    out: List[Sample] = []
    dropped = 0
    for name, fn, n_train in generators(v3, v4):
        calls = max(50, int(n_train * scale)) if role == "train" else EVAL_CALLS[role]
        base = stable_hash("argent_router", name) & 0xFFFF
        # 15 % extra calls: the leak filter below throws some away and the target counts should still be met
        made = SY.generate(fn, int(calls * 1.15), seed + base + SY.ROLE_SEED[role])
        kept = [s for s in made if not leaks(s.state, blocked)]
        dropped += len(made) - len(kept)
        out.extend(kept)
    log(f"{role:<6} {len(out):>7} samples ({dropped} dropped as exam look-alikes)")
    return out


def build(out_dir: Path, seed: int = 4242, scale: float = 1.0, exam_dir: Optional[Path] = None, v3: bool = False, v4: bool = False) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    blocked = exam_texts(exam_dir or Path(__file__).resolve().parents[3] / "scripts")
    log(f"{len(blocked)} exam requests/commands are off limits")
    rng = random.Random(seed)
    counts: Dict[str, int] = {}
    taken: Set[str] = set()                                   # (ptype, state, question) already used by an evaluation role
    for role in ("test", "calib", "dev", "train"):            # evaluation first: train may not repeat any of it
        used: Set[str] = set()
        rows: List[str] = []
        for s in generate_role(role, seed, scale, blocked, v3, v4):
            k = f"{s.ptype}|{s.state}|{s.question}"
            if k in taken or k in used:
                continue
            used.add(k)
            rows.append(s.to_json())
        if role != "train":
            taken |= used
        rng.shuffle(rows)
        with open(out_dir / f"{role}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(rows) + "\n")
        counts[role] = len(rows)
    (out_dir / "manifest.json").write_text(json.dumps({"counts": counts, "seed": seed, "scale": scale, "v3": v3, "v4": v4}, indent=1), encoding="utf-8")
    log(f"DONE {counts}")
    return counts


def _lines(path: Path) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line


def _question(line: str) -> str:
    m = re.search(r'"question":\s*"((?:[^"\\]|\\.)*)"', line)
    return json.loads('"' + m.group(1) + '"') if m else ""


def mix(targeted: Path, base: Path, out: Path, replay: int = 450000, seed: int = 4243, dev_cap: int = 6000, v3: bool = False, drop_substr: Tuple[str, ...] = ()) -> Dict[str, int]:
    superseded = SUPERSEDED_V3 if v3 else SUPERSEDED_QUESTIONS
    from ..datafile import build_index

    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    new_train = [l.rstrip("\n") for l in _lines(targeted / "train.jsonl")]
    # reservoir sample of the v1 training rows that are not superseded
    reservoir: List[str] = []
    seen = skipped = 0
    for line in _lines(base / "train.jsonl"):
        if _question(line) in superseded or any(t in line for t in drop_substr):   # drop_substr: e.g. '"task":"game_' when new data replaces a whole task family
            skipped += 1
            continue
        seen += 1
        if len(reservoir) < replay:
            reservoir.append(line.rstrip("\n"))
        else:
            j = rng.randrange(seen)
            if j < replay:
                reservoir[j] = line.rstrip("\n")
    log(f"replay: {len(reservoir)} of {seen} v1 rows kept ({skipped} superseded rows left out)")
    train = new_train + reservoir
    rng.shuffle(train)
    counts = {"train": len(train)}
    with open(out / "train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(train) + "\n")
    for role in ("dev", "calib", "test"):
        rows = [l.rstrip("\n") for l in _lines(targeted / f"{role}.jsonl")]
        old = [l.rstrip("\n") for l in _lines(base / f"{role}.jsonl")] if (base / f"{role}.jsonl").exists() else []
        old = [l for l in old if _question(l) not in superseded and not any(t in l for t in drop_substr)]
        rng.shuffle(old)
        if role == "dev":
            old = old[: max(0, dev_cap - len(rows))]
        allr = rows + old
        rng.shuffle(allr)
        with open(out / f"{role}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(allr) + "\n")
        counts[role] = len(allr)
    build_index(out / "train.jsonl")
    (out / "manifest.json").write_text(json.dumps({"counts": counts, "replay": len(reservoir), "targeted_train": len(new_train)}, indent=1), encoding="utf-8")
    log(f"MIX DONE {counts}")
    return counts


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--scale", type=float, default=1.0)
    b.add_argument("--seed", type=int, default=4242)
    b.add_argument("--exams", default="")
    b.add_argument("--v3", action="store_true", help="add the production-state generators (session context, 16 options, real-style requests)")
    b.add_argument("--v4", action="store_true", help="v3 plus the false-dispatch clusters (argent_hardening.py)")
    m = sub.add_parser("mix")
    m.add_argument("--targeted", required=True)
    m.add_argument("--base", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--replay", type=int, default=450000)
    m.add_argument("--seed", type=int, default=4243)
    m.add_argument("--v3", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(Path(a.out), a.seed, a.scale, Path(a.exams) if a.exams else None, a.v3 or a.v4, a.v4)
    else:
        mix(Path(a.targeted), Path(a.base), Path(a.out), a.replay, a.seed, v3=a.v3)


if __name__ == "__main__":
    main()
