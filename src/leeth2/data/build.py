"""Corpus builder: Hub datasets (+ local Argent data) -> train / dev / calib / test JSONL.

    python -m leeth2.data.build --out /kaggle/working/corpus --workers 3

Per-spec shards make the build resumable and failure-tolerant: one dataset that cannot be loaded is
reported in ``manifest.json`` instead of killing a 40-minute run.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..schema import Sample
from .common import humanize, stable_hash
from .kinds import KINDS, Spec

ROLES = ("train", "dev", "calib", "test")
CARVE_PERMILLE = {"dev": 3, "calib": 6, "test": 12}  # cumulative: 0.3% dev, 0.3% calib, 0.6% test of a train-only source
CARVE_MIN_CAP = {"dev": 150, "calib": 300, "test": 600}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- loading
def _retry(fn, tries: int = 5, base: float = 8.0):
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # HF 429 / transient network errors
            last = e
            msg = str(e)
            wait = base * (2**k) if ("429" in msg or "Too Many" in msg or "timed out" in msg.lower() or "Connection" in msg) else 2.0
            if k < tries - 1:
                time.sleep(min(wait, 120.0))
    raise last  # type: ignore[misc]


def open_split(spec: Spec, split: str):
    from datasets import load_dataset

    def go():
        kw: Dict[str, Any] = dict(split=split, streaming=True)
        if spec.revision:
            kw["revision"] = spec.revision
        return load_dataset(spec.repo, spec.cfg, **kw) if spec.cfg else load_dataset(spec.repo, **kw)

    return _retry(go)


def iter_rows(spec: Spec, split: str, seed: int, limit: Optional[int] = None) -> Iterator[Tuple[int, Dict[str, Any]]]:
    ds = open_split(spec, split)
    drop = spec.opts.get("drop_cols")
    if drop:  # heavy columns (screenshots, raw HTML) are never read
        try:
            ds = ds.remove_columns([c for c in drop if c in (ds.column_names or drop)])
        except Exception:
            pass
    try:
        ds = ds.shuffle(seed=seed, buffer_size=2000 if drop else 10000)
    except Exception:
        pass
    it = iter(ds)
    i = 0
    while True:
        if limit is not None and i >= limit:
            return
        try:
            row = next(it)
        except StopIteration:
            return
        except Exception as e:  # a broken shard mid-stream: keep what we have
            log(f"  {spec.id}[{split}] stream error after {i} rows: {type(e).__name__}: {str(e)[:120]}")
            return
        yield i, row
        i += 1


def _feature_names(features: Any, col: str) -> Optional[List[str]]:
    try:
        f = features[col]
    except Exception:
        return None
    names = getattr(f, "names", None)
    if names:
        return list(names)
    inner = getattr(f, "feature", None)  # Sequence(ClassLabel) / List(ClassLabel)
    if inner is not None:
        return list(getattr(inner, "names", None) or []) or None
    if isinstance(f, list) and f:
        return list(getattr(f[0], "names", None) or []) or None
    return None


def make_ctx(spec: Spec, seed: int) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    if spec.kind in ("cls", "nli"):
        label_col = spec.opts.get("label", "label")
        names = spec.opts.get("names")
        probe_split = spec.train or spec.dev or spec.test
        if names is None and spec.kind == "cls":
            feats = None
            try:
                feats = open_split(spec, probe_split).features
            except Exception:
                pass
            names = _feature_names(feats, label_col) if feats else None
            if names is None or spec.opts.get("string_labels"):
                seen: Counter = Counter()
                for _, row in iter_rows(spec, probe_split, seed, limit=30000):
                    v = row.get(label_col)
                    if v is not None and v != "":
                        seen[str(v)] += 1
                order = spec.opts.get("label_order") or sorted(seen)
                labels = [l for l in order if l in seen] + [l for l in sorted(seen) if l not in order]
                ctx["label_to_id"] = {l: i for i, l in enumerate(labels)}
                names = labels
        if spec.opts.get("string_labels") and "label_to_id" not in ctx:
            order = spec.opts.get("label_order") or (names or [])
            ctx["label_to_id"] = {l: i for i, l in enumerate(order)}
            names = list(order)
        if spec.kind == "nli":
            if spec.opts.get("string_labels"):
                order = spec.opts.get("label_order") or ["entailment", "neutral", "contradiction"]
                ctx["label_to_id"] = {l: i for i, l in enumerate(order)}
            names = names or ["entailment", "neutral", "contradiction"]
        mp = spec.opts.get("label_names_map")
        if mp:
            ctx["names"] = [mp.get(n, humanize(n)) for n in names]
            mr = spec.opts.get("names_ru_map")
            if mr:
                ctx["names_ru"] = [mr.get(n, mp.get(n, humanize(n))) for n in names]
        else:
            ctx["names"] = [humanize(n) for n in names] if names else []
        if spec.kind == "cls" and ctx["names"] and all(re.fullmatch(r"\d+", str(n)) for n in ctx["names"]):
            raise RuntimeError("class names are bare numbers: this source ships no label names")
        ctx["n_classes"] = len(ctx["names"])
    if spec.kind == "ner":
        tag_col = spec.opts.get("tags", "ner_tags")
        feats = None
        try:
            feats = open_split(spec, spec.train or spec.dev).features
        except Exception:
            pass
        ctx["tag_names"] = _feature_names(feats, tag_col) if feats else None
    return ctx


# ---------------------------------------------------------------------------------------------- one spec
def convert_rows(spec: Spec, rows: Iterator[Tuple[int, Dict[str, Any]]], rng: random.Random, ctx: Dict[str, Any], want: int, stats: Counter) -> List[Sample]:
    fn = KINDS[spec.kind]
    out: List[Sample] = []
    for i, row in rows:
        stats["rows"] += 1
        try:
            samples = fn(spec, row, rng, ctx)
        except Exception as e:
            stats[f"conv_err:{type(e).__name__}"] += 1
            continue
        if not samples:
            stats["empty"] += 1
        for s in samples:
            try:
                s.validate()
            except ValueError:
                stats["invalid"] += 1
                continue
            s.meta = {"lic": spec.lic, **({"heldout": True} if spec.heldout else {})}
            out.append(s)
        if len(out) >= want:
            break
    return out


def run_spec(spec: Spec, out_dir: Path, seed: int, scale: float, force: bool = False) -> Dict[str, Any]:
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    marker = shard_dir / f"{spec.id}.done.json"
    if marker.exists() and not force:
        return json.loads(marker.read_text(encoding="utf-8"))
    t0 = time.time()
    rng = random.Random(stable_hash(seed, spec.id))
    stats: Counter = Counter()
    pools: Dict[str, List[Sample]] = {r: [] for r in ROLES}
    err = None
    try:
        caps = [int(c * scale) if c else 0 for c in spec.caps]
        if scale < 1.0:
            caps = [max(c, 20) if c else 0 for c in caps]
        ctx = make_ctx(spec, seed)
        if spec.kind in ("cls", "nli") and not ctx.get("names") and spec.kind == "cls":
            raise RuntimeError("could not determine class names")
        # ---- train (and carved eval pools when the source has no official eval split)
        if spec.train and caps[0] > 0 and not spec.eval_only:
            carve = spec.dev is None
            ecap = {r: max(caps[k], int(CARVE_MIN_CAP[r] * (scale if scale < 1.0 else 1.0)), 5) for k, r in ((1, "dev"), (2, "calib"), (3, "test"))}
            fn = KINDS[spec.kind]
            n_rows = 0
            for i, row in iter_rows(spec, spec.train, seed):
                n_rows += 1
                stats["rows"] += 1
                role = "train"
                if carve:
                    h = stable_hash(spec.id, i) % 1000
                    role = "dev" if h < CARVE_PERMILLE["dev"] else "calib" if h < CARVE_PERMILLE["calib"] else "test" if h < CARVE_PERMILLE["test"] else "train"
                try:
                    samples = fn(spec, row, rng, ctx)
                except Exception as e:
                    stats[f"conv_err:{type(e).__name__}"] += 1
                    continue
                if not samples:
                    stats["empty"] += 1
                for s in samples:
                    try:
                        s.validate()
                    except ValueError:
                        stats["invalid"] += 1
                        continue
                    s.meta = {"lic": spec.lic}
                    cap = caps[0] if role == "train" else ecap[role]
                    if len(pools[role]) < cap:
                        pools[role].append(s)
                if len(pools["train"]) >= caps[0] and (not carve or n_rows >= 3000 or all(len(pools[r]) >= ecap[r] for r in ecap)):
                    break
        # ---- dev / calib / test from official splits
        if spec.dev and (caps[1] + caps[2] > 0 or (spec.test is None and caps[3] > 0)):
            need = caps[1] + caps[2] + (caps[3] if spec.test is None else 0)
            got = convert_rows(spec, iter_rows(spec, spec.dev, seed + 1), rng, ctx, need, stats)
            pools["dev"] = got[: caps[1]]
            pools["calib"] = got[caps[1] : caps[1] + caps[2]]
            if spec.test is None:
                pools["test"] = got[caps[1] + caps[2] : caps[1] + caps[2] + caps[3]]
        if spec.test and caps[3] > 0:
            pools["test"] = convert_rows(spec, iter_rows(spec, spec.test, seed + 2), rng, ctx, caps[3], stats)
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:300]}"
        log(f"FAIL {spec.id}: {err}")
        if not pools["train"] and not pools["test"] and not pools["dev"]:
            traceback.print_exc()
    for role, ss in pools.items():
        if ss:
            with open(shard_dir / f"{spec.id}.{role}.jsonl", "w", encoding="utf-8") as f:
                for s in ss:
                    f.write(s.to_json() + "\n")
    info = {
        "id": spec.id, "repo": spec.repo, "cfg": spec.cfg, "kind": spec.kind, "family": spec.family, "lang": spec.lang, "lic": spec.lic,
        "heldout": spec.heldout, "eval_only": spec.eval_only, "counts": {r: len(v) for r, v in pools.items()}, "stats": dict(stats),
        "ptypes": dict(Counter(s.ptype for v in pools.values() for s in v)), "seconds": round(time.time() - t0, 1), "error": err,
    }
    marker.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    log(f"{spec.id:<28} train={info['counts']['train']:>6} dev={info['counts']['dev']:>4} calib={info['counts']['calib']:>4} test={info['counts']['test']:>5} ({info['seconds']}s){' ERR ' + err if err else ''}")
    return info


# ---------------------------------------------------------------------------------------------- assembly
def assemble(out_dir: Path, seed: int, infos: List[Dict[str, Any]], max_train: int = 0, extra: Optional[Dict[str, List[Path]]] = None) -> Dict[str, Any]:
    shard_dir = out_dir / "shards"
    rng = random.Random(seed)
    lines: Dict[str, List[str]] = {r: [] for r in ROLES}
    for p in sorted(shard_dir.glob("*.jsonl")):
        task, role = p.name[: -len(".jsonl")].rsplit(".", 1)
        if role in lines:
            with open(p, "r", encoding="utf-8") as f:
                lines[role].extend(l for l in f if l.strip())
    for role, paths in (extra or {}).items():
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                lines[role].extend(l for l in f if l.strip())

    def key(line: str) -> int:
        d = json.loads(line)
        return stable_hash(d.get("state", ""), d.get("question", ""), json.dumps(d.get("options") or []))

    eval_keys = set()
    for role in ("dev", "calib", "test"):
        for l in lines[role]:
            eval_keys.add(key(l))
    before = len(lines["train"])
    seen = set()
    kept = []
    for l in lines["train"]:
        k = key(l)
        if k in eval_keys or k in seen:  # no train/eval leakage, no exact duplicates
            continue
        seen.add(k)
        kept.append(l)
    lines["train"] = kept
    rng.shuffle(lines["train"])
    from . import augment as AUG

    pool = AUG.make_pool([json.loads(l)["state"] for l in lines["train"][:60000]], rng)
    n_rot = 0
    for i, l in enumerate(lines["train"]):
        d = json.loads(l)
        new = AUG.augment(d, pool, rng)
        if new is not None:
            lines["train"][i] = json.dumps(new, ensure_ascii=False, separators=(",", ":")) + chr(10)
            n_rot += 1
    if max_train and len(lines["train"]) > max_train:
        lines["train"] = lines["train"][:max_train]
    for role in ROLES:
        if role != "train":
            rng.shuffle(lines[role])
        with open(out_dir / f"{role}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(lines[role])
    from ..datafile import build_index

    build_index(out_dir / "train.jsonl")
    # a few samples of every task/role for eyeballing prompts without opening gigabytes
    per: Counter = Counter()
    with open(out_dir / "preview.jsonl", "w", encoding="utf-8") as pf:
        for role in ROLES:
            for l in lines[role]:
                d = json.loads(l)
                k = (role, d["task"], d["ptype"])
                if per[k] < 3:
                    per[k] += 1
                    d["_role"] = role
                    pf.write(json.dumps(d, ensure_ascii=False) + chr(10))
    summary: Dict[str, Any] = {"counts": {r: len(lines[r]) for r in ROLES}, "train_removed_leak_or_dup": before - len(kept), "train_context_rot": n_rot}
    fam: Dict[str, Counter] = defaultdict(Counter)
    for role in ROLES:
        for l in lines[role]:
            d = json.loads(l)
            fam[role][f"{d['ptype']}|{d.get('family','')}"] += 1
    summary["by_ptype_family"] = {r: dict(c.most_common()) for r, c in fam.items()}
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every cap (pilot builds)")
    ap.add_argument("--only", default="", help="comma-separated spec ids")
    ap.add_argument("--skip", default="", help="comma-separated spec ids")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--argent", default="", help="dir with local Argent data to convert (see data/argent.py)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop waiting for slow/hung datasets after this long and assemble what is done")
    ap.add_argument("--no-synth", action="store_true", help="skip the procedural domains (games, NPC, sorting, browser, tools)")
    a = ap.parse_args(argv)
    from .specs import SPECS

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {x for x in a.only.split(",") if x}
    skip = {x for x in a.skip.split(",") if x}
    specs = [s for s in SPECS if (not only or s.id in only) and s.id not in skip]
    log(f"{len(specs)} specs, workers={a.workers}, scale={a.scale}")
    infos: List[Dict[str, Any]] = []
    ex = ThreadPoolExecutor(max_workers=a.workers)
    futs = {ex.submit(run_spec, s, out, a.seed, a.scale, a.force): s for s in specs}
    pending = set(futs)
    deadline = time.time() + a.max_minutes * 60 if a.max_minutes else None
    while pending:
        wait_s = None if deadline is None else max(1.0, deadline - time.time())
        done, pending = wait(pending, timeout=wait_s, return_when=FIRST_COMPLETED)
        for fu in done:
            try:
                infos.append(fu.result())
            except Exception as e:
                s = futs[fu]
                log(f"FATAL {s.id}: {e}")
                infos.append({"id": s.id, "error": f"fatal: {e}", "counts": {r: 0 for r in ROLES}})
        if deadline is not None and time.time() >= deadline:
            break
    hung = [futs[fu].id for fu in pending]
    for sid in hung:
        log(f"TIMEOUT {sid}: not finished within {a.max_minutes} min — skipped")
        infos.append({"id": sid, "error": "timeout", "counts": {r: 0 for r in ROLES}})
    # shards of timed-out specs may be half written by a thread that is still running: ignore them
    for sid in hung:
        for p in (out / "shards").glob(f"{sid}.*.jsonl"):
            p.unlink(missing_ok=True)
    extra: Dict[str, List[Path]] = {}
    if a.argent:
        from . import argent as A

        for k, v in A.build(Path(a.argent), out / "shards_extra", seed=a.seed, scale=a.scale).items():
            extra.setdefault(k, []).extend(v)
    if not a.no_synth:
        from . import synth as SY

        for k, v in SY.build(out / "shards_extra", seed=a.seed, scale=a.scale).items():
            extra.setdefault(k, []).extend(v)
    summary = assemble(out, a.seed, infos, max_train=a.max_train, extra=extra)
    if not a.keep_shards:  # the assembled files hold everything; shards would double the output size
        for p in list((out / "shards").glob("*.jsonl")) + list((out / "shards_extra").glob("*.jsonl")):
            p.unlink()
    manifest = {"specs": sorted(infos, key=lambda i: i["id"]), "summary": summary, "failed": [i["id"] for i in infos if i.get("error")]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"DONE counts={summary['counts']} failed={manifest['failed']}")
    if hung:  # a hung network thread would keep the interpreter alive forever
        sys.stdout.flush()
        import os as _os

        _os._exit(0)


if __name__ == "__main__":
    main()
