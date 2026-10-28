"""The honest arena: Leeth against Laya on the same items (src/leeth2/fairarena.py says why and how).

    python scripts/leeth_arena_fair.py prepare                       # fetch the public rows, write the shared items (network)
    python scripts/leeth_arena_fair.py contam                        # do the test items occur in the Leeth 3 corpus?
    python scripts/leeth_arena_fair.py run-laya --repro              # step 1: Laya's own protocol must give its published numbers
    python scripts/leeth_arena_fair.py run-laya                      # all checkpoints x modes, dev + test
    python scripts/leeth_arena_fair.py run-leeth NAME MODEL_DIR      # one Leeth candidate, both modes, dev + test (+ repro items)
    python scripts/leeth_arena_fair.py report [NAME ...]             # selection on dev, test with intervals, verdict -> reports/

Everything lives in .argent/arena_fair (items, raw rows, answers); the report holds numbers and item ids only.  The rows come from
the Hugging Face dataset viewer API (JSON pages of 100 rows), so neither ``datasets`` nor a parquet reader is needed.  Runs on CPU
by default; ``--device cuda`` on a GPU machine.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.leeth2 import fairarena as fa  

WORK = ROOT / ".argent" / "arena_fair"
ITEMS = WORK / "items.jsonl"
PREDS = WORK / "preds"
LAYA_DIR = ROOT / ".argent" / "laya" / "laya"          # snapshot of convaiinnovations/laya
LAYA_PKG = ROOT / ".argent" / "laya" / "pkg"           # the laya 0.4.12 wheel, unpacked (not installed into .venv)
LAYA_CKPTS = {"en": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
LAYA_MODES = ("single", "wide", "tournament")          # README: one question / head_max_len=512,max_len=1024 / predict_tournament(16)
LAYA_ORDER = LAYA_MODES + ("score",)                   # tie-break order of the dev selection (simpler first)
LEETH_MODES = ("single", "split16")
API = "https://datasets-server.huggingface.co"


def _lines(path: Path) -> List[str]:
    """A JSONL file's lines.  Not ``read_text().splitlines()``: that also cuts at U+2028 / U+0085, which XNLI has inside its texts
    (json.dumps(ensure_ascii=False) writes them raw), and a cut record no longer parses."""
    with open(path, encoding="utf-8") as fh:
        return [line for line in fh]


def _records(path: Path) -> List[Dict[str, Any]]:
    """The answers written so far.  A run killed mid-write leaves its last line cut short: that answer is skipped
    (and asked again by the next run), the rest are kept."""
    out = []
    for line in _lines(path):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ------------------------------------------------------------------------------------------------ fetching
_SESSION = None


def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    global _SESSION
    import requests

    _SESSION = _SESSION or requests.Session()
    err: Optional[str] = None
    for attempt in range(10):
        try:
            r = _SESSION.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}"                 # 429 / 5xx from the viewer
            if r.status_code == 429:                      # rate limit: wait as long as the viewer asks
                try:
                    time.sleep(min(300, float(r.headers.get("Retry-After", 30))))
                except ValueError:
                    time.sleep(30)
                continue
        except requests.RequestException as e:          # a reset connection: a new session, then retry
            err = f"{type(e).__name__}: {e}"
            _SESSION = requests.Session()
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"{url} {params}: {err}")


def class_names(task: fa.Task) -> Optional[List[str]]:
    info = _get(f"{API}/info", {"dataset": task.dataset, "config": task.config})["dataset_info"]
    feat = info["features"].get(task.label) or {}
    return list(feat["names"]) if feat.get("_type") == "ClassLabel" else None


def split_size(task: fa.Task, split: str) -> int:
    info = _get(f"{API}/info", {"dataset": task.dataset, "config": task.config})["dataset_info"]
    return int(info["splits"][split]["num_examples"])


def fetch_rows(task: fa.Task, split: str, offsets: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Rows by index, cached in WORK/raw; consecutive offsets are fetched as pages of up to 100."""
    cache = WORK / "raw" / f"{task.dataset.replace('/', '__')}__{task.config}__{split}.jsonl"
    have: Dict[int, Dict[str, Any]] = {}
    if cache.exists():
        for line in _lines(cache):
            if line.strip():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:          # a line cut short when an earlier run was killed: fetched again below
                    continue
                have[d["i"]] = d["row"]
    need = sorted(set(offsets) - set(have))
    pages: List[List[int]] = []
    for i in need:                                    # a run of near offsets (gaps of < 4 rows, at most 100) is one request;
        if pages and i - pages[-1][-1] < 4 and i - pages[-1][0] < 100:   # a sparse sample asks for its rows alone
            pages[-1].append(i)
        else:
            pages.append([i])
    cache.parent.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor

    def one(pg):
        return _get(f"{API}/rows", {"dataset": task.dataset, "config": task.config, "split": split, "offset": pg[0], "length": pg[-1] - pg[0] + 1})

    with cache.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(2) as pool:
        for got in pool.map(one, pages):
            for r in got.get("rows", []):
                i = int(r["row_idx"])
                if i not in have:
                    have[i] = r["row"]
                    fh.write(json.dumps({"i": i, "row": r["row"]}, ensure_ascii=False) + "\n")
    return {i: have[i] for i in offsets if i in have}


def full_split(task: fa.Task, split: str) -> Dict[int, Dict[str, Any]]:
    return fetch_rows(task, split, range(split_size(task, split)))


def task_items(task: fa.Task, split_role: str, want: int, names: Optional[List[str]], labels: List[str]) -> List[Dict[str, Any]]:
    """``want`` items of the task from the test split (role "test") or the dev split (role "dev"), sampled with the fixed seed.
    Small splits are read whole (rows out of the task, e.g. CLINC's out-of-scope, are dropped before sampling); a big one
    (DBpedia's 70 000 / 560 000) is read at 1.2x the wanted offsets, row by row."""
    split = task.test_split if split_role == "test" else task.dev_split
    n = split_size(task, split)
    if n <= 16000:
        rows = full_split(task, split)
    else:
        rows = fetch_rows(task, split, fa.sample_indices(n, want + want // 5, task.name, split_role, "pool"))
    items = [it for i, r in sorted(rows.items()) if (it := fa.make_item(task, split_role, i, r, labels, names))]
    pick = set(fa.sample_indices(len(items), want, task.name, split_role))
    return [it for j, it in enumerate(items) if j in pick]


def repro_items(name: str, spec: Dict[str, Any], names: Optional[List[str]], labels: List[str]) -> List[Dict[str, Any]]:
    task = fa.TASKS[spec["task"]]
    n = split_size(task, task.test_split)
    offs = list(range(min(n, spec["rows"] * 2))) if spec["first"] else fa.sample_indices(n, spec["rows"] * 2, name)
    rows = fetch_rows(task, task.test_split, offs)
    out = []
    for i, r in sorted(rows.items()):
        it = fa.make_item(task, "repro", i, r, labels, names, k=spec.get("k", 0), seed=13 if spec.get("k") else fa.SEED)
        if it:
            it["id"] = f"{name}:{i}"
            it["repro"] = name
            out.append(it)
        if spec["first"] and len(out) >= spec["rows"]:
            break
    if not spec["first"]:
        pick = set(fa.sample_indices(len(out), spec["rows"], name, "pick"))
        out = [it for j, it in enumerate(out) if j in pick]
    return out


def cmd_prepare(a) -> None:
    items: List[Dict[str, Any]] = []
    meta: Dict[str, Dict[str, Any]] = {}
    for name, task in fa.TASKS.items():
        try:                                          # one unreachable dataset must not cost the others: rows are cached, a rerun resumes
            names = class_names(task) if task.kind == "cls" else None
            labels = fa.label_set(task, full_split(task, task.test_split).values() if names is None and task.kind == "cls" else [], names)
            test = task_items(task, "test", fa.N_TEST, names, labels)
            dev = task_items(task, "dev", fa.N_DEV, names, labels)
        except RuntimeError as e:
            meta[name] = {"missing": str(e)[:300], "group": task.group}
            print(f"{name:<12} NOT FETCHED ({str(e)[:160]}) -- run prepare again", flush=True)
            continue
        items += dev + test
        meta[name] = {"labels": len(labels), "dev": len(dev), "test": len(test), "group": task.group}
        print(f"{name:<12} labels {len(labels):>3}  dev {len(dev):>3}  test {len(test):>3}", flush=True)
        for rname, spec in fa.REPRO.items():
            if spec["task"] == name:
                rep = repro_items(rname, spec, names, labels)
                items += rep
                meta[rname] = {"items": len(rep), "of": name}
                print(f"  {rname:<20} {len(rep)}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    ITEMS.write_text(body, encoding="utf-8")
    import hashlib

    (WORK / "manifest.json").write_text(json.dumps({"seed": fa.SEED, "sha256": hashlib.sha256(body.encode()).hexdigest(), "tasks": meta},
                                                   indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} items -> {ITEMS}")


def load_items(split: Optional[Sequence[str]] = None, tasks: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    out = [json.loads(line) for line in _lines(ITEMS) if line.strip()]
    return [it for it in out if (not split or it["split"] in split) and (not tasks or it["task"] in tasks or it.get("repro") in tasks)]


# ------------------------------------------------------------------------------------------------ contamination
def cmd_contam(a) -> None:
    """Do the test texts occur in what Leeth 3 trains on?  The local Leeth 3 corpus (teacher records + their expansions) is checked
    directly; the Leeth 2 corpora are built by leeth2.data.build, which takes no train rows of a held-out task (caps[0] = 0 or
    eval_only) -- their dev/validation rows feed only Leeth's dev and temperature pools, never its weights."""
    data = ROOT / "leeth3_data"
    corpus = set()
    for p in [data / "accepted" / "records.jsonl", *sorted((data / "expanded").glob("*.jsonl"))]:
        if p.exists():
            for line in _lines(p):
                if line.strip():
                    corpus |= fa.shingles(json.loads(line).get("state", ""))
    hits = Counter()
    n = Counter()
    for it in load_items(["test"]):
        txt = it.get("text") or (it["premise"] + " " + it["hypothesis"])
        n[it["task"]] += 1
        if fa.contaminated(txt, corpus):
            hits[it["task"]] += 1
            print("  contaminated:", it["id"])
    for t in fa.TASKS:
        print(f"{t:<12} {hits[t]:>3} of {n[t]}")


# ------------------------------------------------------------------------------------------------ running
def _write(name: str, rows: List[Dict[str, Any]]) -> None:
    PREDS.mkdir(parents=True, exist_ok=True)
    with (PREDS / f"{name}.jsonl").open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _done(name: str) -> set:
    p = PREDS / f"{name}.jsonl"
    if not p.exists():
        return set()
    return {(d["config"], d["id"]) for d in _records(p)}


def cmd_run_leeth(a) -> None:
    import numpy as np
    import torch

    from src.leeth2.evaluate import predict
    from src.leeth2.runtime import Leeth2Runtime
    from src.leeth2.schema import Sample

    torch.set_num_threads(a.threads)
    rt = Leeth2Runtime.load(a.model_dir, device=a.device)
    items = load_items(a.splits, a.tasks)
    done = _done(a.name)
    for mode in LEETH_MODES:
        rt.model.cfg.split_k = 0 if mode == "single" else 16
        todo = [it for it in items if (mode, it["id"]) not in done and (mode == "single" or len(it["options"]) > 16)]
        for k in range(0, len(todo), 256):
            chunk = todo[k:k + 256]
            samples = []
            for it in chunk:
                task = fa.TASKS[it["task"]]
                state, q = fa.leeth_question(task, it)
                samples.append(Sample(it["task"], task.group, it["lang"], "choice", state, q, list(it["options"]), it["gold"], split="test"))
            preds = predict(rt.model, rt.encoder, samples, device=a.device, calibrated=True, tokens_per_batch=6000)
            _write(a.name, [{"model": a.name, "config": mode, "task": it.get("repro") or it["task"], "split": it["split"],
                             **fa.outcome(it, list(map(float, p.probs)))} for it, p in zip(chunk, preds)])
            print(f"{a.name} {mode}: {k + len(chunk)}/{len(todo)}", flush=True)
    lat = []
    one = [it for it in items if it["split"] == "test"][:40]
    rt.model.cfg.split_k = 0
    for it in one:
        task = fa.TASKS[it["task"]]
        state, q = fa.leeth_question(task, it)
        t0 = time.perf_counter()
        predict(rt.model, rt.encoder, [Sample(it["task"], task.group, it["lang"], "choice", state, q, list(it["options"]), it["gold"])], device=a.device)
        lat.append((time.perf_counter() - t0) * 1000)
    _write(a.name + ".latency", [{"model": a.name, "ms_p50": float(np.median(lat[5:])), "n": len(lat) - 5, "threads": a.threads}])


def cmd_run_laya(a) -> None:
    import numpy as np
    import torch

    sys.path.insert(0, str(LAYA_PKG))
    import laya

    torch.set_num_threads(a.threads)
    items = load_items(["repro"] if a.repro else ["test"] if a.selected else a.splits, a.tasks)
    chosen: Dict[str, tuple] = {}
    if a.selected:                                   # the test split only for the checkpoint and mode each task chose on dev
        tab = _merged([f"laya-{c}" for c in LAYA_CKPTS])
        for t in fa.TASKS:
            ch = _choose(tab, [f"laya-{c}" for c in LAYA_CKPTS], t, LAYA_ORDER)
            if ch:
                chosen[t] = ch[:2]
                print(f"{t:<12} chosen on dev: {ch[0]} / {ch[1]} ({ch[2]:.3f})", flush=True)
        items = [it for it in items if it["split"] == "test" and it["task"] in chosen]
    for ck in a.ckpts:
        name = f"laya-{ck}"
        if a.selected and not any(m == name for m, _ in chosen.values()):
            continue
        agent = laya.load(str(LAYA_DIR), subfolder=LAYA_CKPTS[ck], device=a.device)
        done = _done(name)
        modes = ("single", "tournament") if a.repro else LAYA_MODES
        for mode in modes + ("score",):                # score: SST-5 only, the primitive Laya publishes SST-5 with
            todo = []
            for it in items:
                task = fa.TASKS[it["task"]]
                if a.selected and chosen[task.name] != (name, mode) and not (mode == "single" and chosen[task.name][0] == name
                                                                              and len(it["options"]) <= 16):
                    continue                         # (a wide/tournament choice answers small items as "single" does)
                if mode == "score" and task.name != "sst5":
                    continue
                if mode in ("wide", "tournament") and len(it["options"]) <= 16:
                    continue                         # identical to "single" when the options fit a group
                if (mode, it["id"]) not in done:
                    todo.append(it)
            lat = []
            for j, it in enumerate(todo):
                task = fa.TASKS[it["task"]]
                state, qs = fa.laya_request(task, it, score=(mode == "score"))
                t0 = time.perf_counter()
                try:
                    if mode == "tournament":
                        res = laya.predict_tournament(agent, state, qs, group_size=16)
                    elif mode == "wide":
                        res = agent.predict(state, qs, head_max_len=512, max_len=1024)
                    else:
                        res = agent.predict(state, qs)
                    out = fa.outcome(it, fa.laya_probs(it, res["answers"]["q"]))
                except Exception as e:              # counted as an error AND a wrong answer, never dropped
                    out = fa.outcome(it, None, error=f"{type(e).__name__}: {str(e)[:160]}")
                lat.append((time.perf_counter() - t0) * 1000)
                _write(name, [{"model": name, "config": mode, "task": it.get("repro") or it["task"], "split": it["split"], **out}])
                if j % 100 == 0:
                    print(f"{name} {mode}: {j}/{len(todo)}  ({np.median(lat):.0f} ms)", flush=True)
            if lat:
                _write(name + ".latency", [{"model": name, "config": mode, "ms_p50": float(np.median(lat)), "n": len(lat), "threads": a.threads}])
        del agent


# ------------------------------------------------------------------------------------------------ report
def _load_preds(name: str) -> fa.Table:
    tab = fa.Table()
    p = PREDS / f"{name}.jsonl"
    if p.exists():
        seen = set()
        for d in _records(p):
            if (d["config"], d["id"]) not in seen:
                seen.add((d["config"], d["id"]))
                tab.add(d["model"], d["config"], d["task"], d["split"], [d])
    return tab


def _merged(names: Sequence[str]) -> fa.Table:
    tab = fa.Table()
    for n in names:
        tab.rows.update(_load_preds(n).rows)
    return tab


def _choose(tab: fa.Table, models: Sequence[str], task: str, order: Sequence[str]) -> Optional[tuple]:
    """(model, config, dev accuracy) with the best dev accuracy over ``models`` x their configs; ties to the earlier model/config."""
    best = None
    for m in models:
        for c in order:
            dev = tab.get(m, c, task, "dev")
            if not dev:
                continue
            acc = fa.accuracy(dev)
            if best is None or acc > best[2] + 1e-12:
                best = (m, c, acc)
    return best


def _fill_small(tab: fa.Table, m: str, c: str, task: str, split: str) -> List[Dict[str, Any]]:
    """Modes that skip items with <= 16 options answer them as "single" does."""
    rows = tab.get(m, c, task, split)
    if c != "single":
        have = {r["id"] for r in rows}
        rows = rows + [r for r in tab.get(m, "single", task, split) if r["id"] not in have]
    return rows


def cmd_report(a) -> None:
    import numpy as np

    items = {it["id"]: it for it in load_items()}
    gold = {i: it["gold"] for i, it in items.items()}
    laya_models = [f"laya-{c}" for c in LAYA_CKPTS]
    tab = _merged(laya_models + list(a.names))
    rep: Dict[str, Any] = {"date": time.strftime("%Y-%m-%d"), "seed": fa.SEED, "repro": {}, "tasks": {}, "verdicts": {}}
    # 1) Laya's reproduction of its own numbers
    for rname, spec in fa.REPRO.items():
        for key, pub in spec["published"].items():
            ck, _, mode = key.partition("/")
            rows = tab.get(f"laya-{ck}", "score" if spec.get("score") else (mode or "single"), rname, "repro")
            if rows:
                acc = fa.accuracy(rows)
                rep["repro"][f"{rname} {key}"] = {"measured": round(acc, 3), "published": pub, "n": len(rows),
                                                  "errors": sum(r["error"] is not None for r in rows), "ok": fa.repro_ok(acc, pub, len(rows))}
    # 2) per task: each side's config chosen on dev, scored on test
    lines = [f"{'task':<12}{'side':<22}{'config (dev)':<34}{'test acc':>9}{'err':>5}{'ECE ship':>9}{'ECE refit':>10}"]
    picks: Dict[str, Dict[str, tuple]] = defaultdict(dict)
    for task in fa.TASKS:
        rep["tasks"][task] = {}
        sides = {"laya": laya_models, **{n: [n] for n in a.names}}
        for side, models in sides.items():
            order = LAYA_ORDER if side == "laya" else LEETH_MODES
            ch = _choose(tab, models, task, order)
            if not ch:
                continue
            m, c, dacc = ch
            test = _fill_small(tab, m, c, task, "test")
            dev = _fill_small(tab, m, c, task, "dev")
            ok = [r for r in test if r["probs"]]
            ship = fa.ece([r["conf"] for r in test], [r["ok"] for r in test])
            t, refit = fa.refit_ece(dev, test, gold)
            picks[task][side] = (m, c, {r["id"]: r["ok"] for r in test})
            rep["tasks"][task][side] = {"model": m, "config": c, "dev_acc": round(dacc, 3), "test_acc": round(fa.accuracy(test), 3), "n": len(test),
                                        "errors": len(test) - len(ok), "ece_shipped": round(ship, 3), "ece_refit": round(refit, 3), "temperature": round(t, 3)}
            lines.append(f"{task:<12}{side:<22}{m + ' / ' + c:<34}{fa.accuracy(test):>9.3f}{len(test) - len(ok):>5}{ship:>9.3f}{refit:>10.3f}")
    # 3) verdicts, held-out group and home group separately
    for name in a.names:
        for group, tasks in (("heldout", fa.HELDOUT), ("home", fa.HOME)):
            aa, bb = {}, {}
            for t in tasks:
                if name in picks[t] and "laya" in picks[t]:
                    ids = sorted(set(picks[t][name][2]) & set(picks[t]["laya"][2]))
                    aa[t] = [picks[t][name][2][i] for i in ids]
                    bb[t] = [picks[t]["laya"][2][i] for i in ids]
            if aa:
                boot = fa.paired_bootstrap(aa, bb)
                rep["verdicts"][f"{name} / {group}"] = {"verdict": fa.verdict(boot), **boot}
    for f in sorted((PREDS).glob("*.latency.jsonl")):
        rep.setdefault("latency", {})[f.stem.replace(".latency", "")] = [json.loads(x) for x in _lines(f) if x.strip()]
    out = ROOT / "reports" / f"leeth_arena_fair_{a.tag or time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print("\nreproduction of Laya's own numbers:")
    for k, v in rep["repro"].items():
        print(f"  {k:<36} {v['measured']:.3f} vs {v['published']:.3f}  n {v['n']}  errors {v['errors']}  {'OK' if v['ok'] else 'MISS'}")
    for k, v in rep["verdicts"].items():
        print(f"\n{k}: {v['verdict']}  mean {v['mean']['diff']:+.3f} [{v['mean']['lo']:+.3f}, {v['mean']['hi']:+.3f}]")
        for t, d in v["tasks"].items():
            print(f"  {t:<12} {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  n {d['n']}")
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("contam")
    def runner(p, splits):
        p.add_argument("--splits", nargs="+", default=splits)
        p.add_argument("--tasks", nargs="+", default=None)
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=8)

    p = sub.add_parser("run-leeth")
    p.add_argument("name")
    p.add_argument("model_dir")
    runner(p, ["dev", "test", "repro"])
    p = sub.add_parser("run-laya")
    p.add_argument("--repro", action="store_true", help="only Laya's own protocol (the reproduction check)")
    p.add_argument("--ckpts", nargs="+", default=list(LAYA_CKPTS), choices=list(LAYA_CKPTS))
    p.add_argument("--selected", action="store_true", help="test split only, for the checkpoint+mode each task chose on dev")
    runner(p, ["dev"])
    p = sub.add_parser("report")
    p.add_argument("names", nargs="*", help="Leeth candidates (their run-leeth names)")
    p.add_argument("--tag", default="")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "contam": cmd_contam, "run-leeth": cmd_run_leeth, "run-laya": cmd_run_laya, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
