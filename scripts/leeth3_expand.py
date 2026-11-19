"""Rebuild the programmatic expansion of the accepted teacher corpus (src/leeth3/expand.py).

    python scripts/leeth3_expand.py [--data leeth3_data] [--seed 0]

Writes ``<data>/expanded/catalog.jsonl`` (tool choice on 32-255 options), ``long.jsonl`` (states of 2k-32k tokens) and
``manifest.json`` (seed, source hash, counts, sha256 of the files). Run after every ingest: the expansion is derived from
``accepted/records.jsonl`` and is rebuilt whole, so records removed from the corpus disappear from it too. Prints numbers only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leeth3.expand import approx_tokens, catalogs, long_variants  
from src.leeth3.records import parse_record  

VERSION = "expand/v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(data: Path, seed: int = 0) -> dict:
    src = data / "accepted" / "records.jsonl"
    records = [parse_record(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_dir = data / "expanded"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = {"catalog": catalogs(records, seed), "long": long_variants(records, seed)}
    man = {"version": VERSION, "seed": seed, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source": {"file": "accepted/records.jsonl", "records": len(records), "sha256": sha256(src)}, "files": {}}
    for name, recs in parts.items():
        path = out_dir / f"{name}.jsonl"
        path.write_text("".join(r.to_json() + "\n" for r in recs), encoding="utf-8")
        info = {"records": len(recs), "sha256": sha256(path), "by_domain": dict(Counter(r.domain for r in recs))}
        if name == "catalog":
            info["by_size"] = dict(sorted(Counter(len(r.options) + r.abstain_ok for r in recs).items()))
        else:
            tiers = Counter(r.id.split("-", 1)[0][2:] for r in recs)
            info["by_tier"] = dict(sorted(tiers.items()))
            info["max_tokens"] = max((approx_tokens(r.state) for r in recs), default=0)
        man["files"][f"{name}.jsonl"] = info
    (out_dir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "leeth3_data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    man = build(a.data, a.seed)
    print(json.dumps({k: v for k, v in man.items() if k != "built_at"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
