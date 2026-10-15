"""
scripts/leeth_inspect.py — dry-run inspector for Leeth fast-path decisions.

Usage:
  python scripts/leeth_inspect.py "прочитай config.py"            # dry-run
  python scripts/leeth_inspect.py "q1" "q2" --json                # machine-readable
  python scripts/leeth_inspect.py --log "q1" "q2"                 # training batch

Dry-run (default) shows what the heads say and what the dispatcher would do
WITHOUT executing anything and WITHOUT touching the feedback buffer
(LEETH_FEEDBACK_DISABLED=1 is forced). --log records the decisions into the
buffer for the review/labeling loop — they will need honest verdicts.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _human(info: dict) -> str:
    lines = [f"'{info['query']}'"]
    if "need_tool" in info:
        nt = info["need_tool"]
        tool = info["tool"]
        lines.append(f"  need_tool : {nt['probability']:.4f} (value={nt['value']})")
        lines.append(f"  tool      : {tool['choice']} / {tool['confidence']:.4f}")
        lines.append(f"  param     : {info.get('param', info.get('param_raw', ''))!r}")
        if info.get("safety"):
            s = info["safety"]
            lines.append(f"  safety    : safe={s['is_safe']} action={s['action']}")
    if info.get("dispatch"):
        d = info["dispatch"]
        lines.append(f"  VERDICT   : DISPATCH {d['tool']} {json.dumps(d['arguments'], ensure_ascii=False)} (conf={d['confidence']:.2f})")
    else:
        reason = info.get("skip_reason") or "unknown"
        lines.append(f"  VERDICT   : SKIP ({reason}) -> delegated to LLM")
    if "elapsed_ms" in info:
        lines.append(f"  latency   : {info['elapsed_ms']} ms")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Leeth fast-path decisions (dry-run by default).")
    parser.add_argument("queries", nargs="+", help="Queries to inspect")
    parser.add_argument("--log", action="store_true",
                        help="Record decisions into the feedback buffer for labeling (training mode)")
    parser.add_argument("--triage", action="store_true",
                        help="Batch-triage all queries in ONE fan-out forward (dry-run, compact output)")
    parser.add_argument("--json", action="store_true", help="Machine-readable output (dry-run only)")
    args = parser.parse_args()

    if not args.log:
        # Dry-run must never pollute the training buffer.
        os.environ["LEETH_FEEDBACK_DISABLED"] = "1"

    from src.leeth.dispatcher import dispatcher

    if args.triage:
        infos = dispatcher.triage(args.queries)
        for info in infos:
            if info.get("dispatch"):
                d = info["dispatch"]
                print(f"DISPATCH {d['tool']:<14} {json.dumps(d['arguments'], ensure_ascii=False)} | {info['query']}")
            else:
                print(f"SKIP ({info.get('skip_reason')}) | {info['query']}")
        return

    for q in args.queries:
        if args.log:
            res = dispatcher.try_fast_dispatch(q)
            if res:
                print(f"'{q}' -> DISPATCH {res.tool_name} {json.dumps(res.arguments, ensure_ascii=False)} (logged)")
            else:
                print(f"'{q}' -> SKIP (decision logged, needs labeling)")
        else:
            info = dispatcher.inspect_query(q)
            print(args.json and json.dumps(info, ensure_ascii=False) or _human(info))


if __name__ == "__main__":
    main()