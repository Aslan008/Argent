"""Command-line entry point for replay.

    python -m argent.replay                 # list traces
    python -m argent.replay --latest        # newest trace
    python -m argent.replay <trace-or-session> [--json] [--content]
"""

from __future__ import annotations

import argparse
import json
import sys

from argent import features
from argent.replay.reader import latest_source, list_traces, load_source, normalize_source
from argent.replay.render import render
from argent.replay.recorder import default_trace_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="argent replay")
    parser.add_argument("source", nargs="?", help="trace .jsonl, session id, or session .json.gz")
    parser.add_argument("--latest", action="store_true", help="use the newest trace")
    parser.add_argument("--json", action="store_true", help="print raw rows as JSON")
    parser.add_argument("--content", action="store_true", help="show answer and thinking text")
    parser.add_argument("--list", action="store_true", help="list traces")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if not features.enabled("replay"):
        print("replay is disabled. Enable it with ARGENT_FEATURE_REPLAY=1 "
              "(or set features.replay=true in the Argent config).")
        return 2

    if args.list or (not args.source and not args.latest):
        traces = list_traces()
        if not traces:
            print(f"no traces in {default_trace_dir()}")
            return 0
        print(f"traces in {default_trace_dir()}:")
        for path in traces[:30]:
            print(f"  {path.name}  {path.stat().st_size} bytes")
        return 0

    source = latest_source() if args.latest and not args.source else normalize_source(args.source or "")
    if source is None:
        print(f"trace or session not found: {args.source or '(latest)'}")
        return 1

    rows = load_source(source)
    if not rows:
        print(f"no readable rows in {source}")
        return 1

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(render(rows, content=args.content))
    return 0


if __name__ == "__main__":
    sys.exit(main())
