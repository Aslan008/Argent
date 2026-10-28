"""Command-line entry point for the agent bench.

    python -m argent.bench list
    python -m argent.bench run bench/cases/*.yaml
    python -m argent.bench run --case bench/cases/x.yaml --json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from argent import features
from argent.bench.case import CaseError, list_cases, load_case
from argent.bench.report import to_json, to_markdown
from argent.bench.runner import run_case


def _default_cases_dir() -> Path:
    return Path.cwd() / "bench" / "cases"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="argent bench")
    sub = parser.add_subparsers(dest="command")
    p_list = sub.add_parser("list", help="list case files")
    p_list.add_argument("--dir", default=str(_default_cases_dir()))
    p_run = sub.add_parser("run", help="run cases")
    p_run.add_argument("cases", nargs="*", help="case files or directories")
    p_run.add_argument("--case", action="append", default=[], help="a case file (repeatable)")
    p_run.add_argument("--repo", default=".", help="repository to work from")
    p_run.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    p_run.add_argument("--keep", action="store_true", help="keep worktrees for inspection")
    p_run.add_argument("--report", default="", help="write the markdown report to this path")
    p_promote = sub.add_parser("promote", help="turn a saved session into a draft case")
    p_promote.add_argument("--session", required=True, help="session id or .json.gz path")
    p_promote.add_argument("--turn", type=int, default=None, help="1-based user turn (default: last)")
    p_promote.add_argument("--out", default="bench/cases", help="directory for the draft")
    p_promote.add_argument("--start", default="HEAD", help="git commit/checkpoint")
    p_promote.add_argument("--force", action="store_true", help="overwrite an existing draft")
    p_mine = sub.add_parser("mine", help="find failures in traces/sessions and draft cases")
    p_mine.add_argument("--traces", default=str(Path.home() / ".argent" / "replay"))
    p_mine.add_argument("--sessions", default=str(Path.home() / ".argent" / "sessions"))
    p_mine.add_argument("--out", default="bench/cases/_drafts")
    p_mine.add_argument("--limit", type=int, default=20)
    return parser


def _case_paths(args) -> list[Path]:
    values = list(args.case) + list(args.cases)
    if not values:
        values = [str(_default_cases_dir())]
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(list_cases(path))
        elif path.exists():
            paths.append(path)
    return paths


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if not features.enabled("bench"):
        print("bench is disabled. Enable it with ARGENT_FEATURE_BENCH=1 "
              "(or set features.bench=true in the Argent config).")
        return 2
    if args.command in (None, "list"):
        cases = list_cases(Path(getattr(args, "dir", _default_cases_dir())))
        if not cases:
            print(f"no cases in {getattr(args, 'dir', _default_cases_dir())}")
            return 0
        for path in cases:
            try:
                case = load_case(path)
                checks = len(case.checks)
            except CaseError as e:
                print(f"  {path.name}: INVALID ({e})")
                continue
            print(f"  {path.name}: {case.id} ({checks} checks)")
        return 0

    if args.command == "promote":
        from argent.bench.promote import promote_session
        from argent.replay.reader import normalize_source

        source = normalize_source(args.session)
        if source is None:
            print(f"session not found: {args.session}")
            return 1
        try:
            path = promote_session(
                source, turn=args.turn, out_dir=Path(args.out),
                start=args.start, force=args.force,
            )
        except CaseError as e:
            print(str(e))
            return 1
        print(f"draft written: {path}")
        print("Add checks before running it; a case without checks is not runnable.")
        return 0

    if args.command == "mine":
        from argent.bench.miner import mine

        traces = Path(args.traces) if args.traces else None
        sessions = Path(args.sessions) if args.sessions else None
        drafts = mine(traces, sessions, Path(args.out), limit=args.limit)
        if not drafts:
            print("no new failure episodes found")
            return 0
        for path in drafts:
            print(path)
        return 0

    paths = _case_paths(args)
    if not paths:
        print("no case files found")
        return 1
    repo = Path(args.repo).resolve()
    results = []
    for path in paths:
        try:
            case = load_case(path)
        except CaseError as e:
            print(f"{path}: {e}")
            continue
        print(f"running {case.id} ...", flush=True)
        try:
            results.append(run_case(case, repo, keep=args.keep))
        except CaseError as e:
            print(f"{path}: {e}")
            continue
    if not results:
        return 1
    if args.report:
        from argent.bench.report import write_report

        write_report(results, Path(args.report))
    print(to_json(results) if args.json else to_markdown(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
