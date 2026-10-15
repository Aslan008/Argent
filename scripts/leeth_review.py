"""
scripts/leeth_review.py — Reviewer CLI for the Leeth active-learning feedback loop.

The main agent (LLM) uses this to review pending Leeth fast-path decisions and
record verdicts that become real training signal (see src/leeth/finetune.py).

Usage:
  python scripts/leeth_review.py list [--all]
  python scripts/leeth_review.py show <id>
  python scripts/leeth_review.py accept  <id> [--reason R]
  python scripts/leeth_review.py reject  <id> [--reason R]
  python scripts/leeth_review.py partial <id> [--tool T] [--param P] [--reason R]
  python scripts/leeth_review.py stats

Verdict semantics (consistent with feedback.record_to_samples):
  accept  — the proposal (or the implicit skip) was correct; for run_command
            the safety verdict was also correct.
  reject  — this query must NOT be fast-dispatched (conversational/dangerous);
            teaches need_tool=0 and tool=NO_TOOL.
  partial — a tool IS needed but tool/param was wrong (or a legit query was
            wrongly skipped); provide --tool and/or --param correction.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.leeth import feedback  # noqa: E402


def _brief(rec: Dict[str, Any], idx: int) -> str:
    p = rec.get("proposal") or {}
    d = rec.get("dispatch") or {}
    tool = d.get("tool_name") or (p.get("tool") or {}).get("choice")
    conf = (p.get("tool") or {}).get("confidence")
    need = (p.get("need_tool") or {}).get("probability")
    skip = rec.get("skip_reason")
    verdict = rec.get("verdict") or "PENDING"
    lines = [
        f"#{idx} {rec['id']} [{verdict}]"
        f" need={need} tool={tool}/{conf} skip={skip or '-'}"
    ]
    lines.append(f"   query: {rec.get('query', '')[:120]}")
    if d:
        lines.append(f"   args:  {json.dumps(d.get('arguments'), ensure_ascii=False)}")
    if rec.get("safety"):
        s = rec["safety"]
        lines.append(f"   safety: safe_prob={s.get('is_safe_prob')} action={s.get('action')}")
    return "\n".join(lines)


def cmd_list(args: argparse.Namespace) -> None:
    records = feedback.load_records()
    if args.all:
        shown = records
    else:
        shown = [r for r in records if not r.get("verdict")]
    print(f"Showing {len(shown)} record(s) ({'all' if args.all else 'pending only'}; total {len(records)})")
    for i, rec in enumerate(shown, 1):
        print(_brief(rec, i))


def cmd_show(args: argparse.Namespace) -> None:
    rec = feedback.get_record(args.id)
    if rec is None:
        print(f"Record not found: {args.id}")
        sys.exit(1)
    print(json.dumps(rec, ensure_ascii=False, indent=2))


def cmd_verdict(args: argparse.Namespace) -> None:
    ok = feedback.submit_verdict(
        args.id,
        args.command,
        correction_tool=getattr(args, "tool", None),
        correction_param=getattr(args, "param", None),
        source="agent",
        reason=getattr(args, "reason", "") or "",
    )
    print(("OK: " if ok else "FAILED: ") + f"{args.command} -> {args.id}")
    sys.exit(0 if ok else 1)


def cmd_batch(args: argparse.Namespace) -> None:
    """Submits multiple verdicts from a JSON file: [{"id", "verdict", "tool", "param", "reason"}]."""
    path = Path(args.file)
    items = json.loads(path.read_text(encoding="utf-8"))
    ok_count = 0
    for item in items:
        ok = feedback.submit_verdict(
            item["id"],
            item["verdict"],
            correction_tool=item.get("tool"),
            correction_param=item.get("param"),
            source=item.get("source", "agent"),
            reason=item.get("reason", ""),
        )
        ok_count += bool(ok)
        print(("OK   " if ok else "FAIL ") + item["verdict"].ljust(8) + item["id"])
    print(f"Submitted {ok_count}/{len(items)} verdict(s).")


def cmd_stats(args: argparse.Namespace) -> None:
    print(json.dumps(feedback.stats(), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Leeth feedback reviewer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list pending (or all) records")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="show one record in full")
    p.add_argument("id")
    p.set_defaults(func=cmd_show)

    for verb in ("accept", "reject", "partial"):
        p = sub.add_parser(verb)
        p.add_argument("id")
        p.add_argument("--tool", default=None, help="correction tool (partial)")
        p.add_argument("--param", default=None, help="correction param (partial)")
        p.add_argument("--reason", default="", help="short reason")
        p.set_defaults(func=cmd_verdict)

    p = sub.add_parser("batch", help="submit verdicts from a JSON file")
    p.add_argument("file")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("stats")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()