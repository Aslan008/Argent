"""Headless worker for one bench case.

Runs in a separate process, in the worktree, so a hung model call can be killed
by the parent's timeout. Writes normalized events and a small result JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _safe_approval(action: str, destructive: bool, grant_key) -> str:
    return "deny" if destructive else "once"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="argent.bench.worker")
    parser.add_argument("--task", default="")
    parser.add_argument("--task-file", default="")
    parser.add_argument("--events", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--allowed-tools", default="")
    args = parser.parse_args(argv)

    task = args.task
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8")
    allowed = [x for x in (args.allowed_tools or "").split(",") if x] or None

    result = {"ok": False, "events": 0, "error": ""}
    events_path = Path(args.events)
    result_path = Path(args.result)
    try:
        import argent.approval as approval
        from argent.agent import ArgentAgent
        from src.bridge.session import AgentSession

        approval.set_approval_backend(_safe_approval)
        session = AgentSession(agent=ArgentAgent())
        count = 0
        with events_path.open("w", encoding="utf-8") as out:
            for event in session.handle(task, allowed_tools=allowed):
                out.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                out.flush()
                count += 1
        result.update(ok=True, events=count)
        try:
            session.close_trace()
        except Exception:
            pass
    except BaseException as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
    finally:
        try:
            result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
