"""Running a room with nobody at the keyboard: from a schedule or from a webhook.

A run like that cannot ask. A step marked `confirm` is therefore NOT run (it leaves "declined" in the
state), a question to the person ends the run as "needs a person", and a room may not propose new
rooms. Everything else is the ordinary run: the same engine, the same budgets and rules, the same run
record on disk, so the run shows up in `/rooms history` like any other.

`run_headless` is the function; `main` is its entry as a child process (`python -m src.rooms.headless`),
which is how a webhook runs a room without putting the tools' output on the host's screen.
"""

import json
import re
import sys
import time

#: The child prints its verdict on one line that starts with this, after whatever the tools printed.
RESULT_MARKER = "ARGENT_ROOM_RESULT "

#: How the run ended, in the words an automation uses: ok | error | denied (it needed a person).
STATUS_BY_OUTCOME = {"done": "ok", "escalated": "error", "budget": "error"}

_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def split_task(text: str) -> tuple:
    """`"night_run take a look key=value"` -> ("night_run", "take a look", {"key": "value"}): the first
    word is the room, `key=value` words are parameters, the rest is the task."""
    from src.rooms.session import parse_params
    rest, params = parse_params(text or "")
    parts = rest.split(None, 1)
    return (parts[0] if parts else ""), (parts[1] if len(parts) > 1 else ""), params


def _never(node, state) -> bool:
    return False


def run_headless(room: str, task: str = "", params: dict = None, *, library=None, runner=None,
                 record_dir=None, journal=None) -> dict:
    """Run `room` unattended. Never raises: a failure comes back as status "error" with the reason.

    Returns {status, outcome, exit, summary, history, seconds, needs_person, declined}."""
    from src.rooms.session import build_library, build_runner, execute_room_run
    started = time.monotonic()
    out = {"status": "error", "outcome": "", "exit": "", "summary": "", "history": [], "seconds": 0.0,
           "needs_person": False, "declined": [], "room": room}
    if not room or not _NAME.match(room):
        out["summary"] = f"'{room}' is not a room name"
        return out
    try:
        library = library if library is not None else build_library()
        if library.get(room) is None:
            out["summary"] = f"no room named '{room}'"
            return out
        if runner is None:
            runner = build_runner(None, library=library, confirm=_never, allow_spawn=False)
        summary = execute_room_run(task or "", start=room, params=dict(params or {}), library=library,
                                   runner=runner, journal=journal, show_progress=False,
                                   record_dir=record_dir)
    except Exception as exc:                 # a scheduler / a listener must outlive its runs
        out["summary"] = f"the run failed: {exc}"[:2000]
        out["seconds"] = round(time.monotonic() - started, 2)
        return out
    result = summary.result
    state = result.state
    declined = sorted(k[: -len(".declined")] for k, v in (state.data if state else {}).items()
                      if k.endswith(".declined") and v)
    needs_person = result.last_exit == "human_pause"
    status = STATUS_BY_OUTCOME.get(result.outcome, "error")
    if needs_person:
        status = "denied"
    out.update(status=status, outcome=result.outcome, exit=result.last_exit, summary=summary.report,
               history=[list(h) for h in result.history], seconds=round(summary.seconds, 2),
               needs_person=needs_person, declined=declined)
    return out


def main(argv=None) -> int:
    """python -m src.rooms.headless <room> [--task TEXT] [--params-file FILE]"""
    args = list(sys.argv[1:] if argv is None else argv)
    task, params_file = "", None
    for flag in ("--task", "--params-file"):
        if flag in args:
            at = args.index(flag)
            value = args[at + 1] if at + 1 < len(args) else ""
            del args[at:at + 2]
            if flag == "--task":
                task = value
            else:
                params_file = value
    if len(args) != 1:
        print("usage: python -m src.rooms.headless <room> [--task TEXT] [--params-file FILE]", file=sys.stderr)
        return 2
    params = {}
    if params_file:
        try:
            with open(params_file, encoding="utf-8") as fh:
                loaded = json.load(fh)
            params = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError) as exc:
            print(f"cannot read the parameters: {exc}", file=sys.stderr)
            return 2
    result = run_headless(args[0], task, params)
    # The tools print freely; the marker is how the parent finds the verdict.
    print(RESULT_MARKER + json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
