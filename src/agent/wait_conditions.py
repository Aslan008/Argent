"""Conditions an autonomous run can wait ON, instead of waiting a guessed time.

`wait_heartbeat(60, "check if the build finished")` sleeps a guess and then
spends a FULL MODEL TURN — re-prefilling the whole context and generating —
just to look at a file and go back to sleep. Ten polls cost ten turns, which on
a local model is minutes of wall time. A condition evaluated here costs
essentially nothing, so the agent sleeps once and wakes when the thing actually
happened.

The condition language is deliberately tiny and evaluated by walking the AST
against a whitelist — never `eval`. Two rules shape what is in it:

* **No new authority.** Every predicate is a local read. Running a command in a
  poll loop with nobody watching is exactly the authority unattended runs
  refuse elsewhere, so there is no `command_succeeds`.
* **Nothing destructive.** Reading a background process's output DRAINS its
  queues, so a predicate that peeked at output would eat what the model reads
  next. Process state is therefore checked with poll(), which is non-invasive.
"""

import ast
import time
from pathlib import Path

from logger import get_logger

log = get_logger("agent")

# A wait without a deadline is the worst kind of failure — silent and forever.
DEFAULT_TIMEOUT_SECONDS = 600
MAX_TIMEOUT_SECONDS = 6 * 3600
DEFAULT_POLL_SECONDS = 5


class ConditionError(ValueError):
    """The condition could not be parsed or uses something not allowed."""


def _resolve(path: str) -> Path:
    from tools._helpers import _resolve_path
    return _resolve_path(path)


def _file_exists(path: str) -> bool:
    try:
        return _resolve(path).is_file()
    except OSError:
        return False


def _file_missing(path: str) -> bool:
    return not _file_exists(path)


def _file_contains(path: str, text: str) -> bool:
    try:
        p = _resolve(path)
        if not p.is_file():
            return False
        # Logs grow; read the tail rather than the whole file every poll.
        data = p.read_text(encoding="utf-8", errors="replace")
        return text in data
    except OSError:
        return False


def _process_finished(pid) -> bool:
    """True once the background process has exited.

    A pid missing from the registry counts as finished: read_background_command
    removes a process once it reports completion, and treating "gone" as "still
    running" would hang the wait forever.
    """
    from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
    key = str(pid)
    with ACTIVE_PROCESSES_LOCK:
        info = ACTIVE_PROCESSES.get(key)
    if info is None:
        return True
    process = info.get("process")
    if process is None:
        return True
    return process.poll() is not None       # non-destructive, unlike reading output


def _process_running(pid) -> bool:
    return not _process_finished(pid)


PREDICATES = {
    "file_exists": _file_exists,
    "file_missing": _file_missing,
    "file_contains": _file_contains,
    "process_finished": _process_finished,
    "process_running": _process_running,
}


def _eval_node(node) -> bool:
    if isinstance(node, ast.BoolOp):
        results = [_eval_node(v) for v in node.values]
        return all(results) if isinstance(node.op, ast.And) else any(results)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_node(node.operand)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ConditionError("only direct calls to the listed checks are allowed")
        name = node.func.id
        if name not in PREDICATES:
            raise ConditionError(
                f"unknown check '{name}'. Available: {', '.join(sorted(PREDICATES))}")
        if node.keywords:
            raise ConditionError("keyword arguments are not supported")
        args = []
        for arg in node.args:
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, (str, int)):
                raise ConditionError("check arguments must be plain strings or numbers")
            args.append(arg.value)
        try:
            return bool(PREDICATES[name](*args))
        except TypeError as e:
            raise ConditionError(f"wrong arguments for '{name}': {e}")

    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value

    raise ConditionError(f"unsupported expression element: {type(node).__name__}")


def evaluate(condition: str) -> bool:
    """Evaluate a wait condition. Raises ConditionError on anything invalid."""
    if not condition or not condition.strip():
        raise ConditionError("condition is empty")
    try:
        tree = ast.parse(condition.strip(), mode="eval")
    except SyntaxError as e:
        raise ConditionError(f"could not parse the condition: {e}")
    return _eval_node(tree.body)


def describe_predicates() -> str:
    """One-line summary for prompts and error messages."""
    return ("file_exists(path), file_missing(path), file_contains(path, text), "
            "process_finished(pid), process_running(pid); combine with and / or / not")


def wait_for(condition: str, timeout: float = DEFAULT_TIMEOUT_SECONDS,
             poll_seconds: float = DEFAULT_POLL_SECONDS, sleep=None, clock=None) -> dict:
    """Poll until the condition holds, the deadline passes, or it breaks.

    Returns {"met": bool, "reason": str, "waited": float}. A condition that
    cannot be evaluated stops the wait immediately rather than burning the whole
    timeout on an expression that will never work — and says so, so the model
    can fix it instead of guessing why it slept for ten minutes.

    ``sleep`` and ``clock`` are injected together on purpose: stubbing only the
    sleep would leave the deadline on the real clock, turning the loop into a
    busy-wait for the full timeout.
    """
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    timeout = max(1.0, min(float(timeout or DEFAULT_TIMEOUT_SECONDS), MAX_TIMEOUT_SECONDS))
    poll_seconds = max(0.1, float(poll_seconds))
    started = clock()

    while True:
        try:
            if evaluate(condition):
                return {"met": True, "reason": "condition met",
                        "waited": round(clock() - started, 1)}
        except ConditionError as e:
            return {"met": False, "reason": f"invalid condition: {e}",
                    "waited": round(clock() - started, 1)}

        waited = clock() - started
        if waited >= timeout:
            return {"met": False, "reason": f"timed out after {int(waited)}s",
                    "waited": round(waited, 1)}
        sleep(min(poll_seconds, timeout - waited))
