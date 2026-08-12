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
    except Exception:
        return False


def _file_missing(path: str) -> bool:
    return not _file_exists(path)


def _file_contains(path: str, text: str) -> bool:
    try:
        p = _resolve(path)
        if not p.is_file():
            return False
        
        # Read the tail in binary to avoid encoding mismatch with search string
        text_bytes = text.encode("utf-8")
        chunk_size = max(8192, len(text_bytes) * 2)
        size = p.stat().st_size
        
        with p.open("rb") as f:
            if size > chunk_size:
                f.seek(size - chunk_size)
            data = f.read()
            return text_bytes in data
    except Exception:
        return False


def _process_finished(pid) -> bool:
    """True once the background process has exited.

    A pid missing from the registry counts as finished: read_background_command
    removes a process once it reports completion, and treating "gone" as "still
    running" would hang the wait forever.
    """
    try:
        from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
    except ImportError:
        return True
        
    if ACTIVE_PROCESSES is None:
        return True

    # Normalize float PIDs to int strings
    try:
        key = str(int(float(pid)))
    except (ValueError, TypeError):
        key = str(pid)

    with ACTIVE_PROCESSES_LOCK:
        info = ACTIVE_PROCESSES.get(key)
        
    if info is None:
        # A missing PID counts as finished because we don't track history
        # and would hang forever otherwise. The false positive risk is
        # preferable to an infinite hang.
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
    # A module with just a comment is parsed as an empty Module body.
    # evaluate() catches that before calling _eval_node.

    if isinstance(node, ast.BoolOp):
        results = [_eval_node(v) for v in node.values]
        return all(results) if isinstance(node.op, ast.And) else any(results)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_node(node.operand)
        
    if isinstance(node, ast.Compare):
        # Allow `condition == False` or `condition == True`
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            left = _eval_node(node.left)
            right = node.comparators[0]
            if isinstance(right, ast.Constant) and isinstance(right.value, bool):
                right_val = right.value
                return left == right_val if isinstance(node.ops[0], ast.Eq) else left != right_val
        raise ConditionError("only boolean constants are allowed in comparisons")

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
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, (str, int, float)):
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
        tree = ast.parse(condition.strip(), mode="exec")
    except SyntaxError as e:
        raise ConditionError(f"could not parse the condition: {e}")
        
    # Support comment-only condition
    if not tree.body:
        return True
        
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ConditionError("condition must be a single expression")
        
    return _eval_node(tree.body[0].value)


def describe_predicates() -> str:
    """One-line summary for prompts and error messages."""
    return ("file_exists(path), file_missing(path), file_contains(path, text), "
            "process_finished(pid), process_running(pid); combine with and / or / not")


def wait_for(condition: str, timeout: float = None,
             poll_seconds: float = None, sleep=None, clock=None) -> dict:
    """Poll until the condition holds, the deadline passes, or it breaks."""
    
    if timeout is None:
        timeout = DEFAULT_TIMEOUT_SECONDS
    timeout = max(0.0, min(float(timeout), MAX_TIMEOUT_SECONDS))
    
    if poll_seconds is None:
        poll_seconds = DEFAULT_POLL_SECONDS
    poll_seconds = max(0.0, float(poll_seconds))

    is_sleep_mocked = sleep is not None
    is_clock_mocked = clock is not None
    
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    
    # If sleep is mocked but clock is not, the loop will spin instantly in real time
    # We must track logical time instead of real time.
    track_logical_time = is_sleep_mocked and not is_clock_mocked
    
    started = clock()
    logical_waited = 0.0

    while True:
        current_clock = clock()
        waited = logical_waited if track_logical_time else (current_clock - started)
        
        # Guard against pathological clock returning massive values
        if waited > MAX_TIMEOUT_SECONDS * 2:
            waited = MAX_TIMEOUT_SECONDS * 2
            
        try:
            if evaluate(condition):
                return {"met": True, "reason": "condition met",
                        "waited": round(waited, 1)}
        except ConditionError as e:
            return {"met": False, "reason": f"invalid condition: {e}",
                    "waited": round(waited, 1)}

        if waited >= timeout:
            return {"met": False, "reason": f"timed out after {int(waited)}s",
                    "waited": round(waited, 1)}
                    
        sleep_duration = min(poll_seconds, timeout - waited) if poll_seconds > 0 else 0
        
        # Guard against constant clock busy spin
        if not track_logical_time and current_clock == started and logical_waited > 20 * poll_seconds:
            # If time isn't advancing and we've spun 20+ times, clock is stalled
            return {"met": False, "reason": "clock is stalled (constant time)", "waited": round(waited, 1)}
            
        if sleep:
            sleep(sleep_duration)
        logical_waited += sleep_duration
