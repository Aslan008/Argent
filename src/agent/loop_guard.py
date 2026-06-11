"""
Deterministic detector of repeated identical tool calls.

Small models routinely fail to notice their own loops: the same failing
command gets retried verbatim until the context fills up. Prompt-level
instructions ("if stuck, stop") do not work on them, so this guard acts at
the harness level: identical call+result pairs first earn an injected
warning, then force-end the turn.
"""

import hashlib
import json
from collections import deque

WARN_REPEATS = 2   # same call with the same result seen this many times -> warn
STOP_REPEATS = 4   # -> force-end the turn

# Tools whose verbatim repetition is legitimate by design.
EXCLUDED_TOOLS = {"wait_heartbeat"}


def build_loop_note(level: str) -> str:
    if level == "warn":
        return (
            "\n\n[LOOP GUARD]: You have ALREADY executed this exact tool call and received "
            "the IDENTICAL result. Repeating it will not change anything. Choose a DIFFERENT "
            "approach or different arguments, or explain the blocker to the user."
        )
    return (
        f"\n\n[LOOP GUARD]: This exact call has repeated {STOP_REPEATS} times with the same "
        "result. Execution is stopped. Explain to the user what you were trying to achieve "
        "and why it keeps failing."
    )


class LoopGuard:
    """Tracks recent (tool, arguments, result) signatures in a sliding window."""

    def __init__(self, window: int = 10):
        self._recent = deque(maxlen=window)

    def reset(self):
        self._recent.clear()

    @staticmethod
    def _signature(func_name: str, arguments, result: str) -> str:
        try:
            args_part = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            args_part = str(arguments)
        raw = f"{func_name}|{args_part}|{str(result)[:2000]}"
        return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()

    def record(self, func_name: str, arguments, result: str) -> str | None:
        """Register an executed call. Returns None, "warn" or "stop"."""
        if func_name in EXCLUDED_TOOLS:
            return None
        sig = self._signature(func_name, arguments, result)
        self._recent.append(sig)
        count = sum(1 for s in self._recent if s == sig)
        if count >= STOP_REPEATS:
            return "stop"
        if count >= WARN_REPEATS:
            return "warn"
        return None
