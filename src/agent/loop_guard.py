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
import re
from collections import deque

WARN_REPEATS = 2   # same call with the same result seen this many times -> warn
STOP_REPEATS = 4   # -> force-end the turn

# Tools whose verbatim repetition is legitimate by design.
EXCLUDED_TOOLS = {"wait_heartbeat"}

# Volatile tokens that change every run even when the call is genuinely stuck
# (a failing command that keeps printing a fresh timestamp / PID / address).
# Without stripping these, the result hash differs each time and the loop is
# never caught. We deliberately touch only clearly-volatile shapes — long digit
# runs, hex/0x blobs, GUIDs, HH:MM:SS times — and leave short numbers alone so
# a legitimate re-read whose content changed by "1 -> 2" still reads as
# different (not a loop). Order matters: specific shapes before the generic
# digit-run collapse.
_VOLATILE_PATTERNS = [
    (re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"), "<guid>"),
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "<time>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<hex>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<hex>"),
    (re.compile(r"\d{4,}"), "<n>"),
]


def _normalize_volatile(text: str) -> str:
    for rx, repl in _VOLATILE_PATTERNS:
        text = rx.sub(repl, text)
    return text


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
        result_part = _normalize_volatile(str(result)[:2000])
        raw = f"{func_name}|{args_part}|{result_part}"
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
