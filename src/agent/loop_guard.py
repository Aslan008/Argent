"""
Deterministic detector of repeated identical tool calls.

Small models routinely fail to notice their own loops: the same failing
command gets retried verbatim until the context fills up. Prompt-level
instructions ("if stuck, stop") do not work on them, so this guard acts at
the harness level: identical call+result pairs first earn an injected
warning, then force-end the turn.

Two different loops are caught, because they look different from outside:

* the same call returning the same answer — the failure that never moves;
* the same call returning a slightly different answer each time — the failure
  that always looks like progress (a file re-read after being rewritten, a check
  that keeps "improving"), which the result-based signature alone cannot see.
"""

import hashlib
import json
import re
from collections import deque
from difflib import SequenceMatcher

WARN_REPEATS = 2   # same call with the same result seen this many times -> warn
NUDGE_REPEATS = 4  # for smart models: inject sobering alert (nudge), don't kill the turn
STOP_REPEATS = 4   # default / small models -> force-end the turn
STOP_REPEATS_SMALL = 4
STOP_REPEATS_SMART = 6  # for smart models: emergency stop

# Tools whose verbatim repetition is legitimate by design.
EXCLUDED_TOOLS = {"wait_heartbeat"}

# The same call answered slightly differently each time is still a loop: the model
# re-reads a file it just wrote, re-runs a check that keeps "improving", and never
# stops. The signature below cannot see it — the results differ — so a run of
# identical calls is counted on its own, whatever came back. This is what keeps an
# unattended run from spinning until the context or the budget runs out.
REPEAT_ARGS_WARN = 3
REPEAT_ARGS_STOP = 5
# Asking the same thing again is how you wait rather than how you spin.
# `send_background_command` belongs here for the same reason as the two reads: the
# INPUT is often identical on purpose — a wizard asks five questions and every one of
# them is answered "y". The state advances between the calls, so the result differs;
# a genuine stuck loop (same input, same complaint back) is still caught by the
# result-based signature, which is what this exemption leaves in place.
REPEAT_ARGS_EXEMPT = {
    "wait_heartbeat", "list_background_commands", "read_background_command",
    "send_background_command", "subagents",
}

# A model that keeps issuing a *slightly different* shell one-liner (a new
# slice index, another quote, one more grep) is still looping. Compare the
# command text itself, not just the exact arguments hash: the screenshot loop
# changed the Python expression on every call, so the old guard never fired.
_COMMAND_TOOLS = {"run_command", "start_background_command"}
_COMMAND_SIMILARITY = 0.88
_COMMAND_WARN_REPEATS = 3


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
    # Collapse "RUNNING for NNs" so that consecutive idle polls of the same
    # background process produce identical signatures for the loop guard.
    (re.compile(r"RUNNING for \d+s"), "RUNNING for <elapsed>"),
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
    if level == "nudge":
        return (
            "\n\n[LOOP GUARD ALERT - REPEATED CALL DETECTED]: You have executed this exact call "
            f"{NUDGE_REPEATS} times with the exact same outcome. The environment is STATIC and will NOT change by repeating this. "
            "Do NOT repeat this call again. Reflect on why this is failing: switch tools (e.g. use browser_action(action=\"wait_for\"), read error logs, check inputs) "
            "or explain the situation to the user directly."
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
        self._commands: deque[tuple[str, str]] = deque(maxlen=window)
        self._last_call_key: str | None = None
        self._same_call_streak = 0

    def reset(self):
        self._recent.clear()
        self._commands.clear()
        self._last_call_key = None
        self._same_call_streak = 0

    @staticmethod
    def _call_key(func_name: str, arguments) -> str:
        """Identity of the call alone: same tool, same arguments, no result."""
        try:
            args_part = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            args_part = str(arguments)
        return hashlib.sha1(
            f"{func_name}|{args_part}".encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _signature(func_name: str, arguments, result: str) -> str:
        try:
            args_part = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            args_part = str(arguments)
        result_part = _normalize_volatile(str(result)[:2000])
        raw = f"{func_name}|{args_part}|{result_part}"
        return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _command_text(func_name: str, arguments) -> str:
        if func_name not in _COMMAND_TOOLS or not isinstance(arguments, dict):
            return ""
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ""
        # Volatile values (elapsed times, IDs) must not make two runs of the
        # same probe look different; whitespace/case are irrelevant too.
        text = _normalize_volatile(command)
        return re.sub(r"\s+", " ", text).strip().lower()[:4000]

    def _command_repeats(self, func_name: str, arguments) -> int:
        """How many recent commands of this tool are near-identical to this one."""
        text = self._command_text(func_name, arguments)
        if not text:
            return 0
        repeats = 0
        for prev_name, prev_text in self._commands:
            if prev_name != func_name:
                continue
            if SequenceMatcher(None, text, prev_text).ratio() >= _COMMAND_SIMILARITY:
                repeats += 1
        self._commands.append((func_name, text))
        return repeats + 1

    @staticmethod
    def _level_for_count(count: int, is_smart_model: bool) -> str | None:
        if is_smart_model:
            if count >= STOP_REPEATS_SMART:
                return "stop"
            if count >= NUDGE_REPEATS:
                return "nudge"
            if count >= WARN_REPEATS:
                return "warn"
            return None
        if count >= STOP_REPEATS_SMALL:
            return "stop"
        if count >= WARN_REPEATS:
            return "warn"
        return None

    @staticmethod
    def _command_level_for_count(count: int, is_smart_model: bool) -> str | None:
        """Similar-but-not-identical commands need one extra observation first."""
        if is_smart_model:
            if count >= STOP_REPEATS_SMART:
                return "stop"
            if count >= NUDGE_REPEATS:
                return "nudge"
            if count >= _COMMAND_WARN_REPEATS:
                return "warn"
            return None
        if count >= STOP_REPEATS_SMALL:
            return "stop"
        if count >= _COMMAND_WARN_REPEATS:
            return "warn"
        return None

    def record(self, func_name: str, arguments, result: str, enabled: bool = True, category: str | None = None) -> str | None:
        """Register an executed call. Returns None, "warn", "nudge" or "stop"."""
        if not enabled:
            return None
        try:
            from argent.config import get_loop_guard_mode
            mode = get_loop_guard_mode()
        except Exception:
            mode = "auto"
        if mode == "off":
            return None
        if func_name in EXCLUDED_TOOLS:
            return None

        key = self._call_key(func_name, arguments)
        if key == self._last_call_key:
            self._same_call_streak += 1
        else:
            self._last_call_key = key
            self._same_call_streak = 1

        sig = self._signature(func_name, arguments, result)
        self._recent.append(sig)
        count = sum(1 for s in self._recent if s == sig)

        is_smart_model = category in ("cloud", "large", "moe", "standard", "medium")
        level = self._level_for_count(count, is_smart_model)

        # Near-identical shell commands: the model rewrites a quote/index each
        # time but is still doing the same thing. Exact signatures miss this.
        command_count = self._command_repeats(func_name, arguments)
        if command_count:
            command_level = self._command_level_for_count(command_count, is_smart_model)
            rank = {"warn": 1, "nudge": 2, "stop": 3}
            if rank.get(command_level or "", 0) > rank.get(level or "", 0):
                level = command_level

        # Same call again and again, however the answer shifts: that is the spin the
        # signature cannot see. It follows the same ladder as the rest, so a capable
        # model still gets its nudge before the turn is taken away from it.
        if func_name not in REPEAT_ARGS_EXEMPT and self._same_call_streak >= REPEAT_ARGS_WARN:
            if is_smart_model:
                if self._same_call_streak >= STOP_REPEATS_SMART:
                    streak_level = "stop"
                elif self._same_call_streak >= NUDGE_REPEATS:
                    streak_level = "nudge"
                else:
                    streak_level = "warn"
            else:
                streak_level = "stop" if self._same_call_streak >= REPEAT_ARGS_STOP else "warn"
            rank = {"warn": 1, "nudge": 2, "stop": 3}
            if rank[streak_level] > rank.get(level or "", 0):
                level = streak_level

        return level

