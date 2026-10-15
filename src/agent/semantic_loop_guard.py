"""
Semantic Loop Guard for Argent.

Detects high-level regression cycles and edit-revert loops that escape
verbatim tool-call detectors:
1. File-state oscillation (State A <-> State B): The model edits a file, breaks
   a test, edits another file (or reverts), and returns the files to a state
   seen 2-3 steps ago.
2. Error-signature ping-pong: The model alternates between Error X and Error Y,
   where fixing one directly reproduces the other.

Unlike low-level LoopGuard, SemanticLoopGuard maintains session-level history
across file writes and test runs.
"""

import hashlib
import os
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from logger import get_logger

log = get_logger("semantic_loop_guard")

# How many past state snapshots to retain in the sliding window
SNAPSHOT_WINDOW = 8
# Consecutive oscillations needed to trigger warning / stop
OSCILLATION_WARN_THRESHOLD = 2
OSCILLATION_STOP_THRESHOLD = 4

# Patterns to normalize volatile parts of compiler/test outputs
_VOLATILE_REGEXES = [
    (re.compile(r"0x[0-9a-fA-F]+"), "<hex>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<hash>"),
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b"), "<time>"),
    (re.compile(r"in\s+\d+(?:\.\d+)?s", re.IGNORECASE), "in <elapsed>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|seconds|s)\b", re.IGNORECASE), "<duration>"),
    (re.compile(r"(?:line\s+|:)\d+(?::\d+)?", re.IGNORECASE), ":<line>"),
    (re.compile(r"[A-Za-z]:\\[^:\s\n]+"), "<abspath>"),
    (re.compile(r"/(?:[^:\s\n]+/)+[^:\s\n]+"), "<abspath>"),
]


def normalize_error_signature(text: str) -> str:
    """Extract a canonical signature from a test failure or compiler error log."""
    if not text:
        return ""

    # 1. Pytest / unittest failures: extract failed test names and assertions
    failed_tests = re.findall(r"FAILED\s+([^\s:]+(?:::[\w_]+)+)", text)
    if failed_tests:
        return "pytest_failed:" + ",".join(sorted(set(failed_tests)))

    # 2. Python exception: capture exception class and core message
    exc_matches = re.findall(r"([A-Z]\w*(?:Error|Exception|Warning)):\s*([^\n]+)", text)
    if exc_matches:
        last_exc_type, last_msg = exc_matches[-1]
        norm_msg = last_msg
        for rx, repl in _VOLATILE_REGEXES:
            norm_msg = rx.sub(repl, norm_msg)
        return f"py_exc:{last_exc_type}:{norm_msg.strip()[:100]}"

    # 3. C/C++ / MSVC / Clang errors: capture error code / category
    c_errors = re.findall(r"(?:error|fatal error)\s*(?:C\d+|CS\d+|LNK\d+|:\s*[^\n]+)", text, re.IGNORECASE)
    if c_errors:
        norm_errs = []
        for err in c_errors[:3]:
            cleaned = err
            for rx, repl in _VOLATILE_REGEXES:
                cleaned = rx.sub(repl, cleaned)
            norm_errs.append(cleaned.strip())
        return "c_error:" + ";".join(norm_errs)

    # 4. Fallback: normalize volatile patterns on the first 500 chars
    cleaned = text[:500]
    for rx, repl in _VOLATILE_REGEXES:
        cleaned = rx.sub(repl, cleaned)
    return "raw:" + hashlib.sha1(cleaned.encode("utf-8", errors="replace")).hexdigest()[:16]


class SemanticLoopGuard:
    """Tracks high-level project states and error signatures across turns."""

    def __init__(self, window: int = SNAPSHOT_WINDOW):
        self._window = window
        self._file_state_history: deque[Dict[str, str]] = deque(maxlen=window)
        self._error_signature_history: deque[str] = deque(maxlen=window)
        self._browser_eval_history: deque[Dict[str, Any]] = deque(maxlen=window)
        self._revert_count = 0
        self._error_ping_pong_count = 0
        self._tracked_files: set[str] = set()
        self._hash_cache: Dict[str, Tuple[float, int, str]] = {}

    def reset(self) -> None:
        """Reset all tracked state."""
        self._file_state_history.clear()
        self._error_signature_history.clear()
        self._browser_eval_history.clear()
        self._revert_count = 0
        self._error_ping_pong_count = 0
        self._tracked_files.clear()
        self._hash_cache.clear()

    def _hash_file(self, path: str) -> Optional[str]:
        """Compute SHA-1 hash of a file's content with mtime/size caching."""
        try:
            p = Path(path)
            if not p.is_file():
                return None
            st = p.stat()
            mtime, size = st.st_mtime, st.st_size
            cached = self._hash_cache.get(path)
            if cached and cached[0] == mtime and cached[1] == size:
                return cached[2]

            data = p.read_bytes()
            h = hashlib.sha1(data).hexdigest()
            self._hash_cache[path] = (mtime, size, h)
            return h
        except Exception:
            return None

    def _snapshot_tracked_files(self) -> Dict[str, str]:
        """Snapshot current state of all files touched in this session."""
        snapshot = {}
        for f in sorted(self._tracked_files):
            h = self._hash_file(f)
            snapshot[f] = h if h is not None else "<deleted>"
        return snapshot

    def record_file_edit(self, file_path: str) -> Optional[str]:
        """Record a successful file modification and check for edit-revert oscillations.

        Returns:
            "warn" | "stop" | None
        """
        if not file_path:
            return None

        norm_path = os.path.normcase(os.path.abspath(file_path))
        self._tracked_files.add(norm_path)
        self._hash_cache.pop(norm_path, None)

        current_snapshot = self._snapshot_tracked_files()
        if not current_snapshot:
            return None

        # Check if current snapshot matches a past snapshot in the window
        # We check period 2 (State A -> B -> A) and period 3 (State A -> B -> C -> A)
        history_len = len(self._file_state_history)
        match_period = None

        if history_len >= 2:
            # Check period 2: did we just revert to 2 steps ago?
            if current_snapshot == self._file_state_history[-2]:
                match_period = 2
            elif history_len >= 3 and current_snapshot == self._file_state_history[-3]:
                match_period = 3

        self._file_state_history.append(current_snapshot)

        if match_period is not None:
            self._revert_count += 1
            log.warning(
                "SemanticLoopGuard: Edit-revert oscillation detected (period %d, count %d)",
                match_period,
                self._revert_count,
            )
            if self._revert_count >= OSCILLATION_STOP_THRESHOLD:
                return "stop"
            if self._revert_count >= OSCILLATION_WARN_THRESHOLD:
                return "warn"
        else:
            self._revert_count = max(0, self._revert_count - 1)

        return None

    def record_command_result(self, command: str, output: str, exit_code: int) -> Optional[str]:
        """Record command/test outcome and detect alternating error loops.

        Returns:
            "warn" | "stop" | None
        """
        if exit_code == 0:
            self._error_ping_pong_count = 0
            self._error_signature_history.clear()
            return None

        sig = normalize_error_signature(output)
        if not sig:
            return None

        # Check for error ping-pong: Error A -> Error B -> Error A
        history_len = len(self._error_signature_history)
        is_ping_pong = False

        if history_len >= 2:
            prev_prev = self._error_signature_history[-2]
            prev = self._error_signature_history[-1]
            if sig == prev_prev and sig != prev:
                is_ping_pong = True

        self._error_signature_history.append(sig)

        if is_ping_pong:
            self._error_ping_pong_count += 1
            log.warning(
                "SemanticLoopGuard: Error ping-pong detected between '%s' and '%s' (count %d)",
                sig,
                self._error_signature_history[-2],
                self._error_ping_pong_count,
            )
            if self._error_ping_pong_count >= OSCILLATION_STOP_THRESHOLD:
                return "stop"
            if self._error_ping_pong_count >= OSCILLATION_WARN_THRESHOLD:
                return "warn"
        else:
            self._error_ping_pong_count = max(0, self._error_ping_pong_count - 1)

        return None

    def reset_browser_eval(self) -> None:
        """Reset browser evaluation tracking when an action changes page state."""
        self._browser_eval_history.clear()

    @staticmethod
    def _is_unproductive_eval_result(result_str: str) -> bool:
        """Classify whether an eval_js outcome represents an unproductive probe."""
        res = (result_str or "").strip()
        # Explicit void / null / undefined / errors
        if res in ("JS result: null", "JS result: undefined", 'JS result: "__ARGENT_UNDEFINED__"', "JS result:"):
            return True
        if res.startswith("Error evaluating JS:") or res.startswith("Error:"):
            return True
        if "Cannot read properties" in res or "is not defined" in res or "is not a function" in res:
            return True
        # Real DOM element or collection indicates legitimate inspection
        if '"__type": "Element"' in res or '"__type":"Element"' in res:
            return False

        # Primitive echo probes (e.g. '{"ping":1}', '"sync works"', 'typeof fetch: function', 'x42', boolean, short scalars)
        # These are commonly emitted during brute-force store probing
        norm_val = res.removeprefix("JS result:").strip().strip('"\'')
        echo_tokens = ("ping", "pong", "test", "ok", "sync works", "typeof", "function", "true", "false", "undefined", "object", "null")
        if len(norm_val) <= 40 and any(tok in norm_val.lower() for tok in echo_tokens):
            return True
        if len(norm_val) <= 10:  # e.g. "x42", "42", "0"
            return True

        return False

    def record_browser_eval(self, expression: str, result: str) -> Optional[str]:
        """Track browser_action(action="eval_js") calls to detect futile reverse-engineering loops.

        Returns:
            "warn" | "stop" | None
        """
        is_unproductive = self._is_unproductive_eval_result(result)
        self._browser_eval_history.append({
            "expression": expression,
            "unproductive": is_unproductive,
        })

        hist_len = len(self._browser_eval_history)
        if hist_len < 5:
            return None

        unproductive_count = sum(1 for item in self._browser_eval_history if item["unproductive"])

        # Hard stop: 8+ unproductive probes in the sliding window or 10 consecutive
        if unproductive_count >= 8 or hist_len >= 10 and unproductive_count >= 7:
            return "stop"

        # Warning: 5+ unproductive probes in the window
        if unproductive_count >= 5:
            return "warn"

        return None


# --- reading a browser_action call -----------------------------------------
# One tool, many verbs: the guards used to key on a tool NAME per verb
# (browser_eval_js, browser_click, ...). Those names are gone, and a guard that
# matches a name nothing calls is silently off — the eval_js probing guard was
# exactly that. The verb now lives in the `action` argument.

_EVAL_ACTIONS = frozenset({"eval_js", "eval"})

# Actions that change the page, so earlier JavaScript probes no longer describe
# it. Reading actions (eval_js, scroll, switch_tab, close_tab) are not here.
_PAGE_CHANGING_ACTIONS = frozenset({
    "click", "type", "input", "select", "upload_file", "upload", "reload", "back",
    "press_key", "press", "wait_for", "wait", "act", "stagehand_act", "ai_act",
})


def _browser_action_name(arguments) -> str:
    if not isinstance(arguments, dict):
        return ""
    return str(arguments.get("action") or "").strip().lower()


def is_browser_eval_call(arguments) -> bool:
    """True for browser_action(action="eval_js")."""
    return _browser_action_name(arguments) in _EVAL_ACTIONS


def browser_eval_expression(arguments) -> str:
    """The JavaScript a browser_action(eval_js) call carries.

    Mirrors the tool: the expression travels in `text`, else `value`, else
    `instruction`.
    """
    if not isinstance(arguments, dict):
        return ""
    for key in ("text", "value", "instruction"):
        if arguments.get(key) is not None:
            return str(arguments[key])
    return ""


def is_page_changing_browser_action(arguments) -> bool:
    """True when a browser_action call alters the page (a click, typing, a reload…)."""
    return _browser_action_name(arguments) in _PAGE_CHANGING_ACTIONS


def build_semantic_loop_note(level: str, loop_type: str = "edit_revert") -> str:
    """Build actionable intervention directive for the LLM."""
    if loop_type == "browser_probing":
        if level == "warn":
            return (
                "\n\n[SEMANTIC BROWSER GUARD]: Unproductive JavaScript inspection loop detected. "
                "Multiple consecutive browser_action(action=\"eval_js\") calls returned null, empty, or dummy test results without modifying page state. "
                "STOP probing obfuscated internal closures or private stores via JS. "
                "Switch to user-visible UI interactions: use semantic actions with browser_action(action=\"act\", instruction=\"...\") or direct interaction via browser_action(action=\"click\")."
            )
        return (
            "\n\n[SEMANTIC BROWSER GUARD]: Hard loop stop triggered. Page state is static and repeated JS evaluation attempts are failing. "
            "Execution paused. Explain to the user what elements or actions are required on this web page instead of repeating JS calls."
        )

    if loop_type == "edit_revert":
        if level == "warn":
            return (
                "\n\n[SEMANTIC REGRESSION GUARD]: You are in an edit-revert loop (State A <-> State B). "
                "Your latest change restored the files to a state from 2 steps ago. "
                "STOP repeating this edit. Do NOT alternate back and forth. "
                "Step back, read the error message carefully, and find an alternative solution."
            )
        return (
            "\n\n[SEMANTIC REGRESSION GUARD]: Hard loop stop triggered. The codebase has oscillated "
            "repeatedly between the same conflicting states. Execution paused. "
            "Explain the architectural conflict to the user."
        )

    # error_ping_pong
    if level == "warn":
        return (
            "\n\n[SEMANTIC REGRESSION GUARD]: You are caught in an error ping-pong cycle. "
            "Fixing error A causes error B, and fixing error B reproduces error A. "
            "STOP modifying the same lines back and forth. "
            "Re-evaluate the contract that satisfies BOTH requirements simultaneously."
        )
    return (
        "\n\n[SEMANTIC REGRESSION GUARD]: Hard loop stop triggered. Alternating test failures detected. "
        "Execution paused. Explain to the user why the two tests or compiler constraints conflict."
    )

