"""
Detection of failed tool calls for the auto-healing loop.

The old implementation matched substrings like "FAILED" anywhere in the tool
output, which misfired on legitimate text ("0 tests failed", linter summaries,
git messages). Failure is now derived from structured signals: the command
exit code and the error prefixes that Argent's own tools produce.
"""

import re

# Tools whose failures trigger the auto-healing loop.
HEALING_TOOLS = ("run_command", "write_file", "replace_in_file", "replace_python_function")

# Commands launched fire-and-forget (UI apps): their exit codes are meaningless.
_FIRE_AND_FORGET_PREFIXES = ("explorer", "start ", "start.", 'start"')

_EXIT_CODE_RE = re.compile(r"^Exit code:\s*(-?\d+)")


def detect_tool_failure(func_name: str, result: str, command: str | None = None) -> bool:
    """Return True when a tool result is an actual failure worth auto-fixing."""
    if func_name not in HEALING_TOOLS:
        return False
    text = str(result).strip()

    if func_name == "run_command":
        if command and any(command.strip().lower().startswith(p) for p in _FIRE_AND_FORGET_PREFIXES):
            return False
        m = _EXIT_CODE_RE.match(text)
        if m:
            return int(m.group(1)) != 0
        # No exit code in the output: only explicit execution errors count.
        # User denial ("Execution aborted by user...") is NOT a failure.
        return text.startswith("Error running command")

    # File tools format real failures with an "Error" prefix; syntax validation
    # appends a "COMPILATION FAILED" marker after a syntactically broken write.
    return text.startswith("Error") or "COMPILATION FAILED" in text


def build_healing_hint(attempt: int, func_name: str = None, max_attempts: int = 3) -> str:
    """Instruction appended to a failed tool result to drive the self-repair loop.

    The guidance is tailored to where the error actually is: a failed
    `run_command` is usually a problem with the COMMAND (wrong shell, typo,
    missing tool, bad path) — telling the model to "fix the code" there is
    actively misleading. File/edit tool failures are genuine code problems.
    """
    if attempt > max_attempts:
        return "\n\n[AUTO-HEALING FAILED]: You have failed 3 times. Stop trying and explain the failure to the user."

    if func_name == "run_command":
        return (
            f"\n\n[AUTO-HEALING MODE TRIGGERED]: Attempt {attempt}/{max_attempts} to fix this. "
            f"Do NOT stop or apologize. First read the error output and any [DIAGNOSIS] line above — "
            f"decide whether the COMMAND itself was wrong (bad shell/syntax, typo, missing tool, wrong "
            f"path) and fix the COMMAND, OR whether the command ran but a compiler/test reported real "
            f"failures, in which case fix the CODE with replace_in_file and re-run. Do not assume it's the code."
        )
    return (
        f"\n\n[AUTO-HEALING MODE TRIGGERED]: Attempt {attempt}/{max_attempts} to automatically fix this error. "
        f"Do NOT stop or apologize. Read the error (especially if it is a Pytest AssertionError), "
        f"use `read_file` if needed to see the context, and use `replace_in_file` to fix the syntax or logic immediately. "
        f"If you just ran tests and they failed, you MUST fix the code and re-run the tests."
    )
