import sys
import os
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# --- Robust import setup to avoid 'agent is not a package' conflicts ---
_THIS = Path(__file__).resolve()
_CANDIDATE = _THIS.parent
while _CANDIDATE != _CANDIDATE.parent:
    if (_CANDIDATE / "agent" / "wait_conditions.py").exists():
        break
    _CANDIDATE = _CANDIDATE.parent
_PROJECT_ROOT = _CANDIDATE

# Remove any conflicting 'agent' module cached from a previous import
for key in list(sys.modules.keys()):
    if key == "agent" or key.startswith("agent."):
        del sys.modules[key]

# Drop sys.path entries that contain a stray agent.py shadowing the package
sys.path = [p for p in sys.path
            if not (Path(p) / "agent.py").is_file()
            or Path(p) == _PROJECT_ROOT]

# Ensure project root is first so the real 'agent' package wins
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.wait_conditions import (
    _file_exists, _file_missing, _file_contains,
    _process_finished, _process_running, _eval_node,
    evaluate, wait_for, ConditionError,
    DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS,
)


class TestBlindSpots:

    @pytest.mark.lmtrust(layer="L4", direction="D7")
    def test_process_finished_none_registry_attribute_error(self):
        # Assumption: ACTIVE_PROCESSES supports .get(key) and LOCK is a context manager
        # Violation: ACTIVE_PROCESSES is None -> None.get(key) raises AttributeError
        fake_lock = MagicMock()
        with patch("tools.ACTIVE_PROCESSES", None), \
             patch("tools.ACTIVE_PROCESSES_LOCK", fake_lock):
            result = _process_finished(123)
        # Correct behavior: handle None registry gracefully
        assert result is True



    @pytest.mark.lmtrust(layer="L4", direction="D4")
    def test_file_exists_propagates_value_error_from_resolve(self):
        # Assumption: _resolve_path only raises OSError on failure
        # Violation: _resolve_path raises ValueError -> propagates up
        with patch("tools._helpers._resolve_path",
                   side_effect=ValueError("bad path")):
            result = _file_exists("/some/path")
        # Correct behavior: catch all exceptions, return False
        assert result is False

    @pytest.mark.lmtrust(layer="L1", direction="D9")
    def test_file_contains_reads_whole_file_not_tail(self):
        # Assumption: docstring says "read the tail rather than the whole file"
        # Violation: code calls p.read_text() -> reads the entire file every poll
        # Put target at the start and lots of data after; a true tail-read misses it.
        fd, path = tempfile.mkstemp(suffix=".log")
        try:
            os.write(fd, b"START_TARGET_TEXT")
            os.write(fd, b"X" * 200_000)
            os.close(fd)
            result = _file_contains(path, "START_TARGET_TEXT")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        # Correct behavior: a tail-only reader would NOT find text at offset 0
        assert result is False, (
            f"Expected tail-read to miss START marker, got {result} "
            "(code reads whole file via read_text)")

    @pytest.mark.lmtrust(layer="L2", direction="D7")
    def test_wait_for_timeout_zero_silently_becomes_default(self):
        # Assumption: timeout=0 is either rejected or means "no wait"
        # Violation: `timeout or DEFAULT_TIMEOUT_SECONDS` -> 0 is falsy -> 600s default
        clock_val = [0.0]
        def fake_clock():
            return clock_val[0]
        def fake_sleep(s):
            # Jump the clock so the test doesn't run for the full default timeout
            clock_val[0] += 1000.0
        with patch("src.agent.wait_conditions.evaluate", return_value=False):
            result = wait_for(
                "file_exists('/nope')",
                timeout=0,            # caller expects ~no wait
                poll_seconds=0.1,
                sleep=fake_sleep,
                clock=fake_clock,
            )
        # Correct behavior: timeout=0 should not produce a long wait
        assert result["waited"] < 1.0, (
            f"timeout=0 produced a {result['waited']}s wait "
            "(silently became DEFAULT_TIMEOUT_SECONDS)")

    @pytest.mark.lmtrust(layer="L2", direction="D7")
    def test_wait_for_poll_seconds_zero_silently_clamped(self):
        # Assumption: poll_seconds=0 is honored (or rejected loudly)
        # Violation: max(0.1, float(0)) silently bumps it to 0.1
        sleep_args = []
        clock_val = [0.0]
        def fake_clock():
            return clock_val[0]
        def fake_sleep(s):
            sleep_args.append(s)
            clock_val[0] += 1.0
        eval_n = [0]
        def fake_eval(*args):
            eval_n[0] += 1
            return eval_n[0] > 3
        with patch("src.agent.wait_conditions.evaluate", side_effect=fake_eval):
            wait_for(
                "file_exists('/x')",
                timeout=10,
                poll_seconds=0,   # caller expects no sleep
                sleep=fake_sleep,
                clock=fake_clock,
            )
        # Correct behavior: every sleep should be 0 seconds
        assert all(s == 0 for s in sleep_args), (
            f"poll_seconds=0 was silently clamped; sleeps were {sleep_args}")

    @pytest.mark.lmtrust(layer="L4", direction="D1")
    def test_wait_for_constant_clock_busy_spin(self):
        # Assumption: clock() strictly increases (monotonic, non-stalling)
        # Violation: clock returns a constant -> waited never grows -> busy spin
        eval_n = [0]
        def fake_eval(*args):
            eval_n[0] += 1
            # bail-out cap so the test terminates; the bug is that we GET here
            return eval_n[0] > 200
        with patch("src.agent.wait_conditions.evaluate", side_effect=fake_eval):
            wait_for(
                "file_exists('/x')",
                timeout=10,
                poll_seconds=0.1,
                sleep=lambda s: None,
                clock=lambda: 42.0,   # CONSTANT
            )
        # Correct behavior: detect stalled clock, exit well before the poll cap
        assert eval_n[0] < 50, (
            f"Constant clock caused {eval_n[0]} busy-spin iterations")

    @pytest.mark.lmtrust(layer="L8", direction="D6")
    def test_wait_for_sleep_injected_clock_not(self):
        # Assumption: sleep and clock injected together
        # Violation: only sleep injected; real clock drives the deadline -> spin
        start = time.monotonic()
        result = wait_for(
            "file_exists('/definitely_not_here_xyz')",
            timeout=0.5,
            poll_seconds=0.05,
            sleep=lambda s: None,   # no-op sleep
            # clock defaults to time.monotonic -> loop spins for real 0.5s
        )
        elapsed = time.monotonic() - start
        # Correct behavior: should not busy-wait for the full real-time timeout
        assert elapsed < 0.1, (
            f"Loop busy-waited {elapsed:.2f}s of real time "
            "because only sleep was stubbed")

    @pytest.mark.lmtrust(layer="L4", direction="D3")
    def test_process_finished_float_pid_key_mismatch(self):
        # Assumption: str(pid) coercion matches the registry's key format
        # Violation: pid=12.0 -> key='12.0' never matches registry key '12'
        fake_process = MagicMock()
        fake_process.poll.return_value = 0     # the real process IS finished
        fake_lock = MagicMock()
        fake_registry = {"12": {"process": fake_process}}
        with patch("tools.ACTIVE_PROCESSES", fake_registry), \
             patch("tools.ACTIVE_PROCESSES_LOCK", fake_lock):
            result = _process_finished(12.0)
        # Correct behavior: should normalize numeric pid, or report finished
        # when the same int is registered under any numeric form
        assert result is True, (
            f"Float pid 12.0 missed registry key '12' -> got {result}")

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_eval_node_rejects_trivial_comparison(self):
        # Assumption: only BoolOp/UnaryOp(not)/Call/Constant(bool) accepted
        # Violation: a trivial '== False' Compare node is rejected
        condition = "file_exists('/x') and not process_running(123) == False"
        # Correct behavior: this is semantically equivalent to
        #   file_exists('/x') and process_running(123)
        # and should be accepted (or at worst normalized).
        result = evaluate(condition)
        assert result is False or result is True   # i.e. did NOT raise



    @pytest.mark.lmtrust(layer="L2", direction="D7")
    def test_evaluate_comment_only_condition(self):
        # Assumption: empty/whitespace-only is rejected; comment-only is
        # the same "no expression" case and also rejected.
        # Violation: a harmless comment-only expression raises ConditionError.
        # Correct behavior: treat as a no-op success (vacuously true) or
        # at minimum do not raise.
        try:
            evaluate("# nothing")
        except ConditionError:
            pytest.fail(
                "Comment-only condition raised ConditionError; "
                "should be treated as a no-op (vacuously true)")

    @pytest.mark.lmtrust(layer="L2", direction="D1")
    def test_wait_for_pathological_clock_values(self):
        # Assumption: clock values are reasonable; round(waited, 1) is meaningful
        # Violation: clock() returns 1e20 -> waited is reported as 1e20 (useless)
        def fake_clock():
            return 1e20
        result = wait_for(
            "file_exists('/nope')",
            timeout=1,
            poll_seconds=0.1,
            sleep=lambda s: None,
            clock=fake_clock,
        )
        # Correct behavior: report a waited value in the same order as timeout
        assert result["waited"] < 100, (
            f"Pathological clock produced meaningless waited={result['waited']}")