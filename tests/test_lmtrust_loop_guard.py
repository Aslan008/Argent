"""
Deep mutation-killing tests for src.agent.loop_guard.

These tests target every branch and boundary of _normalize_volatile,
_signature, LoopGuard.record / reset and build_loop_note so that simple
mutations (off-by-one, wrong regex, dropped fallback, wrong constant)
are reliably caught.
"""

import hashlib
import json

import pytest

from src.agent.loop_guard import (
    EXCLUDED_TOOLS,
    LoopGuard,
    STOP_REPEATS,
    WARN_REPEATS,
    _normalize_volatile,
    build_loop_note,
)

# _signature is a staticmethod on LoopGuard
_sig = LoopGuard._signature


# ---------------------------------------------------------------------------
# _normalize_volatile
# ---------------------------------------------------------------------------
class TestNormalizeVolatile:
    def test_guid_at_start(self):
        assert _normalize_volatile(
            "550e8400-e29b-41d4-a716-446655440000 failed"
        ) == "<guid> failed"

    def test_guid_in_middle(self):
        assert _normalize_volatile(
            "error guid 550e8400-e29b-41d4-a716-446655440000 here"
        ) == "error guid <guid> here"

    def test_guid_at_end(self):
        assert _normalize_volatile(
            "trace 550e8400-e29b-41d4-a716-446655440000"
        ) == "trace <guid>"

    def test_guid_uppercase(self):
        assert _normalize_volatile(
            "id 550E8400-E29B-41D4-A716-446655440000"
        ) == "id <guid>"

    def test_time_hhmmss_with_milliseconds_dot(self):
        assert _normalize_volatile("at 10:31:07.123 we crashed") == "at <time> we crashed"

    def test_time_hhmmss_with_milliseconds_comma(self):
        assert _normalize_volatile("at 10:31:07,123 we crashed") == "at <time> we crashed"

    def test_time_hhmmss_plain(self):
        assert _normalize_volatile("at 10:31:07 we crashed") == "at <time> we crashed"

    def test_time_single_digit_hour(self):
        assert _normalize_volatile("at 1:02:03 we crashed") == "at <time> we crashed"

    def test_hex_with_0x_prefix(self):
        assert _normalize_volatile("addr 0x7ffab3cd") == "addr <hex>"

    def test_hex_0x_uppercase(self):
        assert _normalize_volatile("addr 0xDEADBEEF") == "addr <hex>"

    def test_bare_hex_8_chars(self):
        assert _normalize_volatile("hash deadbeef done") == "hash <hex> done"

    def test_bare_hex_long(self):
        assert _normalize_volatile("blob a1b2c3d4e5f6 done") == "blob <hex> done"

    def test_bare_hex_uppercase_long(self):
        assert _normalize_volatile("blob ABCDEF1234567890 done") == "blob <hex> done"

    def test_four_digit_run(self):
        assert _normalize_volatile("pid 4812 exited") == "pid <n> exited"

    def test_long_digit_run(self):
        assert _normalize_volatile("pid 4812345 exited") == "pid <n> exited"

    def test_short_number_not_replaced(self):
        # numbers with fewer than 4 digits must survive untouched
        assert _normalize_volatile("count = 1") == "count = 1"

    def test_short_numbers_not_replaced(self):
        assert _normalize_volatile("values 12 and 999 ok") == "values 12 and 999 ok"

    def test_three_digit_number_not_replaced(self):
        assert _normalize_volatile("port 808 here") == "port 808 here"

    def test_combination_guid_time_hex_digits(self):
        text = "job 550e8400-e29b-41d4-a716-446655440000 at 10:31:07.5 pid 48123 addr 0xdead"
        out = _normalize_volatile(text)
        assert out == "job <guid> at <time> pid <n> addr <hex>"

    def test_multiple_guids_all_replaced(self):
        text = "a 550e8400-e29b-41d4-a716-446655440000 b 11111111-2222-3333-4444-555555555555"
        assert _normalize_volatile(text) == "a <guid> b <guid>"


# ---------------------------------------------------------------------------
# _signature
# ---------------------------------------------------------------------------
class TestSignature:
    def test_identical_args_result_same_hash(self):
        a = _sig("run_command", {"command": "pytest"}, "Exit code: 1")
        b = _sig("run_command", {"command": "pytest"}, "Exit code: 1")
        assert a == b

    def test_different_args_different_hash(self):
        a = _sig("run_command", {"command": "pytest a"}, "Exit code: 1")
        b = _sig("run_command", {"command": "pytest b"}, "Exit code: 1")
        assert a != b

    def test_different_func_name_different_hash(self):
        a = _sig("run_command", {"command": "x"}, "r")
        b = _sig("read_file", {"command": "x"}, "r")
        assert a != b

    def test_different_result_different_hash(self):
        a = _sig("run_command", {"command": "x"}, "result A")
        b = _sig("run_command", {"command": "x"}, "result B")
        assert a != b

    def test_args_key_order_independent(self):
        a = _sig("t", {"a": 1, "b": 2}, "r")
        b = _sig("t", {"b": 2, "a": 1}, "r")
        assert a == b

    def test_non_serializable_set_falls_back_to_str(self):
        # set is not JSON-serializable -> default=str path used; should still
        # be deterministic and equal across calls.
        a = _sig("t", {"items": {1, 2, 3}}, "r")
        b = _sig("t", {"items": {1, 2, 3}}, "r")
        assert a == b

    def test_non_serializable_custom_object_falls_back_to_str(self):
        class Widget:
            def __str__(self):
                return "widget-instance"

        a = _sig("t", {"w": Widget()}, "r")
        b = _sig("t", {"w": Widget()}, "r")
        assert a == b
        # And differs from a plain string arg
        c = _sig("t", {"w": "other"}, "r")
        assert a != c

    def test_result_truncated_to_2000_chars(self):
        base = "x" * 2000
        long1 = "x" * 2000 + "AAAA"
        long2 = "x" * 2000 + "BBBB"
        # base, long1 and long2 all share the first 2000 chars, so after
        # truncation they produce the same signature.
        a = _sig("t", {}, base)
        b = _sig("t", {}, long1)
        assert a == b
        c = _sig("t", {}, long2)
        assert b == c

    def test_result_truncation_boundary_exact_2000(self):
        exactly = "y" * 2000
        plus_one = "y" * 2000 + "z"
        assert _sig("t", {}, exactly) == _sig("t", {}, plus_one)

    def test_signature_is_sha1_hex(self):
        sig = _sig("t", {"a": 1}, "r")
        assert len(sig) == 40
        int(sig, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# LoopGuard.record
# ---------------------------------------------------------------------------
class TestRecord:
    def test_below_warn_returns_none(self):
        g = LoopGuard()
        assert g.record("run_command", {"command": "x"}, "boom") is None

    def test_exactly_warn_repeats_returns_warn(self):
        g = LoopGuard()
        first = g.record("run_command", {"command": "x"}, "boom")
        second = g.record("run_command", {"command": "x"}, "boom")
        assert first is None
        assert second == "warn"
        # sanity: WARN_REPEATS really is 2
        assert WARN_REPEATS == 2

    def test_exactly_stop_repeats_returns_stop(self):
        g = LoopGuard()
        levels = [g.record("run_command", {"command": "x"}, "boom") for _ in range(STOP_REPEATS)]
        assert levels[0] is None
        assert levels[1] == "warn"
        assert levels[2] == "warn"
        assert levels[3] == "stop"
        assert STOP_REPEATS == 4

    def test_above_stop_still_stop(self):
        g = LoopGuard()
        for _ in range(STOP_REPEATS):
            g.record("run_command", {"command": "x"}, "boom")
        assert g.record("run_command", {"command": "x"}, "boom") == "stop"

    def test_different_results_returns_none(self):
        g = LoopGuard()
        g.record("read_file", {"file_path": "a.py"}, "old content")
        assert g.record("read_file", {"file_path": "a.py"}, "new content") is None

    def test_different_args_returns_none(self):
        g = LoopGuard()
        g.record("run_command", {"command": "a"}, "boom")
        assert g.record("run_command", {"command": "b"}, "boom") is None

    def test_window_overflow_drops_old_entries(self):
        g = LoopGuard(window=3)
        g.record("run_command", {"command": "x"}, "boom")  # slot 0
        g.record("read_file", {"file_path": "1"}, "a")     # pushes out
        g.record("read_file", {"file_path": "2"}, "b")
        g.record("read_file", {"file_path": "3"}, "c")
        # original sig evicted, so this is count==1 again -> None
        assert g.record("run_command", {"command": "x"}, "boom") is None

    def test_window_keeps_entries_within_size(self):
        g = LoopGuard(window=10)
        g.record("run_command", {"command": "x"}, "boom")
        # 8 more distinct calls -> 9 entries total, original still in window
        for i in range(8):
            g.record("read_file", {"file_path": str(i)}, "r")
        assert g.record("run_command", {"command": "x"}, "boom") == "warn"

    def test_excluded_tool_always_none(self):
        assert "wait_heartbeat" in EXCLUDED_TOOLS
        g = LoopGuard()
        for _ in range(STOP_REPEATS + 5):
            assert g.record("wait_heartbeat", {"delay_seconds": 60}, "ok") is None

    def test_non_excluded_tool_with_similar_name_not_skipped(self):
        # 'wait_heartbeat2' must not be treated as excluded
        g = LoopGuard()
        g.record("wait_heartbeat2", {}, "ok")
        assert g.record("wait_heartbeat2", {}, "ok") == "warn"


# ---------------------------------------------------------------------------
# LoopGuard.reset
# ---------------------------------------------------------------------------
class TestReset:
    def test_reset_clears_history(self):
        g = LoopGuard()
        g.record("run_command", {"command": "x"}, "boom")
        g.reset()
        assert g.record("run_command", {"command": "x"}, "boom") is None

    def test_after_reset_same_call_warns_again_on_second(self):
        g = LoopGuard()
        g.record("run_command", {"command": "x"}, "boom")
        g.record("run_command", {"command": "x"}, "boom")  # would be warn
        g.reset()
        first = g.record("run_command", {"command": "x"}, "boom")
        assert first is None
        second = g.record("run_command", {"command": "x"}, "boom")
        assert second == "warn"

    def test_reset_idempotent(self):
        g = LoopGuard()
        g.reset()
        g.reset()
        assert g.record("t", {}, "r") is None


# ---------------------------------------------------------------------------
# build_loop_note
# ---------------------------------------------------------------------------
class TestBuildLoopNote:
    def test_warn_note_exact_text(self):
        expected = (
            "\n\n[LOOP GUARD]: You have ALREADY executed this exact tool call and received "
            "the IDENTICAL result. Repeating it will not change anything. Choose a DIFFERENT "
            "approach or different arguments, or explain the blocker to the user."
        )
        assert build_loop_note("warn") == expected

    def test_stop_note_exact_text(self):
        expected = (
            f"\n\n[LOOP GUARD]: This exact call has repeated {STOP_REPEATS} times with the same "
            "result. Execution is stopped. Explain to the user what you were trying to achieve "
            "and why it keeps failing."
        )
        assert build_loop_note("stop") == expected

    def test_stop_note_contains_stop_repeats_number(self):
        assert str(STOP_REPEATS) in build_loop_note("stop")

    def test_warn_note_does_not_contain_stop_number(self):
        assert str(STOP_REPEATS) not in build_loop_note("warn")

    def test_unknown_level_falls_through_to_stop_text(self):
        # any non-"warn" level returns the stop branch
        assert build_loop_note("nonsense") == build_loop_note("stop")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_string_result(self):
        g = LoopGuard()
        assert g.record("run_command", {"command": "x"}, "") is None
        assert g.record("run_command", {"command": "x"}, "") == "warn"

    def test_none_arguments(self):
        g = LoopGuard()
        assert g.record("run_command", None, "boom") is None
        assert g.record("run_command", None, "boom") == "warn"

    def test_very_long_result_over_2000_chars(self):
        g = LoopGuard()
        long_a = "A" * 5000
        long_b = "A" * 2000 + "B" * 3000
        # After truncation to 2000 chars both are identical -> loop detected
        g.record("run_command", {"command": "x"}, long_a)
        assert g.record("run_command", {"command": "x"}, long_b) == "warn"

    def test_long_result_difference_within_first_2000(self):
        g = LoopGuard()
        a = "A" * 1500 + "X" + "A" * 3499
        b = "A" * 1500 + "Y" + "A" * 3499
        g.record("run_command", {"command": "x"}, a)
        # difference is within the first 2000 chars -> NOT a loop
        assert g.record("run_command", {"command": "x"}, b) is None

    def test_empty_string_normalization(self):
        assert _normalize_volatile("") == ""

    def test_none_result_coerced_to_string(self):
        # record() calls str(result); None -> "None"
        g = LoopGuard()
        assert g.record("run_command", {"command": "x"}, None) is None
        assert g.record("run_command", {"command": "x"}, None) == "warn"

    def test_volatile_noise_in_result_still_loops(self):
        g = LoopGuard()
        outs = [
            "fail at 12:00:01.100 pid 9999 guid 550e8400-e29b-41d4-a716-446655440000",
            "fail at 12:00:05.200 pid 12345 guid 660e8400-e29b-41d4-a716-446655440000",
            "fail at 12:00:09.300 pid 67890 guid 770e8400-e29b-41d4-a716-446655440000",
            "fail at 12:00:13.400 pid 11111 guid 880e8400-e29b-41d4-a716-446655440000",
        ]
        levels = [g.record("run_command", {"command": "x"}, o) for o in outs]
        assert levels[-1] == "stop"