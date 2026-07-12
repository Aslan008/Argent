from src.agent.loop_guard import LoopGuard, build_loop_note, WARN_REPEATS, STOP_REPEATS


class TestLoopGuard:
    def test_first_call_is_clean(self):
        g = LoopGuard()
        assert g.record("run_command", {"command": "pytest"}, "Exit code: 1") is None

    def test_second_identical_call_warns(self):
        g = LoopGuard()
        g.record("run_command", {"command": "pytest"}, "Exit code: 1")
        assert g.record("run_command", {"command": "pytest"}, "Exit code: 1") == "warn"

    def test_fourth_identical_call_stops(self):
        g = LoopGuard()
        levels = [g.record("run_command", {"command": "pytest"}, "Exit code: 1") for _ in range(STOP_REPEATS)]
        assert levels[-1] == "stop"
        assert levels[WARN_REPEATS - 1] == "warn"

    def test_different_arguments_do_not_trigger(self):
        g = LoopGuard()
        g.record("run_command", {"command": "pytest tests/a.py"}, "Exit code: 1")
        assert g.record("run_command", {"command": "pytest tests/b.py"}, "Exit code: 1") is None

    def test_different_results_do_not_trigger(self):
        # Same call but the world changed (e.g. re-read after an edit) — not a loop.
        g = LoopGuard()
        g.record("read_file", {"file_path": "a.py"}, "old content")
        assert g.record("read_file", {"file_path": "a.py"}, "new content") is None

    def test_window_eviction(self):
        g = LoopGuard(window=3)
        g.record("run_command", {"command": "x"}, "boom")
        # Push the first record out of the window with unrelated calls.
        g.record("read_file", {"file_path": "1"}, "a")
        g.record("read_file", {"file_path": "2"}, "b")
        g.record("read_file", {"file_path": "3"}, "c")
        assert g.record("run_command", {"command": "x"}, "boom") is None

    def test_excluded_tools_never_trigger(self):
        g = LoopGuard()
        for _ in range(STOP_REPEATS + 1):
            assert g.record("wait_heartbeat", {"delay_seconds": 60, "condition_to_check": "x"}, "ok") is None

    def test_reset_clears_history(self):
        g = LoopGuard()
        g.record("run_command", {"command": "x"}, "boom")
        g.reset()
        assert g.record("run_command", {"command": "x"}, "boom") is None

    def test_unserializable_arguments_handled(self):
        g = LoopGuard()
        weird = {"obj": object()}
        assert g.record("t", weird, "r") is None
        assert g.record("t", weird, "r") == "warn"


class TestVolatileNormalization:
    def test_timestamp_noise_no_longer_defeats_the_guard(self):
        # Same failing command, output differs only by a timestamp each run.
        g = LoopGuard()
        outs = [
            "Build failed at 10:31:07",
            "Build failed at 10:31:09",
            "Build failed at 10:31:12",
            "Build failed at 10:31:15",
        ]
        levels = [g.record("run_command", {"command": "build"}, o) for o in outs]
        assert levels[-1] == "stop"          # caught despite the changing time

    def test_pid_and_hex_noise_collapses(self):
        g = LoopGuard()
        a = g.record("run_command", {"command": "x"}, "failed pid 48120 at 0x7ffab3")
        b = g.record("run_command", {"command": "x"}, "failed pid 51999 at 0x1c2d90")
        assert a is None and b == "warn"

    def test_small_number_changes_stay_distinct(self):
        # A genuine re-read where content changed 1 -> 2 is NOT a loop; single
        # digits are left alone so the results still hash differently.
        g = LoopGuard()
        g.record("read_file", {"file_path": "a.py"}, "count = 1")
        assert g.record("read_file", {"file_path": "a.py"}, "count = 2") is None


class TestLoopNotes:
    def test_warn_note_mentions_different_approach(self):
        assert "DIFFERENT" in build_loop_note("warn")

    def test_stop_note_mentions_stop(self):
        assert "stopped" in build_loop_note("stop")
