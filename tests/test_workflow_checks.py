"""Judging a check: red is only the worker's fault if it was green before.

These tests are the contract the engine now relies on — the difference between
"your patch broke this" and "this repository was already red when you arrived".
"""

from src.agent.workflow import checks as C

PYTEST_RED = """\
=========================== short test summary info ===========================
FAILED tests/test_a.py::test_one - AssertionError: expected 1
ERROR tests/test_b.py::test_two - ModuleNotFoundError: No module named 'transformers'
2 failed, 40 passed
Exit code: 1
"""

PYTEST_GREEN = "20 passed in 0.74s\nExit code: 0\n"


def result(ok, output):
    return C.Result(ok=ok, failed=C.failure_signature(output), output=output)


# ── what counts as "the same failure" ───────────────────────────────────────

def test_pytest_failures_are_named_by_their_test_id():
    failed = C.failure_signature(PYTEST_RED)
    assert "tests/test_a.py::test_one" in failed
    assert "tests/test_b.py::test_two" in failed


def test_a_green_run_has_no_failures():
    assert C.failure_signature(PYTEST_GREEN) == frozenset()


def test_the_same_breakage_keeps_the_same_face():
    # Timings, addresses and paths move between runs; the failure must not.
    first = C.failure_signature("FAILED tests/x.py::test_a - boom at 0x7ff3 in /tmp/abc123\n")
    second = C.failure_signature("FAILED tests/x.py::test_a - boom at 0x1aa4 in /tmp/def456\n")
    assert first == second == frozenset({"tests/x.py::test_a"})


def test_an_unknown_dialect_is_read_by_its_error_lines():
    out = "ModuleNotFoundError: No module named 'scipy'\n"
    assert any("scipy" in f for f in C.failure_signature(out))


def test_cargo_and_go_failures_are_recognised():
    assert C.failure_signature("test engine::tests::drops_it ... FAILED\n") == frozenset(
        {"engine::tests::drops_it"})
    assert C.failure_signature("--- FAIL: TestThing (0.00s)\n") == frozenset({"TestThing"})


# ── the verdict ─────────────────────────────────────────────────────────────

def test_green_stays_green():
    verdict = C.judge(result(True, PYTEST_GREEN), result(True, PYTEST_GREEN))
    assert verdict.ok and verdict.reason == "passed"


def test_the_same_red_before_and_after_is_not_the_patchs_fault():
    verdict = C.judge(result(False, PYTEST_RED), result(False, PYTEST_RED))
    assert verdict.ok, verdict.reason
    assert verdict.pre_existing == 2
    assert "already there" in verdict.reason


def test_a_new_failure_is_the_patchs_fault_and_is_named():
    after = PYTEST_RED + "FAILED tests/test_a.py::test_three - broke\n"
    verdict = C.judge(result(False, PYTEST_RED), result(False, after))
    assert not verdict.ok
    assert verdict.new == ["tests/test_a.py::test_three"]
    assert "new failure" in verdict.reason


def test_green_before_and_red_after_is_a_regression():
    verdict = C.judge(result(True, PYTEST_GREEN), result(False, PYTEST_RED))
    assert not verdict.ok
    assert "broke it" in verdict.reason


def test_a_patch_that_makes_a_red_check_green_is_accepted():
    verdict = C.judge(result(False, PYTEST_RED), result(True, PYTEST_GREEN))
    assert verdict.ok and verdict.fixed == 2


def test_fewer_failures_than_before_is_still_accepted():
    after = "FAILED tests/test_a.py::test_one - boom\n1 failed, 41 passed\nExit code: 1\n"
    verdict = C.judge(result(False, PYTEST_RED), result(False, after))
    assert verdict.ok and verdict.fixed == 1


def test_an_unreadable_red_is_not_blamed_on_the_patch():
    # A check can fail without naming a single test (a script, a build). When the same
    # text comes back with and without the patch, nothing new can be blamed on it.
    verdict = C.judge(result(False, "boom\n"), result(False, "boom\n"))
    assert verdict.ok, verdict.reason
    assert "names no particular failure" in verdict.reason


def test_describe_names_the_new_failures_and_counts_the_rest():
    after = "".join(f"FAILED tests/t.py::test_{i}\n" for i in range(8))
    verdict = C.judge(result(False, PYTEST_RED), result(False, after))
    text = C.describe(verdict, limit=3)
    assert "new failure" in text and "test_0" in text and "ещё 5" in text


# ── the unasked question: steps with no checks ──────────────────────────────

def test_checks_are_derived_from_the_files_a_step_touched(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_widget.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_widget_extra.py").write_text("", encoding="utf-8")

    commands = C.derive_checks(["src/widget.py"], str(tmp_path))

    assert len(commands) == 1
    assert commands[0].startswith("python -m pytest ")
    assert "tests/test_widget.py" in commands[0]
    assert "-q" in commands[0]


def test_a_test_file_is_not_its_own_check(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_widget.py").write_text("", encoding="utf-8")

    assert C.derive_checks(["tests/test_widget.py"], str(tmp_path)) == []


def test_nothing_is_invented_when_no_test_matches(tmp_path):
    (tmp_path / "tests").mkdir()
    assert C.derive_checks(["src/nowhere.py"], str(tmp_path)) == []
    assert C.derive_checks(["src/nowhere.py"], str(tmp_path / "missing")) == []


def test_the_whole_suite_is_never_derived(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_many.py").write_text("", encoding="utf-8")
    commands = C.derive_checks(["src/a.py", "src/b.py", "src/c.py", "src/d.py"], str(tmp_path))
    assert commands == [] or all("tests/test_" in c for c in commands)


def test_a_rust_file_is_checked_by_its_own_crate(tmp_path):
    commands = C.derive_checks(["crates/argent-tui/src/app.rs"], str(tmp_path))
    assert commands == ["cargo test -p argent-tui -q"]


def test_a_python_file_and_a_rust_file_get_their_own_commands(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_widget.py").write_text("", encoding="utf-8")

    commands = C.derive_checks(["src/widget.py", "crates/argent-tui/src/ui.rs"], str(tmp_path))

    assert commands[0].startswith("python -m pytest ")
    assert "cargo test -p argent-tui -q" in commands


def test_a_result_survives_the_round_trip_through_the_plan():
    before = result(False, PYTEST_RED)
    again = C.Result.from_dict(before.to_dict())
    assert again.ok == before.ok and again.failed == before.failed
    assert C.Result.from_dict(None) is None
