import sys

import pytest

from argent.bench.case import Case, CaseError, Check, load_case


def test_a_case_loads_checks_and_limits(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text(
        """
id: demo
task: "write ok"
start: HEAD
setup: ["echo setup"]
limits: {timeout_s: 42}
checks:
  - cmd: "python -c 'print(1)'"
    stdout_contains: "1"
  - "echo second"
""",
        encoding="utf-8",
    )
    case = load_case(path)
    assert case.id == "demo"
    assert case.timeout_s == 42
    assert len(case.checks) == 2
    assert case.checks[0].stdout_contains == "1"
    assert case.checks[1].cmd == "echo second"


def test_a_case_without_a_task_is_rejected(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text("id: x\nchecks: [echo hi]\n", encoding="utf-8")
    with pytest.raises(CaseError):
        load_case(path)


def test_a_case_without_checks_is_rejected_by_validate():
    case = Case(id="x", task="do it")
    with pytest.raises(CaseError):
        case.validate_runnable()


def test_a_string_check_gets_default_exit_zero():
    assert Check.from_value("echo hi").expect_exit == 0
