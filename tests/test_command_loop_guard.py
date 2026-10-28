from src.agent.loop_guard import LoopGuard


def test_near_identical_shell_probes_eventually_stop():
    guard = LoopGuard()
    levels = []
    for i in range(6):
        command = (
            "python -c \"from argent.agent import AgentAgent; "
            f"p=a.build_system_prompt(); print(p[{i}:{i + 3}])\""
        )
        levels.append(
            guard.record(
                "run_command",
                {"command": command},
                f"Exit code: 0\nOUTPUT:\nslice {i}",
                enabled=True,
                category="cloud",
            )
        )
    assert "nudge" in levels
    assert levels[-1] == "stop"


def test_unrelated_commands_do_not_trigger():
    guard = LoopGuard()
    commands = [
        "git status --short",
        "python -m pytest tests/test_one.py -q",
        "cargo check",
        "git diff --stat",
        "python -m pytest tests/test_two.py -q",
        "cargo test",
    ]
    for command in commands:
        assert guard.record(
            "run_command",
            {"command": command},
            "Exit code: 0",
            enabled=True,
            category="cloud",
        ) is None


def test_command_guard_can_be_disabled():
    guard = LoopGuard()
    for i in range(8):
        assert guard.record(
            "run_command",
            {"command": f"python -c \"print({i})\""},
            "Exit code: 0",
            enabled=False,
            category="cloud",
        ) is None


def test_reset_forgets_command_history():
    guard = LoopGuard()
    for i in range(4):
        guard.record(
            "run_command",
            {"command": f"python -c \"print({i})\""},
            "Exit code: 0",
            enabled=True,
            category="cloud",
        )
    guard.reset()
    assert guard.record(
        "run_command",
        {"command": "python -c \"print(99)\""},
        "Exit code: 0",
        enabled=True,
        category="cloud",
    ) is None
