"""`/guard block` must hold on every execution path, not just the plain one.

It was checked inside `run_command` only — after the `admin` shortcut, and with no
counterpart in `start_background_command`. So `admin=True` went straight to a UAC
prompt, and a background start never consulted the gate at all.
"""

import pytest

import argent.tools.command_ops as co

CATASTROPHIC = "rm -rf /"


@pytest.fixture
def guard(monkeypatch):
    def _set(value):
        monkeypatch.setattr("argent.config.get_command_guard", lambda: value)

    return _set


class TestGuardRefusal:
    def test_block_refuses_a_catastrophic_command(self, guard):
        guard("block")
        message = co._guard_refusal(CATASTROPHIC)
        assert message and "BLOCKED" in message

    def test_block_lets_an_ordinary_command_through(self, guard):
        guard("block")
        assert co._guard_refusal("git status") is None

    def test_warn_never_refuses(self, guard):
        guard("warn")
        assert co._guard_refusal(CATASTROPHIC) is None

    def test_off_never_refuses(self, guard):
        guard("off")
        assert co._guard_refusal(CATASTROPHIC) is None


class TestEveryPathAsksTheGate:
    def test_admin_is_refused_before_uac(self, guard, monkeypatch):
        guard("block")
        monkeypatch.setattr(co, "_run_elevated",
                            lambda c: pytest.fail("the UAC prompt must not be reached"))
        out = co.run_command(CATASTROPHIC, admin=True)
        assert "BLOCKED" in out

    def test_plain_path_is_refused(self, guard, monkeypatch):
        guard("block")
        monkeypatch.setattr(co, "_spawn",
                            lambda *a, **k: pytest.fail("nothing may be spawned"))
        out = co.run_command(CATASTROPHIC)
        assert "BLOCKED" in out

    def test_background_path_is_refused(self, guard, monkeypatch):
        guard("block")
        out = co.start_background_command(CATASTROPHIC)
        assert "BLOCKED" in out
        assert not co.ACTIVE_PROCESSES, "nothing was started"

    def test_the_refusal_names_the_way_out(self, guard):
        guard("block")
        assert "/guard warn" in co._guard_refusal(CATASTROPHIC)
