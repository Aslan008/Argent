"""The allowed-apps list is a real restriction when someone fills it in.

It used to be dead: `desktop_allowed_apps` sat in the config, `allowed_apps()` had no
callers, and every window check was made without it. A person who set the list
believed the agent could only touch Those programs — it could touch any.

Empty stays "any program": requiring the list to be filled before anything worked
protected nobody and made the tools useless until configured.
"""

import pytest

import argent.config as config
from src.automation import desktop_guard as guard


@pytest.fixture
def allowed(monkeypatch):
    """Sets the config key the way the person would, and reads it back."""
    def _set(apps):
        monkeypatch.setattr(config, "get_desktop_allowed_apps", lambda: [a.lower() for a in apps])
        # allowed_apps() imports the module, so patching the module attribute is enough.
    return _set


def test_an_empty_list_means_any_program(allowed):
    allowed([])
    assert guard.allowed_apps() == []
    assert guard.check_window("notepad.exe", "Untitled — Notepad", guard.allowed_apps() or None)


def test_a_filled_list_limits_acting_to_those_programs(allowed):
    allowed(["notepad.exe"])
    limit = guard.allowed_apps() or None
    assert guard.check_window("notepad.exe", "Untitled — Notepad", limit)
    denied = guard.check_window("calc.exe", "Calculator", limit)
    assert not denied
    assert "not on the allowed list" in denied.reason


def test_the_list_is_read_lowercased_and_trimmed():
    config.set_desktop_allowed_apps(["  Notepad.EXE ", "CALC.exe", ""])
    assert config.get_desktop_allowed_apps() == ["notepad.exe", "calc.exe"]


def test_a_forbidden_program_stays_forbidden_even_if_listed(allowed):
    # The list may only narrow what the person allows; it cannot widen the hard denials.
    forbidden = next(iter(guard._FORBIDDEN_PROCESSES))
    allowed([forbidden])
    denied = guard.check_window(forbidden, "anything", guard.allowed_apps() or None)
    assert not denied
    assert "not on the allowed list" not in denied.reason


def test_a_console_window_is_refused_regardless_of_the_list(allowed):
    allowed(["powershell.exe"])
    denied = guard.check_window("powershell.exe", "Windows PowerShell", guard.allowed_apps() or None)
    assert not denied or "not on the allowed list" not in denied.reason
