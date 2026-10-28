"""`run_command` and `desktop_act`: the parts where a wrong reading is expensive.

Both tools report what they did in prose, and the prose is what the agent acts on.
Two of those readings were wrong in the field:

* In PowerShell a non-zero exit code is common and often harmless, and a command that
  *succeeds* can still print to stderr. A result that calls stderr "a failure" makes the
  agent retry working code; a result that hides the diagnosis makes it guess.
* `desktop_act` could only tell you what it would press *by pressing it*. There was no
  way to check the target — or to show the person what was about to happen — without
  the click already landing.
"""

import sys
from types import SimpleNamespace

import pytest

import argent.tools.command_ops as co

PY = sys.executable


@pytest.fixture(autouse=True)
def _approve(monkeypatch):
    monkeypatch.setattr(co, "request_approval", lambda *a, **k: True)


class TestRunCommandSemantics:
    def test_stderr_with_exit_zero_is_not_a_failure(self):
        out = co.run_command(f'"{PY}" -c "import sys; sys.stderr.write(\'warning: deprecated\\n\')"')
        assert "Exit code: 0" in out
        assert "warning: deprecated" in out
        assert "[DIAGNOSIS]" not in out
        assert "Error running command" not in out

    def test_a_real_failure_carries_a_diagnosis(self):
        out = co.run_command("argent_no_such_command_xyz --version")
        assert "Exit code:" in out
        assert "[DIAGNOSIS]" in out

    def test_the_exit_code_is_the_first_line(self):
        """Whatever a caller reads first, it must be the verdict."""
        out = co.run_command(f'"{PY}" -c "print(1)"')
        assert out.splitlines()[0].startswith("Exit code:")

    def test_a_missing_directory_is_refused_before_running(self, tmp_path):
        out = co.run_command("echo hi", cwd=str(tmp_path / "nope"))
        assert out.startswith("Error:")
        assert "Directory does not exist" in out


class TestDesktopDryRun:
    """`dry_run=True` answers "what would be pressed" and touches nothing."""

    @pytest.fixture
    def desktop(self, monkeypatch, tmp_path):
        import time

        import argent.config as config
        import argent.tools.desktop_control_tools as tools
        from src.automation import desktop_guard as guard
        from src.automation import uia

        stored = {}
        monkeypatch.setattr(config, "_get", lambda k, d=None: stored.get(k, d))
        monkeypatch.setattr(config, "_set", lambda k, v: stored.__setitem__(k, v))
        guard.SERIES.reset()
        monkeypatch.setattr(guard, "audit_path", lambda: tmp_path / "a.jsonl")
        monkeypatch.setattr(guard, "enabled", lambda: True)

        window = _Window()
        monkeypatch.setattr(uia, "available", lambda: True)
        monkeypatch.setattr(uia, "find_window", lambda t: window if "notes" in str(t).lower() else None)
        monkeypatch.setattr(uia, "foreground_window", lambda: window)
        monkeypatch.setattr(uia, "bring_to_front", lambda info: True)
        monkeypatch.setattr(uia, "resolve", lambda info, elem: _Control(elem))

        typed = []
        monkeypatch.setattr(uia, "send_keys", lambda keys: typed.append(keys))

        # The observation an inspect would have left behind: ids e1..e3, «Delete» last.
        obs = _Observation([
            _Element(1, "Text area", role="Edit"),
            _Element(2, "Save"),
            _Element(3, "Delete"),
        ])
        obs.window = window
        obs.taken_at = time.monotonic()
        tools._LAST.clear()
        tools._LAST[window.hwnd] = obs

        return SimpleNamespace(tools=tools, window=window, typed=typed)

    def test_dry_run_reports_the_target_and_presses_nothing(self, desktop):
        out = desktop.tools.desktop_act(action="click", window="Notes", element="e2", dry_run=True)
        assert out.startswith("DRY RUN")
        assert "Save" in out
        assert "Nothing was done" in out
        assert desktop.typed == []                   # no keys, no clicks
        assert "role=Button" in out and "observation #" in out

    def test_dry_run_says_when_the_person_would_have_to_approve(self, desktop):
        out = desktop.tools.desktop_act(action="click", window="Notes", element="e3", dry_run=True)
        assert out.startswith("DRY RUN")
        assert "needs the person's word" in out      # «Delete»

    def test_a_stale_element_is_refused_before_anything_else(self, desktop):
        out = desktop.tools.desktop_act(action="click", window="Notes", element="e99")
        assert out.startswith("Error:")
        assert "not in the latest observation" in out
        assert "desktop_inspect" in out              # tells the model how to recover

    def test_a_coordinate_click_outside_the_window_is_refused(self, desktop):
        out = desktop.tools.desktop_act(action="click", window="Notes", x=5000, y=5000)
        assert out.startswith("Error:")
        assert "outside the window" in out

    def test_a_coordinate_click_inside_the_window_is_described_as_a_point(self, desktop):
        out = desktop.tools.desktop_act(action="click", window="Notes", x=150, y=120, dry_run=True)
        assert out.startswith("DRY RUN")
        assert "point=(150,120)" in out

    def test_an_unknown_window_says_how_to_list_them(self, desktop):
        out = desktop.tools.desktop_act(action="click", window="Excel", element="e1")
        assert out.startswith("Error:")
        assert "no visible window" in out


class _Rect:
    def __init__(self):
        self.left, self.top, self.width, self.height = 100, 100, 400, 300


class _Control:
    def __init__(self, elem):
        self.elem = elem

    def click(self):
        raise AssertionError("dry run must not click")


class _Window:
    hwnd = 4242
    title = "Notes"
    process = "notepad.exe"
    rect = (100, 100, 400, 300)

    def __init__(self):
        self.clicks = []


class _Element:
    def __init__(self, index, name, role="Button"):
        self.index = index
        self.id = f"e{index}"
        self.name = name
        self.role = role
        self.automation_id = f"id{index}"
        self.password = False
        self.focused = False
        self.rect = (110, 110, 80, 30)


class _Observation:
    """Mirrors src/automation/uia.Observation, including its id lookup."""

    def __init__(self, elements):
        import time

        self.elements = elements
        self.number = 1
        self.taken_at = time.monotonic()
        self.truncated = False

    def get(self, element_id):
        wanted = str(element_id or "").strip().lower().lstrip("[").rstrip("]")
        if wanted.isdigit():
            wanted = f"e{wanted}"
        return next((e for e in self.elements if e.id == wanted), None)
