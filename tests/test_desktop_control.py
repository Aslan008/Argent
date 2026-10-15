"""Desktop control: the rules (pure), the tools (with a stand-in for Windows) and one real window (Windows only)."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import config
from src.automation import desktop_guard as guard
from src.automation import uia

WINDOWS = os.name == "nt"


@pytest.fixture
def values(monkeypatch):
    stored = {}
    monkeypatch.setattr(config, "_get", lambda key, default=None: stored.get(key, default))
    monkeypatch.setattr(config, "_set", lambda key, val: stored.__setitem__(key, val))
    return stored


@pytest.fixture(autouse=True)
def _fresh_guard(tmp_path, monkeypatch, values):
    guard.SERIES.reset()
    guard._session_enabled = False
    monkeypatch.setattr(guard, "audit_path", lambda: tmp_path / "run" / "desktop_audit.jsonl")
    yield
    guard._session_enabled = False
    try:
        from src.automation import desktop_stop
        desktop_stop.stop()
    except Exception:
        pass
    guard.SERIES.reset()


# --- keys ------------------------------------------------------------------------------------------------------

class TestKeys:
    def test_chords(self):
        assert uia.parse_keys("ctrl+s") == [0x11, ord("S")]
        assert uia.parse_keys("Alt + F4") == [0x12, 0x73]
        assert uia.parse_keys("enter") == [0x0D]

    @pytest.mark.parametrize("bad", ["", "ctrl+", "hyper+x", "ctrl+ж"])
    def test_unknown_keys_are_refused(self, bad):
        with pytest.raises(ValueError):
            uia.parse_keys(bad)


# --- the rules -------------------------------------------------------------------------------------------------

class TestWindows:
    def test_unlisted_program_is_refused_with_the_way_to_allow_it(self):
        d = guard.check_window("notepad.exe", "Untitled - Notepad", allowed=[])
        assert not d and "/desktop allow notepad.exe" in d.reason

    def test_listed_program_passes(self):
        assert guard.check_window("Notepad.exe", "Untitled", allowed=["notepad.exe"])

    @pytest.mark.parametrize("proc", ["KeePass.exe", "consent.exe", "1Password.exe", "regedit.exe"])
    def test_secret_holders_are_never_acted_in_even_when_listed(self, proc):
        assert not guard.check_window(proc, "anything", allowed=[proc.lower()])

    @pytest.mark.parametrize("title", ["Enter your password", "Контроль учетных записей", "Windows Security",
                                       "Номер карты", "Sign in - Password"])
    def test_sensitive_titles_are_refused(self, title):
        assert not guard.check_window("app.exe", title, allowed=["app.exe"])


class TestDestructive:
    @pytest.mark.parametrize("name", ["Delete", "Удалить файл", "Send", "Отправить", "Pay now", "Format disk",
                                      "Sign out", "Подтвердить"])
    def test_irreversible_controls(self, name):
        assert guard.is_destructive(name)

    @pytest.mark.parametrize("name", ["Save", "Сохранить", "Open", "Name", "Next", "Cancel"])
    def test_ordinary_controls(self, name):
        assert not guard.is_destructive(name)


class TestChain:
    def ok(self, **over):
        base = dict(action="click", process="app.exe", title="App", element_name="Save", observed_number=3,
                    current_number=3, observation_age=1.0, foreground_ok=True, allowed=["app.exe"])
        base.update(over)
        return guard.check_action(**base)

    def test_off_by_default(self):
        d = self.ok()
        assert not d and "off" in d.reason

    def test_a_normal_action_passes_once_on(self):
        guard._session_enabled = True
        d = self.ok()
        assert d and not d.destructive

    def test_destructive_flag_comes_from_the_control(self):
        guard._session_enabled = True
        assert self.ok(element_name="Delete").destructive

    def test_password_field_is_never_typed_into(self):
        guard._session_enabled = True
        d = self.ok(action="type", is_password=True, typing=True)
        assert not d and "password" in d.reason

    def test_blind_actions_are_refused(self):
        guard._session_enabled = True
        assert "observe" in self.ok(observed_number=None, current_number=None).reason
        assert "older" in self.ok(observed_number=2).reason
        assert "old" in self.ok(observation_age=guard.OBSERVATION_MAX_AGE + 1).reason

    def test_locked_desktop(self):
        guard._session_enabled = True
        assert "locked" in self.ok(foreground_ok=False).reason

    def test_series_limit_and_new_turn(self):
        guard._session_enabled = True
        for _ in range(guard.MAX_SERIES):
            guard.SERIES.note_action()
        assert not self.ok()
        guard.SERIES.begin_turn()
        assert self.ok()

    def test_emergency_stop_holds_until_switched_on_again(self):
        guard._session_enabled = True
        guard.SERIES.stop("test")
        assert "stopped" in self.ok().reason
        guard.set_enabled(True)                     # the person says on again
        assert self.ok()
        guard.set_enabled(False)


class TestPace:
    def test_actions_are_spaced(self):
        series = guard.Series()
        series.note_action()
        start = time.monotonic()
        waited = series.wait_turn()
        assert waited > 0 and time.monotonic() - start >= waited * 0.8


class TestAudit:
    def test_text_is_recorded_by_length_and_a_redacted_head(self):
        key = "sk-" + "Qq1Ww2Ee3Rr4Tt5Yy6Uu7Ii8Oo9Pp"
        guard.audit("done", action="type", text=f"{key} and more text after")
        row = guard.read_audit(1)[0]
        assert row["text_len"] == len(f"{key} and more text after")
        assert key not in json.dumps(row)

    def test_read_back_in_order(self):
        for i in range(3):
            guard.audit("done", n=i)
        assert [r["n"] for r in guard.read_audit(5)] == [0, 1, 2]


class TestSettings:
    def test_allow_list_is_normalized_and_deduplicated(self, values):
        config.set_desktop_allowed_apps(["Notepad.EXE", "notepad.exe", " calc.exe ", ""])
        assert config.get_desktop_allowed_apps() == ["notepad.exe", "calc.exe"]
        assert guard.normalize_app("Notepad") == "notepad.exe"

    def test_default_chord_and_override(self, values):
        assert config.get_desktop_stop_chord() == "ctrl+alt+shift+f12"
        config.set_desktop_stop_chord("Ctrl+Alt+F11")
        assert guard.stop_chord() == "ctrl+alt+f11"

    def test_commands(self, values, monkeypatch):
        from src.cli.commands import CommandContext, dispatch_command
        monkeypatch.setattr(uia, "available", lambda: True)

        def run(text):
            return dispatch_command(text, CommandContext(agent=MagicMock(), user_input=text))

        assert run("/desktop allow Notepad").handled
        assert values["desktop_allowed_apps"] == ["notepad.exe"]
        run("/desktop allow keepass.exe")                      # refused: a password manager
        assert "keepass.exe" not in values["desktop_allowed_apps"]
        run("/desktop on")
        assert guard.enabled()
        run("/desktop off")
        assert not guard.enabled()
        run("/desktop key ctrl+alt+f11")
        assert values["desktop_stop_chord"] == "ctrl+alt+f11"
        run("/desktop deny notepad")
        assert values["desktop_allowed_apps"] == []


class TestEmergencyKey:
    def test_the_watcher_stops_the_series_when_the_chord_is_down(self):
        from src.automation import desktop_stop
        series = guard.Series()
        pressed = {"down": False}
        desktop_stop.start(series, chord_provider=lambda: "ctrl+x", is_down=lambda chord: pressed["down"])
        try:
            time.sleep(0.15)
            assert not series.stopped
            pressed["down"] = True
            deadline = time.time() + 3
            while not series.stopped and time.time() < deadline:
                time.sleep(0.05)
            assert series.stopped
        finally:
            desktop_stop.stop()


# --- the tools, with Windows replaced by a stand-in --------------------------------------------------------

def _element(id="e1", role="Button", name="Save", **kw):
    return uia.Element(id=id, role=role, name=name, rect=(10, 10, 80, 30), path=(0,), **kw)


@pytest.fixture
def fake(monkeypatch, values):
    """A window 'Notes' of 'notes.exe' with a field and two buttons; every call is recorded."""
    from tools import desktop_control_tools as tools
    window = uia.WindowInfo(hwnd=77, title="Notes", pid=1, process="notes.exe", rect=(0, 0, 400, 300))
    elements = [_element("e1", "Edit", "Title"), _element("e2", "Button", "Save"), _element("e3", "Button", "Delete"),
                _element("e4", "Edit", "Secret", password=True)]
    calls = []
    state = {"front": window.hwnd, "approve": True, "approvals": []}

    monkeypatch.setattr(uia, "available", lambda: True)
    monkeypatch.setattr(uia, "windows", lambda: [window])
    monkeypatch.setattr(uia, "find_window", lambda target: window if str(target).lower() in "notes" else None)
    monkeypatch.setattr(uia, "observe", lambda w, **k: uia.Observation(number=len(calls) + 1 + 100, window=w,
                                                                       elements=list(elements)))
    monkeypatch.setattr(uia, "resolve", lambda w, e: SimpleNamespace(element=e))
    monkeypatch.setattr(uia, "invoke", lambda c: calls.append(("invoke", c.element.name)) or "invoke")
    monkeypatch.setattr(uia, "click_at", lambda x, y, **k: calls.append(("click_at", x, y)))
    monkeypatch.setattr(uia, "set_value", lambda c, t: calls.append(("set_value", c.element.name, t)) or True)
    monkeypatch.setattr(uia, "focus", lambda c: None)
    monkeypatch.setattr(uia, "send_text", lambda t, interval=0.0: calls.append(("send_text", t)) or len(t))
    monkeypatch.setattr(uia, "send_keys", lambda k: calls.append(("send_keys", k)) or [])
    monkeypatch.setattr(uia, "scroll", lambda c, p, d, n=3: calls.append(("scroll", d, n, p)) or "scroll pattern")
    monkeypatch.setattr(uia, "drag", lambda x1, y1, x2, y2: calls.append(("drag", x1, y1, x2, y2)))
    monkeypatch.setattr(uia, "bring_to_front", lambda w: True)
    monkeypatch.setattr(uia, "foreground_window", lambda: window if state["front"] == window.hwnd else
                        uia.WindowInfo(hwnd=1, title="Other", pid=2, process="other.exe"))
    import approval
    monkeypatch.setattr(approval, "get_policy", lambda: approval.POLICY_ASK)

    def approve(action, *, destructive=False, grant_key=None):
        state["approvals"].append((action, destructive, grant_key))
        return state["approve"]
    monkeypatch.setattr(approval, "request_approval", approve)
    monkeypatch.setattr(guard.SERIES, "wait_turn", lambda: 0.0)
    tools._LAST.clear()
    config.set_desktop_allowed_apps(["notes.exe"])
    guard._session_enabled = True
    return SimpleNamespace(tools=tools, window=window, calls=calls, state=state)


class TestTools:
    def test_off_means_off(self, fake):
        guard._session_enabled = False
        assert "off" in fake.tools.desktop_inspect(action="windows")
        assert "off" in fake.tools.desktop_act(action="click", window="Notes", element="e2")

    def test_windows_are_marked_by_permission(self, fake):
        out = fake.tools.desktop_inspect(action="windows")
        assert "'Notes'" in out and "[allowed]" in out

    def test_a_program_that_is_not_allowed_is_not_even_read(self, fake):
        config.set_desktop_allowed_apps([])
        out = fake.tools.desktop_inspect(action="tree", window="Notes")
        assert out.startswith("Refused") and "/desktop allow notes.exe" in out

    def test_tree_find_and_ids(self, fake):
        tree = fake.tools.desktop_inspect(action="tree", window="Notes")
        assert "[e2] Button 'Save'" in tree and "(password field)" in tree
        found = fake.tools.desktop_inspect(action="find", window="Notes", query="save")
        assert "'Save'" in found and "'Delete'" not in found

    def test_acting_without_looking_is_refused(self, fake):
        out = fake.tools.desktop_act(action="click", window="Notes", element="e2")
        assert out.startswith("Error") and "desktop_inspect" in out
        assert fake.calls == []

    def test_click_goes_through_approval_and_shows_the_window_after(self, fake):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        out = fake.tools.desktop_act(action="click", window="Notes", element="e2")
        assert out.startswith("Done") and "[e2]" in out
        assert fake.calls == [("invoke", "Save")]
        action, destructive, grant = fake.state["approvals"][0]
        assert "«Save»" in action and not destructive and grant == "desktop:notes.exe"

    def test_a_destructive_control_is_asked_as_destructive(self, fake):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        fake.tools.desktop_act(action="click", window="Notes", element="e3")
        assert fake.state["approvals"][0][1] is True

    def test_declined_means_nothing_happens(self, fake):
        fake.state["approve"] = False
        fake.tools.desktop_inspect(action="tree", window="Notes")
        out = fake.tools.desktop_act(action="click", window="Notes", element="e2")
        assert out.startswith("Not done") and fake.calls == []

    def test_typing_into_a_password_field_is_refused(self, fake):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        out = fake.tools.desktop_act(action="type", window="Notes", element="e4", text="hunter2")
        assert out.startswith("Refused") and "password" in out and fake.calls == []

    def test_typing_and_setting_values(self, fake):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        assert fake.tools.desktop_act(action="type", window="Notes", element="e1", text="Привет {мир}").startswith("Done")
        assert ("send_text", "Привет {мир}") in fake.calls                  # no key-syntax: the braces stay
        fake.tools.desktop_inspect(action="tree", window="Notes")
        assert fake.tools.desktop_act(action="set_value", window="Notes", element="e1", text="abc").startswith("Done")
        assert ("set_value", "Title", "abc") in fake.calls

    def test_keys_are_refused_while_a_password_field_has_focus(self, fake, monkeypatch):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        latest = fake.tools._LAST[fake.window.hwnd]
        latest.elements[3].focused = True                                    # the password field
        out = fake.tools.desktop_act(action="hotkey", window="Notes", keys="ctrl+v")
        assert out.startswith("Refused") and "password" in out and fake.calls == []

    def test_keys_are_checked_before_asking(self, fake):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        assert fake.tools.desktop_act(action="hotkey", window="Notes", keys="hyper+x").startswith("Error")
        assert fake.state["approvals"] == []
        assert fake.tools.desktop_act(action="hotkey", window="Notes", keys="ctrl+s").startswith("Done")
        assert ("send_keys", "ctrl+s") in fake.calls

    def test_scrolling_a_control_and_the_window(self, fake):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        out = fake.tools.desktop_act(action="scroll", window="Notes", element="e1", direction="down", amount=5)
        assert out.startswith("Done") and ("scroll", "down", 5, (50, 25)) in fake.calls
        assert "прокрутить" in fake.state["approvals"][0][0] and "вниз ×5" in fake.state["approvals"][0][0]
        fake.tools.desktop_inspect(action="tree", window="Notes")
        assert fake.tools.desktop_act(action="scroll", window="Notes", direction="up").startswith("Done")
        assert ("scroll", "up", 3, (200, 150)) in fake.calls, "no element: the middle of the window"

    def test_scrolling_needs_a_real_direction_and_a_look_first(self, fake):
        assert fake.tools.desktop_act(action="scroll", window="Notes").startswith("Refused")
        fake.tools.desktop_inspect(action="tree", window="Notes")
        assert fake.tools.desktop_act(action="scroll", window="Notes", direction="diagonal").startswith("Error")
        assert fake.calls == []

    def test_dragging_between_elements_and_points(self, fake):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        out = fake.tools.desktop_act(action="drag", window="Notes", element="e1", to_element="e2")
        assert out.startswith("Done") and ("drag", 50, 25, 50, 25) in fake.calls
        fake.tools.desktop_inspect(action="tree", window="Notes")
        out = fake.tools.desktop_act(action="drag", window="Notes", element="e1", to_x=300, to_y=200)
        assert out.startswith("Done") and ("drag", 50, 25, 300, 200) in fake.calls

    def test_a_drop_onto_a_destructive_target_is_asked_as_destructive(self, fake):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        fake.tools.desktop_act(action="drag", window="Notes", element="e1", to_element="e3")      # «Delete»
        assert fake.state["approvals"][0][1] is True

    def test_a_drag_leaving_the_window_is_refused(self, fake):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        out = fake.tools.desktop_act(action="drag", window="Notes", element="e1", to_x=900, to_y=900)
        assert out.startswith("Error") and "outside" in out and fake.calls == []
        assert fake.tools.desktop_act(action="drag", window="Notes", element="e1").startswith("Error")

    def test_input_never_goes_to_another_window(self, fake):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        fake.state["front"] = 999                                           # something stole the focus
        out = fake.tools.desktop_act(action="click", window="Notes", element="e2")
        assert out.startswith("Refused") and "another window" in out
        assert fake.calls == []

    def test_emergency_stop_cuts_the_action(self, fake):
        fake.tools.desktop_inspect(action="tree", window="Notes")
        guard.SERIES.stop("test")
        out = fake.tools.desktop_act(action="click", window="Notes", element="e2")
        assert "stopped" in out.lower() and fake.calls == []

    def test_auto_mode_never_acts(self, fake, monkeypatch):
        import approval
        monkeypatch.setattr(approval, "get_policy", lambda: approval.POLICY_AUTO)
        fake.tools.desktop_inspect(action="tree", window="Notes")
        out = fake.tools.desktop_act(action="click", window="Notes", element="e2")
        assert out.startswith("Refused") and "auto mode" in out and fake.calls == []

    def test_plan_mode_and_sub_agents(self, fake):
        from src.agent import plan_mode
        from src.agent.subagents import FORBIDDEN_TOOLS
        plan_mode.PLAN.enter()
        try:
            assert plan_mode.violation("desktop_act", {"action": "click"})
            assert plan_mode.violation("desktop_inspect", {"action": "tree"}) is None
        finally:
            plan_mode.PLAN.leave()
        assert {"desktop_act", "desktop_inspect"} <= FORBIDDEN_TOOLS
        guard._session_enabled = False
        assert plan_mode.hides("desktop_act") and plan_mode.hides("desktop_inspect")

    def test_registered(self):
        from tools.lazy_registry import DEFAULT_TOOL_LOADERS
        from tools.schemas import TOOL_SCHEMAS
        names = [t["function"]["name"] for t in TOOL_SCHEMAS]
        for tool in ("desktop_inspect", "desktop_act"):
            assert names.count(tool) == 1 and tool in DEFAULT_TOOL_LOADERS


# --- one real window ---------------------------------------------------------------------------------------

TARGET = r'''
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$form = New-Object System.Windows.Forms.Form
$form.Text = "ArgentTarget"; $form.Width = 640; $form.Height = 300
$box = New-Object System.Windows.Forms.TextBox
$box.Name = "nameBox"; $box.AccessibleName = "Name"; $box.Location = New-Object System.Drawing.Point(20, 20); $box.Width = 200
$form.Controls.Add($box)
$secret = New-Object System.Windows.Forms.TextBox
$secret.Name = "secretBox"; $secret.AccessibleName = "Secret field"; $secret.UseSystemPasswordChar = $true
$secret.Location = New-Object System.Drawing.Point(20, 60); $secret.Width = 200
$form.Controls.Add($secret)
$label = New-Object System.Windows.Forms.Label
$label.Name = "resultLabel"; $label.Text = "idle"; $label.Location = New-Object System.Drawing.Point(20, 100); $label.Width = 300
$form.Controls.Add($label)
$save = New-Object System.Windows.Forms.Button
$save.Name = "saveButton"; $save.Text = "Save"; $save.Location = New-Object System.Drawing.Point(20, 140)
$save.Add_Click({ $label.Text = "saved: " + $box.Text })
$form.Controls.Add($save)
$del = New-Object System.Windows.Forms.Button
$del.Name = "deleteButton"; $del.Text = "Delete"; $del.Location = New-Object System.Drawing.Point(120, 140)
$del.Add_Click({ $label.Text = "deleted" })
$form.Controls.Add($del)
$list = New-Object System.Windows.Forms.ListBox
$list.Name = "longList"; $list.AccessibleName = "Long list"; $list.Location = New-Object System.Drawing.Point(440, 20)
$list.Width = 160; $list.Height = 100
1..60 | ForEach-Object { [void]$list.Items.Add(("Item {0:D2}" -f $_)) }
$form.Controls.Add($list)
$top = New-Object System.Windows.Forms.Label
$top.Name = "topLabel"; $top.Text = "top:0"; $top.Location = New-Object System.Drawing.Point(440, 130); $top.Width = 160
$form.Controls.Add($top)
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 100
$timer.Add_Tick({ $top.Text = "top:" + $list.TopIndex })
$timer.Start()
$src = New-Object System.Windows.Forms.Panel
$src.Name = "dragSource"; $src.AccessibleName = "Drag me"; $src.BorderStyle = "FixedSingle"; $src.BackColor = "LightSteelBlue"
$src.Location = New-Object System.Drawing.Point(240, 20); $src.Size = New-Object System.Drawing.Size(120, 40)
$script:moves = 0
$src.Add_MouseDown({ $script:moves = 0; $label.Text = "pressed" })
$src.Add_MouseMove({ if ($_.Button -eq "Left") { $script:moves++ } })
$src.Add_MouseUp({ $label.Text = "released: moves=" + $script:moves + " dx=" + $_.X + " dy=" + $_.Y })
$form.Controls.Add($src)
$dst = New-Object System.Windows.Forms.Panel
$dst.Name = "dropTarget"; $dst.AccessibleName = "Drop here"; $dst.BorderStyle = "FixedSingle"; $dst.BackColor = "Khaki"
$dst.Location = New-Object System.Drawing.Point(240, 100); $dst.Size = New-Object System.Drawing.Size(120, 40)
$form.Controls.Add($dst)
[void]$form.ShowDialog()
'''


@pytest.fixture
def target(tmp_path):
    if not (WINDOWS and uia.available()):
        pytest.skip("needs Windows and the uiautomation package")
    script = tmp_path / "target.ps1"
    script.write_text(TARGET, encoding="utf-8")
    proc = subprocess.Popen(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        deadline = time.time() + 40
        while time.time() < deadline and uia.find_window("ArgentTarget") is None:
            time.sleep(0.4)
        if uia.find_window("ArgentTarget") is None:
            pytest.skip("the test window did not appear (no interactive desktop?)")
        yield proc
    finally:
        proc.kill()


@pytest.mark.skipif(not WINDOWS, reason="Windows only")
class TestRealWindow:
    @pytest.fixture
    def armed(self, target, monkeypatch, values):
        import approval
        from tools import desktop_control_tools as tools
        monkeypatch.setattr(approval, "get_policy", lambda: approval.POLICY_ASK)
        asked = []

        def approve(action, *, destructive=False, grant_key=None):
            asked.append((action, destructive))
            return True
        monkeypatch.setattr(approval, "request_approval", approve)
        monkeypatch.setattr(guard.SERIES, "wait_turn", lambda: 0.0)
        config.set_desktop_allowed_apps(["powershell.exe"])
        guard._session_enabled = True
        tools._LAST.clear()
        return SimpleNamespace(tools=tools, asked=asked)

    def test_read_fill_and_press(self, armed):
        tree = armed.tools.desktop_inspect(action="tree", window="ArgentTarget")
        assert "'Save'" in tree and "(password field)" in tree
        name_id = next(line.split("]")[0].strip()[1:] for line in tree.splitlines() if "id=nameBox" in line)
        save_id = next(line.split("]")[0].strip()[1:] for line in tree.splitlines() if "id=saveButton" in line)

        out = armed.tools.desktop_act(action="set_value", window="ArgentTarget", element=name_id, text="Anna")
        assert out.startswith("Done"), out
        out = armed.tools.desktop_act(action="click", window="ArgentTarget", element=save_id)
        assert out.startswith("Done"), out
        assert "saved: Anna" in out                          # the answer is the window as it looks after

    def test_typed_text_arrives(self, armed):
        tree = armed.tools.desktop_inspect(action="tree", window="ArgentTarget")
        name_id = next(line.split("]")[0].strip()[1:] for line in tree.splitlines() if "id=nameBox" in line)
        out = armed.tools.desktop_act(action="type", window="ArgentTarget", element=name_id, text="Привет {мир}")
        assert out.startswith("Done"), out
        assert "Привет {мир}" in out

    def test_scrolling_a_real_list(self, armed):
        tree = armed.tools.desktop_inspect(action="tree", window="ArgentTarget")
        assert "top:0" in tree
        list_id = next(line.split("]")[0].strip()[1:] for line in tree.splitlines() if "id=longList" in line)
        out = armed.tools.desktop_act(action="scroll", window="ArgentTarget", element=list_id,
                                      direction="down", amount=6)
        assert out.startswith("Done"), out
        time.sleep(0.5)
        tree = armed.tools.desktop_inspect(action="tree", window="ArgentTarget")
        top = next(int(line.split("top:")[1].split()[0].strip("'")) for line in tree.splitlines() if "top:" in line)
        assert top > 0, tree

    def test_dragging_one_control_onto_another(self, armed):
        tree = armed.tools.desktop_inspect(action="tree", window="ArgentTarget")
        src_id = next(line.split("]")[0].strip()[1:] for line in tree.splitlines() if "id=dragSource" in line)
        dst_id = next(line.split("]")[0].strip()[1:] for line in tree.splitlines() if "id=dropTarget" in line)
        out = armed.tools.desktop_act(action="drag", window="ArgentTarget", element=src_id, to_element=dst_id)
        assert out.startswith("Done"), out
        # the form repaints on its own time (a busy machine takes a moment): look for up to three seconds
        deadline = time.time() + 3
        while True:
            tree = armed.tools.desktop_inspect(action="tree", window="ArgentTarget")
            if "released:" in tree or time.time() > deadline:
                break
            time.sleep(0.3)
        # the button went down on the source, the pointer travelled in steps, and it was let go over the
        # target: 80 logical pixels below the point where it was pressed (the source is 120x40, the target
        # starts 80 below it), in whatever scale the display runs
        import re
        m = re.search(r"released: moves=(\d+) dx=(-?\d+) dy=(-?\d+)", tree)
        assert m, tree
        moves, dx, dy = (int(g) for g in m.groups())
        assert moves >= 5, "a drag is a series of moves, not a jump"
        assert abs(dx - 60) <= 3 and abs(dy - 100) <= 3, (dx, dy)

    def test_secret_field_is_refused_and_delete_is_asked_as_destructive(self, armed):
        tree = armed.tools.desktop_inspect(action="tree", window="ArgentTarget")
        secret_id = next(line.split("]")[0].strip()[1:] for line in tree.splitlines() if "id=secretBox" in line)
        delete_id = next(line.split("]")[0].strip()[1:] for line in tree.splitlines() if "id=deleteButton" in line)
        refused = armed.tools.desktop_act(action="type", window="ArgentTarget", element=secret_id, text="x")
        assert refused.startswith("Refused") and "password" in refused
        armed.tools.desktop_act(action="click", window="ArgentTarget", element=delete_id)
        assert armed.asked[-1][1] is True
