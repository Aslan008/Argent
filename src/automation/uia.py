"""Another program's window as text, and the means to press its buttons: Windows UI Automation.

Why UIA and not pixels. A control tree names things (a button called "Save", a field called "Name"), survives a
different theme, a different DPI, a moved window, and is text — a model with no vision reads it as well as one
with. Coordinates stay as the way out when a program publishes no tree.

What is here is mechanism only: observe a window, find a control again, press / type / set. WHO may act on WHAT,
how often, and who is asked first lives in `desktop_guard.py`; nothing here is called by a tool without it.

Typed text and keys go in as `SendInput` events (Unicode text, virtual keys) built with `ctypes`: no key-syntax
language to escape, so a `{` in a sentence is a `{`.
"""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from logger import get_logger

log = get_logger("uia")

MAX_NODES = 150
MAX_DEPTH = 7
TIME_BUDGET = 6.0                   # seconds to walk one tree: a huge application must not hold the agent

_SKIP_ROLES = {"TitleBarControl", "ScrollBarControl", "ThumbControl", "SeparatorControl", "ImageControl"}
_CONTAINER_ROLES = {"PaneControl", "GroupControl", "CustomControl", "WindowControl", "ToolBarControl",
                    "MenuBarControl", "StatusBarControl", "HeaderControl", "TableControl", "ListControl",
                    "TreeControl", "TabControl", "DocumentControl"}


class UiaUnavailable(RuntimeError):
    pass


def available() -> bool:
    if os.name != "nt":
        return False
    try:
        import uiautomation  # noqa: F401
        return True
    except Exception:
        return False


def _lib():
    try:
        import uiautomation
        return uiautomation
    except Exception as e:
        raise UiaUnavailable(f"the 'uiautomation' package is not installed ({e})") from e


# --- what is observed ------------------------------------------------------------------------------------------

@dataclass
class Element:
    id: str                       # e1, e2 ... valid for the observation it came from
    role: str                     # Button, Edit, Text ...
    name: str
    automation_id: str = ""
    value: str = ""
    enabled: bool = True
    focused: bool = False
    password: bool = False
    rect: Tuple[int, int, int, int] = (0, 0, 0, 0)        # x, y, width, height (screen pixels)
    path: Tuple[int, ...] = ()    # child indexes from the window: how to find it again

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.rect
        return x + w // 2, y + h // 2


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    pid: int
    process: str = ""
    rect: Tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class Observation:
    number: int
    window: WindowInfo
    elements: List[Element] = field(default_factory=list)
    truncated: bool = False
    taken_at: float = field(default_factory=time.monotonic)

    def get(self, element_id: str) -> Optional[Element]:
        wanted = str(element_id or "").strip().lower().lstrip("[").rstrip("]")
        if wanted.isdigit():
            wanted = f"e{wanted}"
        return next((e for e in self.elements if e.id == wanted), None)


_counter = 0


def process_name(pid: int) -> str:
    try:
        import psutil
        return psutil.Process(pid).name()
    except Exception:
        return ""


def windows() -> List[WindowInfo]:
    """Visible top-level windows of the person's desktop."""
    from src.automation.desktop import DesktopAutomation
    out = []
    for w in DesktopAutomation.list_windows():
        out.append(WindowInfo(hwnd=int(w["hwnd"]), title=w["title"], pid=int(w["pid"]),
                              process=process_name(int(w["pid"])),
                              rect=(w["x"], w["y"], w["width"], w["height"])))
    return out


def find_window(target) -> Optional[WindowInfo]:
    """By handle (an int, or digits) or by part of the title (the topmost match in z-order wins)."""
    items = windows()
    text = str(target or "").strip()
    if not text:
        return None
    if text.isdigit():
        return next((w for w in items if w.hwnd == int(text)), None)
    low = text.lower()
    exact = [w for w in items if w.title.lower() == low]
    return (exact or [w for w in items if low in w.title.lower()] or [None])[0]


def _control_for(window: WindowInfo):
    auto = _lib()
    control = auto.ControlFromHandle(window.hwnd)
    if control is None:
        raise UiaUnavailable(f"window '{window.title}' cannot be reached through UI Automation")
    return control


def _short_role(control_type_name: str) -> str:
    return control_type_name[:-7] if control_type_name.endswith("Control") else control_type_name


def observe(window: WindowInfo, max_nodes: int = MAX_NODES, max_depth: int = MAX_DEPTH) -> Observation:
    """Walk the control tree of `window` and keep what a person could see and touch."""
    global _counter
    auto = _lib()
    root = _control_for(window)
    _counter += 1
    obs = Observation(number=_counter, window=window)
    deadline = time.monotonic() + TIME_BUDGET
    seq = [0]

    def walk(control, path: Tuple[int, ...], depth: int) -> None:
        if obs.truncated:
            return
        try:
            children = control.GetChildren()
        except Exception:
            return
        for index, child in enumerate(children):
            if time.monotonic() > deadline or len(obs.elements) >= max_nodes:
                obs.truncated = True
                return
            try:
                role_full = child.ControlTypeName
                if role_full in _SKIP_ROLES:
                    continue
                if child.IsOffscreen:
                    continue
                name = (child.Name or "").strip()
                aid = (child.AutomationId or "").strip()
                rect = child.BoundingRectangle
                width, height = rect.width(), rect.height()
                if width <= 0 or height <= 0:
                    continue
                here = path + (index,)
                interesting = role_full not in _CONTAINER_ROLES or bool(name)
                if interesting:
                    seq[0] += 1
                    value, password = "", False
                    if role_full in ("EditControl", "ComboBoxControl", "SpinnerControl", "DocumentControl"):
                        try:
                            password = bool(child.GetPropertyValue(auto.PropertyId.IsPasswordProperty))
                        except Exception:
                            password = False
                        if not password:
                            try:
                                value = str(child.GetValuePattern().Value or "")[:200]
                            except Exception:
                                value = ""
                    obs.elements.append(Element(
                        id=f"e{seq[0]}", role=_short_role(role_full), name=name[:120], automation_id=aid[:60],
                        value=value, enabled=bool(child.IsEnabled), focused=bool(child.HasKeyboardFocus),
                        password=password, rect=(rect.left, rect.top, width, height), path=here))
                if depth < max_depth:
                    walk(child, here, depth + 1)
            except Exception as e:                       # a control that vanished while we looked
                log.debug("uia: skipped a control: %s", e)

    walk(root, (), 1)
    return obs


def render(obs: Observation, limit: int = 80) -> str:
    """The observation as lines a model can read and refer to by id."""
    w = obs.window
    lines = [f"Window '{w.title}' ({w.process or 'pid ' + str(w.pid)}), observation #{obs.number}, "
             f"{len(obs.elements)} controls{' (cut: more exist)' if obs.truncated else ''}:"]
    for e in obs.elements[:limit]:
        bits = [f"[{e.id}] {e.role}"]
        if e.name:
            bits.append(f"'{e.name}'")
        if e.automation_id:
            bits.append(f"id={e.automation_id}")
        if e.password:
            bits.append("(password field)")
        elif e.value:
            bits.append(f"value='{e.value[:60]}'")
        if not e.enabled:
            bits.append("disabled")
        if e.focused:
            bits.append("focused")
        bits.append(f"@{e.rect[0]},{e.rect[1]} {e.rect[2]}x{e.rect[3]}")
        lines.append("  " + " ".join(bits))
    if len(obs.elements) > limit:
        lines.append(f"  … {len(obs.elements) - limit} more: use desktop_inspect(action='find', query=...)")
    return "\n".join(lines)


def match(obs: Observation, query: str, role: str = "") -> List[Element]:
    low = str(query or "").strip().lower()
    want = str(role or "").strip().lower()
    out = []
    for e in obs.elements:
        if want and want not in e.role.lower():
            continue
        if low and low not in e.name.lower() and low not in e.automation_id.lower() and low not in e.value.lower():
            continue
        out.append(e)
    return out


# --- finding a control again, acting on it ---------------------------------------------------------------------

def resolve(window: WindowInfo, element: Element):
    """The live control for an element of an earlier observation, or None when the window changed under it."""
    control = _control_for(window)
    try:
        for index in element.path:
            control = control.GetChildren()[index]
        if _short_role(control.ControlTypeName) != element.role:
            return None
        if element.name and (control.Name or "").strip()[:120] != element.name:
            return None
        return control
    except Exception:
        return None


def invoke(control) -> str:
    """Press it the way a screen reader would; a click at its centre when it has no invoke pattern."""
    for getter in ("GetInvokePattern", "GetTogglePattern", "GetSelectionItemPattern", "GetExpandCollapsePattern"):
        try:
            pattern = getattr(control, getter)()
            if pattern is None:
                continue
            if getter == "GetInvokePattern":
                pattern.Invoke()
            elif getter == "GetTogglePattern":
                pattern.Toggle()
            elif getter == "GetSelectionItemPattern":
                pattern.Select()
            else:
                pattern.Expand()
            return getter[3:-7].lower()
        except Exception:
            continue
    control.Click(simulateMove=False)
    return "click"


def click_at(x: int, y: int, button: str = "left", double: bool = False) -> None:
    auto = _lib()
    auto.SetCursorPos(int(x), int(y))
    time.sleep(0.03)
    if button == "right":
        auto.RightClick(int(x), int(y), waitTime=0.05)
    elif double:
        auto.DoubleClick(int(x), int(y), waitTime=0.05)
    else:
        auto.Click(int(x), int(y), waitTime=0.05)


_DIRECTIONS = ("up", "down", "left", "right")
MAX_SCROLL_STEPS = 20


def scroll(control, point, direction: str, amount: int = 3) -> str:
    """Scroll a scrollable control by `amount` small steps: through its own scroll pattern when it has one,
    else with the mouse wheel over `point` (vertical only: the wheel has no sideways here)."""
    direction = (direction or "down").strip().lower()
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of: {', '.join(_DIRECTIONS)}")
    steps = max(1, min(int(amount or 1), MAX_SCROLL_STEPS))
    auto = _lib()
    if control is not None:
        try:
            pattern = control.GetScrollPattern()
        except Exception:
            pattern = None
        if pattern is not None:
            none, dec, inc = (auto.ScrollAmount.NoAmount, auto.ScrollAmount.SmallDecrement,
                              auto.ScrollAmount.SmallIncrement)
            h, v = {"up": (none, dec), "down": (none, inc), "left": (dec, none), "right": (inc, none)}[direction]
            for _ in range(steps):
                pattern.Scroll(h, v, waitTime=0.03)
            return "scroll pattern"
    if direction in ("left", "right"):
        raise ValueError("sideways scrolling needs a control with a scroll pattern: give its element id")
    if point is None:
        raise ValueError("nothing to scroll: give an element or a point")
    auto.SetCursorPos(int(point[0]), int(point[1]))
    time.sleep(0.03)
    (auto.WheelUp if direction == "up" else auto.WheelDown)(steps, interval=0.03, waitTime=0.1)
    return "mouse wheel"


def drag(x1: int, y1: int, x2: int, y2: int) -> None:
    """Press at the first point, move to the second in small steps, release.

    A program starts a drag only after the pointer has moved a few pixels with the button down, and only a target
    that sees the pointer arrive (a move message over it) accepts a drop: a jump from one point to the other is
    not a drag to it."""
    user32 = ctypes.windll.user32
    left_down, left_up, move = 0x0002, 0x0004, 0x0001
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    user32.SetCursorPos(x1, y1)
    time.sleep(0.1)
    user32.mouse_event(left_down, 0, 0, 0, 0)
    try:
        time.sleep(0.1)
        steps = max(15, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 12))
        for i in range(1, steps + 1):
            user32.SetCursorPos(x1 + (x2 - x1) * i // steps, y1 + (y2 - y1) * i // steps)
            user32.mouse_event(move, 0, 0, 0, 0)               # the move message that makes the target notice
            time.sleep(0.02)
        time.sleep(0.15)
    finally:
        user32.mouse_event(left_up, 0, 0, 0, 0)               # never leave the button held


def set_value(control, text: str) -> bool:
    try:
        control.GetValuePattern().SetValue(text)
        return True
    except Exception:
        return False


def focus(control) -> None:
    control.SetFocus()


def _is_front(hwnd: int) -> bool:
    return int(ctypes.windll.user32.GetForegroundWindow() or 0) == int(hwnd)


def bring_to_front(window: WindowInfo) -> bool:
    """Make the window the one that receives input. Windows lets only the foreground process do that, so the
    thread's input is attached to the current foreground window's for the call; a tap of Alt is the fallback."""
    try:
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        hwnd = window.hwnd
        if _is_front(hwnd):
            return True
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)                         # SW_RESTORE
        front = user32.GetForegroundWindow()
        mine = kernel32.GetCurrentThreadId()
        theirs = user32.GetWindowThreadProcessId(front, None) if front else 0
        attached = bool(theirs and theirs != mine and user32.AttachThreadInput(mine, theirs, True))
        try:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(mine, theirs, False)
        for _ in range(10):
            if _is_front(hwnd):
                return True
            time.sleep(0.05)
        user32.keybd_event(0x12, 0, 0, 0)                      # Alt down/up: counts as the input that frees the lock
        user32.keybd_event(0x12, 0, 2, 0)
        user32.SetForegroundWindow(hwnd)
        for _ in range(10):
            if _is_front(hwnd):
                return True
            time.sleep(0.05)
        return False
    except Exception as e:
        log.debug("bring_to_front failed: %s", e)
        return False


# --- SendInput: text and keys --------------------------------------------------------------------------------

_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_INPUT_KEYBOARD = 1

_VK = {
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B, "windows": 0x5B,
    "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B, "tab": 0x09, "space": 0x20,
    "backspace": 0x08, "delete": 0x2E, "del": 0x2E, "insert": 0x2D, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
}
for _n in range(1, 25):
    _VK[f"f{_n}"] = 0x70 + _n - 1
_MODIFIERS = {"ctrl", "control", "alt", "shift", "win", "windows"}


class _KeyboardInput(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _MouseInput(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _InputUnion(ctypes.Union):
    _fields_ = [("ki", _KeyboardInput), ("mi", _MouseInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _InputUnion)]


def parse_keys(spec: str) -> List[int]:
    """'ctrl+shift+s' -> virtual-key codes in press order. Raises ValueError for a key it does not know."""
    codes = []
    text = str(spec or "").lower().replace(" ", "")
    if not text:
        raise ValueError("no keys given")
    for part in text.split("+"):
        if not part:
            raise ValueError(f"an empty key in '{spec}'")
        if part in _VK:
            codes.append(_VK[part])
        elif len(part) == 1 and part.isalnum() and part.isascii():
            codes.append(ord(part.upper()))
        else:
            raise ValueError(f"unknown key '{part}'")
    if all(code in (0x10, 0x11, 0x12, 0x5B) for code in codes):
        raise ValueError("a chord needs a key besides the modifiers")
    return codes


def _send(events: list) -> None:
    array = (_Input * len(events))(*events)
    sent = ctypes.windll.user32.SendInput(len(events), array, ctypes.sizeof(_Input))
    if sent != len(events):
        raise OSError(f"SendInput delivered {sent} of {len(events)} events (a program with higher rights is in front?)")


def _key_event(vk: int = 0, scan: int = 0, flags: int = 0) -> _Input:
    union = _InputUnion()
    union.ki = _KeyboardInput(vk, scan, flags, 0, None)
    return _Input(_INPUT_KEYBOARD, union)


def send_text(text: str, interval: float = 0.0) -> int:
    """Type `text` exactly as given (any language, any symbol). Returns the number of characters sent."""
    count = 0
    for ch in text:
        if ch == "\n":
            send_keys("enter")
            count += 1
            continue
        units = ch.encode("utf-16-le")
        events = []
        for i in range(0, len(units), 2):
            unit = int.from_bytes(units[i:i + 2], "little")
            events.append(_key_event(0, unit, _KEYEVENTF_UNICODE))
            events.append(_key_event(0, unit, _KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP))
        _send(events)
        count += 1
        if interval:
            time.sleep(interval)
    return count


def send_keys(spec: str) -> List[int]:
    """Press a chord: modifiers down, the key, everything up in reverse."""
    codes = parse_keys(spec)
    events = [_key_event(vk) for vk in codes] + [_key_event(vk, 0, _KEYEVENTF_KEYUP) for vk in reversed(codes)]
    _send(events)
    return codes


def key_is_down(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def chord_is_down(spec: str) -> bool:
    try:
        return all(key_is_down(vk) for vk in parse_keys(spec))
    except ValueError:
        return False


def foreground_window() -> Optional[WindowInfo]:
    """The window the person is looking at (None when the desktop is locked or on a secure desktop)."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return WindowInfo(hwnd=int(hwnd), title=buf.value, pid=int(pid.value), process=process_name(int(pid.value)),
                          rect=(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top))
    except Exception:
        return None
