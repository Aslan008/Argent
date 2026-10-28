"""Desktop Window Automation and Screen Capture for Argent.

Provides safe, targeted window enumeration, full-screen capture, and clipboard
image retrieval using Win32 API and Pillow without global display-grabber crashes.
"""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import struct
import time
from typing import Any, Dict, List, Optional, Union

from argent.logger import get_logger

log = get_logger("desktop_automation")

PW_RENDERFULLCONTENT = 2
DESKTOP_ALL = 0x01FF


def _bind_win32() -> None:
    """Declare the Win32 signatures this module depends on, once, at import.

    Without `argtypes`/`restype`, ctypes assumes a 32-bit int for every argument and
    for the return value. A window handle on a 64-bit system is much larger than that,
    so two failures follow from the same cause:

    * passing one raises `OverflowError: int too long to convert` — capture died on the
      very first `GetWindowDC(hwnd)`, and `capture_desktop_screen` with it;
    * receiving one silently truncates it — an HDC from `GetDC` came back cut in half,
      which is worse, because a bogus handle fails later and looks unrelated.

    Both were live. Declaring the signatures here, instead of wrapping each call site,
    is what keeps a new call from forgetting them.
    """
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    dwmapi = ctypes.windll.dwmapi
    kernel32 = ctypes.windll.kernel32

    # Handles: HWND/HDC/HBITMAP are pointers, never ints.
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.GetWindowDC.argtypes = [wintypes.HWND]
    user32.GetWindowDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    user32.PrintWindow.restype = wintypes.BOOL
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    # The buffer is a c_void_p: callers pass a c_wchar array, not a Python str.
    user32.GetWindowTextW.argtypes = [wintypes.HWND, ctypes.c_void_p, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.GetThreadDesktop.argtypes = [wintypes.DWORD]
    user32.GetThreadDesktop.restype = wintypes.HANDLE
    user32.OpenDesktopW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    user32.OpenDesktopW.restype = wintypes.HANDLE
    user32.SetThreadDesktop.argtypes = [wintypes.HANDLE]
    user32.SetThreadDesktop.restype = wintypes.BOOL
    user32.CloseDesktop.argtypes = [wintypes.HANDLE]
    user32.CloseDesktop.restype = wintypes.BOOL

    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
    gdi32.BitBlt.restype = wintypes.BOOL
    # The pixel buffer is a c_void_p (a c_char array) and the header is passed as bytes.
    gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                                ctypes.c_void_p, ctypes.c_char_p, wintypes.UINT]
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL

    dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long       # HRESULT

    kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    # The executable name is the stable identity of a window: a browser's title is the
    # page it currently shows, so "Яндекс Браузер" matches nothing while the browser is
    # plainly there.
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                    ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL


if os.name == "nt":
    _bind_win32()


@contextmanager
def _attached_user_desktop():
    """Ensure thread is attached to the user's interactive 'default' desktop.

    When running inside IDEs (e.g. Antigravity IDE), services, or sandboxed
    subprocesses, the worker thread may belong to an isolated desktop (e.g.
    'exebox-...'), causing EnumWindows to return 0 user windows. This context
    manager safely switches to 'default' and guarantees restoration in finally.
    """
    if os.name != "nt":
        yield
        return

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    old_desk = None
    hdesk_default = None

    try:
        cur_thread = kernel32.GetCurrentThreadId()
        old_desk = user32.GetThreadDesktop(cur_thread)
        hdesk_default = user32.OpenDesktopW("default", 0, False, DESKTOP_ALL)
        if hdesk_default and hdesk_default != old_desk:
            user32.SetThreadDesktop(hdesk_default)
    except Exception as e:
        log.debug("Desktop attach attempt bypassed: %s", e)

    try:
        # Every window walk, capture and click in this module shares one
        # coordinate space: physical pixels. Otherwise a listing taken here and a
        # PrintWindow frame disagree by the monitor's scaling factor.
        with _dpi_aware_thread():
            yield
    finally:
        if old_desk and hdesk_default and hdesk_default != old_desk:
            try:
                user32.SetThreadDesktop(old_desk)
            except Exception:
                pass
        if hdesk_default:
            try:
                user32.CloseDesktop(hdesk_default)
            except Exception:
                pass


# ── coordinate space ────────────────────────────────────────────────────────
#
# Argent's Python process is normally DPI-unaware: GetWindowRect answers in
# virtualised pixels while PrintWindow paints at the window's physical size. The
# bitmap is then smaller than the window and the frame comes back clipped — the
# top-left slice only. The switch below is per-thread, so nothing else in the
# process changes its scaling.

DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
DWMWA_EXTENDED_FRAME_BOUNDS = 9


@contextmanager
def _dpi_aware_thread():
    if os.name != "nt":
        yield
        return
    user32 = ctypes.windll.user32
    previous = None
    try:
        get_ctx = getattr(user32, "GetThreadDpiAwarenessContext", None)
        set_ctx = getattr(user32, "SetThreadDpiAwarenessContext", None)
        if get_ctx and set_ctx:
            get_ctx.restype = ctypes.c_void_p
            set_ctx.argtypes = [ctypes.c_void_p]
            set_ctx.restype = ctypes.c_void_p
            previous = get_ctx()
            set_ctx(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    except Exception as e:  # Windows older than 10 1607
        log.debug("DPI context switch unavailable: %s", e)
        previous = None
    try:
        yield
    finally:
        if previous:
            try:
                user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(previous))
            except Exception:
                pass


def _visible_rect(hwnd: int) -> tuple:
    """(x, y, width, height) the user actually sees.

    GetWindowRect includes the invisible resize border DWM draws around a window
    (about 7 px per side), so a report built from it disagrees with the pixels.
    """
    rect = wintypes.RECT()
    try:
        dwm = ctypes.windll.dwmapi
        hr = dwm.DwmGetWindowAttribute(
            wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect))
        if hr == 0 and rect.right > rect.left and rect.bottom > rect.top:
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    except Exception as e:
        log.debug("DWM frame bounds unavailable for %s: %s", hwnd, e)
    fallback = wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(fallback))
    return fallback.left, fallback.top, fallback.right - fallback.left, fallback.bottom - fallback.top


def _virtual_screen() -> Dict[str, int]:
    """The whole desktop, in the coordinates this thread works in."""
    user32 = ctypes.windll.user32
    return {
        "x": user32.GetSystemMetrics(76),
        "y": user32.GetSystemMetrics(77),
        "width": user32.GetSystemMetrics(78),
        "height": user32.GetSystemMetrics(79),
    }


def _offscreen_pixels(left: int, top: int, width: int, height: int) -> int:
    """How much of the rectangle lies outside the desktop: a partly off-screen
    window can only be photographed as far as it is visible."""
    screen = _virtual_screen()
    right = min(left + width, screen["x"] + screen["width"])
    bottom = min(top + height, screen["y"] + screen["height"])
    visible_w = max(0, right - max(left, screen["x"]))
    visible_h = max(0, bottom - max(top, screen["y"]))
    return max(0, width * height - visible_w * visible_h)


def _clamp_rect(x, y, w, h, width: int, height: int) -> tuple:
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def _parse_region(region: Optional[str], width: int, height: int) -> Optional[tuple]:
    """`bottom:200`, `top:120`, `left:300`, `right:400` or `x,y,w,h`.

    The interesting part of a terminal is where the answer ends, not its top: a
    strip near the prompt is often the only row a person is asking about.
    """
    if not region:
        return None
    text = str(region).strip().lower().replace(" ", "")
    try:
        if "," in text:
            parts = [int(float(p)) for p in text.split(",")]
            if len(parts) == 4:
                return _clamp_rect(*parts, width, height)
            return None
        side, _, value = text.partition(":")
        if not value:
            return None
        amount = int(float(value))
        if side == "bottom":
            return _clamp_rect(0, height - amount, width, amount, width, height)
        if side == "top":
            return _clamp_rect(0, 0, width, amount, width, height)
        if side == "left":
            return _clamp_rect(0, 0, amount, height, width, height)
        if side == "right":
            return _clamp_rect(width - amount, 0, amount, height, width, height)
    except (ValueError, TypeError):
        return None
    return None


def _frame_is_blank(buffer: bytes, pixel_bytes: int = 4, samples: int = 4096) -> bool:
    """True when the frame is one flat colour — an empty PrintWindow frame.

    Sampling is pixel-aligned: a byte stride can step over a small but real piece
    of the window and declare a live frame blank.
    """
    if not buffer:
        return True
    pixels = len(buffer) // pixel_bytes
    if pixels <= 0:
        return True
    first = buffer[0:3]
    step = max(1, pixels // samples)
    for p in range(0, pixels, step):
        i = p * pixel_bytes
        if buffer[i:i + 3] != first:
            return False
    return True


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

#: pid -> executable file name. Pids are reused after a process dies, but a stale
#: name is only a worse guess than asking again, and the cost of asking per window
#: per listing is real.
_PROCESS_NAMES: Dict[int, str] = {}


def _process_name(pid: int) -> str:
    """The executable file name behind a pid, e.g. "browser.exe".

    This is the name that stays put: a browser calls its window after the page it is
    showing, so a person says "browser" while the title bar says "Gemini — ...".
    """
    if not pid:
        return ""
    if pid in _PROCESS_NAMES:
        return _PROCESS_NAMES[pid]
    name = ""
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            try:
                size = wintypes.DWORD(32768)
                buf = ctypes.create_unicode_buffer(size.value)
                if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    name = os.path.basename(buf.value)
            finally:
                kernel32.CloseHandle(handle)
    except Exception as exc:
        log.debug("process name for pid %s unavailable: %s", pid, exc)
    _PROCESS_NAMES[pid] = name
    return name


def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def match_windows(windows: List[Dict[str, Any]], target: str):
    """(window, how) pairs, strongest identity first.

    Title matching alone is not enough: for a browser the title is the current page,
    so the obvious name — the one a person uses — matches nothing. The order below is
    "what the caller most likely meant": an exact title, then any title containing it,
    then the executable name, then loose word overlap.
    """
    raw = str(target or "").strip()
    if not raw:
        return []
    want = _norm(raw)
    want_stem = want[:-4] if want.endswith(".exe") else want
    found, seen = [], set()

    def add(win: Dict[str, Any], how: str) -> None:
        if win.get("hwnd") not in seen:
            seen.add(win.get("hwnd"))
            found.append((win, how))

    for win in windows:
        if _norm(win.get("title")) == want:
            add(win, "title")
    for win in windows:
        if want and want in _norm(win.get("title")):
            add(win, "title")
    for win in windows:
        exe = _norm(win.get("process"))
        stem = exe[:-4] if exe.endswith(".exe") else exe
        if exe and (exe == want or stem == want_stem):
            add(win, "process")
    for win in windows:
        exe = _norm(win.get("process"))
        stem = exe[:-4] if exe.endswith(".exe") else exe
        if stem and want_stem and (want_stem in stem or stem in want_stem):
            add(win, "process")
    words = [word for word in want.split() if len(word) > 2]
    if len(words) >= 2:
        for win in windows:
            title = _norm(win.get("title"))
            if all(word in title for word in words):
                add(win, "words")
    return found


def _describe_windows(windows: List[Dict[str, Any]], limit: int = 8) -> str:
    """What exists, so the next call can name one instead of guessing again."""
    lines = [f"Visible windows ({len(windows)}):"]
    for win in windows[:limit]:
        lines.append(
            f"- [HWND: {win.get('hwnd')}] [PID: {win.get('pid')}] \"{win.get('title')}\" "
            f"<{win.get('process') or 'unknown process'}>")
    if len(windows) > limit:
        lines.append(f"- … and {len(windows) - limit} more (list_desktop_windows)")
    return "\n".join(lines)


def _pick_among(matches, foreground: Optional[int] = None):
    """One window out of several, or None when the choice is genuinely ambiguous.

    Two windows of the SAME program are not ambiguous: when someone says the program's
    name, they mean the one in front. Two different programs are ambiguous, and there
    the call must refuse — guessing is how the wrong window gets photographed and
    described as if it were the right one.
    """
    if not matches:
        return None, ""
    processes = {_norm(win.get("process")) for win, _ in matches}
    if len(processes) != 1 or "" in processes:
        return None, ""
    if foreground is None and os.name == "nt":
        foreground = int(ctypes.windll.user32.GetForegroundWindow() or 0)
    for win, how in matches:
        if foreground and int(win["hwnd"]) == int(foreground):
            return (win, how), "foreground"
    biggest = max(matches, key=lambda pair: pair[0]["width"] * pair[0]["height"])
    return biggest, "largest"


def _blit_screen_rect(left: int, top: int, width: int, height: int, mem_dc) -> bool:
    """Copy what is really on screen into `mem_dc` (fallback after a blank frame)."""
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hdc_screen = user32.GetDC(0)
    if not hdc_screen:
        return False
    try:
        return bool(gdi32.BitBlt(mem_dc, 0, 0, width, height, hdc_screen, left, top, 0x00CC0020))
    finally:
        user32.ReleaseDC(0, hdc_screen)


class DesktopAutomation:
    """Win32-based desktop window, screen, and clipboard inspector."""

    @staticmethod
    def is_available() -> bool:
        return os.name == "nt"

    @classmethod
    def list_windows(cls) -> List[Dict[str, Any]]:
        """List all visible top-level desktop windows with titles and geometry."""
        if not cls.is_available():
            return []

        user32 = ctypes.windll.user32
        windows = []

        EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _enum_cb(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return 1

            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return 1

            title_buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, title_buf, length + 1)
            title = title_buf.value.strip()

            if not title:
                return 1

            left, top, width, height = _visible_rect(hwnd)

            # Filter out zero-size or off-screen helper windows
            if width <= 0 or height <= 0:
                return 1

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

            windows.append({
                "hwnd": int(hwnd),
                "title": title,
                "pid": int(pid.value),
                "process": _process_name(int(pid.value)),
                "x": left,
                "y": top,
                "width": width,
                "height": height,
            })
            return 1

        with _attached_user_desktop():
            cb = EnumWindowsProc(_enum_cb)
            user32.EnumWindows(cb, 0)
        return windows

    @classmethod
    def find_window(cls, title_substring: str) -> Optional[Dict[str, Any]]:
        """First window matching a title fragment, executable name or numeric HWND."""
        matches = cls.find_windows(title_substring)
        return matches[0] if matches else None

    @classmethod
    def find_window_matches(cls, target: str):
        """(window, how) pairs for a name: exact title, title fragment, executable, words."""
        return match_windows(cls.list_windows(), target)

    @classmethod
    def find_windows(cls, target: str) -> List[Dict[str, Any]]:
        """Every window the name could mean: an ambiguous name must never be guessed.

        A title is not the only identity. A browser names its window after the page, so
        "browser" — the obvious word for a person — used to match nothing at all.
        """
        return [win for win, _how in cls.find_window_matches(target)]

    @classmethod
    def describe_window(cls, hwnd: int) -> Optional[Dict[str, Any]]:
        """Title, pid and visible rectangle of one window.

        A capture asked for by HWND used to report the literal string "HWND 984082"
        as its title, so nobody could tell which window had been photographed.
        """
        if not cls.is_available():
            return None
        user32 = ctypes.windll.user32
        if not user32.IsWindow(wintypes.HWND(hwnd)):
            return None
        length = user32.GetWindowTextLengthW(wintypes.HWND(hwnd))
        title = ""
        if length:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(wintypes.HWND(hwnd), buf, length + 1)
            title = buf.value.strip()
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
        left, top, width, height = _visible_rect(hwnd)
        return {
            "hwnd": int(hwnd),
            "title": title or f"HWND {hwnd}",
            "pid": int(pid.value),
            "x": left,
            "y": top,
            "width": width,
            "height": height,
        }

    @classmethod
    def capture_window(
        cls,
        target: Union[str, int],
        output_path: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Capture an image of a specific window by HWND or title substring.

        Returns a dictionary with status, file path, dimensions, and vision preview.
        """
        if not cls.is_available():
            return {"success": False, "error": "Desktop capture is only available on Windows."}

        hwnd = None
        win_info = None

        if isinstance(target, int):
            hwnd = target
        elif isinstance(target, str):
            s = target.strip()
            if s.lower().startswith("0x"):
                try:
                    hwnd = int(s, 16)
                except ValueError:
                    hwnd = None
            elif s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                hwnd = int(s)

        chosen_how = None
        chosen_reason = ""
        other_titles: List[str] = []

        if hwnd is None:
            matches = cls.find_window_matches(str(target))
            if not matches:
                # A miss must hand back the names that do exist: "not found" alone
                # costs a whole round trip to list_desktop_windows, and the useful
                # name is the executable, not the page title the window shows.
                return {"success": False,
                        "error": f"No visible window matches '{target}'.\n"
                                 + _describe_windows(cls.list_windows())}
            if len(matches) > 1:
                picked, chosen_reason = _pick_among(matches)
                if picked is None:
                    listing = "\n".join(
                        f"- [HWND: {w['hwnd']}] [PID: {w['pid']}] \"{w['title']}\" "
                        f"<{w.get('process') or 'unknown process'}> "
                        f"({w['width']}x{w['height']} at {w['x']},{w['y']})"
                        for w, _ in matches
                    )
                    return {
                        "success": False,
                        "candidates": [w for w, _ in matches],
                        "error": (f"'{target}' matches {len(matches)} windows — pick one by HWND:\n{listing}"),
                    }
                win_info, chosen_how = picked
                hwnd = win_info["hwnd"]
                other_titles = [w["title"] for w, _ in matches if w["hwnd"] != hwnd]
            else:
                win_info, chosen_how = matches[0]
                hwnd = win_info["hwnd"]

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        if not user32.IsWindow(hwnd):
            return {"success": False, "error": f"Invalid or closed HWND: {hwnd}."}

        # A numeric HWND carries no title: look the window up so the report says
        # what was photographed instead of "HWND 984082".
        if win_info is None:
            win_info = cls.describe_window(hwnd)

        with _attached_user_desktop():
            left, top, width, height = _visible_rect(hwnd)

            if width <= 0 or height <= 0:
                return {"success": False, "error": f"Target window has invalid geometry: {width}x{height}."}

            # Setup destination directory
            if not output_path:
                out_dir = Path(".argent") / "artifacts" / "desktop"
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time() * 1000)
                output_path = str(out_dir / f"window_{hwnd}_{timestamp}.png")
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            hwnd_dc = user32.GetWindowDC(hwnd)
            if not hwnd_dc:
                return {"success": False, "error": f"Could not acquire Device Context for HWND {hwnd}."}

            mem_dc = None
            bitmap = None
            old_bmp = None

            try:
                mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
                if not mem_dc:
                    return {"success": False, "error": f"CreateCompatibleDC failed for HWND {hwnd}."}

                bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
                if not bitmap:
                    return {"success": False, "error": f"CreateCompatibleBitmap failed ({width}x{height}) for HWND {hwnd}."}

                old_bmp = gdi32.SelectObject(mem_dc, bitmap)

                # Use PrintWindow to capture even if covered or partially off-screen
                method = "printwindow"
                printed = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
                if not printed:
                    method = "bitblt"
                    # Fallback to BitBlt
                    gdi32.BitBlt(mem_dc, 0, 0, width, height, hwnd_dc, 0, 0, 0x00CC0020)  # SRCCOPY

                from PIL import Image

                bmi = bytearray(40)  # BITMAPINFOHEADER
                # Use signed 'l' for biHeight to correctly pack negative height in Python 3.14
                struct.pack_into("<LllHHLLLLLL", bmi, 0, 40, width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
                buffer = bytearray(width * height * 4)

                lines = gdi32.GetDIBits(mem_dc, bitmap, 0, height, (ctypes.c_char * len(buffer)).from_buffer(buffer), bytes(bmi), 0)
                if lines == 0:
                    return {"success": False, "error": "GetDIBits failed to read pixel data."}

                if _frame_is_blank(buffer):
                    # Windows drawn by DirectComposition (Windows Terminal is one) can
                    # answer PrintWindow with an empty frame. What is really on screen
                    # is then the only truth left.
                    if _blit_screen_rect(left, top, width, height, mem_dc):
                        method = "bitblt-fallback"
                        gdi32.GetDIBits(mem_dc, bitmap, 0, height, (ctypes.c_char * len(buffer)).from_buffer(buffer), bytes(bmi), 0)

                img = Image.frombuffer("RGBA", (width, height), bytes(buffer), "raw", "BGRA", 0, 1)
                img = img.convert("RGB")
                crop = _parse_region(region, width, height)
                if crop:
                    cx, cy, cw, ch = crop
                    img = img.crop((cx, cy, cx + cw, cy + ch))
                img.save(output_path, "PNG")

                from argent.vision import encode_image
                b64, media_type, _ = encode_image(output_path)

                return {
                    "success": True,
                    "path": output_path,
                    "width": img.width,
                    "height": img.height,
                    "frame": {"width": width, "height": height},
                    "region": list(crop) if crop else None,
                    "title": win_info["title"] if win_info else f"HWND {hwnd}",
                    "pid": win_info.get("pid") if win_info else None,
                    "match": {"by": chosen_how, "reason": chosen_reason,
                              "others": other_titles},
                    "rect": {"x": left, "y": top, "width": width, "height": height},
                    "screen": _virtual_screen(),
                    "foreground": bool(user32.GetForegroundWindow() == hwnd),
                    "offscreen_px": _offscreen_pixels(left, top, width, height),
                    "method": method,
                    "base64": b64,
                    "media_type": media_type,
                }

            except Exception as e:
                log.error("Failed capturing window: %s", e)
                return {"success": False, "error": f"Window capture exception: {e}"}
            finally:
                if mem_dc and old_bmp:
                    gdi32.SelectObject(mem_dc, old_bmp)
                if bitmap:
                    gdi32.DeleteObject(bitmap)
                if mem_dc:
                    gdi32.DeleteDC(mem_dc)
                if hwnd_dc:
                    user32.ReleaseDC(hwnd, hwnd_dc)

    @classmethod
    def capture_screen(
        cls,
        output_path: Optional[str] = None,
        monitor: str = "primary",
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Capture the entire screen (primary monitor or all virtual monitors).

        Args:
            output_path: Optional custom path to save PNG.
            monitor: "primary" (default) or "all" (entire virtual multi-monitor desktop).
        """
        if not cls.is_available():
            return {"success": False, "error": "Desktop capture is only available on Windows."}

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        with _attached_user_desktop():
            if str(monitor).lower() == "all":
                x = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
                y = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
                width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
                height = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
            else:
                x = 0
                y = 0
                width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
                height = user32.GetSystemMetrics(1)  # SM_CYSCREEN

            if width <= 0 or height <= 0:
                return {"success": False, "error": f"Invalid screen geometry: {width}x{height}."}

            if not output_path:
                out_dir = Path(".argent") / "artifacts" / "desktop"
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time() * 1000)
                output_path = str(out_dir / f"screen_{timestamp}.png")
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            hdc_screen = user32.GetDC(0)
            if not hdc_screen:
                return {"success": False, "error": "Could not acquire DC for Desktop Screen."}

            mem_dc = None
            bitmap = None
            old_bmp = None

            try:
                mem_dc = gdi32.CreateCompatibleDC(hdc_screen)
                if not mem_dc:
                    return {"success": False, "error": "CreateCompatibleDC failed for Screen DC."}

                bitmap = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
                if not bitmap:
                    return {"success": False, "error": f"CreateCompatibleBitmap failed ({width}x{height}) for screen."}

                old_bmp = gdi32.SelectObject(mem_dc, bitmap)
                gdi32.BitBlt(mem_dc, 0, 0, width, height, hdc_screen, x, y, 0x00CC0020)  # SRCCOPY

                from PIL import Image

                bmi = bytearray(40)  # BITMAPINFOHEADER
                struct.pack_into("<LllHHLLLLLL", bmi, 0, 40, width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
                buffer = bytearray(width * height * 4)

                lines = gdi32.GetDIBits(mem_dc, bitmap, 0, height, (ctypes.c_char * len(buffer)).from_buffer(buffer), bytes(bmi), 0)
                if lines == 0:
                    return {"success": False, "error": "GetDIBits failed to read screen pixels."}

                img = Image.frombuffer("RGBA", (width, height), bytes(buffer), "raw", "BGRA", 0, 1)
                img = img.convert("RGB")
                crop = _parse_region(region, width, height)
                if crop:
                    cx, cy, cw, ch = crop
                    img = img.crop((cx, cy, cx + cw, cy + ch))
                img.save(output_path, "PNG")

                from argent.vision import encode_image
                b64, media_type, _ = encode_image(output_path)

                return {
                    "success": True,
                    "path": output_path,
                    "width": img.width,
                    "height": img.height,
                    "frame": {"width": width, "height": height},
                    "region": list(crop) if crop else None,
                    "title": "Full Desktop Screen" if monitor != "all" else "All Monitors Desktop",
                    "rect": {"x": x, "y": y, "width": width, "height": height},
                    "screen": _virtual_screen(),
                    "method": "bitblt-screen",
                    "base64": b64,
                    "media_type": media_type,
                }
            except Exception as e:
                log.error("Failed capturing full screen: %s", e)
                return {"success": False, "error": f"Full screen capture exception: {e}"}
            finally:
                if mem_dc and old_bmp:
                    gdi32.SelectObject(mem_dc, old_bmp)
                if bitmap:
                    gdi32.DeleteObject(bitmap)
                if mem_dc:
                    gdi32.DeleteDC(mem_dc)
                if hdc_screen:
                    user32.ReleaseDC(0, hdc_screen)

    @classmethod
    def get_clipboard_image(cls, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve an image from the Windows clipboard (e.g. copied via Win+Shift+S / PrintScreen).

        Returns:
            Dictionary with success status, saved file path, and vision metadata.
        """
        try:
            from PIL import ImageGrab, Image

            clip_data = ImageGrab.grabclipboard()
            if clip_data is None:
                return {
                    "success": False,
                    "error": (
                        "Clipboard does not contain an image. "
                        "Take a screenshot using Win+Shift+S or PrintScreen, then try again."
                    ),
                }

            img = None
            if isinstance(clip_data, Image.Image):
                img = clip_data
            elif isinstance(clip_data, list) and clip_data:
                first_file = Path(clip_data[0])
                if first_file.is_file():
                    from argent.vision import SUPPORTED_SUFFIXES
                    if first_file.suffix.lower() in SUPPORTED_SUFFIXES:
                        img = Image.open(first_file)

            if img is None:
                return {
                    "success": False,
                    "error": f"Clipboard content ({type(clip_data).__name__}) could not be decoded as an image.",
                }

            if not output_path:
                out_dir = Path(".argent") / "artifacts" / "desktop"
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time() * 1000)
                output_path = str(out_dir / f"clipboard_{timestamp}.png")
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            img = img.convert("RGB")
            img.save(output_path, "PNG")

            from argent.vision import encode_image
            b64, media_type, _ = encode_image(output_path)

            return {
                "success": True,
                "path": output_path,
                "width": img.width,
                "height": img.height,
                "title": "Clipboard Screenshot",
                "base64": b64,
                "media_type": media_type,
            }
        except Exception as e:
            log.error("Failed retrieving image from clipboard: %s", e)
            return {"success": False, "error": f"Clipboard image retrieval failed: {e}"}
