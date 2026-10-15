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

from logger import get_logger

log = get_logger("desktop_automation")

PW_RENDERFULLCONTENT = 2
DESKTOP_ALL = 0x01FF


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

            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            width = rect.right - rect.left
            height = rect.bottom - rect.top

            # Filter out zero-size or off-screen helper windows
            if width <= 0 or height <= 0:
                return 1

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

            windows.append({
                "hwnd": int(hwnd),
                "title": title,
                "pid": int(pid.value),
                "x": rect.left,
                "y": rect.top,
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
        """Find the first window whose title matches title_substring (case-insensitive)."""
        sub = title_substring.strip().lower()
        for win in cls.list_windows():
            if sub in win["title"].lower():
                return win
        return None

    @classmethod
    def capture_window(
        cls,
        target: Union[str, int],
        output_path: Optional[str] = None,
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

        if hwnd is None:
            win_info = cls.find_window(str(target))
            if not win_info:
                return {"success": False, "error": f"Window matching '{target}' not found among visible windows."}
            hwnd = win_info["hwnd"]

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        if not user32.IsWindow(hwnd):
            return {"success": False, "error": f"Invalid or closed HWND: {hwnd}."}

        with _attached_user_desktop():
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            width = rect.right - rect.left
            height = rect.bottom - rect.top

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
                printed = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
                if not printed:
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

                img = Image.frombuffer("RGBA", (width, height), bytes(buffer), "raw", "BGRA", 0, 1)
                img = img.convert("RGB")
                img.save(output_path, "PNG")

                from vision import encode_image
                b64, media_type, _ = encode_image(output_path)

                return {
                    "success": True,
                    "path": output_path,
                    "width": width,
                    "height": height,
                    "title": win_info["title"] if win_info else f"HWND {hwnd}",
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
                img.save(output_path, "PNG")

                from vision import encode_image
                b64, media_type, _ = encode_image(output_path)

                return {
                    "success": True,
                    "path": output_path,
                    "width": width,
                    "height": height,
                    "title": "Full Desktop Screen" if monitor != "all" else "All Monitors Desktop",
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
                    from vision import SUPPORTED_SUFFIXES
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

            from vision import encode_image
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
