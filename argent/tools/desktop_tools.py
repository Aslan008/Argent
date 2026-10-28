"""Tools for native desktop window inspection, screen capture, and clipboard in Argent."""

from typing import Optional
from argent.logger import get_logger

log = get_logger("tools")


def _maybe_auto_attach(res: dict, default_label: str) -> str:
    """Attach captured image to vision queue if the current model supports vision."""
    from argent.config import get_current_model, get_provider
    from argent.vision import model_supports_vision, queue_image

    b64 = res.get("base64")
    media_type = res.get("media_type")
    path = res.get("path")
    label = res.get("title") or default_label

    supports = model_supports_vision(get_current_model(), get_provider())
    if supports is False:
        return (
            f"\n\n[VISION NOTICE]: Captured image saved to disk ({path or label}). "
            f"Image was NOT attached to model context because the current model "
            f"('{get_current_model()}') does not support image/multimodal input."
        )

    if supports is not False and b64 and media_type:
        queue_image(b64, media_type, str(path or label))
        return (
            "\n\n[VISION ATTACHED]: The image has been attached to the conversation. "
            "You will SEE it in your very next message — continue and describe what is visible."
        )
    return ""


def list_desktop_windows() -> str:
    """List all currently visible desktop windows with their HWND, PID, and dimensions."""
    try:
        from src.automation.desktop import DesktopAutomation
        if not DesktopAutomation.is_available():
            return "Desktop window inspection is only available on Windows."

        windows = DesktopAutomation.list_windows()
        if not windows:
            return "No visible top-level windows found."

        lines = [f"Found {len(windows)} visible desktop window(s):\n"]
        for w in windows:
            lines.append(
                f"- [HWND: {w['hwnd']}] [PID: {w['pid']}] \"{w['title']}\" "
                f"({w['width']}x{w['height']} at {w['x']},{w['y']})"
            )
        return "\n".join(lines)
    except Exception as e:
        log.error("list_desktop_windows error: %s", e)
        return f"Error listing desktop windows: {e}"


def _capture_report(res: dict, kind: str) -> str:
    """What was photographed, in the terms needed to trust the frame.

    A capture that reports only "1168x195" cannot be checked: the reader does not
    know which window it was, whether the window was on screen, or how the picture
    was obtained. Every one of those has already misled a reader once.
    """
    rect = res.get("rect") or {}
    frame = res.get("frame") or {}
    region = res.get("region")
    head = f"Captured {kind} '{res.get('title')}'"
    if res.get("pid"):
        head += f" [pid {res['pid']}]"
    lines = [head]
    if rect:
        lines.append(
            f"  window: {rect.get('width')}x{rect.get('height')} at {rect.get('x')},{rect.get('y')}"
            + (" (foreground)" if res.get("foreground") else " (in the background)")
        )
    shot = f"  image: {res.get('width')}x{res.get('height')}"
    if region and frame:
        rx, ry, rw, rh = region
        shot += f" — region {rw}x{rh} at {rx},{ry} of the {frame.get('width')}x{frame.get('height')} frame"
    elif frame:
        shot += " — the whole frame"
    shot += f", method: {res.get('method')}"
    lines.append(shot)
    off = res.get("offscreen_px") or 0
    if off:
        lines.append(
            f"  ! {off} px of the window lie outside the desktop — that part of the frame may be incomplete"
        )
    if res.get("method") == "bitblt-fallback":
        lines.append(
            "  ! PrintWindow returned an empty frame, so the picture was taken from the screen: "
            "anything covering the window is in it"
        )
    lines.append(f"Saved screenshot to: {res.get('path')}")
    return "\n".join(lines)


def capture_desktop_window(target: str, output_path: Optional[str] = None, region: Optional[str] = None) -> str:
    """Capture a screenshot of a desktop window by its title substring or numeric HWND.
    
    Args:
        target: Window title substring (e.g. 'Unity', 'Notepad') or numeric HWND.
        output_path: Optional custom path to save PNG image.
        region: Optional strip of the window: 'bottom:200', 'top:120', 'left:300',
            'right:400' or 'x,y,w,h' in pixels. Use it when only one part matters
            (the panel near a terminal's prompt, a dialog's buttons).
    """
    try:
        from src.automation.desktop import DesktopAutomation
        if not DesktopAutomation.is_available():
            return "Desktop window capture is only available on Windows."

        res = DesktopAutomation.capture_window(target, output_path, region=region)
        if not res.get("success"):
            return f"Error capturing window: {res.get('error', 'Unknown error')}"

        attach_note = _maybe_auto_attach(res, f"Window '{res.get('title')}'")
        return _capture_report(res, "window") + attach_note
    except Exception as e:
        log.error("capture_desktop_window error: %s", e)
        return f"Error capturing desktop window: {e}"


def capture_desktop_screen(monitor: str = "primary", output_path: Optional[str] = None,
                           region: Optional[str] = None) -> str:
    """Capture a screenshot of the entire desktop screen (primary monitor or all monitors).

    Args:
        monitor: 'primary' (default) for main display, or 'all' for the full multi-monitor virtual desktop.
        output_path: Optional custom path to save PNG image.
        region: Optional strip of the screen: 'bottom:400', 'top:300', 'left:500', 'right:500' or 'x,y,w,h'.
    """
    try:
        from src.automation.desktop import DesktopAutomation
        if not DesktopAutomation.is_available():
            return "Desktop screen capture is only available on Windows."

        res = DesktopAutomation.capture_screen(output_path=output_path, monitor=monitor, region=region)
        if not res.get("success"):
            return f"Error capturing screen: {res.get('error', 'Unknown error')}"

        attach_note = _maybe_auto_attach(res, res.get("title"))
        return _capture_report(res, "screen") + attach_note
    except Exception as e:
        log.error("capture_desktop_screen error: %s", e)
        return f"Error capturing desktop screen: {e}"


def get_clipboard_image(output_path: Optional[str] = None) -> str:
    """Retrieve and inspect an image from the system clipboard (e.g. copied via Win+Shift+S or PrintScreen).

    Args:
        output_path: Optional custom path to save PNG image.
    """
    try:
        from src.automation.desktop import DesktopAutomation
        res = DesktopAutomation.get_clipboard_image(output_path=output_path)
        if not res.get("success"):
            return f"Clipboard: {res.get('error', 'No image found.')}"

        path = res.get("path")
        w = res.get("width")
        h = res.get("height")
        attach_note = _maybe_auto_attach(res, "Clipboard Image")
        return f"Successfully retrieved clipboard image ({w}x{h}).\nSaved to: {path}{attach_note}"
    except Exception as e:
        log.error("get_clipboard_image error: %s", e)
        return f"Error retrieving clipboard image: {e}"
