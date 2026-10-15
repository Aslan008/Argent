"""Tools for native desktop window inspection, screen capture, and clipboard in Argent."""

from typing import Optional
from logger import get_logger

log = get_logger("tools")


def _maybe_auto_attach(res: dict, default_label: str) -> str:
    """Attach captured image to vision queue if the current model supports vision."""
    from config import get_current_model, get_provider
    from vision import model_supports_vision, queue_image

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


def capture_desktop_window(target: str, output_path: Optional[str] = None) -> str:
    """Capture a screenshot of a desktop window by its title substring or numeric HWND.
    
    Args:
        target: Window title substring (e.g. 'Unity', 'Notepad') or numeric HWND.
        output_path: Optional custom path to save PNG image.
    """
    try:
        from src.automation.desktop import DesktopAutomation
        if not DesktopAutomation.is_available():
            return "Desktop window capture is only available on Windows."

        res = DesktopAutomation.capture_window(target, output_path)
        if not res.get("success"):
            return f"Error capturing window: {res.get('error', 'Unknown error')}"

        path = res.get("path")
        title = res.get("title")
        w = res.get("width")
        h = res.get("height")
        attach_note = _maybe_auto_attach(res, f"Window '{title}'")
        return f"Successfully captured window '{title}' ({w}x{h}).\nSaved screenshot to: {path}{attach_note}"
    except Exception as e:
        log.error("capture_desktop_window error: %s", e)
        return f"Error capturing desktop window: {e}"


def capture_desktop_screen(monitor: str = "primary", output_path: Optional[str] = None) -> str:
    """Capture a screenshot of the entire desktop screen (primary monitor or all monitors).

    Args:
        monitor: 'primary' (default) for main display, or 'all' for the full multi-monitor virtual desktop.
        output_path: Optional custom path to save PNG image.
    """
    try:
        from src.automation.desktop import DesktopAutomation
        if not DesktopAutomation.is_available():
            return "Desktop screen capture is only available on Windows."

        res = DesktopAutomation.capture_screen(output_path=output_path, monitor=monitor)
        if not res.get("success"):
            return f"Error capturing screen: {res.get('error', 'Unknown error')}"

        path = res.get("path")
        title = res.get("title")
        w = res.get("width")
        h = res.get("height")
        attach_note = _maybe_auto_attach(res, title)
        return f"Successfully captured {title} ({w}x{h}).\nSaved screenshot to: {path}{attach_note}"
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
