"""Telling you something happened, without you having to go look.

A background agent you have to remember to check is not a background agent —
it's a log file. But the opposite failure is worse: a toast after every tick
trains you to dismiss them without reading, and then the one that mattered goes
by unread too. So the rule here is **speak only when there is something to
say**: new items found, an error, or an action that needed a human.

Delivery is best-effort by design. A desktop toast can fail for a dozen
environment-specific reasons, and none of them should turn a successful
automation into a failed one — the run log stays the source of truth, and this
is a nudge toward it.
"""

import json
import os
import subprocess
import sys

from logger import get_logger

log = get_logger("automation")

MAX_TITLE = 60
MAX_BODY = 220
_TOAST_TIMEOUT = 15


def _clean(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text[:limit - 1] + "…" if len(text) > limit else text


def _windows_toast(title: str, message: str) -> bool:
    """WinRT toast through PowerShell.

    Borrowing PowerShell's registered AppID avoids shipping a dependency or
    registering our own Start Menu entry just to show a line of text.
    """
    script = (
        '[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications,'
        ' ContentType=WindowsRuntime] > $null;'
        '$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent('
        '[Windows.UI.Notifications.ToastTemplateType]::ToastText02);'
        '$n = $t.GetElementsByTagName("text");'
        '$n.Item(0).AppendChild($t.CreateTextNode($env:ARGENT_TOAST_TITLE)) > $null;'
        '$n.Item(1).AppendChild($t.CreateTextNode($env:ARGENT_TOAST_BODY)) > $null;'
        '$toast = [Windows.UI.Notifications.ToastNotification]::new($t);'
        '[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('
        '"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe"'
        ').Show($toast);'
    )
    # The text goes through the environment, not the script body: a summary can
    # contain quotes, $ or backticks, and interpolating it would be both broken
    # and an injection into a shell.
    env = dict(os.environ, ARGENT_TOAST_TITLE=title, ARGENT_TOAST_BODY=message)
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, timeout=_TOAST_TIMEOUT, env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return result.returncode == 0


def _macos_toast(title: str, message: str) -> bool:
    # json.dumps gives AppleScript-compatible quoting for the two strings that
    # come from a model's output, so a quote in a summary cannot break the line.
    script = (f"display notification {json.dumps(message)} "
              f"with title {json.dumps(title)}")
    result = subprocess.run(["osascript", "-e", script],
                            capture_output=True, timeout=_TOAST_TIMEOUT)
    return result.returncode == 0


def _linux_toast(title: str, message: str) -> bool:
    result = subprocess.run(["notify-send", title, message],
                            capture_output=True, timeout=_TOAST_TIMEOUT)
    return result.returncode == 0


def desktop_notify(title: str, message: str) -> bool:
    """Show a desktop notification. False means it did not get through."""
    from config import get_desktop_notifications
    if not get_desktop_notifications():
        return False
    title, message = _clean(title, MAX_TITLE), _clean(message, MAX_BODY)
    try:
        if sys.platform == "win32":
            return _windows_toast(title, message)
        if sys.platform == "darwin":
            return _macos_toast(title, message)
        return _linux_toast(title, message)
    except Exception as e:
        log.info("desktop notification unavailable: %s", e)
        return False


def notification_for(name: str, result: dict):
    """(title, body) worth interrupting you for, or None to stay quiet.

    Ordered by how much it wants your attention: a broken run first, then one
    that hit a wall it cannot pass without you, then actual news.
    """
    status = (result or {}).get("status", "ok")
    new_items = int((result or {}).get("new_items") or 0)
    denied = (result or {}).get("denied_actions") or []
    summary = (result or {}).get("summary") or ""

    if status == "error":
        return f"⚠ {name}: сбой", summary or "Прогон завершился ошибкой — /tasks runs"
    if denied:
        actions = ", ".join(d.get("action", "?") for d in denied[:3])
        return f"🔒 {name}: нужен ты", f"Отклонено без человека: {actions}"
    if new_items > 0:
        word = "новое" if new_items == 1 else "новых"
        return f"🔔 {name}: {new_items} {word}", summary or "Подробности — /tasks runs"
    return None


def notify_run(name: str, result: dict) -> bool:
    """Announce a finished run if it earned it. Returns True if anything was
    sent."""
    payload = notification_for(name, result)
    if payload is None:
        return False
    return desktop_notify(*payload)
