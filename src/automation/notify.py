"""Telling you something happened, without you having to go look.

A background agent you have to remember to check is not a background agent —
it's a log file. But the opposite failure is worse: a toast after every tick
trains you to dismiss them without reading, and then the one that mattered goes
by unread too. So the default rule here is **speak only when there is something
to say**: new items found, an error, or an action that needed a human. Each
automation can ask for more (`always`, `changes`) or less (`failures`, `never`).

Delivery is best-effort by design. A desktop toast can fail for a dozen
environment-specific reasons, and none of them should turn a successful
automation into a failed one — the run log stays the source of truth, and this
is a nudge toward it.
"""

import html
import json
import os
import subprocess
import sys
import urllib.request

from logger import get_logger

log = get_logger("automation")

MAX_TITLE = 60
MAX_BODY = 220
_TOAST_TIMEOUT = 15
_TELEGRAM_TIMEOUT = 10
TELEGRAM_LIMIT = 3500


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


def telegram_notify(title: str, message: str) -> bool:
    """Send the notification to the owner's Telegram chat.

    Needs only the bot token and chat id from the config — not the polling
    bridge — so a job running in a child process, or with Argent closed, can
    still reach a phone. The token never reaches a log line.
    """
    try:
        from config import get_telegram_chat_id, get_telegram_token
        token, chat_id = get_telegram_token(), get_telegram_chat_id()
        if not token or not chat_id:
            return False
        # The title in bold, the body as it is: both are escaped, so a "<" in a summary cannot
        # make Telegram refuse the whole message.
        # Cut the raw text, not the escaped one: a cut through "&lt;" is a parse error.
        title, message = (title or "")[:200], (message or "")[:TELEGRAM_LIMIT // 2]
        title_html, body_html = html.escape(title, quote=False), html.escape(message, quote=False)
        text = f"<b>{title_html}</b>\n\n{body_html}".strip()
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                             "link_preview_options": {"is_disabled": True}}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=_TELEGRAM_TIMEOUT) as response:
            return 200 <= response.status < 300
    except Exception as e:
        log.info("telegram notification failed: %s", type(e).__name__)
        return False


def should_alert(policy: str, status: str, denied: int = 0, new_items: int = 0,
                 silent: bool = False, changed: bool = False) -> bool:
    """Does this finished run deserve your attention under ``policy``?

        auto      a failure, an action that needed you, or something new
        failures  a failure or an action that needed you
        changes   all of the above, or a report that differs from the last one
        always    every run (a daily digest)
        never     nothing — the run log still has it
    """
    if policy == "never":
        return False
    if status == "error" or status == "denied" or denied:
        return True
    if policy == "failures":
        return False
    if policy == "always":
        return True
    if new_items:
        return True
    if policy == "changes":
        return bool(changed) and not silent
    return False


def notification_for(name: str, result: dict):
    """(title, body) worth interrupting you for, or None to stay quiet.

    Ordered by how much it wants your attention: a broken run first, then one
    that hit a wall it cannot pass without you, then actual news.
    """
    result = result or {}
    status = result.get("status", "ok")
    new_items = int(result.get("new_items") or 0)
    denied = result.get("denied_actions") or []
    summary = result.get("summary") or ""

    if "alert" in result:
        wanted = bool(result["alert"])
    else:
        wanted = should_alert(result.get("notify_policy", "auto"), status, len(denied), new_items,
                              bool(result.get("silent")), bool(result.get("changed")))
    if not wanted:
        return None

    if status == "error":
        return f"{name}: сбой", summary or "Прогон завершился ошибкой — /tasks runs"
    if denied:
        actions = ", ".join(d.get("action", "?") for d in denied[:3])
        return f"{name}: нужен ты", f"Отклонено без человека: {actions}"
    if result.get("mode") == "remind":
        return f"{name}", summary or "Напоминание"
    if new_items > 0:
        word = "новое" if new_items == 1 else "новых"
        return f"{name}: {new_items} {word}", summary or "Подробности — /tasks runs"
    if result.get("silent"):
        return f"{name}", summary or "Без изменений"
    return f"{name}", summary or "Прогон завершён — /tasks runs"


def notify_run(name: str, result: dict, channels=()) -> bool:
    """Announce a finished run if it earned it. Returns True if anything was
    sent."""
    payload = notification_for(name, result)
    if payload is None:
        return False
    sent = desktop_notify(*payload)
    if "telegram" in (channels or ()):
        sent = telegram_notify(*payload) or sent
    return sent
