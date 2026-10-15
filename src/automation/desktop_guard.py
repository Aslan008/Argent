"""Who may touch what on the person's desktop, how fast, and who has been asked — before `uia.py` moves anything.

The rules, in the order they are applied to every action:

  1. switched on         `/desktop on` (off at start of every session: a mouse that moves by itself is an event
                         the person must have chosen)
  2. emergency stop      Ctrl+Alt+Shift+F12 (or the chord set in config) held anywhere ends the series at once and
                         keeps control off until `/desktop on`
  3. a place to act      only a program on the allow-list (`/desktop allow notepad.exe`; empty at the start)
  4. never in            password managers, the Windows security and UAC dialogs, a window named like a password
                         or a payment step, a field that is a password field
  5. fresh sight         an element id or a point must come from the observation of the SAME window made after
                         the last action; no blind series of clicks
  6. the person's yes    through the same approval gate as every command; a control called Delete / Удалить /
                         Pay / Send ... is "destructive": asked every time, refused in auto mode
  7. pace                at most N actions in a series and a pause between them
  8. a record            every decision, allowed or refused, goes to `.argent/run/desktop_audit.jsonl`

Pure decisions (no Windows calls) so they can be tested anywhere; the tools pass in what they observed.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from logger import get_logger

log = get_logger("desktop_guard")

DEFAULT_STOP_CHORD = "ctrl+alt+shift+f12"
MAX_SERIES = 25                  # actions without a new instruction from the person
MIN_INTERVAL = 0.25              # seconds between two actions
OBSERVATION_MAX_AGE = 90.0       # an observation older than this is not "fresh"

# Programs that hold what must never pass through a model.
_FORBIDDEN_PROCESSES = {
    "consent.exe", "logonui.exe", "credwiz.exe", "winlogon.exe", "securityhealthsystray.exe", "lockapp.exe",
    "1password.exe", "bitwarden.exe", "keepass.exe", "keepassxc.exe", "lastpass.exe", "dashlane.exe",
    "enpass.exe", "nordpass.exe", "roboform.exe", "keeper.exe", "mmc.exe", "regedit.exe", "taskmgr.exe",
}
_FORBIDDEN_TITLE = re.compile(
    r"(password|passphrase|credential|пароль|парол|учётных записей|учетных записей|user account control|"
    r"windows security|безопасность windows|ввод данных карты|card number|номер карты|cvv|cvc)", re.IGNORECASE)
# A control whose name says it cannot be taken back or moves money / messages out.
_DESTRUCTIVE_NAME = re.compile(
    r"(delete|remove|erase|format|wipe|uninstall|discard|reset|clear all|empty|purge|send|submit|publish|post|"
    r"pay|buy|purchase|order|confirm|transfer|sign out|log ?out|shut ?down|restart|"
    r"удал|стереть|очист|форматир|сброс|отправ|опубликов|оплат|купить|заказ|подтверд|перевод|выйти|выключ|"
    r"перезагруз)", re.IGNORECASE)

ACTION_PREFIX = "действие на рабочем столе: "


@dataclass
class Decision:
    ok: bool
    reason: str = ""
    destructive: bool = False

    def __bool__(self) -> bool:
        return self.ok


# --- state of the series ---------------------------------------------------------------------------------------

class Series:
    """What has happened since the person last gave an instruction."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self.count = 0
        self.last_at = 0.0
        self.stopped = False                  # the emergency stop was used: stays until `/desktop on`
        self.stop_reason = ""

    def reset(self) -> None:
        with self._lock:
            self.count = 0
            self.last_at = 0.0
            self.stopped = False
            self.stop_reason = ""

    def stop(self, reason: str = "аварийная остановка") -> None:
        with self._lock:
            self.stopped = True
            self.stop_reason = reason

    def begin_turn(self) -> None:
        """A new instruction from the person: the series starts again (the stop stays)."""
        with self._lock:
            self.count = 0

    def note_action(self) -> None:
        with self._lock:
            self.count += 1
            self.last_at = self._clock()

    def wait_turn(self) -> float:
        """The pause that keeps actions human-paced. Returns what was waited."""
        with self._lock:
            gap = self._clock() - self.last_at
        wait = max(0.0, MIN_INTERVAL - gap) if self.last_at else 0.0
        if wait:
            time.sleep(wait)
        return wait


SERIES = Series()


# --- settings --------------------------------------------------------------------------------------------------

_session_enabled = False


def enabled() -> bool:
    """On only for this run of Argent, and only after the person said so."""
    return _session_enabled


def set_enabled(value: bool) -> None:
    global _session_enabled
    _session_enabled = bool(value)
    if value:
        SERIES.reset()
        from src.automation import desktop_stop
        desktop_stop.start(SERIES)
    else:
        try:
            from src.automation import desktop_stop
            desktop_stop.stop()
        except Exception:
            pass


def allowed_apps() -> List[str]:
    try:
        import config
        return [a.lower() for a in config.get_desktop_allowed_apps()]
    except Exception:
        return []


def stop_chord() -> str:
    try:
        import config
        return config.get_desktop_stop_chord() or DEFAULT_STOP_CHORD
    except Exception:
        return DEFAULT_STOP_CHORD


# --- decisions -------------------------------------------------------------------------------------------------

def normalize_app(name: str) -> str:
    name = str(name or "").strip().lower()
    if name and not name.endswith(".exe") and "." not in name:
        name += ".exe"
    return name


def is_destructive(element_name: str) -> bool:
    return bool(_DESTRUCTIVE_NAME.search(element_name or ""))


def check_window(process: str, title: str, allowed: Optional[List[str]] = None) -> Decision:
    """May anything be done in this window at all?"""
    proc = (process or "").lower()
    if proc in _FORBIDDEN_PROCESSES:
        return Decision(False, f"'{process}' holds secrets or system security: Argent does not act there")
    if _FORBIDDEN_TITLE.search(title or ""):
        return Decision(False, f"the window '{title}' looks like a password, payment or security step: "
                               f"do it yourself")
    apps = allowed_apps() if allowed is None else [a.lower() for a in allowed]
    if proc not in apps:
        hint = f"/desktop allow {proc}" if proc else "/desktop allow <программа.exe>"
        return Decision(False, f"'{process or title}' is not on the list of programs Argent may act in "
                               f"(the person adds one with: {hint})")
    return Decision(True)


def check_action(*, action: str, process: str, title: str, element_name: str = "", is_password: bool = False,
                 typing: bool = False, observed_number: Optional[int] = None, current_number: Optional[int] = None,
                 observation_age: Optional[float] = None, foreground_ok: bool = True,
                 allowed: Optional[List[str]] = None, series: Optional[Series] = None) -> Decision:
    """The whole chain of rules for one action. The first that says no is the answer."""
    series = series or SERIES
    if not enabled():
        return Decision(False, "desktop control is off: the person switches it on with /desktop on")
    if series.stopped:
        return Decision(False, f"stopped ({series.stop_reason}); the person re-enables with /desktop on")
    if not foreground_ok:
        return Decision(False, "the desktop is locked or a secure screen (UAC, sign-in) is in front: nothing is done")
    where = check_window(process, title, allowed)
    if not where:
        return where
    if is_password and (typing or action in ("type", "set_value")):
        return Decision(False, "this is a password field: Argent never types into one")
    if series.count >= MAX_SERIES:
        return Decision(False, f"{MAX_SERIES} actions without a word from the person: ask them to continue "
                               f"(a new message resets the count)")
    if action not in ("focus",):
        if current_number is None or observed_number is None:
            return Decision(False, "observe the window first (desktop_inspect), then act on what was seen")
        if observed_number != current_number:
            return Decision(False, "that element comes from an older observation: observe the window again")
        if observation_age is not None and observation_age > OBSERVATION_MAX_AGE:
            return Decision(False, "the observation is old: observe the window again")
    return Decision(True, destructive=is_destructive(element_name))


# --- the record ------------------------------------------------------------------------------------------------

def audit_path() -> Path:
    from project_paths import project_root_or_cwd
    return Path(project_root_or_cwd()) / ".argent" / "run" / "desktop_audit.jsonl"


_AUDIT_LOCK = threading.Lock()
_AUDIT_MAX_BYTES = 1_000_000


def audit(event: str, **fields) -> None:
    """One line per decision. Typed text is recorded by length and a redacted head, never whole."""
    try:
        from src.agent.secret_redactor import redact
        text = fields.pop("text", None)
        if text is not None:
            fields["text_len"] = len(text)
            fields["text_head"] = redact(text)[:24]
        record = {"at": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
        path = audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK:
            if path.exists() and path.stat().st_size > _AUDIT_MAX_BYTES:
                path.replace(path.with_suffix(".jsonl.old"))
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:                       # a journal that cannot be written must not block, but is said
        log.warning("desktop audit not written: %s", e)


def read_audit(limit: int = 20) -> List[dict]:
    try:
        lines = audit_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out
