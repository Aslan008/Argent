"""When an automation is due.

Deliberately not cron. Cron syntax is powerful and unreadable, and the failure
mode of a misread cron line in an UNATTENDED task is a job that quietly runs at
the wrong time (or 60x too often) while nobody is watching. These two forms
cover the cases that actually come up and can be read back correctly at a
glance:

    "every 15m" / "every 2h" / "every 1d"   — repeating interval
    "daily at 09:00"                        — once a day at a wall-clock time

Everything is evaluated against a caller-supplied ``now``, so the tests drive
time directly instead of sleeping.
"""

import re
from datetime import datetime, timedelta

_INTERVAL_RE = re.compile(r"^every\s+(\d+)\s*(m|min|mins|minute|minutes|h|hour|hours|d|day|days)$",
                          re.IGNORECASE)
_DAILY_RE = re.compile(r"^daily\s+at\s+(\d{1,2}):(\d{2})$", re.IGNORECASE)

_UNIT_SECONDS = {"m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
                 "h": 3600, "hour": 3600, "hours": 3600,
                 "d": 86400, "day": 86400, "days": 86400}

# A minute is the floor: anything tighter turns an unattended agent into a
# runaway token burner, and no useful monitoring needs sub-minute polling.
MIN_INTERVAL_SECONDS = 60

# How late a daily job may still run after its wall-clock time. Argent only runs
# while it is open, so a 09:00 job routinely finds nobody home at 09:00: without
# a catch-up window it would be skipped every single day the app started late.
# Bounded, because the opposite failure is just as bad — a "daily 09:00 report"
# firing at 23:40 because that is when the app happened to open.
DAILY_CATCHUP_SECONDS = 3600


class ScheduleError(ValueError):
    """Unparseable or unsafe schedule."""


def parse_schedule(text: str) -> dict:
    """Parse a schedule string into {"kind": "interval"|"daily", ...}."""
    if not text or not text.strip():
        raise ScheduleError("schedule is empty")
    raw = text.strip()

    m = _INTERVAL_RE.match(raw)
    if m:
        seconds = int(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()]
        if seconds < MIN_INTERVAL_SECONDS:
            raise ScheduleError(
                f"interval too small ({seconds}s); the minimum is {MIN_INTERVAL_SECONDS}s")
        return {"kind": "interval", "seconds": seconds}

    m = _DAILY_RE.match(raw)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ScheduleError(f"invalid time of day: {hour:02d}:{minute:02d}")
        return {"kind": "daily", "hour": hour, "minute": minute}

    raise ScheduleError(
        f"could not read schedule {text!r}. Use 'every 15m', 'every 2h' or 'daily at 09:00'.")


def next_run_after(schedule: str, last_run: datetime | None, now: datetime) -> datetime:
    """When this schedule should fire next, given the previous run."""
    spec = parse_schedule(schedule)

    if spec["kind"] == "interval":
        if last_run is None:
            return now                      # never ran: due immediately
        return last_run + timedelta(seconds=spec["seconds"])

    today = now.replace(hour=spec["hour"], minute=spec["minute"], second=0, microsecond=0)

    if last_run is not None and last_run >= today:
        return today + timedelta(days=1)    # already ran today

    if today > now:
        return today                        # still ahead of us today

    # The time has passed and it hasn't run today: catch up if we are only a
    # little late, otherwise wait for tomorrow rather than firing at a
    # surprising hour.
    late = (now - today).total_seconds()
    return today if late <= DAILY_CATCHUP_SECONDS else today + timedelta(days=1)


def is_due(schedule: str, last_run: datetime | None, now: datetime) -> bool:
    """Whether the automation should run at ``now``."""
    try:
        return next_run_after(schedule, last_run, now) <= now
    except ScheduleError:
        return False                        # a broken schedule never fires
