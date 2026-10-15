"""When an automation is due.

Deliberately not cron. Cron syntax is powerful and unreadable, and the failure
mode of a misread cron line in an UNATTENDED task is a job that quietly runs at
the wrong time (or 60x too often) while nobody is watching. The forms below
cover the cases that actually come up, can be written in English or Russian, and
read back correctly at a glance:

    every 15m | every 2h | every 1d        repeating interval
    every 30m between 09:00 and 18:00      ... only inside a daily window
    every 1h on weekdays                   ... only on some days
    daily at 09:00                         once a day
    weekdays at 09:00 | weekends at 11:00  work days / days off
    mon,wed,fri at 18:30 | every monday at 10:00
    monthly on 15 at 09:00                 a day of the month (31 = the last day)
    once at 2026-10-09 18:00               one time only
    in 2h | tomorrow at 08:30              one time, resolved when it is created
    on start                               every time Argent starts

    каждые 15 минут | каждый час | каждый день в 9:00 | по будням в 9:00
    по пн, ср в 18:30 | каждый понедельник в 10:00 | через 2 часа
    завтра в 8:30 | при запуске | раз в 30 минут с 9:00 до 18:00

Everything is evaluated against a caller-supplied ``now``, so the tests drive
time directly instead of sleeping. Times are local wall-clock time.

Relative one-shots ("in 2h", "tomorrow at 8") mean nothing without a "now", so
they are turned into an absolute ``once at ...`` by `normalize_schedule` when the
automation is created; what is stored is always the canonical form.
"""

import calendar
import re
from datetime import datetime, timedelta

# A minute is the floor: anything tighter turns an unattended agent into a
# runaway token burner, and no useful monitoring needs sub-minute polling.
MIN_INTERVAL_SECONDS = 60

# How late a daily job may still run after its wall-clock time. Argent only runs
# while it is open, so a 09:00 job routinely finds nobody home at 09:00: without
# a catch-up window it would be skipped every single day the app started late.
# Bounded, because the opposite failure is just as bad — a "daily 09:00 report"
# firing at 23:40 because that is when the app happened to open.
DAILY_CATCHUP_SECONDS = 3600

# A "run at start" job that already ran this recently is not run again: opening
# and closing Argent three times in ten minutes is one morning, not three.
STARTUP_COOLDOWN_SECONDS = 600

_FAR_FUTURE = datetime.max


class ScheduleError(ValueError):
    """Unparseable or unsafe schedule."""


# --- vocabulary --------------------------------------------------------------------

_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400}

_UNIT_WORD = (r"(?P<unit>минут\w*|мин\b\.?|час\w*|сут\w*|дн\w*|день|"
              r"minutes?|mins?|hours?|hrs?|days?|[mhdмчд])")


def _unit_seconds(word: str) -> int:
    word = word.lower().rstrip(".")
    if word[:1] in ("m", "м") and not word.startswith("мес"):
        return 60
    if word[:1] in ("h", "ч"):
        return 3600
    return 86400              # d, day(s), д, день, дня, дней, сутки


_WEEKDAY_NAMES = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")
_WEEKDAY_EN = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

_DAY_PATTERNS = [
    (0, re.compile(r"^(mon\w*|пн|пон\w*)$")),
    (1, re.compile(r"^(tue\w*|вт|вто\w*)$")),
    (2, re.compile(r"^(wed\w*|ср|сре\w*)$")),
    (3, re.compile(r"^(thu\w*|чт|чет\w*)$")),
    (4, re.compile(r"^(fri\w*|пт|пят\w*)$")),
    (5, re.compile(r"^(sat\w*|сб|суб\w*)$")),
    (6, re.compile(r"^(sun\w*|вс|вос\w*)$")),
]

_WORKDAYS = re.compile(r"^(weekdays?|workdays?|working|будн\w*|рабоч\w*)$")
_WEEKEND = re.compile(r"^(weekends?|выходн\w*)$")
_EVERYDAY = re.compile(r"^(daily|everyday|ежедневно|ежедневный)$")
_EVERY_DAY = re.compile(r"\b(?:every|each|кажд\w+)\s+(?:day|день|дн\w*)\b")
_FILLER = {"on", "at", "по", "в", "во", "each", "every", "кажд", "каждый", "каждую", "каждое",
           "каждые", "на", "и", "and", "the", "day", "days", "день", "дням", "дни", "weekly",
           "еженедельно", "of", "week", "недели", "неделю"}


def _day_index(token: str):
    for index, pattern in _DAY_PATTERNS:
        if pattern.match(token):
            return index
    return None


def parse_days(phrase: str):
    """A phrase like 'weekdays', 'по пн, ср', 'mon-fri' as a sorted list of
    weekday numbers (Monday = 0). ``None`` when something in it is not a day —
    a typo must fail loudly, not quietly widen the schedule."""
    text = _EVERY_DAY.sub("daily", _normalize(phrase))
    if not text:
        return None
    found = set()
    seen_day_word = False

    # Ranges first: "пн-пт", "mon-fri", "пн–ср".
    def _range(match):
        nonlocal seen_day_word
        first, last = _day_index(match.group(1)), _day_index(match.group(2))
        if first is None or last is None:
            return match.group(0)
        seen_day_word = True
        index = first
        while True:
            found.add(index)
            if index == last:
                break
            index = (index + 1) % 7
        return " "

    text = re.sub(r"([a-zа-я]+)\s*[-–—]\s*([a-zа-я]+)", _range, text)

    for token in re.split(r"[\s,;/]+", text):
        token = token.strip(".")
        if not token:
            continue
        if token in _FILLER:
            continue
        if _WORKDAYS.match(token):
            found.update(range(5))
        elif _WEEKEND.match(token):
            found.update((5, 6))
        elif _EVERYDAY.match(token):
            found.update(range(7))
        else:
            index = _day_index(token)
            if index is None:
                return None
            found.add(index)
        seen_day_word = True
    if not seen_day_word and not found:
        return None
    return sorted(found)


def _days_text(days, russian: bool = False) -> str:
    """Canonical (English) or spoken (Russian) rendering of a day set."""
    days = sorted(days)
    if days == list(range(5)):
        return "по будням" if russian else "weekdays"
    if days == [5, 6]:
        return "по выходным" if russian else "weekends"
    names = _WEEKDAY_NAMES if russian else _WEEKDAY_EN
    return ", ".join(names[d] for d in days) if russian else ",".join(names[d] for d in days)


# --- text helpers ------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".!")


def _clock(hour_text: str, minute_text: str = None):
    hour = int(hour_text)
    minute = int(minute_text) if minute_text is not None else 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ScheduleError(f"неверное время суток: {hour:02d}:{minute:02d}")
    return hour, minute


_TIME_AT_END = re.compile(
    r"(?:^|\s)(?:(?:at|в|во|около)\s+)?(\d{1,2})[:.](\d{2})(?:\s*(?:ч|h|часов|hrs?))?$")
_TIME_WORD_AT_END = re.compile(
    r"(?:^|\s)(?:at|в|во|около)\s+(\d{1,2})(?:\s*(?:ч|h|часов|час|часа|hrs?))?$")


def _split_time(text: str):
    """(rest, hour, minute) when the text ends with a time of day, else None."""
    for pattern in (_TIME_AT_END, _TIME_WORD_AT_END):
        match = pattern.search(text)
        if match:
            minute = match.group(2) if match.re is _TIME_AT_END else None
            hour, minute = _clock(match.group(1), minute)
            return text[:match.start()].strip(), hour, minute
    return None


_WINDOW = re.compile(
    r"\s*(?:between|from|с|в промежутке с|в интервале с)\s+(\d{1,2})[:.](\d{2})\s*"
    r"(?:and|to|до|по|-|–|—)\s*(\d{1,2})[:.](\d{2})")
_WINDOW_BARE = re.compile(r"\s+(\d{1,2})[:.](\d{2})\s*[-–—]\s*(\d{1,2})[:.](\d{2})$")


def _extract_window(text: str):
    match = _WINDOW.search(text) or _WINDOW_BARE.search(text)
    if not match:
        return text, None
    start_h, start_m = _clock(match.group(1), match.group(2))
    end_h, end_m = _clock(match.group(3), match.group(4))
    start, end = start_h * 60 + start_m, end_h * 60 + end_m
    if start == end:
        raise ScheduleError("окно времени пустое: начало и конец совпадают")
    return (text[:match.start()] + " " + text[match.end():]).strip(), [start, end]


_STARTUP = re.compile(
    r"^(?:on|at|every)?\s*(?:start|startup|start-up|launch|argent start|"
    r"при запуске|при старте|при открытии|при запуске argent|при старте argent)$")

_RELATIVE = re.compile(
    r"^(?:in|через)\s+(?:(?P<n>\d+)\s*|an?\s+)?" + _UNIT_WORD + r"$")
_HALF_HOUR = re.compile(r"^(?:in|через)\s+(?:полчаса|half an hour|half\s+hour)$")
_DAY_WORD = re.compile(r"^(today|tomorrow|сегодня|завтра|послезавтра)\s*(.*)$")
_ABS_ISO = re.compile(
    r"^(?:once\s+)?(?:at|on|в|на)?\s*(\d{4})-(\d{2})-(\d{2})(?:\s+|t)(?:at\s+|в\s+)?"
    r"(\d{1,2})[:.](\d{2})$")
_ABS_DOT = re.compile(
    r"^(?:once\s+)?(?:at|on|в|на)?\s*(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(?:at\s+|в\s+)?"
    r"(\d{1,2})[:.](\d{2})$")


def _once_spec(text: str, now):
    """A one-shot spec, or None when the text is not a one-shot at all."""
    match = _ABS_ISO.match(text)
    if match:
        year, month, day = (int(match.group(i)) for i in (1, 2, 3))
        hour, minute = _clock(match.group(4), match.group(5))
    else:
        match = _ABS_DOT.match(text)
        if match:
            day, month, year = (int(match.group(i)) for i in (1, 2, 3))
            hour, minute = _clock(match.group(4), match.group(5))
        else:
            match = None
    if match:
        try:
            at = datetime(year, month, day, hour, minute)
        except ValueError as e:
            raise ScheduleError(f"неверная дата: {e}") from None
        return {"kind": "once", "at": at.isoformat(timespec="minutes")}

    match = _RELATIVE.match(text)
    half = _HALF_HOUR.match(text)
    if match or half:
        if now is None:
            raise ScheduleError(
                f"{text!r} is relative to the moment it is created: use normalize_schedule()")
        seconds = 1800 if half else (
            int(match.group("n") or 1) * _unit_seconds(match.group("unit")))
        if seconds < MIN_INTERVAL_SECONDS:
            raise ScheduleError(
                f"слишком скоро ({seconds} с): минимум {MIN_INTERVAL_SECONDS} с")
        at = (now + timedelta(seconds=seconds)).replace(second=0, microsecond=0)
        if at <= now:
            at += timedelta(minutes=1)
        return {"kind": "once", "at": at.isoformat(timespec="minutes")}

    match = _DAY_WORD.match(text)
    if match:
        word, rest = match.group(1), match.group(2).strip()
        timed = _split_time(" " + rest) if rest else None
        if timed is None or timed[0]:
            raise ScheduleError(f"укажите и время, например: '{word} at 09:00' или '{word} в 9:00'")
        if now is None:
            raise ScheduleError(
                f"{text!r} is relative to the moment it is created: use normalize_schedule()")
        offset = {"today": 0, "сегодня": 0, "tomorrow": 1, "завтра": 1, "послезавтра": 2}[word]
        at = (now + timedelta(days=offset)).replace(
            hour=timed[1], minute=timed[2], second=0, microsecond=0)
        if at <= now:
            raise ScheduleError(f"{at:%H:%M} сегодня уже прошло")
        return {"kind": "once", "at": at.isoformat(timespec="minutes")}
    return None


_MONTHLY = re.compile(r"month|месяц|ежемесячно|числа|числе")


def _monthly_spec(text: str):
    if not _MONTHLY.search(text):
        return None
    timed = _split_time(text)
    if timed is None:
        raise ScheduleError("для ежемесячного расписания нужно время: 'monthly on 15 at 09:00'")
    head, hour, minute = timed
    numbers = re.findall(r"\d{1,2}", head)
    if len(numbers) != 1:
        raise ScheduleError("укажите день месяца: 'monthly on 15 at 09:00'")
    day = int(numbers[0])
    if not 1 <= day <= 31:
        raise ScheduleError(f"неверный день месяца: {day}")
    return {"kind": "monthly", "day": day, "hour": hour, "minute": minute}


_INTERVAL_HEAD = re.compile(
    r"^(?:every|each|кажд\w+|раз в|через каждые)\s*(?P<n>\d+)?\s*" + _UNIT_WORD + r"\s*(?P<rest>.*)$")


# --- parsing -----------------------------------------------------------------------

def _parse(text: str, now):
    if not text or not text.strip():
        raise ScheduleError("расписание пустое")
    raw = _normalize(text)

    if _STARTUP.match(raw):
        return {"kind": "startup"}

    once = _once_spec(raw, now)
    if once is not None:
        return once

    monthly = _monthly_spec(raw)
    if monthly is not None:
        return monthly

    core, window = _extract_window(raw)

    timed = _split_time(core)
    if timed is not None:
        head, hour, minute = timed
        if window is not None:
            raise ScheduleError("окно времени имеет смысл только для повторяющегося интервала (every 30m between 09:00 and 18:00)")
        if re.search(r"(?:^|\s)(?:every|each|кажд\w+|раз в)\s*\d+\s*" + _UNIT_WORD, head):
            raise ScheduleError(
                "интервал ('every 2h') нельзя сочетать со временем суток: "
                "нужно либо 'daily at 09:00', либо 'every 2h between 09:00 and 18:00'")
        days = _days_from_head(head)
        spec = {"kind": "daily", "hour": hour, "minute": minute}
        if days is not None and days != list(range(7)):
            spec["days"] = days
        return spec

    match = _INTERVAL_HEAD.match(core)
    if match:
        count = int(match.group("n")) if match.group("n") else 1
        seconds = count * _unit_seconds(match.group("unit"))
        if seconds < MIN_INTERVAL_SECONDS:
            raise ScheduleError(
                f"интервал слишком мал ({seconds} с): минимум {MIN_INTERVAL_SECONDS} с")
        spec = {"kind": "interval", "seconds": seconds}
        rest = match.group("rest").strip()
        if rest:
            days = parse_days(rest)
            if days is None:
                raise ScheduleError(f"не понял «{rest}» после интервала (ожидалось: по будням, пн,ср, ...)")
            if days != list(range(7)):
                spec["days"] = days
        if window is not None:
            spec["window"] = window
        return spec

    raise ScheduleError(
        f"не понял расписание «{text.strip()}». Примеры: every 15m · каждые 2 часа · "
        f"daily at 09:00 · по будням в 9:00 · пн,пт в 18:30 · monthly on 1 at 09:00 · "
        f"every 1h between 09:00 and 18:00 · через 2 часа · завтра в 8:00 · при запуске")


def _days_from_head(head: str):
    """Days for a 'daily'-style schedule from whatever precedes 'at HH:MM'."""
    head = head.strip()
    if not head:
        return None
    days = parse_days(head)
    if re.search(r"(?:^|\s)(?:weekly|еженедельно)(?:\s|$)", head) and (
            days is None or days == list(range(7))):
        raise ScheduleError("для weekly нужен день: 'weekly on mon at 10:00'")
    if days is None:
        raise ScheduleError(
            f"не понял «{head}» перед временем. Примеры: daily at 09:00 · "
            f"weekdays at 09:00 · mon,wed at 18:30")
    return days


def parse_schedule(text: str, now: datetime = None) -> dict:
    """Parse a schedule string into a spec dict (see the module docstring).

    The keys of the original forms are unchanged: {"kind": "interval", "seconds"}
    and {"kind": "daily", "hour", "minute"}. Optional "days" (Monday = 0) and
    "window" ([start_minute, end_minute]) appear only when the text asked for them.
    """
    return _parse(text, now)


def normalize_schedule(text: str, now: datetime = None) -> str:
    """The canonical English form of a schedule, resolving relative one-shots
    against ``now``. This — not what the user typed — is what gets stored."""
    return canonical(_parse(text, now or datetime.now()))


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def canonical(spec: dict) -> str:
    kind = spec["kind"]
    if kind == "startup":
        return "on start"
    if kind == "once":
        return "once at " + datetime.fromisoformat(spec["at"]).strftime("%Y-%m-%d %H:%M")
    if kind == "monthly":
        return f"monthly on {spec['day']} at {spec['hour']:02d}:{spec['minute']:02d}"
    if kind == "daily":
        when = f"at {spec['hour']:02d}:{spec['minute']:02d}"
        days = spec.get("days")
        return f"{_days_text(days)} {when}" if days else f"daily {when}"
    seconds = spec["seconds"]
    for suffix, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % size == 0:
            text = f"every {seconds // size}{suffix}"
            break
    if spec.get("days"):
        text += f" on {_days_text(spec['days'])}"
    if spec.get("window"):
        text += f" between {_hhmm(spec['window'][0])} and {_hhmm(spec['window'][1])}"
    return text


def _plural(count: int, one: str, few: str, many: str) -> str:
    if 11 <= count % 100 <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(count % 10, many)


def describe_schedule(text: str) -> str:
    """The schedule as a short Russian phrase; the text itself if unreadable."""
    try:
        spec = parse_schedule(text, now=datetime.now())
    except ScheduleError:
        return text
    kind = spec["kind"]
    if kind == "startup":
        return "при каждом запуске Argent"
    if kind == "once":
        return "один раз, " + datetime.fromisoformat(spec["at"]).strftime("%d.%m.%Y %H:%M")
    if kind == "monthly":
        return f"ежемесячно, {spec['day']}-го числа в {spec['hour']:02d}:{spec['minute']:02d}"
    if kind == "daily":
        when = f"в {spec['hour']:02d}:{spec['minute']:02d}"
        days = spec.get("days")
        return f"{_days_text(days, True)} {when}" if days else f"каждый день {when}"
    seconds = spec["seconds"]
    if seconds % 86400 == 0:
        count = seconds // 86400
        phrase = "раз в сутки" if count == 1 else f"раз в {count} {_plural(count, 'день', 'дня', 'дней')}"
    elif seconds % 3600 == 0:
        count = seconds // 3600
        phrase = "каждый час" if count == 1 else f"каждые {count} {_plural(count, 'час', 'часа', 'часов')}"
    else:
        count = seconds // 60
        phrase = "каждую минуту" if count == 1 else (
            f"каждые {count} {_plural(count, 'минуту', 'минуты', 'минут')}")
    if spec.get("days"):
        phrase += ", " + _days_text(spec["days"], True)
    if spec.get("window"):
        phrase += f", с {_hhmm(spec['window'][0])} до {_hhmm(spec['window'][1])}"
    return phrase


# --- when does it fire -------------------------------------------------------------

def _day_ok(spec: dict, moment: datetime) -> bool:
    days = spec.get("days")
    return days is None or moment.weekday() in days


def _in_window(spec: dict, moment: datetime) -> bool:
    window = spec.get("window")
    if not window:
        return True
    start, end = window
    minutes = moment.hour * 60 + moment.minute
    if start < end:
        return start <= minutes < end
    return minutes >= start or minutes < end          # overnight: 22:00 -> 06:00


def _allowed(spec: dict, moment: datetime) -> bool:
    return _day_ok(spec, moment) and _in_window(spec, moment)


def _next_allowed(spec: dict, moment: datetime) -> datetime:
    """The first instant at or after ``moment`` that the day set and the window allow."""
    window = spec.get("window")
    current = moment
    for _ in range(9):
        if _day_ok(spec, current):
            if _in_window(spec, current):
                return current
            if window:
                start_today = current.replace(hour=window[0] // 60, minute=window[0] % 60,
                                              second=0, microsecond=0)
                if start_today > current:
                    return start_today
        current = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return current


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _slot_candidates(spec: dict, now: datetime):
    """Wall-clock slots of a daily/monthly schedule around ``now``, oldest first."""
    if spec["kind"] == "daily":
        for offset in range(-1, 9):
            day = now + timedelta(days=offset)
            if _day_ok(spec, day):
                yield day.replace(hour=spec["hour"], minute=spec["minute"], second=0, microsecond=0)
        return
    # Start one month back so a slot just after midnight on the 1st still counts.
    first = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
    year, month = first.year, first.month
    for _ in range(14):
        day = min(spec["day"], _last_day_of_month(year, month))
        yield datetime(year, month, day, spec["hour"], spec["minute"])
        month += 1
        if month > 12:
            month, year = 1, year + 1


def next_run_after(schedule: str, last_run: datetime | None, now: datetime,
                   catchup_seconds: int = DAILY_CATCHUP_SECONDS) -> datetime:
    """When this schedule should fire next, given the previous run.

    A time in the past means "overdue": the caller decides it is due now.
    ``datetime.max`` means never (a one-shot that has run, or a start trigger).
    """
    spec = parse_schedule(schedule, now=now)
    kind = spec["kind"]

    if kind == "startup":
        return _FAR_FUTURE

    if kind == "once":
        at = datetime.fromisoformat(spec["at"])
        if last_run is not None and last_run >= at:
            return _FAR_FUTURE
        return at                           # late is better than never for a one-shot

    if kind == "interval":
        base = now if last_run is None else last_run + timedelta(seconds=spec["seconds"])
        return _next_allowed(spec, base)

    for slot in _slot_candidates(spec, now):
        if last_run is not None and last_run >= slot:
            continue                        # this slot has already been served
        if slot > now:
            return slot                     # still ahead of us
        if (now - slot).total_seconds() <= catchup_seconds:
            return slot                     # only a little late: catch up
    return _FAR_FUTURE


def next_fire(schedule: str, last_run: datetime | None, now: datetime,
              catchup_seconds: int = DAILY_CATCHUP_SECONDS) -> datetime | None:
    """For display: the next moment it will actually run, never in the past;
    None when it will not run again by itself."""
    try:
        spec = parse_schedule(schedule, now=now)
        nxt = next_run_after(schedule, last_run, now, catchup_seconds)
    except ScheduleError:
        return None
    if nxt == _FAR_FUTURE:
        return None
    if spec["kind"] == "interval":
        return _next_allowed(spec, max(nxt, now))
    return max(nxt, now)


def is_due(schedule: str, last_run: datetime | None, now: datetime,
           starting: bool = False, catchup_seconds: int = DAILY_CATCHUP_SECONDS) -> bool:
    """Whether the automation should run at ``now``.

    ``starting`` is True only on the first tick of a freshly started host: it is
    what makes an 'on start' schedule fire once per launch instead of every tick.
    """
    try:
        spec = parse_schedule(schedule, now=now)
        if spec["kind"] == "startup":
            return bool(starting) and (
                last_run is None
                or (now - last_run).total_seconds() >= STARTUP_COOLDOWN_SECONDS)
        due = next_run_after(schedule, last_run, now, catchup_seconds) <= now
    except ScheduleError:
        return False                        # a broken schedule never fires
    if due and spec["kind"] == "interval":
        # An overdue interval waits for its window rather than firing at 03:00.
        return _allowed(spec, now)
    return due


def is_one_shot(schedule: str) -> bool:
    try:
        return parse_schedule(schedule, now=datetime.now())["kind"] == "once"
    except ScheduleError:
        return False


def humanize_when(moment: datetime | None, now: datetime) -> str:
    """'через 12 мин', 'сегодня в 18:00', 'завтра в 09:00', '12.10 в 09:00'."""
    if moment is None:
        return "—"
    seconds = (moment - now).total_seconds()
    if seconds <= 60:
        return "сейчас"
    if seconds < 3600:
        return f"через {int(seconds // 60)} мин"
    stamp = moment.strftime("%H:%M")
    days = (moment.date() - now.date()).days
    if days == 0:
        return f"сегодня в {stamp}"
    if days == 1:
        return f"завтра в {stamp}"
    return f"{moment:%d.%m} в {stamp}"
