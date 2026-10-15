"""The schedule language: what can be written, how it reads back, when it fires.

A misread schedule in an unattended task is a job that runs at the wrong time
while nobody is looking, so the parser is strict (a typo is an error, never a
quietly wider schedule) and everything is driven by an explicit `now`.
"""

from datetime import datetime, timedelta

import pytest

from src.automation.schedule import (
    ScheduleError, canonical, describe_schedule, humanize_when, is_due, is_one_shot,
    next_fire, next_run_after, normalize_schedule, parse_days, parse_schedule,
)

THU = datetime(2026, 10, 8, 14, 30)          # a Thursday


class TestEnglishAndRussianAgree:
    @pytest.mark.parametrize("text,expected", [
        ("every 15m", "every 15m"), ("каждые 15 минут", "every 15m"),
        ("every hour", "every 1h"), ("каждый час", "every 1h"),
        ("каждые 2 часа", "every 2h"), ("раз в 30 минут", "every 30m"),
        ("every 1d", "every 1d"), ("каждые 2 дня", "every 2d"),
        ("daily at 09:00", "daily at 09:00"), ("каждый день в 9:00", "daily at 09:00"),
        ("every day at 9", "daily at 09:00"), ("ежедневно в 21.45", "daily at 21:45"),
        ("weekdays at 09:00", "weekdays at 09:00"), ("по будням в 9:00", "weekdays at 09:00"),
        ("mon-fri at 8:00", "weekdays at 08:00"), ("пн-пт в 8:00", "weekdays at 08:00"),
        ("weekends at 11:00", "weekends at 11:00"), ("по выходным в 11", "weekends at 11:00"),
        ("mon,wed,fri at 18:30", "mon,wed,fri at 18:30"),
        ("по пн, ср, пт в 18:30", "mon,wed,fri at 18:30"),
        ("каждый понедельник в 10:00", "mon at 10:00"),
        ("every monday at 10:00", "mon at 10:00"),
        ("weekly on fri at 17:00", "fri at 17:00"),
        ("по вторникам и четвергам в 8:15", "tue,thu at 08:15"),
        ("monthly on 15 at 09:00", "monthly on 15 at 09:00"),
        ("ежемесячно 15 числа в 9:00", "monthly on 15 at 09:00"),
        ("on start", "on start"), ("при запуске", "on start"),
    ])
    def test_canonical_form(self, text, expected):
        assert canonical(parse_schedule(text, now=THU)) == expected

    def test_the_original_dict_shapes_are_unchanged(self):
        assert parse_schedule("every 15m") == {"kind": "interval", "seconds": 900}
        assert parse_schedule("daily at 09:30") == {"kind": "daily", "hour": 9, "minute": 30}


class TestModifiers:
    def test_window_and_days_on_an_interval(self):
        spec = parse_schedule("every 30m on weekdays between 09:00 and 18:00")
        assert spec == {"kind": "interval", "seconds": 1800, "days": [0, 1, 2, 3, 4],
                        "window": [540, 1080]}

    def test_russian_window(self):
        spec = parse_schedule("раз в 30 минут с 9:00 до 18:00")
        assert spec["window"] == [540, 1080]

    def test_canonical_text_reparses_to_the_same_spec(self):
        for text in ("every 2h on mon,fri between 22:00 and 06:00", "tue,thu at 08:15",
                     "monthly on 31 at 07:05", "every 5m"):
            spec = parse_schedule(text, now=THU)
            assert parse_schedule(canonical(spec), now=THU) == spec

    def test_a_window_needs_an_interval(self):
        with pytest.raises(ScheduleError, match="интервал"):
            parse_schedule("daily at 09:00 between 09:00 and 18:00")


class TestOneShots:
    def test_relative_forms_become_absolute(self):
        assert normalize_schedule("in 2h", now=THU) == "once at 2026-10-08 16:30"
        assert normalize_schedule("через 30 минут", now=THU) == "once at 2026-10-08 15:00"
        assert normalize_schedule("tomorrow at 8:30", now=THU) == "once at 2026-10-09 08:30"
        assert normalize_schedule("послезавтра в 9", now=THU) == "once at 2026-10-10 09:00"

    @pytest.mark.parametrize("text,expected", [
        ("через час", "once at 2026-10-08 15:30"), ("через минуту", "once at 2026-10-08 14:31"),
        ("через полчаса", "once at 2026-10-08 15:00"), ("in an hour", "once at 2026-10-08 15:30"),
        ("in half an hour", "once at 2026-10-08 15:00"), ("in a day", "once at 2026-10-09 14:30"),
        ("через 3 дня", "once at 2026-10-11 14:30"),
    ])
    def test_everyday_relative_phrases(self, text, expected):
        assert normalize_schedule(text, now=THU) == expected

    def test_absolute_forms(self):
        for text in ("once at 2026-10-09 18:00", "2026-10-09 18:00", "09.10.2026 18:00"):
            assert normalize_schedule(text, now=THU) == "once at 2026-10-09 18:00"

    def test_a_relative_form_without_a_now_is_refused(self):
        """Stored as-is it would be 'two hours from whenever it is read' — never."""
        with pytest.raises(ScheduleError, match="relative"):
            parse_schedule("in 2h")

    def test_a_time_that_already_passed_is_an_error(self):
        with pytest.raises(ScheduleError, match="прошло"):
            normalize_schedule("today at 08:00", now=THU)

    def test_is_one_shot(self):
        assert is_one_shot("once at 2026-10-09 18:00") is True
        assert is_one_shot("daily at 09:00") is False

    def test_fires_when_reached_and_never_again(self):
        stored = "once at 2026-10-09 18:00"
        at = datetime(2026, 10, 9, 18, 0)
        assert is_due(stored, None, at - timedelta(minutes=1)) is False
        assert is_due(stored, None, at) is True
        # Argent was closed at 18:00: a reminder that comes late beats one that never comes.
        assert is_due(stored, None, at + timedelta(hours=5)) is True
        assert is_due(stored, at, at + timedelta(hours=5)) is False


@pytest.mark.parametrize("text", [
    "", "   ", "sometimes", "every banana", "daily at 25:00", "daily at 09:75",
    "every 2h at 9:00", "weekly at 10:00", "every 1h on funday", "monthly at 09:00",
    "monthly on 32 at 09:00", "every 30m between 09:00 and 09:00", "2026-13-40 10:00",
    "mon,xyz at 10:00",
])
def test_garbage_is_rejected(text):
    with pytest.raises(ScheduleError):
        parse_schedule(text, now=THU)


def test_sub_minute_is_refused():
    with pytest.raises(ScheduleError, match="слишком мал"):
        parse_schedule("every 0m")


class TestDays:
    def test_ranges_wrap_around_the_week(self):
        assert parse_days("fri-mon") == [0, 4, 5, 6]

    def test_unknown_word_is_none(self):
        assert parse_days("mon, someday") is None


class TestDueWithDaysAndWindows:
    def test_interval_waits_for_its_window(self):
        spec = "every 30m between 09:00 and 18:00"
        long_ago = datetime(2026, 10, 8, 8, 0)
        assert is_due(spec, long_ago, datetime(2026, 10, 8, 8, 59)) is False
        assert is_due(spec, long_ago, datetime(2026, 10, 8, 9, 0)) is True
        assert is_due(spec, None, datetime(2026, 10, 8, 20, 0)) is False       # after hours

    def test_an_overdue_interval_does_not_fire_after_hours(self):
        spec = "every 1h between 09:00 and 18:00"
        ran = datetime(2026, 10, 8, 12, 0)
        assert is_due(spec, ran, datetime(2026, 10, 8, 22, 0)) is False
        assert is_due(spec, ran, datetime(2026, 10, 9, 9, 0)) is True

    def test_overnight_window(self):
        spec = "every 1h between 22:00 and 06:00"
        assert is_due(spec, None, datetime(2026, 10, 8, 23, 0)) is True
        assert is_due(spec, None, datetime(2026, 10, 8, 3, 0)) is True
        assert is_due(spec, None, datetime(2026, 10, 8, 12, 0)) is False

    def test_interval_skips_the_wrong_days(self):
        spec = "every 1h on weekdays"
        saturday = datetime(2026, 10, 10, 12, 0)
        assert is_due(spec, None, saturday) is False
        assert is_due(spec, None, saturday + timedelta(days=2)) is True       # Monday

    def test_weekday_schedule_skips_the_weekend(self):
        spec = "weekdays at 09:00"
        assert is_due(spec, None, datetime(2026, 10, 9, 9, 5)) is True        # Friday
        assert is_due(spec, None, datetime(2026, 10, 10, 9, 5)) is False      # Saturday
        assert next_run_after(spec, datetime(2026, 10, 9, 9, 0),
                              datetime(2026, 10, 10, 9, 5)) == datetime(2026, 10, 12, 9, 0)

    def test_named_days(self):
        spec = "mon,thu at 18:00"
        assert next_run_after(spec, None, THU) == datetime(2026, 10, 8, 18, 0)
        assert next_run_after(spec, datetime(2026, 10, 8, 18, 0), datetime(2026, 10, 8, 19, 0)) \
            == datetime(2026, 10, 12, 18, 0)


class TestDailyCatchUp:
    def test_yesterdays_late_night_slot_is_still_served_after_midnight(self):
        """A 23:30 job on a machine that was shut at 23:30 and opens at 00:10."""
        spec = "daily at 23:30"
        assert is_due(spec, None, datetime(2026, 10, 9, 0, 10)) is True

    def test_but_not_hours_later(self):
        assert is_due("daily at 23:30", None, datetime(2026, 10, 9, 6, 0)) is False

    def test_catchup_window_is_adjustable(self):
        now = datetime(2026, 10, 8, 11, 0)
        assert is_due("daily at 09:00", None, now) is False
        assert is_due("daily at 09:00", None, now, catchup_seconds=4 * 3600) is True
        assert is_due("daily at 09:00", None, datetime(2026, 10, 8, 9, 5), catchup_seconds=0) is False


class TestMonthly:
    def test_fires_on_the_day(self):
        assert is_due("monthly on 15 at 09:00", None, datetime(2026, 10, 15, 9, 10)) is True
        assert is_due("monthly on 15 at 09:00", None, datetime(2026, 10, 14, 9, 10)) is False

    def test_day_31_means_the_last_day_of_a_short_month(self):
        spec = "monthly on 31 at 09:00"
        assert next_run_after(spec, datetime(2026, 9, 30, 9, 0),
                              datetime(2026, 10, 1, 12, 0)) == datetime(2026, 10, 31, 9, 0)
        assert next_run_after(spec, datetime(2026, 10, 31, 9, 0),
                              datetime(2026, 11, 1, 12, 0)) == datetime(2026, 11, 30, 9, 0)

    def test_once_per_month(self):
        ran = datetime(2026, 10, 15, 9, 0)
        assert is_due("monthly on 15 at 09:00", ran, datetime(2026, 10, 15, 9, 30)) is False

    def test_leap_february(self):
        assert next_run_after("monthly on 30 at 09:00", None,
                              datetime(2028, 2, 10, 8, 0)) == datetime(2028, 2, 29, 9, 0)


class TestStartup:
    def test_fires_only_on_the_first_tick_of_a_launch(self):
        assert is_due("on start", None, THU, starting=True) is True
        assert is_due("on start", None, THU, starting=False) is False

    def test_not_again_within_the_cooldown(self):
        assert is_due("on start", THU - timedelta(minutes=2), THU, starting=True) is False
        assert is_due("on start", THU - timedelta(minutes=30), THU, starting=True) is True


class TestDisplayHelpers:
    def test_next_fire_is_never_in_the_past(self):
        long_ago = datetime(2026, 10, 1, 9, 0)
        assert next_fire("every 1h", long_ago, THU) == THU          # overdue = now

    def test_next_fire_respects_the_window(self):
        spec = "every 1h between 09:00 and 18:00"
        late = datetime(2026, 10, 8, 20, 0)
        assert next_fire(spec, datetime(2026, 10, 8, 17, 0), late) == datetime(2026, 10, 9, 9, 0)

    def test_next_fire_of_a_finished_one_shot_is_none(self):
        at = datetime(2026, 10, 8, 10, 0)
        assert next_fire("once at 2026-10-08 10:00", at, THU) is None
        assert next_fire("on start", None, THU) is None

    def test_humanize(self):
        assert humanize_when(THU + timedelta(seconds=20), THU) == "сейчас"
        assert humanize_when(THU + timedelta(minutes=12), THU) == "через 12 мин"
        assert humanize_when(datetime(2026, 10, 8, 18, 0), THU) == "сегодня в 18:00"
        assert humanize_when(datetime(2026, 10, 9, 9, 0), THU) == "завтра в 09:00"
        assert humanize_when(datetime(2026, 10, 12, 9, 0), THU) == "12.10 в 09:00"
        assert humanize_when(None, THU) == "—"

    @pytest.mark.parametrize("text,words", [
        ("every 1h", "каждый час"), ("every 5m", "каждые 5 минут"), ("every 2m", "каждые 2 минуты"),
        ("every 1m", "каждую минуту"), ("every 3h", "каждые 3 часа"), ("every 1d", "раз в сутки"),
        ("weekdays at 09:00", "по будням в 09:00"), ("on start", "при каждом запуске"),
    ])
    def test_describe_in_russian(self, text, words):
        assert words in describe_schedule(text)

    def test_describe_leaves_unreadable_text_alone(self):
        assert describe_schedule("gibberish") == "gibberish"

    def test_a_broken_schedule_never_fires(self):
        assert is_due("every banana", None, THU) is False
        assert next_fire("every banana", None, THU) is None
