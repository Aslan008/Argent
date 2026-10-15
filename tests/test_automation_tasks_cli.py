"""`/tasks`: parsing, what it writes to disk, and what it says."""

import os
import sys
import types
from datetime import datetime, timedelta

import pytest

from src.automation import manage, notice, service, store, tasks_cli, templates
from src.automation.store import Automation

NOW = datetime(2026, 10, 8, 14, 30)       # a Thursday


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("project_paths.project_root_or_cwd", lambda *a, **k: tmp_path)
    return tmp_path


def run(command, live=None, now=NOW):
    outs, errs = [], []
    tasks_cli.handle(command, outs.append, errs.append, live=live, now=now)
    return "\n".join(outs), "\n".join(errs)


def _a(name="job", **kw):
    kw.setdefault("task", "t")
    kw.setdefault("schedule", "every 1h")
    return Automation(name=name, **kw)


# --- option parsing ------------------------------------------------------------------

class TestParseOptions:
    def test_keys_in_either_language(self):
        fields, errors = tasks_cli.parse_options(
            ["tools: read_file, grep_search", "уведомлять: всегда", "таймаут: 5", "report: r/{date}.md"])
        assert errors == []
        assert fields == {"tools": "read_file, grep_search", "notify": "always", "timeout": "5",
                          "report": "r/{date}.md"}

    @pytest.mark.parametrize("word,expected", [
        ("auto", "auto"), ("всегда", "always"), ("только сбои", "failures"),
        ("изменения", "changes"), ("никогда", "never"), ("CHANGES", "changes"),
    ])
    def test_notify_words(self, word, expected):
        assert tasks_cli.parse_options([f"notify: {word}"])[0]["notify"] == expected

    def test_modes(self):
        assert tasks_cli.parse_options(["mode: команда"])[0]["mode"] == "command"

    def test_all_tools_means_unrestricted(self):
        assert tasks_cli.parse_options(["tools: all"])[0]["tools"] == []
        assert tasks_cli.parse_options(["tools: все"])[0]["tools"] == []

    def test_channels_and_off_switches(self):
        assert tasks_cli.parse_options(["channels: Telegram"])[0]["channels"] == ["telegram"]
        assert tasks_cli.parse_options(["channels: телеграм"])[0]["channels"] == ["telegram"]
        assert tasks_cli.parse_options(["channels: none"])[0]["channels"] == []
        assert tasks_cli.parse_options(["report: off"])[0]["report"] == ""
        assert tasks_cli.parse_options(["commands: -"])[0]["commands"] == []

    @pytest.mark.parametrize("segment", ["notify: loudly", "mode: wizard", "frobnicate: 1", "just words"])
    def test_a_typo_is_an_error_not_silently_dropped(self, segment):
        fields, errors = tasks_cli.parse_options([segment])
        assert errors and fields == {}


class TestSplitAdd:
    def test_basic(self):
        assert tasks_cli.split_add("rep | daily at 9:00 | Do it") == ("rep", "daily at 9:00", "Do it", [])

    def test_options_are_split_off_the_task(self):
        name, schedule, task, rest = tasks_cli.split_add("n | every 1h | check | tools: a | notify: always")
        assert task == "check" and rest == [" tools: a ", " notify: always"]

    def test_a_pipe_inside_the_task_survives(self):
        _, _, task, rest = tasks_cli.split_add('n | every 1h | dir | find "x" | notify: always')
        assert task == 'dir | find "x"' and len(rest) == 1

    def test_too_few_pieces(self):
        assert tasks_cli.split_add("only | two") is None


# --- creating ------------------------------------------------------------------------

class TestAdd:
    def test_a_russian_schedule_is_stored_canonically(self, project):
        out, err = run("/tasks add отчёт | по будням в 9:00 | Собери сводку")
        assert err == ""
        a = store.get_automation("отчёт")
        assert a.schedule == "weekdays at 09:00" and a.mode == "agent"
        assert "сохранена" in out and "по будням в 09:00" in out and "следующий запуск" in out

    def test_options(self, project):
        run("/tasks add r | every 2h | check | tools: read_file, run_command | commands: git log*; npm test"
            " | notify: changes | report: .argent/reports/{date}.md | timeout: 7 | retries: 3 | channels: telegram")
        a = store.get_automation("r")
        assert a.allowed_tools == ["read_file", "run_command"]
        assert a.allowed_commands == ["git log*", "npm test"]
        assert (a.notify, a.timeout_minutes, a.retries, a.channels) == ("changes", 7, 3, ["telegram"])
        assert a.report_file == ".argent/reports/{date}.md"

    def test_commands_pull_in_the_tool_they_need(self, project):
        out, _ = run("/tasks add r | every 2h | check | tools: read_file | commands: git log*")
        assert store.get_automation("r").allowed_tools == ["read_file", "run_command"]
        assert "run_command" in out

    def test_unrestricted_tools_and_fast_polling_are_warned_about(self, project):
        out, _ = run("/tasks add r | every 2m | check")
        assert "не ограничены" in out and "токенов" in out

    def test_run_command_without_commands_is_warned_about(self, project):
        out, _ = run("/tasks add r | every 1h | check | tools: run_command")
        assert "любая будет отклонена" in out

    @pytest.mark.parametrize("command,needle", [
        ("/tasks add r | sometimes | t", "sometimes"),
        ("/tasks add r | every 1h | t | tools: serch_web", "serch_web"),
        ("/tasks add r | every 1h | t | notify: loud", "notify"),
        ("/tasks add r | every 1h | t | report: ../x.md", "внутри проекта"),
        ("/tasks add r | every 1h | t | timeout: soon", "целое"),
        ("/tasks add r | every 1h", "Формат"),
    ])
    def test_a_bad_definition_saves_nothing_and_says_why(self, project, command, needle):
        out, err = run(command)
        assert needle in err and store.load_automations() == []

    def test_duplicate_name(self, project):
        run("/tasks add r | every 1h | t")
        _, err = run("/tasks add r | every 2h | other")
        assert "уже есть" in err and store.get_automation("r").task == "t"

    def test_markup_in_the_task_cannot_break_the_screen(self, project):
        from rich.text import Text
        run("/tasks add r | every 1h | check [red]this[/red] and [/ nonsense")
        out, _ = run("/tasks list")
        assert "check [red]this[/red] and [/ nonsense" in Text.from_markup(out).plain

    def test_cmd_makes_a_command_job(self, project):
        out, err = run("/tasks cmd backup | daily at 03:00 | robocopy a b /MIR | notify: failures")
        a = store.get_automation("backup")
        assert err == "" and a.mode == "command" and a.task == "robocopy a b /MIR"
        assert "без модели" in out

    def test_a_command_job_drops_model_only_settings(self, project):
        out, _ = run("/tasks cmd c | every 1h | echo hi | tools: read_file | commands: ls")
        a = store.get_automation("c")
        assert a.allowed_tools == [] and a.allowed_commands == []
        assert "убраны" in out


class TestRemind:
    def test_relative_time(self, project):
        out, err = run("/tasks remind через 2 часа | позвонить Ане")
        a = store.load_automations()[0]
        assert err == "" and a.mode == "remind" and a.notify == "always"
        assert a.schedule == "once at 2026-10-08 16:30" and a.task == "позвонить Ане"
        assert a.name.startswith("напоминание-") and "позвонить Ане" in out

    def test_names_do_not_clash(self, project):
        run("/tasks remind через 1 час | позвонить")
        run("/tasks remind через 2 часа | позвонить")
        assert len({a.name for a in store.load_automations()}) == 2

    def test_a_recurring_reminder(self, project):
        run("/tasks remind по будням в 9:00 | стендап")
        assert store.load_automations()[0].schedule == "weekdays at 09:00"

    @pytest.mark.parametrize("command", ["/tasks remind", "/tasks remind через час", "/tasks remind | x"])
    def test_usage(self, project, command):
        _, err = run(command)
        assert "Формат" in err

    def test_a_past_time_is_refused(self, project):
        _, err = run("/tasks remind сегодня в 8:00 | поздно")
        assert "прошло" in err or "passed" in err

    def test_the_slash_remind_alias(self, project, monkeypatch):
        from src.cli.commands import agent_commands
        printed = []
        monkeypatch.setattr(agent_commands, "print_system", printed.append)
        monkeypatch.setattr(agent_commands, "print_error", printed.append)
        result = agent_commands.handle_remind(types.SimpleNamespace(
            user_input="/remind через 3 часа | чай", agent=None))
        assert result is not None and any("чай" in p for p in printed)
        assert store.load_automations()[0].mode == "remind"


class TestNew:
    def test_from_a_template_with_an_argument(self, project):
        out, err = run("/tasks new releases astral-sh/uv")
        a = store.get_automation("releases-astral-sh/uv") or store.load_automations()[0]
        assert err == "" and "astral-sh/uv" in a.task and "{arg}" not in a.task
        assert a.allowed_tools == ["read_webpage", "filter_new_items"] or "read_webpage" in a.allowed_tools

    def test_a_command_template_is_a_model_free_job(self, project):
        run("/tasks new tests pytest -q")
        a = store.get_automation("tests")
        assert a.mode == "command" and a.task == "pytest -q" and a.notify == "failures"

    def test_options_override_the_template(self, project):
        run("/tasks new tests pytest -q | schedule: every 30m | name: my-tests")
        assert store.get_automation("my-tests").schedule == "every 30m"

    def test_unknown_and_missing_argument(self, project):
        assert "Нет шаблона" in run("/tasks new nope")[1]
        assert "нужен аргумент" in run("/tasks new page")[1]

    def test_no_arguments_lists_the_templates(self, project):
        out, _ = run("/tasks new")
        assert "morning" in out and "tests" in out

    def test_the_same_template_twice_gets_two_names(self, project):
        run("/tasks new morning")
        run("/tasks new morning")
        assert sorted(a.name for a in store.load_automations()) == ["morning", "morning-2"]


class TestEdit:
    def test_changes_only_what_was_named(self, project):
        run("/tasks add r | every 1h | check | tools: read_file")
        store.update_automation("r", lambda a: setattr(a, "run_count", 7))
        out, err = run("/tasks edit r | schedule: каждые 3 часа | notify: always")
        a = store.get_automation("r")
        assert err == "" and (a.schedule, a.notify) == ("every 3h", "always")
        assert a.allowed_tools == ["read_file"] and a.run_count == 7
        assert "обновлена" in out

    def test_prefix_of_the_name_is_enough(self, project):
        run("/tasks add weekly-digest | every 1d | t")
        assert run("/tasks edit weekly | notify: never")[1] == ""

    def test_errors(self, project):
        run("/tasks add r | every 1h | check")
        assert "не найдена" in run("/tasks edit ghost | notify: always")[1]
        assert "переименовать" in run("/tasks edit r | name: other")[1]
        assert "sometimes" in run("/tasks edit r | schedule: sometimes")[1]
        assert "Формат" in run("/tasks edit r")[1]
        assert store.get_automation("r").schedule == "every 1h"

    def test_a_new_time_revives_a_finished_one_shot(self, project):
        run("/tasks remind завтра в 9:00 | x")
        name = store.load_automations()[0].name
        store.record_run(name, "ok", "x", NOW, NOW, trigger="schedule")
        assert store.get_automation(name).enabled is False
        run(f"/tasks edit {name} | schedule: послезавтра в 10:00")
        assert store.get_automation(name).enabled is True


class TestToggle:
    def test_on_off(self, project):
        run("/tasks add r | every 1h | t")
        assert "выключена" in run("/tasks off r")[0]
        assert store.get_automation("r").enabled is False
        assert "включена" in run("/tasks on r")[0]

    def test_enabling_clears_what_paused_it(self, project):
        store.upsert_automation(_a(max_failures=1))
        store.record_run("job", "error", "boom", NOW, NOW)
        assert store.get_automation("job").enabled is False
        out, _ = run("/tasks on job")
        a = store.get_automation("job")
        assert a.enabled and a.fail_streak == 0 and a.disabled_reason == "" and "была отключена" in out

    def test_all(self, project):
        store.upsert_automation(_a("a"))
        store.upsert_automation(_a("b"))
        assert "Выключено: 2" in run("/tasks off all")[0]
        assert not any(x.enabled for x in store.load_automations())
        assert "Включено: 2" in run("/tasks on все")[0]

    def test_a_finished_one_shot_cannot_just_be_switched_back_on(self, project):
        run("/tasks remind завтра в 9:00 | x")
        name = store.load_automations()[0].name
        store.record_run(name, "ok", "x", NOW, NOW, trigger="schedule")
        assert "разовая" in run(f"/tasks on {name}")[1]


class TestFindingByName:
    def test_ambiguous_prefix(self, project):
        store.upsert_automation(_a("report-a"))
        store.upsert_automation(_a("report-b"))
        assert "несколько" in run("/tasks show report")[1]

    def test_missing_name(self, project):
        assert "Укажите имя" in run("/tasks show")[1]
        assert "не найдена" in run("/tasks show ghost")[1]


class TestReading:
    def test_list_and_show(self, project):
        run("/tasks add r | weekdays at 09:00 | check the build | notify: changes | commands: git log*"
            " | tools: run_command")
        out, _ = run("/tasks list")
        assert "r" in out and "по будням в 09:00" in out and "следующий" in out and "check the build" in out
        shown, _ = run("/tasks show r")
        for needle in ("changes", "git log*", "run_command", "check the build", "повторов при сбое: 1"):
            assert needle in shown

    def test_a_paused_job_says_why(self, project):
        store.upsert_automation(_a(max_failures=1))
        store.record_run("job", "error", "boom", NOW, NOW)
        assert "сбоев подряд" in run("/tasks list")[0]

    def test_runs_last_and_inbox(self, project):
        store.upsert_automation(_a())
        store.mark_seen(NOW - timedelta(hours=1))
        earlier = NOW - timedelta(minutes=10)
        store.record_run("job", "error", "disk is full\nsecond line", earlier,
                         earlier + timedelta(seconds=3),
                         [{"action": "rm x", "destructive": True}], trigger="schedule")
        store.record_run("job", "ok", "fine", earlier, earlier, trigger="manual")
        runs, _ = run("/tasks runs")
        assert "error" in runs and "(вручную)" in runs and "disk is full" in runs
        last, _ = run("/tasks last job")
        assert "fine" in last
        inbox, _ = run("/tasks inbox")
        assert "disk is full" in inbox and "rm x" in inbox and "fine" not in inbox
        assert "Нового нет" in run("/tasks inbox")[0]          # and now it is read

    def test_quiet_runs_are_collapsed(self, project):
        store.upsert_automation(_a())
        store.record_run("job", "ok", "Ничего нового.", NOW, NOW, silent=True)
        out, _ = run("/tasks runs job")
        assert "без новостей" in out and "Ничего нового" not in out

    def test_empty_states(self, project):
        assert "Прогонов ещё не было" in run("/tasks runs")[0]
        store.upsert_automation(_a())
        assert "ещё не запускалась" in run("/tasks last job")[0]

    def test_unknown_subcommand_lists_the_commands(self, project):
        _, err = run("/tasks malicious_payload")
        assert "Команды: /tasks list" in err


class TestDashboard:
    def test_empty_project_gets_a_starting_point(self, project):
        out, _ = run("/tasks")
        assert "/tasks templates" in out and "/tasks remind" in out

    def test_shows_next_up_attention_and_pause(self, project):
        store.upsert_automation(_a("soon", schedule="daily at 18:00"))
        store.upsert_automation(_a("off", enabled=False, disabled_reason="выключена вручную"))
        store.mark_seen(NOW - timedelta(hours=2))
        store.record_run("soon", "error", "boom", NOW - timedelta(hours=1), NOW - timedelta(hours=1))
        store.set_paused(until=NOW + timedelta(hours=2))
        out, _ = run("/tasks")
        assert "Ближайшие" in out and "сегодня в 18:00" in out
        assert "выключена" in out and "Новое для вас: 1" in out and "на паузе" in out

    def test_running_jobs_are_shown(self, project):
        store.upsert_automation(_a("busy"))
        live = types.SimpleNamespace(
            status=lambda: {"running": {"busy": {"since": NOW, "trigger": "schedule"}},
                            "elsewhere": [], "paused_until": None})
        out, _ = run("/tasks", live=live)
        assert "идёт" in out and "busy" in out


class TestPauseResume:
    def test_pause_until_resumed(self, project):
        out, _ = run("/tasks pause")
        assert store.pause_until() == datetime.max and "/tasks resume" in out
        run("/tasks resume")
        assert store.pause_until() is None

    @pytest.mark.parametrize("arg,hours", [("2h", 2), ("90m", 1.5), ("через 3 часа", 3)])
    def test_for_a_while(self, project, arg, hours):
        run(f"/tasks pause {arg}")
        until = datetime.fromisoformat(store.load_state()["paused_until"])
        assert until - NOW == timedelta(hours=hours)

    def test_until_a_time_of_day(self, project):
        run("/tasks pause до 18:00")
        assert datetime.fromisoformat(store.load_state()["paused_until"]) == datetime(2026, 10, 8, 18, 0)
        run("/tasks pause до 9:00")
        assert datetime.fromisoformat(store.load_state()["paused_until"]) == datetime(2026, 10, 9, 9, 0)

    def test_nonsense(self, project):
        assert "Не понял срок" in run("/tasks pause someday")[1]
        assert store.pause_until() is None


class TestRunAndStop:
    def test_a_prompt_run_waits_and_prints_the_outcome(self, project, monkeypatch):
        store.upsert_automation(_a())
        monkeypatch.delenv("ARGENT_TUI", raising=False)
        calls = []

        class Stub:
            def __call__(self, automation, trigger="schedule"):
                calls.append((automation.name, trigger))
                return {"status": "ok", "summary": "all good", "denied_actions": [], "new_items": 2,
                        "started": NOW, "finished": NOW + timedelta(seconds=4),
                        "commands": ["git log -1"], "report": "r.md"}

        monkeypatch.setattr("src.automation.runner.SubprocessRunner", Stub)
        out, err = run("/tasks run job")
        assert err == "" and calls == [("job", "manual")]
        assert "all good" in out and "Новых элементов запомнено: 2" in out
        assert "git log -1" in out and "r.md" in out
        assert store.claim_run("job") is True            # the lock was given back

    def test_a_run_is_not_started_twice(self, project, monkeypatch):
        store.upsert_automation(_a())
        monkeypatch.delenv("ARGENT_TUI", raising=False)
        store.claim_run("job")
        assert "уже выполняется" in run("/tasks run job")[1]

    def test_in_the_tui_it_goes_to_the_background(self, project, monkeypatch):
        store.upsert_automation(_a())
        monkeypatch.setenv("ARGENT_TUI", "1")
        started = []
        live = types.SimpleNamespace(run_now=lambda name: started.append(name) or (True, ""))
        out, err = run("/tasks run job", live=live)
        assert started == ["job"] and "в фоне" in out and err == ""

    def test_in_the_tui_a_refusal_is_reported(self, project, monkeypatch):
        store.upsert_automation(_a())
        monkeypatch.setenv("ARGENT_TUI", "1")
        live = types.SimpleNamespace(run_now=lambda name: (False, "уже выполняется."))
        assert "уже выполняется" in run("/tasks run job", live=live)[1]

    def test_stop(self, project):
        store.upsert_automation(_a())
        live = types.SimpleNamespace(stop_run=lambda name: name == "job")
        assert "Останавливаю" in run("/tasks stop job", live=live)[0]
        assert "не выполняется" in run("/tasks stop job", live=None)[1]


class TestMemoryCommands:
    def test_forget_and_stats(self, project, monkeypatch):
        from src.automation import memory
        memory.filter_new(["a", "b"], scope="job")
        assert "job: 2" in run("/tasks memory")[0]
        assert "Забыто элементов: 2" in run("/tasks forget job")[0]
        assert "Память пуста" in run("/tasks memory")[0]


class TestServiceCommand:
    def test_status_install_remove(self, project, monkeypatch):
        calls = []
        monkeypatch.setattr(service, "install", lambda root, minutes=15: calls.append(("install", minutes)) or (True, "готово"))
        monkeypatch.setattr(service, "remove", lambda root: calls.append(("remove",)) or (True, "убрано"))
        monkeypatch.setattr(service, "status", lambda root: {"installed": False, "name": "X", "supported": True})
        assert "Не установлена" in run("/tasks service")[0]
        assert run("/tasks service install 30")[0] == "готово" and ("install", 30) in calls
        assert run("/tasks service remove")[0] == "убрано"
        assert "минут" in run("/tasks service install abc")[1]
        assert "Формат" in run("/tasks service wat")[1]

    def test_a_failure_goes_to_the_error_channel(self, project, monkeypatch):
        monkeypatch.setattr(service, "install", lambda root, minutes=15: (False, "отказано"))
        out, err = run("/tasks service install")
        assert out == "" and err == "отказано"


class TestTemplates:
    @pytest.mark.parametrize("template", list(templates.TEMPLATES.values()), ids=lambda t: t.id)
    def test_every_template_builds_a_valid_definition(self, template):
        arg = "https://example.com/x" if template.id == "page" else (
            "astral-sh/uv" if template.id == "releases" else "pytest -q")
        fields = templates.fields_for(template, arg if template.needs_arg else "")
        fields["name"] = "t"
        automation, _ = manage.build(fields, None, NOW)
        assert "{arg}" not in automation.task and not any("{arg}" in c for c in automation.allowed_commands)

    def test_agent_templates_use_only_real_tools(self):
        known = manage.known_tool_names()
        for template in templates.TEMPLATES.values():
            assert set(template.tools) <= known, template.id

    def test_agent_templates_that_run_commands_declare_them(self):
        for template in templates.TEMPLATES.values():
            if "run_command" in template.tools:
                assert template.commands, template.id

    def test_the_argument_is_required_where_promised(self):
        with pytest.raises(ValueError, match="нужен аргумент"):
            templates.fields_for(templates.TEMPLATES["tests"], "  ")

    def test_names(self):
        t = templates.TEMPLATES["page"]
        assert templates.default_name(t, "https://example.com/pricing?x=1", []) == "page-example.com/pricing-x-1"
        assert templates.default_name(t, "", []) == "page"
        assert templates.default_name(t, "", ["page", "page-2"]) == "page-3"
        assert templates.default_name(templates.TEMPLATES["tests"], "pytest -q --long", []) == "tests"


class TestManage:
    def test_update_keeps_bookkeeping_that_changed_meanwhile(self, project):
        manage.create({"name": "r", "schedule": "every 1h", "task": "t"}, NOW)
        store.update_automation("r", lambda a: setattr(a, "run_count", 5))
        manage.update("r", {"notify": "never"}, NOW)
        assert store.get_automation("r").run_count == 5

    def test_unknown_parameters_are_refused(self, project):
        with pytest.raises(manage.ManageError, match="неизвестные параметры"):
            manage.create({"name": "r", "schedule": "every 1h", "task": "t", "colour": "red"})

    def test_text_values_are_coerced(self, project):
        a, _ = manage.create({"name": "r", "schedule": "every 1h", "task": "t",
                              "tools": "read_file; run_command", "commands": "git log*; npm test",
                              "timeout": "9", "channels": "telegram"}, NOW)
        assert a.allowed_tools == ["read_file", "run_command"]
        assert a.allowed_commands == ["git log*", "npm test"] and a.timeout_minutes == 9

    def test_summary_lines(self, project):
        a, _ = manage.create({"name": "r", "schedule": "daily at 18:00", "task": "t",
                              "commands": ["ls"], "tools": ["run_command"]}, NOW)
        text = "\n".join(manage.summary_lines(a, NOW))
        assert "сегодня в 18:00" in text and "разрешённые команды: ls" in text


class TestNotice:
    def _event(self, **kw):
        base = {"event": "finished", "name": "job", "status": "ok", "summary": "", "denied": 0,
                "new_items": 0, "alert": False, "trigger": "schedule", "mode": "agent"}
        base.update(kw)
        return base

    def test_a_quiet_scheduled_run_says_nothing(self):
        assert notice.format_event_notice(self._event()) is None
        assert notice.format_event_notice({"event": "started", "name": "x"}) is None

    def test_news_errors_and_refusals_speak(self):
        news = notice.format_event_notice(self._event(alert=True, new_items=2, summary="A, B"))
        assert "нового: 2" in news and "A, B" in news and "/tasks inbox" in news
        assert "⚠" in notice.format_event_notice(self._event(alert=True, status="error", summary="boom"))
        assert "🔒" in notice.format_event_notice(self._event(alert=True, status="denied", denied=1))

    def test_a_reminder_is_just_the_message(self):
        text = notice.format_event_notice(self._event(alert=True, mode="remind", summary="Позвонить Ане"))
        assert text == "⏰ Напоминание «job»: Позвонить Ане"

    def test_a_manual_run_always_answers(self):
        text = notice.format_event_notice(self._event(trigger="manual", summary="done"))
        assert "выполнена" in text and "done" in text and "/tasks inbox" not in text

    def test_events_without_the_flag_fall_back_to_the_old_rule(self):
        event = self._event(alert=None, new_items=1)
        assert notice.format_event_notice(event) is not None
        assert notice.format_event_notice(self._event(alert=None)) is None

    def test_long_summaries_are_cut(self):
        text = notice.format_event_notice(self._event(alert=True, summary="x" * 5000))
        assert len(text) < 800

    def test_first_ever_start_has_no_backlog_and_starts_the_clock(self, project):
        store.upsert_automation(_a())
        store.record_run("job", "error", "old", NOW - timedelta(days=3), NOW - timedelta(days=3))
        assert notice.startup_digest(NOW) is None
        assert store.last_seen() == NOW

    def test_digest_of_what_happened_while_away(self, project):
        store.upsert_automation(_a("a"))
        store.upsert_automation(_a("b"))
        store.mark_seen(NOW - timedelta(hours=8))
        store.record_run("a", "error", "x", NOW - timedelta(hours=5), NOW - timedelta(hours=5))
        store.record_run("b", "ok", "x", NOW - timedelta(hours=4), NOW - timedelta(hours=4), new_items=3)
        store.record_run("b", "ok", "plain", NOW - timedelta(hours=3), NOW - timedelta(hours=3))
        text = notice.startup_digest(NOW)
        assert "прогонов: 2" in text and "нового: 3" in text and "сбоев: 1" in text
        assert "a, b" in text and "/tasks inbox" in text

    def test_nothing_to_report_is_none(self, project):
        store.mark_seen(NOW - timedelta(hours=8))
        assert notice.startup_digest(NOW) is None

    def test_a_pause_is_mentioned(self, project):
        store.set_paused()
        store.mark_seen(NOW)
        assert "на паузе" in notice.startup_digest(NOW)


class TestService:
    class _Run:
        def __init__(self, returncode=0, stderr="", stdout=""):
            self.rc, self.calls = (returncode, stderr, stdout), []

        xml_path = None
        xml_text = None

        def __call__(self, argv, **kw):
            self.calls.append(argv)
            if "/XML" in argv:              # the definition only exists during the call
                from pathlib import Path
                self.xml_path = Path(argv[argv.index("/XML") + 1])
                self.xml_text = self.xml_path.read_text(encoding="utf-16")
            return types.SimpleNamespace(returncode=self.rc[0], stderr=self.rc[1], stdout=self.rc[2])

    @staticmethod
    def _settings(xml_text):
        import xml.etree.ElementTree as ET
        ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
        root = ET.fromstring(xml_text.split("?>", 1)[1])

        def find(path):
            return root.find(path, ns).text
        return find

    def test_install_registers_an_xml_definition(self, tmp_path):
        run_ = self._Run()
        ok, message = service.install(tmp_path, 15, run=run_)
        argv = run_.calls[0]
        assert ok and argv[:2] == ["schtasks", "/Create"]
        assert argv[argv.index("/TN") + 1] == service.task_name(tmp_path) and "/F" in argv
        find = self._settings(run_.xml_text)
        assert find(".//t:Interval") == "PT15M"
        assert find(".//t:Arguments").count("--tick") == 1
        assert str(tmp_path.resolve()) in find(".//t:Arguments") and "runner.py" in find(".//t:Arguments")
        assert find(".//t:WorkingDirectory") == str(tmp_path.resolve())
        assert find(".//t:Command").lower().endswith(("python.exe", "pythonw.exe"))
        assert "15 мин" in message
        assert not run_.xml_path.exists()              # the temp file does not linger

    def test_it_runs_on_battery_and_catches_up_a_missed_tick(self, tmp_path):
        """A task made with schtasks flags refuses to start on battery — on a laptop,
        'never runs when you are away'."""
        run_ = self._Run()
        service.install(tmp_path, run=run_)
        find = self._settings(run_.xml_text)
        assert find(".//t:DisallowStartIfOnBatteries") == "false"
        assert find(".//t:StopIfGoingOnBatteries") == "false"
        assert find(".//t:StartWhenAvailable") == "true"
        assert find(".//t:MultipleInstancesPolicy") == "IgnoreNew"

    def test_paths_with_xml_special_characters_survive(self, tmp_path):
        from pathlib import Path
        odd = tmp_path / "a&b 'proj' é"
        odd.mkdir()
        run_ = self._Run()
        service.install(odd, run=run_)
        assert self._settings(run_.xml_text)(".//t:WorkingDirectory") == str(odd.resolve())
        # '<' cannot be in a real Windows path, but the XML must still be well-formed.
        xml = service.task_xml(Path("C:/x<y>&z"))
        assert "&lt;y&gt;&amp;z" in xml and self._settings(xml)(".//t:Interval") == "PT15M"

    def test_the_interval_is_clamped(self, tmp_path):
        run_ = self._Run()
        service.install(tmp_path, 1, run=run_)
        assert self._settings(run_.xml_text)(".//t:Interval") == "PT5M"

    def test_a_refusal_comes_back_as_a_message(self, tmp_path):
        ok, message = service.install(tmp_path, run=self._Run(1, stderr="Access is denied."))
        assert not ok and "Access is denied" in message

    def test_a_russian_error_from_schtasks_is_decoded_not_fatal(self, tmp_path):
        """schtasks speaks the OEM code page; reading it as UTF-8 killed a reader
        thread with a traceback in the middle of the screen."""
        class Run(self._Run):
            def __call__(self, argv, **kw):
                self.calls.append(argv)
                return types.SimpleNamespace(returncode=1, stdout=b"",
                                             stderr="ОШИБКА: Отказано в доступе.".encode("cp866"))

        ok, message = service.install(tmp_path, run=Run())
        assert not ok and "Отказано в доступе" in message

    def test_remove_and_status(self, tmp_path):
        assert service.remove(tmp_path, run=self._Run())[0] is True
        assert service.remove(tmp_path, run=self._Run(1))[0] is False
        assert service.status(tmp_path, run=self._Run())["installed"] is True
        assert service.status(tmp_path, run=self._Run(1))["installed"] is False

    def test_each_project_gets_its_own_entry(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(); b.mkdir()
        assert service.task_name(a) != service.task_name(b)

    def test_other_systems_get_the_cron_line_instead(self, tmp_path, monkeypatch):
        monkeypatch.setattr(service, "os", types.SimpleNamespace(name="posix"))
        run_ = self._Run()
        ok, message = service.install(tmp_path, 15, run=run_)
        assert not ok and "crontab" in message and "*/15" in message and run_.calls == []
        assert service.status(tmp_path)["supported"] is False

    def test_the_tick_entry_point_exists(self):
        assert service._RUNNER.is_file()


class TestNamelessAdd:
    def test_split_without_name(self):
        from src.automation.tasks_cli import split_add
        assert split_add("по будням в 9:00 | собери сводку") == ("", "по будням в 9:00", "собери сводку", [])
        assert split_add("отчёт | по будням в 9:00 | собери")[0] == "отчёт"

    def test_split_without_name_keeps_options(self):
        from src.automation.tasks_cli import split_add
        name, sched, task, opts = split_add("every 1h | проверь сайт | notify: always")
        assert (name, sched, task) == ("", "every 1h", "проверь сайт") and opts == [" notify: always"]

    def test_add_without_name_gets_a_readable_one(self, tmp_path, monkeypatch):
        monkeypatch.setattr("project_paths.project_root_or_cwd", lambda *a, **k: tmp_path)
        from src.automation import store, tasks_cli
        out, errs = [], []
        tasks_cli.handle("/tasks add по будням в 9:00 | Собери сводку по git за вчера", out.append, errs.append)
        assert not errs
        (a,) = store.load_automations()
        assert a.name.startswith("Собери") and a.schedule
        tasks_cli.handle("/tasks add по будням в 9:00 | Собери сводку по git за вчера", out.append, errs.append)
        assert len(store.load_automations()) == 2 and not errs

    def test_add_with_only_schedule_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("project_paths.project_root_or_cwd", lambda *a, **k: tmp_path)
        from src.automation import tasks_cli
        out, errs = [], []
        tasks_cli.handle("/tasks add по будням в 9:00", out.append, errs.append)
        assert errs
