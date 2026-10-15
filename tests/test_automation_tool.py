"""manage_automations: the agent sets up scheduled work, the person approves it,
and a model never gets to widen its own permissions.
"""

import types

import pytest

import approval
from src.automation import store
from src.automation.runner import UnattendedGate
from src.automation.store import Automation
from tools import automation_tools
from tools.automation_tools import manage_automations as tool


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("project_paths.project_root_or_cwd", lambda *a, **k: tmp_path)
    return tmp_path


@pytest.fixture
def asked(monkeypatch):
    """Records every approval the tool asks for; answers from ``asked.answer``."""
    box = types.SimpleNamespace(calls=[], answer=True)

    def fake(text, destructive):
        box.calls.append((text, destructive))
        return box.answer

    monkeypatch.setattr(automation_tools, "_ask", fake)
    return box


def _a(name="job", **kw):
    kw.setdefault("task", "t")
    kw.setdefault("schedule", "every 1h")
    return Automation(name=name, **kw)


class TestReading:
    def test_reading_never_asks(self, project, monkeypatch):
        monkeypatch.setattr(automation_tools, "_ask", lambda *a: pytest.fail("reading needs no approval"))
        store.upsert_automation(_a())
        store.record_run("job", "ok", "all fine", store.datetime.now(), store.datetime.now())
        assert "job" in tool("list")
        assert "задача: t" in tool("show", name="job")
        assert "all fine" in tool("runs", name="job")
        assert "morning" in tool("templates")

    def test_empty(self, project):
        assert tool("list") == "Автоматизаций нет."
        assert tool("runs") == "Прогонов ещё не было."

    @pytest.mark.parametrize("call", [dict(action="show"), dict(action="update"), dict(action="remove"),
                                      dict(action="enable"), dict(action="run")])
    def test_a_name_is_required(self, project, asked, call):
        assert tool(**call) == "Error: name is required."

    def test_unknown_action(self, project):
        assert tool("explode").startswith("Error: action must be one of")

    def test_unknown_name(self, project, asked):
        assert "не найдена" in tool("show", name="ghost")
        assert "не найдена" in tool("remove", name="ghost")
        assert asked.calls == []


class TestCreate:
    def test_the_person_sees_what_they_approve(self, project, asked):
        out = tool("create", name="digest", schedule="по будням в 9:00", task="Summarise PRs",
                   tools=["read_webpage"], notify="always")
        text, destructive = asked.calls[0]
        assert "«digest»" in text and "по будням в 09:00" in text and "Summarise PRs" in text
        assert "read_webpage" in text and destructive is False
        assert out.startswith("Создана «digest»")
        assert store.get_automation("digest").schedule == "weekdays at 09:00"

    def test_a_refusal_creates_nothing(self, project, asked):
        asked.answer = False
        assert tool("create", name="x", schedule="every 1h", task="t").startswith("Aborted")
        assert store.load_automations() == []

    def test_pre_approved_commands_are_spelled_out_and_asked_as_a_big_deal(self, project, asked):
        tool("create", name="g", schedule="every 1h", task="check", tools=["run_command"],
             commands=["git log*", "npm test"])
        text, destructive = asked.calls[0]
        assert "git log*; npm test" in text and "БУДЕТ РАЗРЕШЕНО" in text
        assert destructive is True

    def test_a_shell_command_job_is_a_big_deal_too(self, project, asked):
        tool("create", name="c", schedule="daily at 03:00", task="robocopy a b", mode="command")
        text, destructive = asked.calls[0]
        assert "ВЫПОЛНЯТЬ КОМАНДУ" in text and "robocopy a b" in text and destructive is True

    def test_a_reminder_is_not(self, project, asked):
        tool("create", name="r", schedule="in 2h", task="tea", mode="remind")
        assert asked.calls[0][1] is False
        assert store.get_automation("r").schedule.startswith("once at ")

    def test_from_a_template(self, project, asked):
        out = tool("create", template="releases", template_arg="astral-sh/uv")
        a = store.load_automations()[0]
        assert a.name.startswith("releases") and "astral-sh/uv" in a.task and "создана" in out.lower()

    def test_explicit_fields_win_over_the_template(self, project, asked):
        tool("create", template="tests", template_arg="pytest -q", name="mine", schedule="every 30m")
        a = store.get_automation("mine")
        assert a.schedule == "every 30m" and a.task == "pytest -q" and a.mode == "command"

    def test_problems_come_back_as_an_error_before_anyone_is_asked(self, project, asked):
        assert "sometimes" in tool("create", name="x", schedule="sometimes", task="t")
        assert "serch_web" in tool("create", name="x", schedule="every 1h", task="t", tools=["serch_web"])
        assert "unknown template" in tool("create", template="nope", name="x")
        assert "нужен аргумент" in tool("create", template="page", name="x")
        assert "name is required" in tool("create", schedule="every 1h", task="t")
        assert asked.calls == [] and store.load_automations() == []

    def test_duplicates(self, project, asked):
        store.upsert_automation(_a())
        assert "already exists" in tool("create", name="job", schedule="every 1h", task="t")


class TestUpdateRemoveToggle:
    def test_update_shows_the_result_not_the_diff(self, project, asked):
        store.upsert_automation(_a(allowed_tools=["run_command"]))
        out = tool("update", name="job", commands=["git log*"])
        text, destructive = asked.calls[0]
        assert "git log*" in text and destructive is True
        assert store.get_automation("job").allowed_commands == ["git log*"] and out.startswith("Обновлена")

    def test_a_plain_update_is_a_plain_approval(self, project, asked):
        store.upsert_automation(_a())
        tool("update", name="job", schedule="every 2h")
        assert asked.calls[0][1] is False and store.get_automation("job").schedule == "every 2h"

    def test_denied_update_changes_nothing(self, project, asked):
        store.upsert_automation(_a())
        asked.answer = False
        assert tool("update", name="job", schedule="every 2h").startswith("Aborted")
        assert store.get_automation("job").schedule == "every 1h"

    def test_update_needs_something_to_change(self, project, asked):
        store.upsert_automation(_a())
        assert "nothing to change" in tool("update", name="job")

    def test_remove_is_always_a_big_ask(self, project, asked):
        store.upsert_automation(_a())
        asked.answer = False
        assert tool("remove", name="job").startswith("Aborted")
        assert store.get_automation("job") is not None
        asked.answer = True
        assert "Удалена" in tool("remove", name="job") and asked.calls[-1][1] is True
        assert store.get_automation("job") is None

    def test_enable_disable(self, project, asked):
        store.upsert_automation(_a())
        assert "выключена" in tool("disable", name="job")
        assert store.get_automation("job").enabled is False
        assert "включена" in tool("enable", name="job")
        assert [c[1] for c in asked.calls] == [False, False]

    def test_a_prefix_finds_the_job(self, project, asked):
        store.upsert_automation(_a("weekly-digest"))
        assert "выключена" in tool("disable", name="weekly")


class TestRun:
    def test_goes_through_the_live_scheduler(self, project, asked, monkeypatch):
        store.upsert_automation(_a())
        started = []
        live = types.SimpleNamespace(run_now=lambda name: started.append(name) or (True, ""))
        monkeypatch.setattr("src.automation.scheduler.get_scheduler", lambda: live)
        assert "в фоне" in tool("run", name="job") and started == ["job"]
        assert asked.calls[0][1] is False

    def test_running_a_command_job_is_a_big_ask(self, project, asked, monkeypatch):
        store.upsert_automation(_a(mode="command", task="echo hi"))
        monkeypatch.setattr("src.automation.scheduler.get_scheduler",
                            lambda: types.SimpleNamespace(run_now=lambda n: (True, "")))
        tool("run", name="job")
        assert asked.calls[0][1] is True

    def test_without_a_scheduler_it_runs_in_a_thread(self, project, asked, monkeypatch):
        import threading
        store.upsert_automation(_a())
        monkeypatch.setattr("src.automation.scheduler.get_scheduler", lambda: None)
        done = threading.Event()

        class Stub:
            def __call__(self, automation, trigger="schedule"):
                done.set()
                return {"status": "ok"}

        monkeypatch.setattr("src.automation.runner.SubprocessRunner", Stub)
        assert "в фоне" in tool("run", name="job")
        assert done.wait(5)

    def test_refused_run(self, project, asked):
        store.upsert_automation(_a())
        asked.answer = False
        assert tool("run", name="job").startswith("Aborted")

    def test_a_job_already_running_is_reported(self, project, asked, monkeypatch):
        store.upsert_automation(_a())
        monkeypatch.setattr("src.automation.scheduler.get_scheduler",
                            lambda: types.SimpleNamespace(run_now=lambda n: (False, "'job' уже выполняется.")))
        assert tool("run", name="job").startswith("Error:")


class TestNoSelfPromotion:
    def test_an_unattended_run_cannot_change_automations(self, project, monkeypatch):
        monkeypatch.setattr(automation_tools, "_ask", lambda *a: pytest.fail("never reaches a prompt"))
        approval.set_approval_backend(UnattendedGate())
        try:
            for call in (dict(action="create", name="x", schedule="every 1h", task="t"),
                         dict(action="remove", name="x"), dict(action="run", name="x")):
                assert "unattended" in tool(**call)
            assert tool("list") == "Автоматизаций нет."        # looking is fine
        finally:
            approval.reset_approval_backend()
        assert store.load_automations() == []

    def test_vibe_mode_cannot_grant_standing_commands(self, project):
        """In /vibe nobody is asked: a destructive approval is refused outright, so
        the model cannot hand itself pre-approved commands there."""
        approval.set_policy(approval.POLICY_AUTO)
        try:
            refused = tool("create", name="g", schedule="every 1h", task="t",
                           tools=["run_command"], commands=["git log*"])
            plain = tool("create", name="p", schedule="every 1h", task="t", tools=["read_file"])
        finally:
            approval.set_policy(approval.POLICY_ASK)
        assert refused.startswith("Aborted") and store.get_automation("g") is None
        assert plain.startswith("Создана") and store.get_automation("p") is not None

    def test_vibe_mode_cannot_schedule_a_shell_command_either(self, project):
        approval.set_policy(approval.POLICY_AUTO)
        try:
            out = tool("create", name="c", schedule="daily at 03:00", task="echo hi", mode="command")
        finally:
            approval.set_policy(approval.POLICY_ASK)
        assert out.startswith("Aborted")

    def test_the_tool_is_on_the_unattended_denylist(self):
        from src.automation.runner import UNATTENDED_TOOL_DENYLIST
        assert "manage_automations" in UNATTENDED_TOOL_DENYLIST


class TestRegistered:
    def test_the_schema_and_the_function_exist(self):
        from tools.schemas import AVAILABLE_TOOLS, TOOL_SCHEMAS
        schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "manage_automations")
        assert schema["function"]["parameters"]["required"] == ["action"]
        assert "manage_automations" in AVAILABLE_TOOLS

    def test_every_schema_parameter_is_a_real_argument(self):
        import inspect
        from tools.schemas import TOOL_SCHEMAS
        schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "manage_automations")
        accepted = set(inspect.signature(tool).parameters)
        assert set(schema["function"]["parameters"]["properties"]) <= accepted

    def test_the_description_names_every_template(self):
        from src.automation import templates
        from tools.schemas import TOOL_SCHEMAS
        schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "manage_automations")
        for template_id in templates.TEMPLATES:
            assert template_id in schema["function"]["description"]
