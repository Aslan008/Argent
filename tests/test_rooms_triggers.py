"""Triggers: a schedule or a webhook starts a room with nobody at the keyboard."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from src.rooms import headless, triggers, webhooks
from src.rooms.runner import NodeRunner
from src.rooms.session import build_library

TOOLS = {"run_command", "read_file"}


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project directory of its own: `.argent/` (rooms, automations, webhooks) lands in it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARGENT_DISABLE_AUTOMATIONS", "0")
    (tmp_path / ".argent" / "rooms").mkdir(parents=True)
    return tmp_path


def put_room(project, name, nodes, edges, **extra):
    room = {"room": name, "description": f"{name} room", "budget": {"max_iterations": 20, "max_tokens": 5000},
            "nodes": nodes, "edges": edges, **extra}
    (project / ".argent" / "rooms" / f"{name}.json").write_text(json.dumps(room), encoding="utf-8")


def library(project):
    return build_library(available_tools=TOOLS, user_dir=project / ".argent" / "rooms",
                         stats_file=project / ".argent" / "stats.json")


@pytest.fixture
def simple_room(project):
    put_room(project, "report", [
        {"id": "s", "type": "set", "values": {"seen": "{{who}}"}},
    ], [{"from": "s", "to": "EXIT:done"}], defaults={"who": "nobody"}, next={"done": "END"})
    return "report"


class TestHeadless:
    def test_runs_a_room_and_reports_how_it_ended(self, project, simple_room):
        out = headless.run_headless("report", "", {"who": "Anna"}, library=library(project),
                                    runner=NodeRunner(), record_dir=project / "rec")
        assert out["status"] == "ok" and out["exit"] == "done" and not out["needs_person"]
        assert "report:done" in out["summary"]

    def test_unknown_room_is_an_error_not_a_crash(self, project):
        out = headless.run_headless("nowhere", library=library(project), runner=NodeRunner())
        assert out["status"] == "error" and "no room" in out["summary"]

    def test_a_bad_name_is_refused_before_anything_runs(self, project):
        assert headless.run_headless("../x")["status"] == "error"

    def test_a_question_ends_the_run_as_needs_a_person(self, project):
        put_room(project, "asks", [{"id": "q", "type": "human_pause", "prompt": "Go on?", "into": "answer"}],
                 [{"from": "q", "to": "EXIT:done"}])
        out = headless.run_headless("asks", library=library(project), runner=NodeRunner(), record_dir=project / "rec")
        assert out["needs_person"] is True and out["status"] == "denied"

    def test_a_step_that_asks_first_is_not_run(self, project):
        put_room(project, "careful", [
            {"id": "danger", "type": "tool", "tool": "run_command", "args": {"command": "echo hi"},
             "confirm": True, "idempotent": False}],
            [{"from": "danger", "to": "EXIT:done"}])
        ran = []
        from src.rooms.session import build_runner
        runner = build_runner(None, library=library(project), confirm=headless._never, allow_spawn=False)
        runner._tool = lambda n, s: ran.append(n.id) or "x"
        out = headless.run_headless("careful", library=library(project), runner=runner, record_dir=project / "rec")
        assert ran == [] and out["declined"] == ["danger"]

    def test_split_task(self):
        assert headless.split_task('night_run look closely k=v p="two words"') == (
            "night_run", "look closely", {"k": "v", "p": "two words"})

    def test_child_entry_prints_the_verdict_line(self, project, simple_room, capsys):
        code = headless.main(["report"])
        assert code == 0
        line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(headless.RESULT_MARKER)][0]
        assert json.loads(line[len(headless.RESULT_MARKER):])["room"] == "report"


class TestSchedule:
    def test_a_room_schedule_is_an_ordinary_automation(self, project, simple_room):
        made = triggers.add_schedule("report", "every day at 03:00", task="be quick", params={"who": "Anna Bell"})
        assert made["kind"] == "schedule" and made["room"] == "report"
        assert made["params"] == {"who": "Anna Bell"} and made["task"] == "be quick"
        from src.automation.store import get_automation
        a = get_automation(made["name"])
        assert a.mode == "room" and a.task.startswith("report ")

    def test_second_schedule_gets_its_own_name(self, project, simple_room):
        a = triggers.add_schedule("report", "every day at 03:00")
        b = triggers.add_schedule("report", "every hour")
        assert a["name"] != b["name"]
        assert {t["name"] for t in triggers.list_triggers("report")} == {a["name"], b["name"]}

    def test_a_bad_schedule_is_refused_with_the_reason(self, project, simple_room):
        with pytest.raises(ValueError):
            triggers.add_schedule("report", "whenever the moon is full")

    def test_remove_and_switch(self, project, simple_room):
        made = triggers.add_schedule("report", "every day at 03:00")
        assert triggers.set_enabled("schedule", made["name"], False)
        assert triggers.list_triggers("report")[0]["enabled"] is False
        assert triggers.remove("schedule", made["name"])
        assert triggers.list_triggers("report") == []

    def test_the_automation_runner_runs_the_room(self, project, simple_room, monkeypatch):
        seen = []
        monkeypatch.setattr(headless, "run_headless", lambda room, text="", params=None, **kw: (
            seen.append((room, text, params)) or
            {"status": "ok", "summary": "all fine", "declined": [], "needs_person": False}))
        made = triggers.add_schedule("report", "every day at 03:00", params={"who": "Anna"})
        from src.automation.runner import run_automation
        from src.automation.store import get_automation
        result = run_automation(get_automation(made["name"]))
        assert result["status"] == "ok" and result["summary"] == "all fine" and result["mode"] == "room"
        assert seen == [("report", "", {"who": "Anna"})]

    def test_a_room_that_needs_a_person_is_reported_as_such(self, project, simple_room, monkeypatch):
        monkeypatch.setattr(headless, "run_headless", lambda *a, **k: {
            "status": "denied", "summary": "asked", "declined": [], "needs_person": True})
        made = triggers.add_schedule("report", "every day at 03:00")
        from src.automation.runner import run_automation
        from src.automation.store import get_automation
        result = run_automation(get_automation(made["name"]))
        assert result["status"] == "denied" and "ответ человека" in result["summary"]


class TestWebhookStore:
    def test_secret_is_shown_once_and_only_a_hash_is_kept(self, project):
        hook, secret = webhooks.add_hook("report", "nightly", params=["who"])
        raw = (project / ".argent" / "webhooks.json").read_text(encoding="utf-8")
        assert secret not in raw and "secret_hash" in raw
        assert hook["name"] == "nightly" and hook["params"] == ["who"]

    def test_names_and_parameters_are_checked(self, project):
        with pytest.raises(ValueError):
            webhooks.add_hook("report", "bad name!")
        with pytest.raises(ValueError):
            webhooks.add_hook("report", "ok", params=["_steps"])
        webhooks.add_hook("report", "ok")
        with pytest.raises(ValueError):
            webhooks.add_hook("report", "ok")

    def test_a_new_secret_replaces_the_old_one(self, project):
        _, old = webhooks.add_hook("report", "h")
        new = webhooks.new_secret("h")
        assert new and new != old


class TestWebhookRequests:
    @pytest.fixture
    def server(self, project, simple_room):
        calls = []
        gate = threading.Event()
        gate.set()

        def runner(room, task, params):
            gate.wait(5)
            calls.append((room, task, params))
            return {"status": "ok", "exit": "done", "summary": "did it", "room": room}
        hook, secret = webhooks.add_hook("report", "nightly", params=["who"])
        srv = webhooks.WebhookServer(port_number=0, runner=runner, library=library(project)).start()
        srv.secret, srv.calls, srv.gate = secret, calls, gate
        yield srv
        gate.set()
        srv.join(5)
        srv.stop()

    def post(self, srv, path="/hook/nightly", body=None, secret=True, headers=None, raw=None):
        data = raw if raw is not None else json.dumps(body or {}).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{srv.port}{path}", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if secret is True:
            req.add_header("X-Argent-Secret", srv.secret)
        elif secret:
            req.add_header("X-Argent-Secret", secret)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode("utf-8"))

    def test_a_good_request_starts_the_room_with_its_parameters(self, server):
        status, payload = self.post(server, body={"who": "Anna", "task": "go"})
        assert status == 202 and payload["status"] == "started"
        server.join(5)
        assert server.calls == [("report", "go", {"who": "Anna"})]

    def test_wait_returns_the_outcome(self, server):
        status, payload = self.post(server, path="/hook/nightly?wait=10", body={"who": "Anna"})
        assert status == 200 and payload["ok"] and payload["exit"] == "done" and payload["summary"] == "did it"

    def test_wrong_secret_and_unknown_hook_look_the_same(self, server):
        a = self.post(server, secret="nope")
        b = self.post(server, path="/hook/missing")
        assert a[0] == 404 and b[0] == 404 and a[1] == b[1]
        assert server.calls == []

    def test_wrong_secrets_are_slowed_down(self, server):
        codes = [self.post(server, secret="nope")[0] for _ in range(webhooks.MAX_BAD_SECRETS + 1)]
        assert codes[-1] == 429
        assert self.post(server)[0] == 429          # even the right secret waits for the window

    def test_only_declared_parameters_come_through(self, server):
        status, payload = self.post(server, body={"who": "Anna", "_steps": 99, "other": "x"})
        assert status == 400 and "_steps" in payload["error"] and "other" in payload["error"]
        assert server.calls == []

    def test_values_must_be_plain_and_short(self, server):
        assert self.post(server, body={"who": {"a": 1}})[0] == 400
        assert self.post(server, body={"who": "x" * (webhooks.MAX_VALUE_CHARS + 1)})[0] == 400

    def test_bad_json_and_a_big_body(self, server):
        assert self.post(server, raw=b"{not json")[0] == 400
        assert self.post(server, raw=b"[1,2]")[0] == 400
        assert self.post(server, raw=b" " * (webhooks.MAX_BODY_BYTES + 10))[0] == 413

    def test_one_run_per_hook_at_a_time(self, server):
        server.gate.clear()
        first = self.post(server)
        second = self.post(server)
        server.gate.set()
        assert first[0] == 202 and second[0] == 429

    def test_a_switched_off_hook_does_not_run(self, server):
        webhooks.set_enabled("nightly", False)
        assert self.post(server)[0] == 409 and server.calls == []

    def test_a_foreign_host_header_is_refused(self, server):
        status, _ = self.post(server, headers={"Host": "evil.example:80"})
        assert status == 403

    def test_a_trusted_host_is_let_in(self, server, project):
        data = json.loads((project / ".argent" / "webhooks.json").read_text(encoding="utf-8"))
        data["trusted_hosts"] = ["hooks.example"]
        (project / ".argent" / "webhooks.json").write_text(json.dumps(data), encoding="utf-8")
        assert self.post(server, headers={"Host": "hooks.example"})[0] == 202

    def test_get_is_not_a_way_in(self, server):
        req = urllib.request.Request(f"http://127.0.0.1:{server.port}/hook/nightly", method="GET")
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req, timeout=5)
        assert err.value.code == 405

    def test_a_failing_run_does_not_take_the_listener_down(self, project, simple_room):
        _, secret = webhooks.add_hook("report", "boom")

        def runner(room, task, params):
            raise RuntimeError("kaboom")
        srv = webhooks.WebhookServer(port_number=0, runner=runner, library=library(project)).start()
        try:
            srv.secret = secret
            status, payload = self.post(srv, path="/hook/boom?wait=5")
            assert status == 500 and "kaboom" in payload["summary"]
            assert self.post(srv, path="/hook/boom?wait=5")[0] == 500      # still answering
        finally:
            srv.stop()

    def test_last_run_is_remembered(self, server):
        self.post(server, path="/hook/nightly?wait=10")
        hook = webhooks.list_hooks("report")[0]
        assert hook["last_status"] == "ok" and hook["last_summary"] == "did it"


class TestFacade:
    def test_lists_both_kinds(self, project, simple_room):
        triggers.add_schedule("report", "every day at 03:00")
        trigger, secret = triggers.add_webhook("report")
        assert secret and trigger["url"].endswith("/hook/report")
        assert {t["kind"] for t in triggers.list_triggers("report")} == {"schedule", "webhook"}
        assert triggers.list_triggers("other") == []

    def test_remove_checks_the_kind(self, project, simple_room):
        made = triggers.add_schedule("report", "every day at 03:00")
        assert triggers.remove("webhook", made["name"]) is False      # not a webhook
        assert triggers.remove("schedule", made["name"]) is True
        with pytest.raises(ValueError):
            triggers.remove("carrier pigeon", "x")

    def test_a_non_room_automation_is_not_touched(self, project, simple_room):
        from src.automation.store import Automation, upsert_automation
        upsert_automation(Automation(name="backup", task="echo hi", schedule="every day at 03:00", mode="command"))
        assert triggers.remove("schedule", "backup") is False
        assert triggers.list_triggers() == []


class TestEditorEndpoints:
    @pytest.fixture
    def editor(self, project, simple_room):
        from src.rooms.editor_server import EditorServer
        srv = EditorServer(lambda: library(project), lambda: TOOLS, project / ".argent" / "rooms")
        return srv

    def test_add_list_remove(self, editor):
        status, made = editor.api_trigger({"action": "add_schedule", "room": "report", "schedule": "every hour"})
        assert status == 200 and made["trigger"]["kind"] == "schedule"
        status, listed = editor.api_triggers("report")
        assert [t["name"] for t in listed["triggers"]] == [made["trigger"]["name"]]
        status, gone = editor.api_trigger({"action": "remove", "kind": "schedule", "name": made["trigger"]["name"]})
        assert gone["ok"] is True

    def test_webhook_secret_comes_back_once(self, editor, monkeypatch):
        monkeypatch.setattr(webhooks, "start_if_configured", lambda: None)
        status, made = editor.api_trigger({"action": "add_webhook", "room": "report", "name": "h1", "params": ["who"]})
        assert status == 200 and made["secret"]
        _, listed = editor.api_triggers("report")
        assert "secret" not in json.dumps(listed)

    def test_unknown_room_and_action(self, editor):
        assert editor.api_trigger({"action": "add_schedule", "room": "ghost", "schedule": "every hour"})[0] == 404
        assert editor.api_trigger({"action": "dance"})[0] == 400
        status, bad = editor.api_trigger({"action": "add_schedule", "room": "report", "schedule": "never o'clock"})
        assert status == 422 and bad["errors"]
