"""A worker's dialogue as it happens, and a retry that continues the worker instead of
hiring a new one.

The child here is a stand-in: `SubagentManager` takes an argv builder, so the test runs a
tiny Python process that prints what a worker would print and then its verdict marker.
"""

import json
import sys

from src.agent import subagents
from src.agent.subagent_worker import load_history, save_history

MARKER = subagents.RESULT_MARKER


def _child(lines, status="ok", report="done"):
    payload = json.dumps({"status": status, "report": report, "denied": []})
    body = ["import json, sys"]
    body += [f"print({line!r}, flush=True)" for line in lines]
    body += [f"print({MARKER!r} + {payload!r}, flush=True)"]
    return [sys.executable, "-c", ";".join(body)]


def manager(lines, **kwargs):
    return subagents.SubagentManager(argv_builder=lambda spec: _child(lines, **kwargs))


# ── the dialogue, line by line ───────────────────────────────────────────────

def test_every_line_a_worker_prints_reaches_the_supervisor(tmp_path):
    seen = []
    mgr = manager(["reading config.py", "running pytest -q"])
    mgr.on_line = lambda run, line: seen.append(line)

    run = mgr.spawn("coder", "do it", isolation="none", cwd=tmp_path)
    assert mgr.wait(run, timeout=60)

    assert seen == ["reading config.py", "running pytest -q"]
    assert run.status == "done"
    assert run.report == "done"


def test_the_verdict_marker_is_not_shown_as_dialogue(tmp_path):
    seen = []
    mgr = manager(["one line"])
    mgr.on_line = lambda run, line: seen.append(line)

    run = mgr.spawn("coder", "do it", isolation="none", cwd=tmp_path)
    mgr.wait(run, timeout=60)

    assert not [line for line in seen if MARKER in line]


def test_a_broken_screen_hook_does_not_kill_the_worker(tmp_path):
    mgr = manager(["still working"])

    def boom(run, line):
        raise RuntimeError("the screen is gone")

    mgr.on_line = boom
    run = mgr.spawn("coder", "do it", isolation="none", cwd=tmp_path)

    assert mgr.wait(run, timeout=60)
    assert run.status == "done", "экран не имеет права валить работу"


def test_the_monitor_sees_the_newest_line_while_the_run_goes(tmp_path):
    mgr = manager(["first", "second"])
    run = mgr.spawn("coder", "do it", isolation="none", cwd=tmp_path)
    mgr.wait(run, timeout=60)

    assert "second" in run.log_tail


# ── continuing a worker ──────────────────────────────────────────────────────

def test_a_retry_is_told_where_the_previous_conversation_is(tmp_path):
    seen_spec = {}

    def build(spec_path):
        seen_spec.update(json.loads(spec_path.read_text(encoding="utf-8")))
        return _child(["continuing"])

    mgr = subagents.SubagentManager(argv_builder=build)
    path = tmp_path / "s1.json"

    run = mgr.spawn("coder", "task", isolation="none", cwd=tmp_path, resume_from=path)
    assert mgr.wait(run, timeout=60)

    assert seen_spec["resume_from"] == str(path)
    assert run.resume_from == str(path)


def test_a_first_attempt_is_not_told_to_resume_anything(tmp_path):
    seen_spec = {}

    def build(spec_path):
        seen_spec.update(json.loads(spec_path.read_text(encoding="utf-8")))
        return _child(["fresh"])

    mgr = subagents.SubagentManager(argv_builder=build)
    run = mgr.spawn("coder", "task", isolation="none", cwd=tmp_path)
    mgr.wait(run, timeout=60)

    assert seen_spec["resume_from"] == ""
    assert run.resume_from is None


def test_history_keeps_the_role_and_drops_the_old_system_prompt(tmp_path):
    path = tmp_path / "h.json"
    path.write_text(json.dumps([
        {"role": "system", "content": "OLD ROLE"},
        {"role": "user", "content": "Start task: x"},
        {"role": "assistant", "content": "I read a.py"},
    ]), encoding="utf-8")
    system = {"role": "system", "content": "NEW ROLE"}

    loaded = load_history(str(path), system)

    assert loaded[0] is system, "роль берётся нынешняя, а не сохранённая"
    assert [m["content"] for m in loaded[1:]] == ["Start task: x", "I read a.py"]


def test_no_saved_conversation_means_a_clean_start(tmp_path):
    system = {"role": "system", "content": "ROLE"}
    assert load_history("", system) == [system]

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_history(str(broken), system) == [system]


def test_a_conversation_survives_the_round_trip(tmp_path):
    path = tmp_path / "nested" / "s1.json"
    messages = [{"role": "system", "content": "R"},
                {"role": "user", "content": "Start task: x"},
                {"role": "assistant", "content": "I changed x.py"}]

    save_history(str(path), messages)

    assert json.loads(path.read_text(encoding="utf-8")) == messages
    assert load_history(str(path), messages[0]) == messages
