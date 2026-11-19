"""Данные панели фоновых задач: хвост вывода, который переживает очередь, список
работающих, который опрашивает TUI, и список завершённых, который печатает кнопка."""

import json
import queue
import time
from types import SimpleNamespace

import pytest

import argent.tools.command_ops as command_ops


class _FakeProc:
    def __init__(self, code=None):
        self._code = code

    def poll(self):
        return self._code


@pytest.fixture(autouse=True)
def clean_registries():
    command_ops.ACTIVE_PROCESSES.clear()
    command_ops.COMPLETED_PROCESSES.clear()
    yield
    command_ops.ACTIVE_PROCESSES.clear()
    command_ops.COMPLETED_PROCESSES.clear()


def _register(pid, code=None, started=None, command="build", with_tail=True):
    info = {
        "process": _FakeProc(code),
        "out_queue": queue.Queue(),
        "err_queue": queue.Queue(),
        "command": command,
        "started": started if started is not None else time.time(),
        "pty": False,
    }
    if with_tail:
        info["tail"] = command_ops._OutputTail()
    command_ops.ACTIVE_PROCESSES[pid] = info
    return info


def _tail(text, limit=command_ops.OUTPUT_TAIL_BYTES):
    tail = command_ops._OutputTail(limit=limit)
    tail.add(text.encode())
    return tail


# ── the buffer ──────────────────────────────────────────────────────────────

def test_the_tail_keeps_the_end_of_a_chatty_process():
    tail = command_ops._OutputTail(limit=10)
    tail.add(b"0123456789")
    tail.add(b"ABCDE")
    assert tail.text() == "56789ABCDE"


def test_a_single_chunk_bigger_than_the_limit_keeps_its_end():
    tail = command_ops._OutputTail(limit=10)
    tail.add(b"x" * 5000)
    assert tail.text() == "x" * 10


def test_the_tail_is_what_survives_the_queue_being_drained():
    """read_background_command empties the queue; the panel needs a second copy."""
    info = _register("t1", command="echo hi")
    info["tail"].add(b"hello\n")
    info["out_queue"].put(b"hello\n")

    first = command_ops.read_background_command("t1")
    assert "hello" in first

    second = command_ops.read_background_command("t1")
    assert "hello" not in second, "queue уже пуста"
    assert "hello" in command_ops.background_output("t1")


def test_a_registry_entry_without_a_tail_still_works():
    """Old callers build the entry by hand; a missing buffer must not raise."""
    _register("t2", with_tail=False, command="legacy")
    assert command_ops.background_output("t2").endswith("(вывод не сохранён)")


# ── the state the TUI polls ─────────────────────────────────────────────────

def test_the_payload_lists_running_tasks_only_and_counts_the_finished():
    from src.bridge.repl_bridge import ReplBridge

    _register("11", started=time.time() - 42, command="npm run dev")
    command_ops.COMPLETED_PROCESSES["10"] = {"exit_code": 0, "command": "echo hi", "output": "hi"}

    bridge = ReplBridge()
    state = bridge.get_state()

    tasks = json.loads(state["background"])
    assert [t["pid"] for t in tasks] == ["11"], "завершённые в панель не едут"
    assert tasks[0]["command"] == "npm run dev"
    assert 41 <= tasks[0]["elapsed"] <= 45
    assert state["background_done"] == "1"


def test_the_payload_names_the_finished_ones_for_the_nested_fold():
    """The panel lists them by name; the text itself comes from /jobs <pid>."""
    from src.bridge.repl_bridge import ReplBridge

    command_ops.COMPLETED_PROCESSES["5"] = {
        "exit_code": 0, "command": "npm run build", "started": 100.0, "finished": 231.0, "output": "x" * 5000}

    state = ReplBridge().get_state()
    finished = json.loads(state["background_finished"])

    assert [f["pid"] for f in finished] == ["5"]
    assert finished[0]["code"] == 0
    assert finished[0]["secs"] == 131
    assert "output" not in finished[0], "вывод не едет в опрос состояния"


def test_the_payload_is_empty_when_nothing_runs():
    from src.bridge.repl_bridge import ReplBridge

    state = ReplBridge().get_state()
    assert json.loads(state["background"]) == []
    assert json.loads(state["background_finished"]) == []
    assert state["background_done"] == "0"


def test_a_process_that_exited_between_turns_leaves_the_running_list():
    """The bg_done event is drained only during a turn, so a finished process used
    to sit in the panel as "9:29" and keep counting while the agent was idle."""
    from src.bridge.repl_bridge import ReplBridge

    info = _register("21", code=0, started=time.time() - 600, command="python -c sleep(240)")
    info["tail"].add(b"done\n")

    state = ReplBridge().get_state()

    assert json.loads(state["background"]) == [], "вышедший процесс не висит в работающих"
    assert state["background_done"] == "1"
    finished = json.loads(state["background_finished"])
    assert [f["pid"] for f in finished] == ["21"]
    assert finished[0]["code"] == 0
    assert finished[0]["secs"] >= 599, "время посчитано от старта, а не от начала опроса"
    assert "done" in command_ops.background_output("21"), "хвост вывода переехал вместе с ним"


def test_a_running_process_stays_with_its_real_elapsed_time():
    from src.bridge.repl_bridge import ReplBridge

    _register("22", code=None, started=time.time() - 7, command="ping -n 300 127.0.0.1")

    state = ReplBridge().get_state()

    tasks = json.loads(state["background"])
    assert [t["pid"] for t in tasks] == ["22"]
    assert 6 <= tasks[0]["elapsed"] <= 9
    assert state["background_done"] == "0"


def test_an_entry_without_a_process_object_does_not_break_the_poll():
    """A hand-built registry entry (an older caller, a pty wrapper without poll)
    must stay in the running list instead of raising inside the state snapshot."""
    from src.bridge.repl_bridge import ReplBridge

    command_ops.ACTIVE_PROCESSES["23"] = {"command": "legacy", "started": time.time()}

    state = ReplBridge().get_state()

    assert [t["pid"] for t in json.loads(state["background"])] == ["23"]


# ── the finished list and its output ────────────────────────────────────────

def test_the_finished_list_names_codes_and_the_way_to_the_output():
    command_ops.COMPLETED_PROCESSES["1"] = {
        "exit_code": 0, "command": "npm run build", "started": 100.0, "finished": 231.0, "output": ""}
    command_ops.COMPLETED_PROCESSES["2"] = {
        "exit_code": 1, "command": "pytest", "started": 200.0, "finished": 202.0, "output": ""}

    out = command_ops.completed_background_commands()

    assert "Завершённые фоновые задачи (2)" in out
    assert "✓ 0" in out and "✗ 1" in out
    # Minutes only from a minute up, the same shape the CLI and the panel use.
    assert "npm run build" in out and "2:11" in out and "· 2s" in out
    assert "/jobs <pid>" in out


def test_the_finished_list_says_the_rest():
    for i in range(15):
        command_ops.COMPLETED_PROCESSES[str(i)] = {"exit_code": 0, "command": f"job {i}"}
    out = command_ops.completed_background_commands(limit=10)
    assert "… ещё 5" in out


def test_only_the_newest_finished_processes_keep_their_output():
    for i in range(command_ops.TAILS_KEPT + 2):
        command_ops._record_completed(
            str(i), {"command": f"job {i}", "tail": _tail(f"out {i}")}, 0)

    with_output = [pid for pid, info in command_ops.COMPLETED_PROCESSES.items() if info["output"]]
    assert len(with_output) == command_ops.TAILS_KEPT
    assert with_output[-1] == str(command_ops.TAILS_KEPT + 1), "хвост есть у самых свежих"


def test_a_finished_process_still_answers_with_its_output():
    command_ops._record_completed(
        "55", {"command": "pytest", "tail": _tail("1 failed\n", limit=64)}, 1)
    out = command_ops.background_output("55")
    assert "завершён, код 1" in out
    assert "1 failed" in out


def test_a_stopped_process_says_it_was_stopped():
    command_ops._record_completed("56", {"command": "sleep 100", "tail": _tail("")}, 1, stopped=True)
    assert "остановлен" in command_ops.background_output("56")


def test_the_record_keeps_fifty_entries_at_most():
    for i in range(command_ops.COMPLETED_KEPT + 5):
        command_ops._record_completed(str(i), {"command": f"job {i}"}, 0)
    assert len(command_ops.COMPLETED_PROCESSES) == command_ops.COMPLETED_KEPT


# ── the slash command behind the panel's button ────────────────────────────

def test_slash_jobs_routes_the_finished_list(monkeypatch):
    import src.cli.commands.config_commands as config_commands

    command_ops.COMPLETED_PROCESSES["9"] = {"exit_code": 0, "command": "echo done"}
    printed = []
    monkeypatch.setattr(config_commands, "print_system", printed.append)

    config_commands.handle_jobs(SimpleNamespace(user_input="/jobs done"))

    assert printed and "Завершённые фоновые задачи" in printed[0]


def test_slash_jobs_routes_a_pid_to_its_output(monkeypatch):
    import src.cli.commands.config_commands as config_commands

    _register("77", command="npm run dev")
    command_ops.ACTIVE_PROCESSES["77"]["tail"].add(b"ready on 3000\n")
    printed = []
    monkeypatch.setattr(config_commands, "print_system", printed.append)

    config_commands.handle_jobs(SimpleNamespace(user_input="/jobs 77"))

    assert printed and "ready on 3000" in printed[0]


def test_the_jobs_entry_matches_its_subcommands():
    from src.cli.commands import registry

    entry = registry._by_name["/jobs"]
    assert entry.matches("/jobs")
    assert entry.matches("/jobs done")
    assert entry.matches("/jobs 42")
