from types import SimpleNamespace

import pytest

import tools.command_ops as co


class FakeProc:
    def __init__(self, ret=None):
        self._ret = ret

    def poll(self):
        return self._ret


@pytest.fixture(autouse=True)
def clean_registry():
    with co.ACTIVE_PROCESSES_LOCK:
        co.ACTIVE_PROCESSES.clear()
    yield
    with co.ACTIVE_PROCESSES_LOCK:
        co.ACTIVE_PROCESSES.clear()


def register(pid, command, ret=None):
    with co.ACTIVE_PROCESSES_LOCK:
        co.ACTIVE_PROCESSES[pid] = {
            "process": FakeProc(ret), "out_queue": None,
            "err_queue": None, "command": command,
        }


class TestListBackgroundCommands:
    def test_empty(self):
        assert "No background processes" in co.list_background_commands()

    def test_running_process_listed(self):
        register("1", "uvicorn app:app", ret=None)
        out = co.list_background_commands()
        assert "PID 1" in out
        assert "RUNNING" in out
        assert "uvicorn app:app" in out

    def test_exited_process_shows_code(self):
        register("2", "npm test", ret=1)
        out = co.list_background_commands()
        assert "PID 2" in out
        assert "EXITED (code 1)" in out

    def test_multiple_processes(self):
        register("1", "server", ret=None)
        register("2", "watcher", ret=0)
        out = co.list_background_commands()
        assert "PID 1" in out and "PID 2" in out

    def test_is_read_only(self):
        # Listing must not reap the registry — that's stop/read's job.
        register("1", "server", ret=0)
        co.list_background_commands()
        with co.ACTIVE_PROCESSES_LOCK:
            assert "1" in co.ACTIVE_PROCESSES

    def test_mentions_companion_tools(self):
        register("1", "server")
        out = co.list_background_commands()
        assert "stop_background_command" in out


class TestRegistration:
    def test_tool_registered_everywhere(self):
        import tools
        from tools.schemas import AVAILABLE_TOOLS, TOOL_SCHEMAS
        assert "list_background_commands" in AVAILABLE_TOOLS
        assert hasattr(tools, "list_background_commands")
        names = [s["function"]["name"] for s in TOOL_SCHEMAS]
        assert "list_background_commands" in names
