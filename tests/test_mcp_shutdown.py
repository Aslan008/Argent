"""MCP stdio servers: shutting one down must never hang, and a dying or chatty
server must not stall the caller.

Regression: quitting Argent (and finishing a pytest run) hung for good whenever
a configured server was started through a launcher (`uv`, `npx`, `cmd`). The
launcher was terminated, the real server behind it kept the pipes open, and
`stop()` then closed the pipes while a reader thread was still blocked on them —
on Windows that close waits for the pending read, so it never returned. The
tests below use real child processes because the failure only exists with real
inherited pipe handles; a mock cannot hold a pipe open.
"""

import json
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import mcp_client
from src.agent import shell

psutil = pytest.importorskip("psutil")

# A minimal MCP server. `mode` picks the misbehaviour under test.
_SERVER = textwrap.dedent('''
    import json, os, subprocess, sys, time

    mode, pidfile, logfile = sys.argv[1], sys.argv[2], sys.argv[3]

    if "grandchild" in mode:
        # What a launcher does: start the real program and keep the pipes open
        # in it by simply inheriting stdout/stderr.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        with open(pidfile, "w") as f:
            f.write(str(child.pid))

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        msg = json.loads(line)
        method = msg.get("method")
        if mode == "ping" and method == "initialize":
            # Server-initiated request whose id collides with the client's.
            send({"jsonrpc": "2.0", "id": msg["id"], "method": "ping"})
            answer = json.loads(sys.stdin.readline())
            with open(logfile, "w") as f:
                f.write(json.dumps(answer))
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": msg["id"],
                  "result": {"serverInfo": {"name": "fake"}}})
        elif mode == "dies_on_request" and "id" in msg:
            os._exit(0)                       # accepted the request, never answers

    if "ignores_eof" in mode:
        time.sleep(300)                       # does not take the hint to exit
''')


@pytest.fixture
def start_server(tmp_path):
    started = []

    def _start(mode: str):
        script = tmp_path / "fake_mcp.py"
        script.write_text(_SERVER, encoding="utf-8")
        pidfile, logfile = tmp_path / "child.pid", tmp_path / "answer.json"
        transport = mcp_client.StdioTransport(
            sys.executable, ["-u", str(script), mode, str(pidfile), str(logfile)])
        assert transport.start() is True, f"fake server ({mode}) did not initialize"
        started.append(transport)
        return transport, pidfile, logfile

    yield _start
    for t in started:                       # never leave a process behind a failed test
        t.stop()


def _stop_returns_within(transport, seconds: float) -> bool:
    worker = threading.Thread(target=transport.stop, daemon=True)
    worker.start()
    worker.join(seconds)
    return not worker.is_alive()


def _child_pid(pidfile) -> int:
    deadline = time.time() + 5
    while time.time() < deadline:
        if pidfile.exists() and pidfile.read_text().strip():
            return int(pidfile.read_text())
        time.sleep(0.05)
    raise AssertionError("the fake server never reported its grandchild")


class TestStopNeverHangs:
    def test_a_grandchild_holding_the_pipes_does_not_hang_stop(self, start_server):
        """The launcher case, the one that hung forever."""
        transport, pidfile, _ = start_server("grandchild")
        grandchild = _child_pid(pidfile)
        assert psutil.pid_exists(grandchild)

        assert _stop_returns_within(transport, 20), "stop() hung with a grandchild alive"

        time.sleep(0.3)
        assert not psutil.pid_exists(grandchild), "the real server behind the launcher leaked"

    def test_a_server_that_ignores_eof_is_killed_with_its_tree(self, start_server):
        transport, pidfile, _ = start_server("grandchild_ignores_eof")
        grandchild = _child_pid(pidfile)

        assert _stop_returns_within(transport, 30), "stop() hung on a server that ignores EOF"

        time.sleep(0.3)
        assert not psutil.pid_exists(grandchild)

    def test_stop_twice_is_harmless(self, start_server):
        transport, _, _ = start_server("plain")
        assert _stop_returns_within(transport, 20)
        assert _stop_returns_within(transport, 5)
        assert transport._process is None

    def test_a_well_behaved_server_stops_quickly(self, start_server):
        """EOF on stdin is enough; nobody should wait out the kill timers."""
        transport, _, _ = start_server("plain")
        started = time.time()
        transport.stop()
        assert time.time() - started < 3


class TestDeadServerDoesNotStallTheCaller:
    def test_a_request_fails_at_once_when_the_server_exits(self, start_server):
        """The reader hits EOF; waiting out the 30 s timeout for an answer that
        can no longer arrive would stall the agent for no reason."""
        transport, _, _ = start_server("dies_on_request")
        started = time.time()
        reply = transport.send_request("tools/list", {}, timeout=30)
        assert "error" in reply
        assert time.time() - started < 10
        assert "closed the connection" in reply["error"]["message"]

    def test_stop_wakes_a_request_that_is_waiting(self, start_server):
        transport, _, _ = start_server("ignores_eof")
        result = {}

        def _call():
            result["reply"] = transport.send_request("tools/list", {}, timeout=30)

        caller = threading.Thread(target=_call, daemon=True)
        caller.start()
        time.sleep(0.5)
        transport.stop()
        caller.join(10)
        assert not caller.is_alive(), "the waiting request was left to run out its timeout"
        assert "error" in result["reply"]


class TestServerInitiatedMessages:
    def test_a_ping_is_answered_and_not_mistaken_for_our_response(self, start_server):
        """Both sides count ids from 1. The server's ping(id=1) arrives while our
        initialize(id=1) is pending; treating it as the response made start()
        fail, and a server that is never answered gives up on its client."""
        transport, _, logfile = start_server("ping")
        assert transport.is_running
        answer = json.loads(logfile.read_text())
        assert answer["id"] == 1
        assert answer["result"] == {}


class TestKillProcessTree:
    def test_an_exited_root_costs_no_os_fallback(self, monkeypatch):
        """Stopping a server that already exited cleanly must not spawn taskkill."""
        spawned = []
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: spawned.append(a))
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()

        shell.kill_process_tree(proc)

        assert spawned == []

    def test_known_descendants_are_killed_even_after_the_root_exited(self):
        """A launcher exits on EOF and leaves its child; the handles taken before
        that are the only way left to find it."""
        parent = subprocess.Popen([sys.executable, "-c", textwrap.dedent('''
            import subprocess, sys, time
            c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
            print(c.pid, flush=True)
            time.sleep(300)
        ''')], stdout=subprocess.PIPE, text=True)
        try:
            grandchild = int(parent.stdout.readline())
            descendants = shell.descendant_processes(parent)
            assert any(d.pid == grandchild for d in descendants)

            parent.kill()
            parent.wait()
            assert psutil.pid_exists(grandchild)

            shell.kill_process_tree(parent, descendants)

            time.sleep(0.3)
            assert not psutil.pid_exists(grandchild)
        finally:
            parent.stdout.close()
            for d in shell.descendant_processes(parent):
                d.kill()

    def test_a_process_sharing_our_group_is_never_group_killed(self, monkeypatch):
        """killpg on a child that shares Argent's own process group would take
        Argent down with it."""
        if sys.platform == "win32":
            pytest.skip("POSIX process groups")
        import os
        monkeypatch.setitem(sys.modules, "psutil", None)      # force the fallbacks
        killed_groups = []
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: killed_groups.append(pgid))
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            shell.kill_process_tree(proc)
            assert killed_groups == []
            assert proc.poll() is not None          # still killed, the plain way
        finally:
            proc.kill()
