"""MCP stdio transport shutdown: reader threads must not stay parked."""

from unittest.mock import MagicMock

import mcp_client


class _FakePipe:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _transport_with_process():
    t = mcp_client.StdioTransport.__new__(mcp_client.StdioTransport)
    proc = MagicMock()
    proc.stdout, proc.stderr, proc.stdin = _FakePipe(), _FakePipe(), _FakePipe()
    t._process = proc
    t._initialized = True
    t._lock = __import__("threading").Lock()
    t._pending_requests = {"1": object()}
    return t, proc


class TestStopClosesPipes:
    def test_pipes_are_closed(self):
        """The readers block in readline(); terminating the child usually wakes
        them, but not when a grandchild inherited the handle. Closing the pipes
        guarantees readline() returns instead of parking a thread per restart."""
        t, proc = _transport_with_process()
        t.stop()
        assert proc.stdout.closed and proc.stderr.closed and proc.stdin.closed

    def test_process_is_terminated_and_cleared(self):
        t, proc = _transport_with_process()
        t.stop()
        proc.terminate.assert_called_once()
        assert t._process is None
        assert t._initialized is False
        assert t._pending_requests == {}

    def test_kill_fallback_still_closes_pipes(self):
        t, proc = _transport_with_process()
        proc.wait.side_effect = Exception("did not exit")
        t.stop()
        proc.kill.assert_called_once()
        assert proc.stdout.closed and proc.stderr.closed

    def test_close_failure_is_survivable(self):
        t, proc = _transport_with_process()
        proc.stdout.close = MagicMock(side_effect=OSError("already closed"))
        t.stop()                      # must not raise
        assert t._process is None

    def test_stop_without_process_is_a_no_op(self):
        t = mcp_client.StdioTransport.__new__(mcp_client.StdioTransport)
        t._process = None
        t._initialized = True
        t._lock = __import__("threading").Lock()
        t._pending_requests = {}
        t.stop()
        assert t._initialized is False


def _server_with_transport(reply):
    srv = mcp_client.MCPServer.__new__(mcp_client.MCPServer)
    srv.name = "srv"
    srv._tools_cache = []
    srv._tools_unavailable = False
    srv.transport = MagicMock()
    srv.transport.send_request.return_value = reply
    return srv


class TestToolListingIsAskedOnce:
    """list_tools() runs once per turn while the system prompt is built, so a
    server that cannot answer must not be re-asked: each miss cost a 10s
    timeout and flipped the prompt text, churning the provider's prefix cache."""

    def test_success_is_cached(self):
        srv = _server_with_transport({"result": {"tools": [{"name": "t"}]}})
        assert srv.list_tools() == [{"name": "t"}]
        assert srv.list_tools() == [{"name": "t"}]
        assert srv.transport.send_request.call_count == 1

    def test_failure_is_cached_too(self):
        srv = _server_with_transport({"error": {"message": "boom"}})
        assert srv.list_tools() == []
        assert srv.list_tools() == []
        assert srv.transport.send_request.call_count == 1

    def test_restart_clears_the_failure(self):
        srv = _server_with_transport({"error": {"message": "boom"}})
        srv.list_tools()
        srv.stop()
        assert srv._tools_unavailable is False
