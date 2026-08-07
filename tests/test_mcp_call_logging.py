"""MCP calls leave evidence, and stop freezing the terminal for ten minutes.

A call that hangs is the one thing you cannot debug afterwards: the terminal
shows a spinner, the server keeps no history, and the turn ends. When one
really did hang for 600 seconds, `grep -i mcp` over every log in ~/.argent/logs
returned nothing about it — the MCP logger had no handler at all.
"""

import logging

import pytest

import mcp_client
from mcp_client import SLOW_CALL_SECONDS, MCPServer


class _Transport:
    """Answers tools/call with whatever the test wants, after a fake delay."""

    def __init__(self, reply, delay=0.0):
        self.reply = reply
        self.delay = delay
        self.timeouts = []

    def send_request(self, method, params=None, timeout=30):
        self.timeouts.append(timeout)
        return self.reply


def _server(reply, delay=0.0):
    srv = MCPServer.__new__(MCPServer)
    srv.name = "unity"
    srv.transport = _Transport(reply, delay)
    return srv


@pytest.fixture
def clock(monkeypatch):
    """Virtual time so a 'slow call' test does not actually take minutes."""
    state = {"now": 0.0, "step": 0.0}
    monkeypatch.setattr(mcp_client.time, "monotonic",
                        lambda: state.__setitem__("now", state["now"] + state["step"])
                        or state["now"])
    return state


OK = {"result": {"content": [{"text": "{\"status\": \"ready\"}"}]}}


class TestTimeout:
    def test_the_default_is_not_ten_minutes(self):
        """A frozen terminal with no output and no way to cancel is never the
        right default; an editor silent for two minutes is stuck, not busy."""
        from config import get_mcp_call_timeout
        assert get_mcp_call_timeout() <= 300

    def test_configured_value_reaches_the_transport(self, monkeypatch):
        monkeypatch.setattr("config.get_mcp_call_timeout", lambda: 45.0)
        srv = _server(OK)
        srv.call_tool("editor_status", {})
        assert srv.transport.timeouts == [45.0]

    def test_absurd_values_are_clamped(self, monkeypatch, tmp_path):
        import config
        monkeypatch.setattr(config, "_get", lambda k, d=None: 0.1)
        assert config.get_mcp_call_timeout() == 5.0
        monkeypatch.setattr(config, "_get", lambda k, d=None: "nonsense")
        assert config.get_mcp_call_timeout() == 120.0


class TestTimeoutMessage:
    def test_the_model_is_told_not_to_immediately_retry(self):
        """The request is still in flight server-side; re-sending queues a
        second copy behind the first and makes the stall worse."""
        srv = _server({"error": {"message": "MCP server request timed out after 120 seconds"}})
        out = srv.call_tool("get_quality_settings", {})
        assert "do NOT immediately repeat" in out
        assert "get_quality_settings" in out and "unity" in out

    def test_other_errors_pass_through_unchanged(self):
        srv = _server({"error": {"message": "Command Not Found"}})
        assert srv.call_tool("nope", {}) == "MCP Error: Command Not Found"


class TestLogging:
    def test_the_logger_actually_writes_somewhere(self):
        """logging.getLogger('argent.mcp') had no handler — every MCP event went
        nowhere, which is why a real 600s hang left no trace to diagnose."""
        assert mcp_client.log.handlers
        assert any(isinstance(h, logging.FileHandler) for h in mcp_client.log.handlers)

    def test_a_call_is_recorded_before_and_after(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=mcp_client.log.name):
            _server(OK).call_tool("editor_status", {"a": 1})
        text = caplog.text
        assert "unity.editor_status ->" in text      # proof it was ever sent
        assert "unity.editor_status ok in" in text

    def test_a_failure_is_recorded_with_its_duration(self, caplog):
        srv = _server({"error": {"message": "boom"}})
        with caplog.at_level(logging.DEBUG, logger=mcp_client.log.name):
            srv.call_tool("editor_status", {})
        assert "FAILED after" in caplog.text and "boom" in caplog.text

    def test_a_slow_but_successful_call_is_flagged(self, caplog, clock):
        clock["step"] = SLOW_CALL_SECONDS + 1
        with caplog.at_level(logging.DEBUG, logger=mcp_client.log.name):
            _server(OK).call_tool("bake_navmesh", {})
        assert "the server is slow to respond" in caplog.text

    def test_a_fast_call_is_not_flagged(self, caplog, clock):
        clock["step"] = 0.01
        with caplog.at_level(logging.DEBUG, logger=mcp_client.log.name):
            _server(OK).call_tool("editor_status", {})
        assert "slow to respond" not in caplog.text
