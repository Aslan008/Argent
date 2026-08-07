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
    srv._tools_cache = []            # nothing known yet, so preflight stands aside
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

    def test_a_tool_level_error_is_logged_as_one(self, caplog):
        """It arrives inside a successful JSON-RPC response, so logging on the
        transport outcome alone recorded 'ok in 0.1s' for calls the server had
        flatly rejected — a run of bad-argument errors left a healthy-looking
        log."""
        srv = _server({"result": {"isError": True,
                                  "content": [{"text": "400 Bad Request"}]}})
        with caplog.at_level(logging.DEBUG, logger=mcp_client.log.name):
            srv.call_tool("find_assets", {})
        assert "tool error" in caplog.text
        assert "ok in" not in caplog.text


# The real Unity schema, trimmed. Note there is no `required` list: the rule
# "at least one of type/name/label" lives in the server, not in the schema.
FIND_ASSETS = {
    "name": "find_assets",
    "description": "Find assets by type and/or name and/or label. Returns paths.",
    "inputSchema": {"type": "object", "properties": {
        "type": {}, "name": {}, "label": {}, "search_in": {}, "limit": {}}},
}
BAKE = {
    "name": "bake_navmesh",
    "description": "Bake the navmesh.",
    "inputSchema": {"type": "object",
                    "properties": {"confirm": {}, "dry_run": {}},
                    "required": ["confirm"]},
}


def _with_tools(tools, reply=OK):
    srv = _server(reply)
    srv._tools_cache = tools
    return srv


class TestPreflight:
    def test_invented_parameter_names_are_answered_with_the_signature(self):
        """Observed for real: the model sent filter/search/max_results when the
        parameters are type/name/label/search_in/limit. Signatures are not in
        the prompt, so a model that skips list_mcp_tools invents plausible ones."""
        srv = _with_tools([FIND_ASSETS])
        out = srv.call_tool("find_assets", {"filter": "t:Prefab", "search": "",
                                            "max_results": 200})
        assert "unknown parameter(s): filter, max_results, search" in out
        assert "find_assets(type?, name?, label?, search_in?, limit?)" in out
        assert srv.transport.timeouts == []          # never left the machine

    def test_a_missing_required_argument_is_caught(self):
        out = _with_tools([BAKE]).call_tool("bake_navmesh", {"dry_run": True})
        assert "missing required: confirm" in out

    def test_a_correct_call_is_sent_untouched(self):
        srv = _with_tools([FIND_ASSETS])
        srv.call_tool("find_assets", {"type": "Scene", "limit": 5})
        assert len(srv.transport.timeouts) == 1

    def test_an_unknown_tool_name_gets_near_matches(self):
        out = _with_tools([FIND_ASSETS, BAKE]).call_tool("find_asset", {"type": "x"})
        assert "has no tool 'find_asset'" in out and "find_assets" in out
        assert "list_mcp_tools" in out

    def test_nothing_is_blocked_before_the_tool_list_is_known(self):
        """A server whose listing failed must still be callable; refusing every
        call because we cannot check it would be worse than the typo."""
        srv = _server(OK)
        srv._tools_cache = []
        srv.call_tool("anything", {"whatever": 1})
        assert len(srv.transport.timeouts) == 1

    def test_a_loose_schema_is_left_to_the_server(self):
        """Plenty of servers accept more than they describe; blocking a call
        that would have worked is worse than the failure this prevents."""
        srv = _with_tools([{"name": "eval", "inputSchema": {}}])
        srv.call_tool("eval", {"code": "1+1"})
        assert len(srv.transport.timeouts) == 1


class TestServerSideExplanation:
    def test_a_validation_error_comes_back_with_the_signature(self):
        """Preflight cannot know 'at least one of type, name or label' — that
        rule is in the server, not the schema."""
        srv = _with_tools([FIND_ASSETS], reply={"result": {
            "isError": True,
            "content": [{"text": "Parameter Validation Failed. At least one of "
                                 "type, name, or label is required."}]}})
        out = srv.call_tool("find_assets", {})
        assert "Signature: find_assets(type?" in out

    def test_an_unrelated_error_is_not_padded(self):
        srv = _with_tools([FIND_ASSETS], reply={"result": {
            "isError": True, "content": [{"text": "Editor is compiling"}]}})
        out = srv.call_tool("find_assets", {"type": "Scene"})
        assert "Signature:" not in out
