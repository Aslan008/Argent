"""Blind-spot tests for mcp_client.py — pure functions and classes with mocked
transports, no real subprocess servers, no real HTTP calls."""

import json
import queue
import subprocess
import threading
from unittest.mock import MagicMock, patch

import pytest

import mcp_client
from mcp_client import (
    _signature,
    _closest_names,
    _ARGUMENT_ERROR_CUES,
    MCPServer,
    MCPClient,
    MCPTransportType,
    StdioTransport,
    RESTTransport,
)


# ════════════════════════════════════════════════════════════════
# 1. _signature
# ════════════════════════════════════════════════════════════════

class TestSignature:
    def test_required_and_optional_with_description(self):
        """name(a, b?) — first sentence of description."""
        tool = {
            "name": "name",
            "description": "Does something. More info.",
            "inputSchema": {
                "properties": {"a": {}, "b": {}},
                "required": ["a"],
            },
        }
        assert _signature(tool) == "name(a, b?) — Does something"

    def test_no_schema(self):
        """No inputSchema → name()."""
        tool = {"name": "name"}
        assert _signature(tool) == "name()"

    def test_no_description(self):
        """Properties present but no description → name(params) without — suffix."""
        tool = {
            "name": "name",
            "inputSchema": {
                "properties": {"x": {}, "y": {}},
                "required": ["x", "y"],
            },
        }
        # No " — " appended when description is empty
        result = _signature(tool)
        assert result == "name(x, y)"
        assert "—" not in result

    def test_empty_properties(self):
        """Schema with empty properties → name()."""
        tool = {
            "name": "name",
            "inputSchema": {"properties": {}, "required": []},
        }
        assert _signature(tool) == "name()"

    def test_all_optional(self):
        """All params optional → each gets ?."""
        tool = {
            "name": "fn",
            "inputSchema": {"properties": {"a": {}, "b": {}}},
        }
        assert _signature(tool) == "fn(a?, b?)"

    def test_description_truncated_at_first_sentence(self):
        """Only the first sentence (up to first '.') is used."""
        tool = {
            "name": "fn",
            "description": "First sentence. Second sentence.",
            "inputSchema": {"properties": {}},
        }
        assert _signature(tool) == "fn() — First sentence"

    def test_parameters_alias(self):
        """'parameters' key works as a fallback for 'inputSchema'."""
        tool = {
            "name": "fn",
            "description": "Desc.",
            "parameters": {"properties": {"a": {}}, "required": ["a"]},
        }
        assert _signature(tool) == "fn(a) — Desc"

    def test_missing_name(self):
        """No name key → '?'."""
        tool = {"inputSchema": {"properties": {}}}
        assert _signature(tool).startswith("?(")

    def test_long_description_truncated(self):
        """Description longer than 140 chars is truncated."""
        long_desc = "A" * 200 + ". trailing."
        tool = {"name": "fn", "description": long_desc}
        result = _signature(tool)
        # split(".")[0] gives the whole 200-char string (no dot in it)
        # then [:140] truncates
        desc_part = result.split(" — ", 1)[1]
        assert len(desc_part) <= 140


# ════════════════════════════════════════════════════════════════
# 2. _closest_names
# ════════════════════════════════════════════════════════════════

class TestClosestNames:
    def test_close_match(self):
        """'read_fil' matches 'read_file' as the closest result."""
        result = _closest_names("read_fil", ["read_file", "write_file", "list_dir"])
        assert result[0] == "read_file"
        assert "read_file" in result

    def test_no_match(self):
        """'xyz' matches nothing."""
        result = _closest_names("xyz", ["read_file", "write_file", "list_dir"])
        assert result == []

    def test_empty_candidates(self):
        """Empty candidate list → empty result."""
        assert _closest_names("anything", []) == []

    def test_limit_parameter(self):
        """limit controls max results."""
        result = _closest_names("read_file", ["read_file", "read_fil", "readfile"], limit=1)
        assert len(result) <= 1

    def test_filters_out_empty_strings(self):
        """Empty strings in candidates are ignored."""
        result = _closest_names("read_file", ["read_file", "", None])
        assert result == ["read_file"]


# ════════════════════════════════════════════════════════════════
# 3. _ARGUMENT_ERROR_CUES
# ════════════════════════════════════════════════════════════════

class TestArgumentErrorCues:
    def test_contains_parameter_validation(self):
        assert "parameter validation" in _ARGUMENT_ERROR_CUES

    def test_contains_invalid_parameter(self):
        assert "invalid parameter" in _ARGUMENT_ERROR_CUES

    def test_contains_is_required(self):
        assert "is required" in _ARGUMENT_ERROR_CUES

    def test_contains_missing_required(self):
        assert "missing required" in _ARGUMENT_ERROR_CUES

    def test_is_iterable(self):
        """Must be iterable so any() can scan it."""
        assert len(list(_ARGUMENT_ERROR_CUES)) >= 4


# ════════════════════════════════════════════════════════════════
# 4. MCPTransportType enum
# ════════════════════════════════════════════════════════════════

class TestMCPTransportType:
    def test_has_stdio(self):
        assert MCPTransportType.STDIO == "stdio"

    def test_has_sse(self):
        assert MCPTransportType.SSE == "sse"

    def test_has_standard(self):
        assert MCPTransportType.STANDARD == "standard"

    def test_has_unity_bridge(self):
        assert MCPTransportType.UNITY_BRIDGE == "unity_bridge"

    def test_str_enum_equality(self):
        """str enum: MCPTransportType.STDIO == 'stdio' is True."""
        assert MCPTransportType.STDIO == "stdio"
        assert MCPTransportType("stdio") is MCPTransportType.STDIO

    def test_all_values(self):
        values = {t.value for t in MCPTransportType}
        assert values == {"stdio", "sse", "standard", "unity_bridge"}


# ════════════════════════════════════════════════════════════════
# 5. MCPServer._explain
# ════════════════════════════════════════════════════════════════

def _make_server(tools=None):
    """Build an MCPServer without calling __init__ side effects."""
    srv = MCPServer.__new__(MCPServer)
    srv.name = "test"
    srv.config = {"type": "stdio", "command": "fake"}
    srv.transport = None
    srv._tools_cache = tools or []
    srv._tools_unavailable = False
    return srv


class TestMCPServerExplain:
    def test_explain_with_argument_error_cue(self):
        """When the message contains an argument-error cue, the signature is
        appended."""
        srv = _make_server([
            {
                "name": "my_tool",
                "inputSchema": {"properties": {"a": {}}, "required": ["a"]},
                "description": "Does thing.",
            }
        ])
        result = srv._explain("my_tool", "parameter validation failed")
        assert "Signature:" in result
        assert "my_tool(a)" in result

    def test_explain_with_invalid_parameter_cue(self):
        srv = _make_server([
            {"name": "t", "inputSchema": {"properties": {"x": {}}}, "description": "Desc."}
        ])
        result = srv._explain("t", "invalid parameter: z")
        assert "Signature:" in result

    def test_explain_with_is_required_cue(self):
        srv = _make_server([
            {"name": "t", "inputSchema": {"properties": {"x": {}}, "required": ["x"]}}
        ])
        result = srv._explain("t", "x is required")
        assert "Signature:" in result

    def test_explain_no_cue_returns_original(self):
        """Message without any cue → returned unchanged."""
        srv = _make_server([{"name": "t"}])
        msg = "some random network error"
        assert srv._explain("t", msg) == msg

    def test_explain_tool_not_found(self):
        """Cue present but tool not in cache → original message."""
        srv = _make_server([])
        msg = "invalid parameter: z"
        assert srv._explain("nonexistent", msg) == msg

    def test_explain_empty_message(self):
        """Empty message, no cue → returns empty string."""
        srv = _make_server([{"name": "t"}])
        assert srv._explain("t", "") == ""

    def test_explain_no_args_tool(self):
        """Tool with no args still gets signature appended."""
        srv = _make_server([
            {"name": "noop", "inputSchema": {"properties": {}}, "description": "No args."}
        ])
        result = srv._explain("noop", "missing required: x")
        assert "Signature:" in result
        assert "noop()" in result

    def test_explain_optional_args_tool(self):
        """Tool with optional args gets ? markers in signature."""
        srv = _make_server([
            {"name": "opt", "inputSchema": {"properties": {"a": {}, "b": {}}}, "description": "Opt."}
        ])
        result = srv._explain("opt", "invalid arguments")
        assert "opt(a?, b?)" in result


# ════════════════════════════════════════════════════════════════
# 6. MCPServer.preflight
# ════════════════════════════════════════════════════════════════

class TestMCPServerPreflight:
    def test_nonexistent_tool_with_close_match(self):
        """Wrong tool name → error with 'Did you mean' hint."""
        srv = _make_server([
            {"name": "read_file", "inputSchema": {"properties": {}}},
            {"name": "write_file", "inputSchema": {"properties": {}}},
            {"name": "list_dir", "inputSchema": {"properties": {}}},
        ])
        result = srv.preflight("read_fil", {})
        assert result is not None
        assert "read_file" in result
        assert "Did you mean" in result

    def test_nonexistent_tool_no_close_match(self):
        """Wrong tool name, no close match → error without hint."""
        srv = _make_server([
            {"name": "read_file"},
            {"name": "write_file"},
        ])
        result = srv.preflight("xyz", {})
        assert result is not None
        assert "no tool" in result
        assert "Did you mean" not in result

    def test_missing_required_arguments(self):
        """Missing required arg → error mentioning 'missing required'."""
        srv = _make_server([
            {
                "name": "my_tool",
                "inputSchema": {"properties": {"a": {}, "b": {}}, "required": ["a"]},
            }
        ])
        result = srv.preflight("my_tool", {})
        assert result is not None
        assert "missing required" in result
        assert "a" in result

    def test_unknown_arguments(self):
        """Unknown arg → error mentioning 'unknown parameter'."""
        srv = _make_server([
            {
                "name": "my_tool",
                "inputSchema": {"properties": {"a": {}}, "required": ["a"]},
            }
        ])
        result = srv.preflight("my_tool", {"a": 1, "z": 2})
        assert result is not None
        assert "unknown parameter" in result
        assert "z" in result

    def test_valid_call_returns_none(self):
        """Correct args → None (no error)."""
        srv = _make_server([
            {
                "name": "my_tool",
                "inputSchema": {"properties": {"a": {}, "b": {}}, "required": ["a"]},
            }
        ])
        assert srv.preflight("my_tool", {"a": 1}) is None

    def test_valid_call_with_optional_only(self):
        """All optional, providing some → None."""
        srv = _make_server([
            {"name": "my_tool", "inputSchema": {"properties": {"a": {}, "b": {}}}},
        ])
        assert srv.preflight("my_tool", {"a": 1}) is None

    def test_empty_cache_returns_none(self):
        """No tools cached → None (nothing to validate against)."""
        srv = _make_server([])
        assert srv.preflight("anything", {}) is None

    def test_loose_schema_returns_none(self):
        """Tool with no inputSchema → None (let server judge)."""
        srv = _make_server([{"name": "my_tool"}])
        assert srv.preflight("my_tool", {"x": 1}) is None

    def test_empty_properties_returns_none(self):
        """Tool with empty properties → None."""
        srv = _make_server([
            {"name": "my_tool", "inputSchema": {"properties": {}}},
        ])
        assert srv.preflight("my_tool", {"x": 1}) is None

    def test_non_dict_arguments_returns_none(self):
        """Non-dict arguments → None."""
        srv = _make_server([
            {"name": "my_tool", "inputSchema": {"properties": {"a": {}}, "required": ["a"]}},
        ])
        assert srv.preflight("my_tool", None) is None

    def test_error_includes_correct_signature(self):
        """The preflight error includes the _signature() output."""
        srv = _make_server([
            {
                "name": "my_tool",
                "inputSchema": {"properties": {"a": {}, "b": {}}, "required": ["a"]},
                "description": "Does thing.",
            }
        ])
        result = srv.preflight("my_tool", {})
        assert "my_tool(a, b?)" in result

    def test_both_unknown_and_missing(self):
        """Both unknown and missing → error mentions both."""
        srv = _make_server([
            {
                "name": "my_tool",
                "inputSchema": {"properties": {"a": {}}, "required": ["a"]},
            }
        ])
        result = srv.preflight("my_tool", {"z": 1})
        assert "unknown parameter" in result
        assert "missing required" in result


# ════════════════════════════════════════════════════════════════
# 7. MCPServer.call_tool
# ════════════════════════════════════════════════════════════════

class TestMCPServerCallTool:
    @pytest.fixture(autouse=True)
    def _mock_timeout(self, monkeypatch):
        """call_tool() imports config.get_mcp_call_timeout at call time."""
        monkeypatch.setattr("config.get_mcp_call_timeout", lambda: 30)

    def _server_with_mock_transport(self, send_request_return, tools=None):
        srv = _make_server(tools or [
            {"name": "my_tool", "inputSchema": {"properties": {"a": {}}, "required": ["a"]}}
        ])
        srv.transport = MagicMock()
        srv.transport.send_request.return_value = send_request_return
        srv.transport.is_running = True
        return srv

    def test_successful_call(self):
        srv = self._server_with_mock_transport({
            "result": {"content": [{"type": "text", "text": "success!"}], "isError": False}
        })
        result = srv.call_tool("my_tool", {"a": 1})
        assert "success!" in result
        srv.transport.send_request.assert_called_once()

    def test_error_response(self):
        srv = self._server_with_mock_transport({
            "error": {"message": "something went wrong"}
        })
        result = srv.call_tool("my_tool", {"a": 1})
        assert "MCP Error" in result
        assert "something went wrong" in result

    def test_preflight_error_before_transport(self):
        """Missing required arg → preflight rejects, transport never called."""
        srv = self._server_with_mock_transport({"result": {"content": []}})
        result = srv.call_tool("my_tool", {})
        assert "MCP Error" in result
        assert "missing required" in result
        srv.transport.send_request.assert_not_called()

    def test_not_started(self):
        """No transport → error about server not started."""
        srv = _make_server([{"name": "my_tool"}])
        srv.transport = None
        result = srv.call_tool("my_tool", {})
        assert "not started" in result

    def test_tool_error_response(self):
        """isError=True in result → MCP Tool Error prefix."""
        srv = self._server_with_mock_transport({
            "result": {"content": [{"type": "text", "text": "tool failed"}], "isError": True}
        })
        result = srv.call_tool("my_tool", {"a": 1})
        assert "MCP Tool Error" in result
        assert "tool failed" in result

    def test_timeout_message(self):
        """Timeout error → special 'did not answer' message."""
        srv = self._server_with_mock_transport({
            "error": {"message": "MCP server request timed out after 30 seconds"}
        })
        result = srv.call_tool("my_tool", {"a": 1})
        assert "did not answer" in result
        assert "do NOT" in result

    def test_non_text_content(self):
        """Content items without 'text' key are JSON-serialized."""
        srv = self._server_with_mock_transport({
            "result": {"content": [{"type": "image", "data": "abc"}], "isError": False}
        })
        result = srv.call_tool("my_tool", {"a": 1})
        assert "abc" in result

    def test_string_content_item(self):
        """Non-dict content items are stringified."""
        srv = self._server_with_mock_transport({
            "result": {"content": ["raw string"], "isError": False}
        })
        result = srv.call_tool("my_tool", {"a": 1})
        assert "raw string" in result

    def test_empty_content(self):
        """Empty content list → empty string output."""
        srv = self._server_with_mock_transport({
            "result": {"content": [], "isError": False}
        })
        result = srv.call_tool("my_tool", {"a": 1})
        assert result == ""

    def test_error_with_argument_cue_appends_signature(self):
        """Error message with argument cue → signature appended."""
        srv = self._server_with_mock_transport(
            {"error": {"message": "invalid parameter: z"}},
            tools=[{
                "name": "my_tool",
                "inputSchema": {"properties": {"a": {}}, "required": ["a"]},
                "description": "Does thing.",
            }],
        )
        result = srv.call_tool("my_tool", {"a": 1})
        assert "Signature:" in result
        assert "my_tool(a)" in result

    def test_send_request_called_with_correct_params(self):
        """Verify the JSON-RPC params passed to transport."""
        srv = self._server_with_mock_transport(
            {
                "result": {"content": [{"type": "text", "text": "ok"}], "isError": False}
            },
            tools=[{
                "name": "my_tool",
                "inputSchema": {"properties": {"a": {}, "b": {}}, "required": ["a"]},
            }],
        )
        srv.call_tool("my_tool", {"a": 1, "b": 2})
        call_args = srv.transport.send_request.call_args
        assert call_args[0][0] == "tools/call"
        params = call_args[0][1]
        assert params["name"] == "my_tool"
        assert params["arguments"] == {"a": 1, "b": 2}


# ════════════════════════════════════════════════════════════════
# 8. MCPClient
# ════════════════════════════════════════════════════════════════

def _mock_rest_response(tools=None):
    """Build a mock requests response for _list_tools."""
    resp = MagicMock()
    resp.json.return_value = {"tools": tools or [{"name": "tool1"}]}
    resp.raise_for_status = MagicMock()
    return resp


class TestMCPClient:
    def test_register_rest_adds_server(self, monkeypatch):
        """register_rest creates and starts a REST server."""
        client = MCPClient()
        monkeypatch.setattr("mcp_client.requests.get", MagicMock(return_value=_mock_rest_response()))
        result = client.register_rest("my_server", "http://localhost:8080", "standard")
        assert "Started" in result
        assert "my_server" in client.servers

    def test_register_rest_loads_tools(self, monkeypatch):
        """register_rest reports the number of tools loaded."""
        client = MCPClient()
        monkeypatch.setattr(
            "mcp_client.requests.get",
            MagicMock(return_value=_mock_rest_response([{"name": "a"}, {"name": "b"}])),
        )
        result = client.register_rest("srv", "http://localhost:8080", "standard")
        assert "2 tools" in result

    def test_unregister_server_removes_it(self, monkeypatch):
        """unregister_server stops and removes the server."""
        client = MCPClient()
        monkeypatch.setattr("mcp_client.requests.get", MagicMock(return_value=_mock_rest_response()))
        client.register_rest("my_server", "http://localhost:8080", "standard")
        result = client.unregister_server("my_server")
        assert "Stopped" in result
        assert "my_server" not in client.servers

    def test_unregister_nonexistent(self):
        """Unregistering a server that doesn't exist → 'not found'."""
        client = MCPClient()
        result = client.unregister_server("nonexistent")
        assert "not found" in result

    def test_get_servers_returns_list(self, monkeypatch):
        """get_servers returns a list of server info dicts."""
        client = MCPClient()
        monkeypatch.setattr("mcp_client.requests.get", MagicMock(return_value=_mock_rest_response()))
        client.register_rest("srv1", "http://localhost:8080", "standard")
        client.register_rest("srv2", "http://localhost:8081", "standard")
        servers = client.get_servers()
        assert isinstance(servers, list)
        assert len(servers) == 2
        names = {s["name"] for s in servers}
        assert names == {"srv1", "srv2"}

    def test_get_servers_empty(self):
        """No servers registered → empty list."""
        client = MCPClient()
        assert client.get_servers() == []

    def test_stop_all_stops_every_server(self, monkeypatch):
        """stop_all clears the servers dict."""
        client = MCPClient()
        monkeypatch.setattr("mcp_client.requests.get", MagicMock(return_value=_mock_rest_response()))
        client.register_rest("srv1", "http://localhost:8080", "standard")
        client.register_rest("srv2", "http://localhost:8081", "standard")
        client.stop_all()
        assert len(client.servers) == 0

    def test_list_tools_for_registered_server(self, monkeypatch):
        """list_tools(name) returns that server's tools."""
        client = MCPClient()
        monkeypatch.setattr(
            "mcp_client.requests.get",
            MagicMock(return_value=_mock_rest_response([{"name": "tool_a"}, {"name": "tool_b"}])),
        )
        client.register_rest("srv1", "http://localhost:8080", "standard")
        tools = client.list_tools("srv1")
        assert isinstance(tools, list)
        assert len(tools) == 2
        assert tools[0]["name"] == "tool_a"

    def test_list_tools_aggregates_across_servers(self, monkeypatch):
        """Tools from all registered servers are accessible via list_tools."""
        client = MCPClient()
        monkeypatch.setattr(
            "mcp_client.requests.get",
            MagicMock(return_value=_mock_rest_response([{"name": "tool_a"}])),
        )
        client.register_rest("srv1", "http://localhost:8080", "standard")
        monkeypatch.setattr(
            "mcp_client.requests.get",
            MagicMock(return_value=_mock_rest_response([{"name": "tool_b"}])),
        )
        client.register_rest("srv2", "http://localhost:8081", "standard")
        tools1 = client.list_tools("srv1")
        tools2 = client.list_tools("srv2")
        all_tool_names = {t["name"] for t in tools1} | {t["name"] for t in tools2}
        assert all_tool_names == {"tool_a", "tool_b"}

    def test_list_tools_unregistered_server(self):
        """list_tools for unregistered server → error dict."""
        client = MCPClient()
        tools = client.list_tools("nonexistent")
        assert isinstance(tools, list)
        assert "error" in tools[0]

    def test_call_tool_routes_to_correct_server(self, monkeypatch):
        """call_tool dispatches to the named server's call_tool."""
        client = MCPClient()
        monkeypatch.setattr("mcp_client.requests.get", MagicMock(return_value=_mock_rest_response()))
        client.register_rest("srv1", "http://localhost:8080", "standard")
        # Mock the server's call_tool to verify routing
        client.servers["srv1"].call_tool = MagicMock(return_value="result from srv1")
        result = client.call_tool("srv1", "tool1", {"x": 1})
        assert result == "result from srv1"
        client.servers["srv1"].call_tool.assert_called_once_with("tool1", {"x": 1})

    def test_call_tool_unregistered_server(self):
        """call_tool with unknown server → error."""
        client = MCPClient()
        result = client.call_tool("nonexistent", "tool1", {})
        assert "not registered" in result

    def test_register_rest_replaces_existing(self, monkeypatch):
        """Re-registering the same name stops the old server first."""
        client = MCPClient()
        monkeypatch.setattr("mcp_client.requests.get", MagicMock(return_value=_mock_rest_response()))
        client.register_rest("srv", "http://localhost:8080", "standard")
        old_server = client.servers["srv"]
        old_server.stop = MagicMock()
        client.register_rest("srv", "http://localhost:8081", "standard")
        old_server.stop.assert_called_once()

    def test_get_servers_includes_type_and_endpoint(self, monkeypatch):
        """get_servers entries include type and endpoint fields."""
        client = MCPClient()
        monkeypatch.setattr("mcp_client.requests.get", MagicMock(return_value=_mock_rest_response()))
        client.register_rest("srv1", "http://localhost:9000", "standard")
        info = client.get_servers()[0]
        assert info["type"] == "standard"
        assert "localhost:9000" in info["endpoint"]


# ════════════════════════════════════════════════════════════════
# 9. StdioTransport
# ════════════════════════════════════════════════════════════════

class TestStdioTransportStart:
    def test_start_with_mock_subprocess(self, monkeypatch):
        """start() launches the subprocess and initializes."""
        mock_popen = MagicMock()
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        transport = StdioTransport("fake_cmd", ["--arg"])
        transport.send_request = MagicMock(return_value={
            "result": {"serverInfo": {"name": "test"}}
        })
        transport.send_notification = MagicMock()

        assert transport.start() is True
        assert transport._initialized is True
        mock_popen.assert_called_once()

    def test_start_failure_returns_false(self, monkeypatch):
        """start() returns False when initialize fails."""
        monkeypatch.setattr(subprocess, "Popen", MagicMock())
        transport = StdioTransport("fake_cmd")
        transport.send_request = MagicMock(return_value={"error": {"message": "nope"}})
        transport.send_notification = MagicMock()
        transport.stop = MagicMock()

        assert transport.start() is False
        transport.stop.assert_called_once()

    def test_start_exception_returns_false(self, monkeypatch):
        """start() returns False on Popen exception."""
        monkeypatch.setattr(subprocess, "Popen", MagicMock(side_effect=OSError("no such command")))
        transport = StdioTransport("fake_cmd")
        transport.stop = MagicMock()
        assert transport.start() is False


class TestStdioTransportStop:
    def test_stop_terminates_process(self):
        transport = StdioTransport("fake_cmd")
        mock_proc = MagicMock()
        transport._process = mock_proc
        transport._initialized = True

        transport.stop()
        mock_proc.terminate.assert_called_once()
        assert transport._process is None
        assert transport._initialized is False

    def test_stop_kills_on_terminate_timeout(self):
        transport = StdioTransport("fake_cmd")
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = Exception("timeout")
        transport._process = mock_proc

        transport.stop()
        mock_proc.kill.assert_called_once()

    def test_stop_without_process_is_noop(self):
        transport = StdioTransport("fake_cmd")
        transport.stop()  # must not raise
        assert transport._process is None


class TestStdioTransportSendRequest:
    def test_send_request_not_running(self):
        """No process → error dict."""
        transport = StdioTransport("fake_cmd")
        result = transport.send_request("tools/list")
        assert "error" in result
        assert "not running" in result["error"]["message"]

    def test_send_request_process_ended(self):
        """Process that exited → error dict."""
        transport = StdioTransport("fake_cmd")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # exited
        transport._process = mock_proc
        result = transport.send_request("tools/list")
        assert "error" in result

    def test_send_request_success(self, monkeypatch):
        """Valid process + mock queue → returns response."""
        transport = StdioTransport("fake_cmd")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = MagicMock()
        transport._process = mock_proc

        fake_response = {"result": {"tools": [{"name": "t"}]}}
        _fake_q = type("FakeQ", (), {
            "put": lambda self, item: None,
            "get": lambda self, timeout=None: fake_response,
        })
        monkeypatch.setattr("mcp_client.queue.Queue", _fake_q)

        result = transport.send_request("tools/list", timeout=5)
        assert result == fake_response
        mock_proc.stdin.write.assert_called_once()

    def test_send_request_write_failure(self, monkeypatch):
        """stdin.write raises → error dict, request cleaned up."""
        transport = StdioTransport("fake_cmd")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write.side_effect = BrokenPipeError("pipe closed")
        transport._process = mock_proc

        _fake_q = type("FakeQ", (), {
            "put": lambda self, item: None,
            "get": lambda self, timeout=None: {},
        })
        monkeypatch.setattr("mcp_client.queue.Queue", _fake_q)

        result = transport.send_request("tools/list")
        assert "error" in result
        assert "Failed to write" in result["error"]["message"]


class TestStdioTransportIsRunning:
    def test_not_running_no_process(self):
        transport = StdioTransport("fake_cmd")
        assert transport.is_running is False

    def test_running_with_process(self):
        transport = StdioTransport("fake_cmd")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        transport._process = mock_proc
        transport._initialized = True
        assert transport.is_running is True

    def test_not_running_process_exited(self):
        transport = StdioTransport("fake_cmd")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        transport._process = mock_proc
        transport._initialized = True
        assert transport.is_running is False

    def test_not_running_not_initialized(self):
        transport = StdioTransport("fake_cmd")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        transport._process = mock_proc
        transport._initialized = False
        assert transport.is_running is False


class TestStdioTransportSendNotification:
    def test_send_notification_writes_to_stdin(self):
        transport = StdioTransport("fake_cmd")
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        transport._process = mock_proc
        transport.send_notification("some/method", {"key": "val"})
        mock_proc.stdin.write.assert_called_once()
        written = mock_proc.stdin.write.call_args[0][0]
        assert b"some/method" in written

    def test_send_notification_swallows_errors(self):
        transport = StdioTransport("fake_cmd")
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write.side_effect = OSError("closed")
        transport._process = mock_proc
        # must not raise
        transport.send_notification("method")


# ════════════════════════════════════════════════════════════════
# 10. RESTTransport
# ════════════════════════════════════════════════════════════════

class TestRESTTransportStart:
    def test_start_is_noop_returns_true(self):
        transport = RESTTransport("http://localhost:8080", MCPTransportType.STANDARD)
        assert transport.start() is True

    def test_start_unity_bridge(self):
        transport = RESTTransport("http://localhost:8080", MCPTransportType.UNITY_BRIDGE)
        assert transport.start() is True


class TestRESTTransportStop:
    def test_stop_is_noop(self):
        transport = RESTTransport("http://localhost:8080", MCPTransportType.STANDARD)
        transport.stop()  # must not raise


class TestRESTTransportSendRequest:
    def test_list_tools_standard_url(self, monkeypatch):
        """_list_tools for STANDARD hits {url}/tools."""
        transport = RESTTransport("http://localhost:8080", MCPTransportType.STANDARD)
        mock_resp = _mock_rest_response([{"name": "t1"}])
        mock_get = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("mcp_client.requests.get", mock_get)

        result = transport.send_request("tools/list")
        assert "result" in result
        assert result["result"]["tools"] == [{"name": "t1"}]
        url_called = mock_get.call_args[0][0]
        assert url_called == "http://localhost:8080/tools"

    def test_list_tools_unity_bridge_url(self, monkeypatch):
        """_list_tools for UNITY_BRIDGE hits {url}/api/tools."""
        transport = RESTTransport("http://localhost:8080", MCPTransportType.UNITY_BRIDGE)
        mock_resp = _mock_rest_response([{"name": "unity_t"}])
        mock_get = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("mcp_client.requests.get", mock_get)

        result = transport.send_request("tools/list")
        assert result["result"]["tools"] == [{"name": "unity_t"}]
        url_called = mock_get.call_args[0][0]
        assert url_called == "http://localhost:8080/api/tools"

    def test_call_tool_standard_url(self, monkeypatch):
        """_call_tool for STANDARD posts to {url}/call."""
        transport = RESTTransport("http://localhost:8080", MCPTransportType.STANDARD)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": "result", "isError": False}
        mock_resp.raise_for_status = MagicMock()
        mock_post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("mcp_client.requests.post", mock_post)

        result = transport.send_request("tools/call", {"name": "my_tool", "arguments": {"x": 1}})
        assert "result" in result
        url_called = mock_post.call_args[0][0]
        assert url_called == "http://localhost:8080/call"
        # Check body
        body = mock_post.call_args[1]["json"]
        assert body["name"] == "my_tool"
        assert body["arguments"] == {"x": 1}

    def test_call_tool_unity_bridge_url(self, monkeypatch):
        """_call_tool for UNITY_BRIDGE posts to {url}/api/tool."""
        transport = RESTTransport("http://localhost:8080", MCPTransportType.UNITY_BRIDGE)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": True, "data": {"key": "val"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("mcp_client.requests.post", mock_post)

        result = transport.send_request("tools/call", {"name": "my_tool", "arguments": {"x": 1}})
        assert "result" in result
        url_called = mock_post.call_args[0][0]
        assert url_called == "http://localhost:8080/api/tool"
        body = mock_post.call_args[1]["json"]
        assert body["tool"] == "my_tool"
        assert body["args"] == {"x": 1}

    def test_call_tool_unity_bridge_error(self, monkeypatch):
        """UNITY_BRIDGE with success=False → isError response."""
        transport = RESTTransport("http://localhost:8080", MCPTransportType.UNITY_BRIDGE)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": False, "error": "bad request"}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr("mcp_client.requests.post", MagicMock(return_value=mock_resp))

        result = transport.send_request("tools/call", {"name": "t", "arguments": {}})
        assert result["result"]["isError"] is True
        assert "bad request" in result["result"]["content"][0]["text"]

    def test_call_tool_standard_error(self, monkeypatch):
        """STANDARD with isError=True → error content."""
        transport = RESTTransport("http://localhost:8080", MCPTransportType.STANDARD)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"isError": True, "content": "failure"}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr("mcp_client.requests.post", MagicMock(return_value=mock_resp))

        result = transport.send_request("tools/call", {"name": "t", "arguments": {}})
        assert result["result"]["isError"] is True

    def test_call_tool_unity_bridge_dict_data(self, monkeypatch):
        """UNITY_BRIDGE with dict data → JSON-serialized text."""
        transport = RESTTransport("http://localhost:8080", MCPTransportType.UNITY_BRIDGE)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": True, "data": {"key": "val"}}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr("mcp_client.requests.post", MagicMock(return_value=mock_resp))

        result = transport.send_request("tools/call", {"name": "t", "arguments": {}})
        text = result["result"]["content"][0]["text"]
        assert "key" in text and "val" in text

    def test_call_tool_unity_bridge_string_data(self, monkeypatch):
        """UNITY_BRIDGE with string data → str() in text."""
        transport = RESTTransport("http://localhost:8080", MCPTransportType.UNITY_BRIDGE)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": True, "data": "hello"}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr("mcp_client.requests.post", MagicMock(return_value=mock_resp))

        result = transport.send_request("tools/call", {"name": "t", "arguments": {}})
        assert result["result"]["content"][0]["text"] == "hello"

    def test_unsupported_method(self):
        """Methods other than tools/list and tools/call → error."""
        transport = RESTTransport("http://localhost:8080", MCPTransportType.STANDARD)
        result = transport.send_request("initialize")
        assert "error" in result
        assert "does not support" in result["error"]["message"]

    def test_request_exception_returns_error(self, monkeypatch):
        """requests exception → error dict."""
        transport = RESTTransport("http://localhost:8080", MCPTransportType.STANDARD)
        monkeypatch.setattr("mcp_client.requests.get", MagicMock(side_effect=ConnectionError("refused")))
        result = transport.send_request("tools/list")
        assert "error" in result

    def test_is_running_always_true(self):
        transport = RESTTransport("http://localhost:8080", MCPTransportType.STANDARD)
        assert transport.is_running is True

    def test_url_trailing_slash_stripped(self, monkeypatch):
        """Trailing slash in URL is stripped."""
        transport = RESTTransport("http://localhost:8080/", MCPTransportType.STANDARD)
        mock_resp = _mock_rest_response()
        mock_get = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("mcp_client.requests.get", mock_get)
        transport.send_request("tools/list")
        url_called = mock_get.call_args[0][0]
        assert url_called == "http://localhost:8080/tools"