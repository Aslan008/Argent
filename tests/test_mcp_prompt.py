"""MCP in the system prompt: a map, not a catalog.

The old section inlined every tool of every server. Measured against a real
Unity MCP server (140 tools) it was 27k characters — ~6.8k tokens riding along
in every single request, of which a turn reads one or two.
"""

import pytest

from src.agent.mcp_prompt import (
    FULL_LIST_LIMIT, build_mcp_section, model_access_warning,
)
from tools import misc_tools
from tools.misc_tools import list_mcp_tools


class _FakeClient:
    def __init__(self, tools_by_server):
        self._tools = tools_by_server

    def list_tools(self, name):
        value = self._tools.get(name, [])
        if isinstance(value, Exception):
            raise value
        return value


def _tools(*names):
    return [{"name": n, "description": f"Does {n}.",
             "inputSchema": {"properties": {"path": {}, "confirm": {}},
                             "required": ["path"]}} for n in names]


class TestSection:
    def test_a_small_server_is_listed_in_full(self):
        client = _FakeClient({"tiny": _tools("a", "b")})
        out = build_mcp_section([{"name": "tiny"}], client=client)
        assert "`tiny` (stdio, 2 tools): a, b" in out

    def test_a_huge_server_is_sampled_not_dumped(self):
        """140 signatures in every request buy nothing: a turn uses one or two."""
        names = [f"tool_{i:03d}" for i in range(140)]
        client = _FakeClient({"unity": _tools(*names)})
        out = build_mcp_section([{"name": "unity"}], client=client)
        assert "140 tools" in out and "and 130 more" in out
        assert len(out) < 1500          # the old version was ~27000
        assert "tool_139" not in out

    def test_the_model_is_told_where_the_signatures_are(self):
        client = _FakeClient({"s": _tools("a")})
        out = build_mcp_section([{"name": "s"}], client=client)
        assert "list_mcp_tools" in out and "call_mcp_tool" in out

    def test_an_unreachable_server_says_so(self):
        """Otherwise the model blames itself for the failure and retries."""
        client = _FakeClient({"dead": [{"error": "connection refused"}]})
        out = build_mcp_section([{"name": "dead"}], client=client)
        assert "UNREACHABLE" in out and "connection refused" in out
        assert "Do not call it" in out

    def test_a_raising_client_does_not_break_the_prompt(self):
        client = _FakeClient({"boom": RuntimeError("pipe closed")})
        out = build_mcp_section([{"name": "boom"}], client=client)
        assert "UNREACHABLE" in out and "pipe closed" in out

    def test_a_connected_but_empty_server(self):
        out = build_mcp_section([{"name": "s"}], client=_FakeClient({"s": []}))
        assert "no tools" in out

    def test_names_are_sorted_so_the_prefix_cache_survives(self):
        """The section is the prefix of every request; reordering it between
        turns throws away the provider's KV cache for nothing."""
        client_a = _FakeClient({"s": _tools("b", "a", "c")})
        client_b = _FakeClient({"s": _tools("c", "b", "a")})
        assert (build_mcp_section([{"name": "s"}], client=client_a) ==
                build_mcp_section([{"name": "s"}], client=client_b))

    def test_full_list_boundary(self):
        names = [f"t{i}" for i in range(FULL_LIST_LIMIT)]
        out = build_mcp_section([{"name": "s"}], client=_FakeClient({"s": _tools(*names)}))
        assert "e.g." not in out and names[-1] in out


class TestAccessWarning:
    def test_silence_when_nothing_is_configured(self, monkeypatch):
        monkeypatch.setattr("config.get_mcp_servers", lambda: [])
        assert model_access_warning() is None

    def test_configured_and_reachable_is_silent(self, monkeypatch):
        monkeypatch.setattr("config.get_mcp_servers", lambda: [{"name": "unity"}])
        monkeypatch.setattr("config.get_disabled_tools", lambda: [])
        assert model_access_warning() is None

    def test_a_disabled_call_mcp_tool_is_named_out_loud(self, monkeypatch):
        """'Server RUNNING, 140 tools' and 'the model can use it' are separate
        switches that both read as ON. This cost a real debugging session."""
        monkeypatch.setattr("config.get_mcp_servers", lambda: [{"name": "unity"}])
        monkeypatch.setattr("config.get_disabled_tools", lambda: ["call_mcp_tool"])
        warning = model_access_warning()
        assert warning and "call_mcp_tool" in warning and "/tools" in warning


class TestListTool:
    @pytest.fixture(autouse=True)
    def servers(self, monkeypatch):
        monkeypatch.setattr("config.get_mcp_servers",
                            lambda: [{"name": "unity", "type": "stdio"}])
        names = [f"tool_{i:03d}" for i in range(140)] + ["bake_navmesh"]
        monkeypatch.setattr(misc_tools, "mcp_client",
                            _FakeClient({"unity": _tools(*names)}))

    def test_no_arguments_maps_the_servers(self):
        out = list_mcp_tools()
        assert "unity" in out and "141 tools" in out

    def test_filter_returns_signatures(self):
        out = list_mcp_tools("unity", "navmesh")
        assert "bake_navmesh(path, confirm?)" in out
        assert "1 of 141" in out

    def test_an_unfiltered_big_server_gives_names_only(self):
        """Signatures for 141 tools would be the very dump this replaced."""
        out = list_mcp_tools("unity")
        assert "(path" not in out
        assert "pass `filter`" in out

    def test_a_miss_shows_what_is_there_instead(self):
        out = list_mcp_tools("unity", "quantum")
        assert "No tool on 'unity' matches" in out and "tool_000" in out

    def test_unknown_server_lists_the_known_ones(self):
        out = list_mcp_tools("untiy")
        assert out.startswith("Error:") and "unity" in out

    def test_no_servers_configured(self, monkeypatch):
        monkeypatch.setattr("config.get_mcp_servers", lambda: [])
        assert "No MCP servers are configured" in list_mcp_tools()

    def test_an_unreachable_server_reports_the_reason(self, monkeypatch):
        monkeypatch.setattr(misc_tools, "mcp_client",
                            _FakeClient({"unity": [{"error": "refused"}]}))
        assert "refused" in list_mcp_tools("unity")


class TestWeakModelProfile:
    def test_mcp_tools_join_the_core_set_only_when_a_server_exists(self, monkeypatch):
        """A fixed core set cannot say 'useful here, dead weight there'. Without
        this, a weak model silently loses its only door to a configured Unity
        editor — the exact failure this change was chasing."""
        from tool_profiles import core_tools_now, slim_tools_for_category

        monkeypatch.setattr("config.get_mcp_servers", lambda: [])
        assert "call_mcp_tool" not in core_tools_now()
        assert slim_tools_for_category(["call_mcp_tool", "read_file"], "tiny") == ["read_file"]

        monkeypatch.setattr("config.get_mcp_servers", lambda: [{"name": "unity"}])
        assert {"call_mcp_tool", "list_mcp_tools"} <= core_tools_now()
        assert "call_mcp_tool" in slim_tools_for_category(["call_mcp_tool"], "tiny")

    def test_a_broken_config_does_not_shrink_the_toolset(self, monkeypatch):
        def boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr("config.get_mcp_servers", boom)
        from tool_profiles import CORE_CHAT_TOOLS, core_tools_now
        assert core_tools_now() == set(CORE_CHAT_TOOLS)

    def test_capable_models_are_untouched(self, monkeypatch):
        from tool_profiles import slim_tools_for_category
        monkeypatch.setattr("config.get_mcp_servers", lambda: [])
        assert slim_tools_for_category(["call_mcp_tool"], "cloud") == ["call_mcp_tool"]
