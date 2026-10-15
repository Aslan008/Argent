"""The Needle fast path must hand the agent tools that exist.

Needle is given five stubs to choose from: read_file, git_status, git_diff,
grep_search, run_command. Argent has no `git_status` and no `git_diff`. The
marker path (a stub's return value) already turned them into `run_command`; the
explicit-call path returned the stub's own name, so a plain "git status" typed
by the user — the textbook fast-path query — became a call to a tool that does
not exist.
"""

import pytest

from src.agent import needle_coprocessor as nc
from tools.schemas import AVAILABLE_TOOLS


class _FakeNeedle:
    def __init__(self, outcome):
        self._outcome = outcome

    def reset(self):
        pass

    def run(self, query, max_steps=1):
        return self._outcome


def _dispatch(monkeypatch, query, function_calls=None, results=None, confidence=0.9):
    coprocessor = nc.NeedleCoprocessor()
    monkeypatch.setattr(coprocessor, "initialize", lambda: True)
    coprocessor._needle = _FakeNeedle({
        "success": True, "confidence": confidence,
        "function_calls": function_calls or [], "results": results or [],
    })
    return coprocessor.try_fast_dispatch(query)


class TestExplicitFunctionCalls:
    def test_git_status_runs_as_a_shell_command(self, monkeypatch):
        got = _dispatch(monkeypatch, "покажи статус гит",
                        function_calls=[{"name": "git_status", "arguments": {}}])
        assert got.tool_name == "run_command"
        assert got.arguments == {"command": "git status"}

    def test_git_diff_runs_as_a_shell_command(self, monkeypatch):
        got = _dispatch(monkeypatch, "покажи дифф",
                        function_calls=[{"name": "git_diff", "arguments": {}}])
        assert got.tool_name == "run_command"
        assert got.arguments == {"command": "git diff"}

    def test_git_diff_of_one_file(self, monkeypatch, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("x", encoding="utf-8")
        got = _dispatch(monkeypatch, "покажи дифф файла",
                        function_calls=[{"name": "git_diff", "arguments": {"file": str(target)}}])
        assert got.tool_name == "run_command"
        assert got.arguments == {"command": f"git diff {target}"}

    def test_git_diff_of_a_file_that_is_not_there_is_not_guessed(self, monkeypatch):
        assert _dispatch(monkeypatch, "покажи дифф файла",
                         function_calls=[{"name": "git_diff",
                                          "arguments": {"file": "no/such/file.py"}}]) is None

    @pytest.mark.parametrize("query, stub", [("сделай что-нибудь", "git_status"),
                                             ("сделай что-нибудь", "git_diff")])
    def test_a_stub_without_the_matching_words_in_the_query_is_refused(self, monkeypatch, query, stub):
        assert _dispatch(monkeypatch, query, function_calls=[{"name": stub, "arguments": {}}]) is None

    def test_whatever_it_returns_is_a_real_tool(self, monkeypatch, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("x", encoding="utf-8")
        for query, call in [
            ("покажи статус гит", {"name": "git_status", "arguments": {}}),
            ("покажи дифф", {"name": "git_diff", "arguments": {}}),
            ("прочитай файл", {"name": "read_file", "arguments": {"path": str(target)}}),
            ("найди handle_think", {"name": "grep_search", "arguments": {"query": "handle_think"}}),
            ("запусти git status", {"name": "run_command", "arguments": {"command": "git status"}}),
        ]:
            got = _dispatch(monkeypatch, query, function_calls=[call])
            assert got is not None, query
            assert got.tool_name in AVAILABLE_TOOLS, f"{query!r} -> {got.tool_name!r} is not a tool"


class TestMarkerResults:
    """The path that already worked; pinned so the two stay the same."""

    def test_git_status_marker(self, monkeypatch):
        got = _dispatch(monkeypatch, "покажи статус гит", results=["[git_status]"])
        assert (got.tool_name, got.arguments) == ("run_command", {"command": "git status"})

    def test_git_diff_marker(self, monkeypatch):
        got = _dispatch(monkeypatch, "покажи дифф", results=["[git_diff:]"])
        assert (got.tool_name, got.arguments) == ("run_command", {"command": "git diff"})
