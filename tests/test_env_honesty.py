"""The environment tells the truth about itself: exit codes, local pages, MCP states.

Three places where Argent lied to the model, and the lie cost a step each time:

* A non-zero exit code was read as "it failed" even when the command printed warnings
  and did its job — a PowerShell habit, and in a merged output stream the code is the
  weakest signal available.
* `file://` was refused outright, so viewing a generated HTML report needed
  `Start-Process` and a detour through the desktop.
* An MCP server that answered "here are my tools: none" was reported as `unreachable`,
  and a dead one kept a line in the system prompt of every request, forever.
"""

import os
import sys
from pathlib import Path

import pytest

import argent.tools.command_ops as co

WINDOWS = os.name == "nt"
PY = sys.executable


@pytest.fixture(autouse=True)
def _approve(monkeypatch):
    monkeypatch.setattr(co, "request_approval", lambda *a, **k: True)


class TestExitCodeHonesty:
    def test_powershell_notes_that_a_nonzero_code_is_not_proof_of_failure(self):
        # Prints something, diagnoses nothing recognisable, returns 1.
        out = co.run_command(f'"{PY}" -c "print(\'all good\'); raise SystemExit(1)"')
        assert "Exit code: 1" in out
        assert "[NOTE]" in out
        assert "does not by itself mean the command failed" in out

    def test_a_recognised_failure_gets_the_diagnosis_and_not_the_note(self):
        out = co.run_command("argent_no_such_command_xyz --version")
        assert "[DIAGNOSIS]" in out
        assert "[NOTE]" not in out, "два объяснения одного провала противоречат друг другу"

    def test_a_successful_command_gets_no_note(self):
        out = co.run_command(f'"{PY}" -c "print(1)"')
        assert "Exit code: 0" in out
        assert "[NOTE]" not in out

    def test_the_note_is_only_for_powershell(self):
        """cargo owns its exit code: cmd runs it and the note would be wrong."""
        from src.agent.shell import choose_shell

        assert choose_shell("cargo build") == "cmd"
        assert co._runs_under_powershell("cargo build") is False
        assert co._runs_under_powershell("Get-ChildItem .") is True


class TestLocalPages:
    def test_a_project_html_page_may_be_opened(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        page = tmp_path / "report.html"
        page.write_text("<h1>hi</h1>", encoding="utf-8")
        from argent.browser_engine import _blocked_url_reason

        assert _blocked_url_reason(page.as_uri()) is None

    def test_a_page_outside_the_project_is_refused(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.chdir(project)
        outside = tmp_path / "elsewhere.html"
        outside.write_text("<h1>x</h1>", encoding="utf-8")
        from argent.browser_engine import _blocked_url_reason

        reason = _blocked_url_reason(outside.as_uri())
        assert reason and "outside" in reason.lower()

    def test_a_secret_file_is_not_a_page(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        secret = tmp_path / "id_rsa"
        secret.write_text("PRIVATE KEY", encoding="utf-8")
        from argent.browser_engine import _blocked_url_reason

        reason = _blocked_url_reason(secret.as_uri())
        assert reason and "rendered pages" in reason
        assert "read_file" in reason

    def test_the_dangerous_schemes_are_still_refused(self):
        from argent.browser_engine import _blocked_url_reason

        for url in ("javascript:alert(1)", "data:text/html,<h1>x</h1>", "chrome://settings"):
            assert _blocked_url_reason(url), url


class TestMcpStates:
    def _section(self, client, servers):
        from src.agent.mcp_prompt import build_mcp_section

        return build_mcp_section(servers, client=client)

    def test_a_server_with_no_tools_says_so_instead_of_unreachable(self):
        class _Client:
            def list_tools(self, name):
                return []

        text = self._section(_Client(), [{"name": "unity", "type": "stdio"}])
        assert "NO TOOLS" in text
        assert "unreachable" not in text.lower()
        assert "Check the server" in text

    def test_an_unreachable_server_no_longer_costs_a_line_every_request(self):
        class _Client:
            def list_tools(self, name):
                return [{"error": "connection closed"}]

        text = self._section(_Client(), [{"name": "blender", "type": "stdio"}])
        assert text == "", "мёртвый сервер не должен занимать промпт"

    def test_a_working_server_is_left_alone(self):
        class _Client:
            def list_tools(self, name):
                return [{"name": "bake_navmesh"}, {"name": "list_assets"}]

        text = self._section(_Client(), [{"name": "unity", "type": "stdio"}])
        assert "bake_navmesh" in text and "2 tools" in text

    def test_only_the_dead_one_is_dropped(self):
        class _Client:
            def list_tools(self, name):
                if name == "blender":
                    return [{"error": "no answer"}]
                return [{"name": "bake_navmesh"}]

        text = self._section(_Client(), [
            {"name": "unity", "type": "stdio"},
            {"name": "blender", "type": "stdio"},
        ])
        assert "unity" in text
        assert "blender" not in text


class TestListMcpToolsWording:
    def _run(self, monkeypatch, tmp_path, tools, state):
        import argent.tools.misc_tools as mt

        class _Client:
            def list_tools(self, name):
                return tools

            def listing_state(self, name):
                return state

        monkeypatch.setattr(mt, "mcp_client", _Client())
        monkeypatch.setattr("argent.config.get_mcp_servers",
                            lambda: [{"name": "srv", "type": "stdio"}])
        monkeypatch.chdir(tmp_path)
        return mt

    def test_no_tools_is_its_own_message(self, monkeypatch, tmp_path):
        mt = self._run(monkeypatch, tmp_path, [], "empty")
        out = mt.list_mcp_tools()
        assert "NO TOOLS" in out

    def test_no_answer_is_unreachable(self, monkeypatch, tmp_path):
        mt = self._run(monkeypatch, tmp_path, [], "unreachable")
        out = mt.list_mcp_tools()
        assert "unreachable" in out

    def test_asking_about_a_server_with_no_tools_does_not_call_it_down(self, monkeypatch, tmp_path):
        mt = self._run(monkeypatch, tmp_path, [], "empty")
        out = mt.list_mcp_tools(server_name="srv")
        assert "NO TOOLS" in out
        assert "could not be reached" not in out
