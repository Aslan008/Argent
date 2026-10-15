"""The slash commands of the workflow features: /plan /todo /permissions /hooks /review /agents."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.agent import permissions, review, shell_hooks, subagents
from src.agent.permissions import Permissions
from src.agent.plan_mode import PLAN
from src.agent.subagents import SubagentManager
from src.cli.commands import CommandContext, dispatch_command, plan_commands
from src.cli.commands import permission_commands, review_commands, subagent_commands
from src.cli.commands import agent_commands

MODULES = (plan_commands, permission_commands, review_commands, subagent_commands, agent_commands)


@pytest.fixture
def said(monkeypatch):
    """Everything the commands print, errors prefixed with '!'."""
    out = []
    for module in MODULES:
        monkeypatch.setattr(module, "print_system", lambda message, *a, _out=out, **k: _out.append(str(message)),
                            raising=False)
        monkeypatch.setattr(module, "print_error", lambda message, *a, _out=out, **k: _out.append("!" + str(message)),
                            raising=False)
    return out


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".argent").mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(permissions, "PERMISSIONS", Permissions(project_dir=tmp_path, home=home))
    monkeypatch.setattr(shell_hooks, "SHELL_HOOKS", shell_hooks.ShellHooks(project_dir=tmp_path, home=home))
    return tmp_path


def run(text):
    return dispatch_command(text, CommandContext(agent=SimpleNamespace(), user_input=text))


class TestPlan:
    def test_on_off_and_status(self, said):
        assert run("/plan").handled and PLAN.active
        run("/plan status")
        assert "ВКЛЮЧЁН" in said[-1]
        run("/plan off")
        assert not PLAN.active
        run("/plan off")
        assert "и так выключен" in said[-1]
        run("/plan status")
        assert "выключен" in said[-1]

    def test_with_a_task_it_turns_planning_on_and_hands_the_task_to_the_agent(self, said):
        result = run("/plan add retry to fetch()")
        assert PLAN.active and result.new_input == "add retry to fetch()"


class TestTodo:
    def test_empty_shows_a_hint_and_clear_empties(self, said):
        from src.agent import todos
        run("/todo")
        assert said[-1].startswith("!")
        todos.write([{"content": "step one", "status": "in_progress"}, "step two"])
        run("/todo")
        assert "0/2" in said[-1] and "[~] step one" in said[-1]
        run("/todo clear")
        assert todos.TODOS.get() == []


class TestPermissions:
    def test_add_list_remove(self, said, project):
        run("/permissions allow run_command(git status*)")
        run("/permissions deny write_file(.env*)")
        run("/permissions")
        shown = "\n".join(said)
        assert "allow" in shown and "run_command(git status*)" in shown and "deny" in shown and ".env*" in shown
        run("/permissions remove run_command(git status*)")
        assert "Убрано правил: 1" in said[-1]
        run("/permissions remove nothing(here)")
        assert "Такого правила нет" in said[-1]

    def test_a_rule_without_text_or_with_garbage_is_an_error_not_a_crash(self, said, project):
        run("/permissions allow")
        assert said[-1].startswith("!")
        run("/permissions deny ((")
        assert said[-1].startswith("!")

    def test_global_scope(self, said, project):
        run("/permissions deny run_command(shutdown*) --global")
        assert (project / "home" / ".argent" / "permissions.json").is_file()
        assert not (project / ".argent" / "permissions.json").exists()

    def test_trust_command(self, said, project):
        path = project / ".argent" / "permissions.json"
        path.write_text(json.dumps({"allow": ["run_command(ls*)"]}), encoding="utf-8")
        assert permissions.PERMISSIONS.command_rule("allow", "ls") is None
        run("/permissions trust")
        assert permissions.PERMISSIONS.command_rule("allow", "ls") is not None


class TestHooks:
    def test_list_trust_untrust(self, said, project, monkeypatch):
        path = project / ".argent" / "hooks.json"
        path.write_text(json.dumps({"hooks": {"Stop": [{"command": "echo done"}]}}), encoding="utf-8")
        run("/hooks list")
        assert "НЕ запускаются" in said[-1]
        run("/hooks trust -y")
        run("/hooks list")
        assert any("Stop" in line and "echo done" in line for line in said)
        run("/hooks untrust")
        run("/hooks list")
        assert "НЕ запускаются" in said[-1]

    def test_with_nothing_configured_it_explains_how(self, said, project):
        run("/hooks list")
        assert "hooks.json" in said[0] and "PreToolUse" in "\n".join(said)


class TestReview:
    def fake(self, monkeypatch, findings, calls):
        def fake_run(target, focus):
            calls.append((target, focus))
            bundle = review.DiffBundle(text="d", description="uncommitted changes", files=["a.py"])
            return review.ReviewResult(findings, bundle, "", True)
        monkeypatch.setattr(review, "run_review", fake_run)
        monkeypatch.setattr(review, "save", lambda result, root=None: "")
        monkeypatch.setattr(review_commands.console, "print", lambda *a, **k: None)

    def test_target_and_focus_are_split(self, said, project, monkeypatch):
        calls = []
        self.fake(monkeypatch, [], calls)
        run("/review staged the SQL escaping")
        run("/review the SQL escaping")
        assert calls == [("staged", "the SQL escaping"), ("", "the SQL escaping")]

    def test_fix_hands_the_findings_to_the_agent(self, said, project, monkeypatch):
        finding = review.Finding("a.py", 3, "high", "oops", "always")
        self.fake(monkeypatch, [finding], [])
        result = run("/review --fix")
        assert result.new_input and "a.py:3" in result.new_input and "oops" in result.new_input
        assert run("/review").new_input is None

    def test_nothing_to_review_is_an_error_line(self, said, project, monkeypatch):
        def refuse(target, focus):
            raise review.ReviewError("Nothing to review: no uncommitted changes.")
        monkeypatch.setattr(review, "run_review", refuse)
        run("/review")
        assert said[-1].startswith("!Nothing to review")


class TestAgents:
    @pytest.fixture
    def manager(self, tmp_path, monkeypatch):
        script = tmp_path / "child.py"
        script.write_text(
            "import json, sys\n"
            "print('ARGENT_SUBAGENT_RESULT ' + json.dumps({'status': 'ok', 'report': 'the report', 'denied': []}))\n",
            encoding="utf-8")
        mgr = SubagentManager(argv_builder=lambda spec: [sys.executable, str(script), str(spec)])
        monkeypatch.setattr(subagents, "MANAGER", mgr)
        yield mgr
        mgr.stop_all()

    def test_list_and_report(self, said, project, manager):
        run("/agents")
        assert said[-1] == "No sub-agents in this session."
        sub = manager.spawn("Researcher", "look", cwd=project)
        manager.wait(sub, 30)
        run("/agents")
        assert "#1" in said[-1] and "Researcher" in said[-1]
        run(f"/agents {sub.id}")
        assert "the report" in said[-1]
        run("/agents #1")
        assert "the report" in said[-1]

    def test_unknown_ids_and_actions_are_errors(self, said, project, manager):
        run("/agents 7")
        assert said[-1].startswith("!Нет подагента")
        run("/agents apply")
        assert said[-1].startswith("!")

    def test_nothing_to_apply(self, said, project, manager):
        sub = manager.spawn("Researcher", "look", cwd=project)
        manager.wait(sub, 30)
        run(f"/agents apply {sub.id}")
        assert "Патч не применился" in said[-1]
        run(f"/agents stop {sub.id}")
        assert "не запущен" in said[-1]
