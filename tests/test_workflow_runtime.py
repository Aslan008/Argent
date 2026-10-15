"""Calibration, the advisor, the session switch, the tool and commands — and one run through the real parts."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import config
from src.agent import subagents, worktrees
from src.agent.workflow import advisor, calibration, control, roles, runtime, store
from src.agent.workflow.calibration import Hardware, Profile, choose_level


def git(repo, *args):
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.fixture(autouse=True)
def _switch_off():
    runtime.set_enabled(False)
    control.reset()
    yield
    runtime.set_enabled(False)
    control.reset()


@pytest.fixture
def cfg(monkeypatch):
    values = {}
    monkeypatch.setattr(config, "_get", lambda key, default=None: values.get(key, default))
    monkeypatch.setattr(config, "_set", lambda key, val: values.__setitem__(key, val))
    return values


def profile(local=True, category="medium", tok=20.0, ram=32.0, parallelism=1):
    return Profile(provider="ollama", model="m", local=local, size_category=category, tok_per_s=tok,
                   parallelism=parallelism, hardware={"ram_gb": ram, "cores": 8})


# ── the ladder ───────────────────────────────────────────────────────────────

class TestLevels:
    def test_a_cloud_model_gets_the_full_machinery(self):
        assert choose_level(profile(local=False))[0] == 3

    def test_a_typical_local_model_gets_the_pipeline(self):
        assert choose_level(profile())[0] == 2

    def test_a_tiny_model_does_not_get_the_mode(self):
        level, why = choose_level(profile(category="tiny"))
        assert level == 0 and "3B" in why

    def test_a_small_or_slow_or_cramped_machine_gets_the_light_level(self):
        assert choose_level(profile(category="small"))[0] == 1
        assert choose_level(profile(tok=2.0))[0] == 1
        assert choose_level(profile(ram=6))[0] == 1

    def test_strong_local_setups_may_run_workers_side_by_side(self):
        assert choose_level(profile(tok=40, ram=32, parallelism=3))[0] == 3
        assert choose_level(profile(tok=40, ram=32, parallelism=1))[0] == 2

    def test_the_persons_setting_wins(self):
        assert choose_level(profile(category="tiny"), forced=3)[0] == 3
        assert choose_level(profile(), forced="1")[0] == 1
        assert choose_level(profile(), forced="auto")[0] == 2

    def test_parallelism_follows_the_level(self):
        assert calibration.parallelism_for(2, profile(parallelism=4)) == 1
        assert calibration.parallelism_for(3, profile(parallelism=3)) == 3
        assert calibration.parallelism_for(3, profile(local=False)) == 3


class TestCalibration:
    @pytest.fixture(autouse=True)
    def _own_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration, "_cache_path", lambda: tmp_path / "calibration.json")

    def test_cloud_is_not_measured(self, cfg):
        probe = MagicMock()
        p = calibration.calibrate("deepseek", "deepseek-chat", probe=probe)
        assert not p.local and probe.call_count == 0

    def test_a_local_model_is_measured_once_and_remembered(self, cfg, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        probe = MagicMock(return_value=(0.8, 24.0))
        first = calibration.calibrate("ollama", "qwen3:8b", probe=probe)
        again = calibration.calibrate("ollama", "qwen3:8b", probe=probe)
        assert first.local and first.tok_per_s == 24.0 and first.ttft_s == 0.8
        assert again.tok_per_s == 24.0
        assert probe.call_count == 1
        calibration.calibrate("ollama", "qwen3:8b", probe=probe, force=True)
        assert probe.call_count == 2

    def test_a_failed_measurement_is_not_cached(self, cfg, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        probe = MagicMock(return_value=(None, None))
        calibration.calibrate("ollama", "qwen3:8b", probe=probe)
        calibration.calibrate("ollama", "qwen3:8b", probe=probe)
        assert probe.call_count == 2

    def test_the_cache_expires(self, cfg, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        probe = MagicMock(return_value=(1.0, 10.0))
        t = [1_000_000.0]
        calibration.calibrate("ollama", "qwen3:8b", probe=probe, now=lambda: t[0])
        t[0] += calibration.CACHE_TTL + 5
        calibration.calibrate("ollama", "qwen3:8b", probe=probe, now=lambda: t[0])
        assert probe.call_count == 2

    def test_hardware_fingerprint_is_stable(self):
        assert Hardware(cores=8, ram_gb=31.6, vram_gb=12.0).fingerprint == "8c-32g-12v"


# ── the advisor ──────────────────────────────────────────────────────────────

class TestAdvisor:
    def test_unconfigured_means_the_main_model(self, cfg, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        t = advisor.resolve("ollama", "qwen3:8b")
        assert (t.provider, t.model, t.configured, t.cloud, t.leaves_pc) == ("ollama", "qwen3:8b", False, False, False)

    def test_a_cloud_advisor_for_a_local_main_model_leaves_the_pc(self, cfg, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        config.set_advisor_provider("claude")
        config.set_advisor_model("claude-opus-4")
        t = advisor.resolve("ollama", "qwen3:8b")
        assert t.configured and t.cloud and t.leaves_pc and t.label == "claude:claude-opus-4"

    def test_a_local_advisor_stays_home(self, cfg, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        config.set_advisor_provider("ollama")
        config.set_advisor_model("qwen3:32b")
        t = advisor.resolve("ollama", "qwen3:8b")
        assert t.configured and not t.cloud and not t.leaves_pc

    def test_consult_redacts_secrets_and_parses_the_verdict(self, cfg, monkeypatch):
        seen = {}

        class P:
            def sync_chat(self, model, messages, temperature=0.3, json_format=False, timeout=None):
                seen["text"] = messages[-1]["content"]
                return "- add a test\nVERDICT: REVISE"
        import providers
        monkeypatch.setattr(providers, "create_provider", lambda name=None: P())
        t = advisor.Target("claude", "big", configured=True, cloud=True, leaves_pc=True)
        out = advisor.consult("plan", "make it", "key is sk-ant-api03-" + "A" * 40, t)
        assert "sk-ant-api03" not in seen["text"]
        assert out.verdict == "REVISE" and "add a test" in out.text and out.target == "claude:big"

    def test_a_failing_advisor_falls_back_and_says_so(self, cfg, monkeypatch):
        calls = []

        class P:
            def __init__(self, name):
                self.name = name

            def sync_chat(self, model, messages, **kw):
                calls.append(self.name)
                if self.name == "claude":
                    raise RuntimeError("quota")
                return "VERDICT: APPROVE"
        import providers
        monkeypatch.setattr(providers, "create_provider", lambda name=None: P(name))
        t = advisor.Target("claude", "big", True, True, True)
        fb = advisor.Target("ollama", "small", False, False, False)
        out = advisor.consult("plan", "g", "b", t, fallback=fb)
        assert out.fallback and out.target == "ollama:small" and calls == ["claude", "ollama"]

    def test_without_a_fallback_the_failure_is_raised(self, cfg, monkeypatch):
        class P:
            def sync_chat(self, *a, **k):
                raise RuntimeError("down")
        import providers
        monkeypatch.setattr(providers, "create_provider", lambda name=None: P())
        with pytest.raises(advisor.AdvisorError):
            advisor.consult("final", "g", "b", advisor.Target("claude", "x", True, True, True))

    def test_stuck_guidance_is_cut_out_of_the_answer(self):
        text = "The worker misread the API.\nGUIDANCE: call connect() before send().\nVERDICT: RETRY"
        assert roles.advisor_guidance(text) == "call connect() before send()."
        assert roles.parse_advisor_verdict(text) == "RETRY"

    def test_the_last_verdict_counts(self):
        assert roles.parse_advisor_verdict("VERDICT: REJECT ... later ... **VERDICT: APPROVE**") == "APPROVE"
        assert roles.parse_advisor_verdict("no verdict") == ""


# ── the switch, the tool, the commands ───────────────────────────────────────

class TestSwitch:
    def test_the_tool_is_hidden_until_the_mode_is_on(self):
        from src.agent import plan_mode
        assert plan_mode.hides("ultracode") is True
        runtime.set_enabled(True)
        assert plan_mode.hides("ultracode") is False

    def test_instructions_only_while_on(self):
        assert runtime.instructions() is None
        runtime.set_enabled(True)
        assert "ultracode(action=\"start\"" in runtime.instructions()

    def test_start_refuses_when_off(self):
        text = runtime.start("do a large refactor of everything")
        assert "выключен" in text

    def test_start_refuses_a_vague_goal(self):
        runtime.set_enabled(True)
        assert "формулировк" in runtime.start("fix")

    def test_a_tiny_model_is_told_to_work_normally(self, monkeypatch):
        runtime.set_enabled(True)
        monkeypatch.setattr(runtime, "_decide_level", lambda: (0, "модель слишком маленькая", profile()))
        assert "обычным способом" in runtime.start("refactor the whole package into modules")

    def test_the_tool_dispatches(self, monkeypatch):
        from tools.workflow_tools import ultracode
        monkeypatch.setattr(runtime, "status", lambda: "S")
        monkeypatch.setattr(runtime, "stop", lambda: "T")
        assert ultracode("status") == "S" and ultracode("stop") == "T"
        assert ultracode("dance").startswith("Error")

    def test_a_workflow_cannot_start_a_workflow(self):
        assert "ultracode" in subagents.FORBIDDEN_TOOLS

    def test_commands_switch_the_mode_and_set_the_level(self, cfg):
        from src.cli.commands import CommandContext, dispatch_command

        def run(text):
            return dispatch_command(text, CommandContext(agent=MagicMock(), user_input=text))

        assert run("/ultracode on").handled and runtime.enabled()
        assert run("/ultracode off").handled and not runtime.enabled()
        run("/ultracode level 2")
        assert cfg["ultracode_level"] == 2
        run("/ultracode level auto")
        assert cfg["ultracode_level"] == "auto"
        run("/ultracode budget 45")
        assert cfg["ultracode_max_minutes"] == 45
        run("/ultracode level 9")
        assert cfg["ultracode_level"] == "auto", "a bad level changes nothing"

    def test_advisor_command(self, cfg):
        from src.cli.commands import CommandContext, dispatch_command

        def run(text):
            return dispatch_command(text, CommandContext(agent=MagicMock(), user_input=text))

        run("/advisor model claude:claude-opus-4")
        assert cfg["advisor_model"] == "claude-opus-4" and cfg["advisor_provider"] == "claude"
        run("/advisor consent on")
        assert cfg["advisor_cloud_consent"] is True
        run("/advisor clear")
        assert cfg["advisor_model"] == "" and cfg["advisor_provider"] == ""


class TestParts:
    def test_check_reads_the_exit_code(self, monkeypatch):
        import tools.command_ops as ops
        monkeypatch.setattr(ops, "run_command", lambda c, cwd=None, timeout=None: "Exit code: 0\n3 passed")
        assert runtime.run_check("pytest", ".") == (True, "Exit code: 0\n3 passed")
        monkeypatch.setattr(ops, "run_command", lambda c, cwd=None, timeout=None: "Exit code: 1\nFAILED")
        assert runtime.run_check("pytest", ".")[0] is False
        monkeypatch.setattr(ops, "run_command", lambda c, cwd=None, timeout=None: "Execution aborted by user.")
        assert runtime.run_check("pytest", ".")[0] is False

    def test_repo_hint_names_the_test_command(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("Use tabs.", encoding="utf-8")
        hint = runtime._repo_hint(str(tmp_path))
        assert "pytest" in hint and "Use tabs." in hint and "pyproject.toml" in hint

    def test_stop_sets_the_switch(self):
        runtime.stop()
        assert control.stop_requested()

    def test_agent_stop_stops_a_workflow(self):
        import agent as agent_module
        a = agent_module.ArgentAgent.__new__(agent_module.ArgentAgent)
        a._stop_requested = False
        agent_module.ArgentAgent.request_stop(a)
        assert control.stop_requested()


# ── one run through the real parts ───────────────────────────────────────────

CHILD = r'''
import json, os, sys
spec = json.load(open(sys.argv[1], encoding="utf-8"))
task = spec["task"]
if "[s1]" in task:
    open("util.py", "w", encoding="utf-8").write("X = 2\n")
    report = "changed util.py"
else:
    report = "looked around"
print("ARGENT_SUBAGENT_RESULT " + json.dumps({"status": "ok", "report": report, "denied": []}))
'''


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@argent.local")
    git(root, "config", "user.name", "T")
    (root / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "util.py").write_text("X = 1\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(worktrees.tempfile, "tempdir", str(tmp_path / "tmp"))
    monkeypatch.chdir(root)
    return root


def test_a_workflow_runs_through_real_subagents_worktrees_and_patches(repo, tmp_path, monkeypatch, cfg):
    script = tmp_path / "fake_child.py"
    script.write_text(CHILD, encoding="utf-8")
    manager = subagents.SubagentManager(argv_builder=lambda spec: [sys.executable, str(script), str(spec)])
    monkeypatch.setattr(subagents, "MANAGER", manager)

    plan_text = json.dumps({"summary": "bump X", "steps": [
        {"id": "s1", "title": "bump X", "goal": "set X to 2 in util.py", "role": "coder", "files": ["util.py"],
         "checks": ["python -c \"import util; assert util.X == 2\""]},
        {"id": "s2", "title": "look", "goal": "check callers", "role": "researcher", "deps": ["s1"]}],
        "final_checks": ["python -c \"import util\""]})

    class Provider:
        def sync_chat(self, model, messages, temperature=0.3, json_format=False, timeout=None):
            text = messages[0]["content"]
            if "PLANNER" in text:
                return plan_text
            return "Looks right.\nVERDICT: APPROVE" if "review the plan" in text else "VERDICT: ACCEPT"

    import approval
    import providers
    monkeypatch.setattr(providers, "create_provider", lambda name=None: Provider())
    monkeypatch.setattr(approval, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(config, "get_provider", lambda: "deepseek")
    monkeypatch.setattr(config, "get_current_model", lambda: "deepseek-chat")

    checks = []

    def fake_check(command, cwd):
        done = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        checks.append((command, cwd))
        return done.returncode == 0, f"Exit code: {done.returncode}\n{done.stdout}{done.stderr}"
    monkeypatch.setattr(runtime, "run_check", fake_check)

    import agent as agent_module
    monkeypatch.setattr(agent_module, "run_plan_critique", lambda target, goal=None, what="": "fine\nVERDICT: SOLID")

    runtime.set_enabled(True)
    text = runtime.start("make util.X equal to two and verify it")

    assert (repo / "util.py").read_text(encoding="utf-8").strip() == "X = 2", text
    states = store.list_runs(str(repo))
    assert len(states) == 1 and states[0].status == "done", text
    assert states[0].level == 3                       # a cloud model: the full ladder
    plan = states[0].get_plan()
    assert [s.status for s in plan.steps] == ["done", "done"]
    # the step's own check ran in the worker's private copy, the final check in the real tree
    assert checks[0][1] != str(repo)
    assert checks[-1][1] == str(repo)
    assert "bump X" in text
    assert git(repo, "worktree", "list").strip().count("\n") == 0, "no private copy is left behind"


# ── what the model is shown ──────────────────────────────────────────────────

class TestWhatTheModelSees:
    @pytest.fixture
    def agent(self, monkeypatch):
        import agent as agent_module
        mem = MagicMock()
        mem.data = {}
        monkeypatch.setattr(agent_module, "memory", mem)
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(config, "get_model_category_override", lambda: None)
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "cloud")
        return agent_module.ArgentAgent()

    def test_the_tool_is_offered_only_while_the_mode_is_on(self, agent):
        assert "ultracode" not in agent.effective_tool_names(None)
        runtime.set_enabled(True)
        assert "ultracode" in agent.effective_tool_names(None)

    def test_the_instructions_ride_with_the_request_only_while_on(self, agent):
        runtime.set_enabled(False)
        assert "ULTRACODE" not in (agent._build_ephemeral_context() or "")
        runtime.set_enabled(True)
        assert "ULTRACODE IS ON" in agent._build_ephemeral_context()
