"""Tests for MoE model strategy, memory error resolution, LoopGuard softening, and platform cleanup."""

import pytest
from unittest.mock import MagicMock

import config
import memory_manager
from memory_manager import MemoryManager
from src.agent.strategy import (
    get_model_strategy, MoELocalStrategy, StandardLocalStrategy,
    CloudStrategy, TinyLocalStrategy
)
from src.agent.loop_guard import LoopGuard
from src.agent.healing import detect_tool_failure


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    target = tmp_path / ".argent" / "memory.json"
    monkeypatch.setattr(memory_manager, "resolve_memory_file", lambda: target)
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)
    return target


# ─── MoE Detection & Specs ───────────────────────────────────────────────────

class TestMoEDetection:
    def test_parses_moe_specs_colon_and_compact(self):
        assert config.get_moe_specs("ORNITH1.5:35A3B") == (35.0, 3.0)
        assert config.get_moe_specs("ornith1.5:35a3b") == (35.0, 3.0)
        assert config.get_moe_specs("35a3b") == (35.0, 3.0)
        assert config.get_moe_specs("a3b") == (24.0, 3.0)

    def test_parses_moe_specs_hyphen_and_deepseek(self):
        assert config.get_moe_specs("lfm2.5-8b-a1b") == (8.0, 1.0)
        assert config.get_moe_specs("qwen2-57b-a14b") == (57.0, 14.0)
        assert config.get_moe_specs("deepseek-v2:16b-a2.4b") == (16.0, 2.4)
        assert config.get_moe_specs("deepseek-v3") == (671.0, 37.0)
        assert config.get_moe_specs("deepseek-r1") == (671.0, 37.0)
        assert config.get_moe_specs("phi-3.5-moe") == (42.0, 6.6)
        assert config.get_moe_specs("dbrx") == (132.0, 36.0)
        assert config.get_moe_specs("snowflake-arctic") == (480.0, 17.0)
        assert config.get_moe_specs("jamba-1.5-mini") == (52.0, 12.0)
        assert config.get_moe_specs("jamba-1.5-large") == (398.0, 94.0)
        assert config.get_moe_specs("grok-1") == (314.0, 86.0)

    def test_parses_moe_specs_mixtral_format(self):
        assert config.get_moe_specs("mixtral-8x7b") == (56.0, 14.0)
        assert config.get_moe_specs("mixtral:8x22b") == (176.0, 44.0)

    def test_is_moe_model_boolean(self):
        assert config.is_moe_model("ORNITH1.5:35A3B") is True
        assert config.is_moe_model("mixtral-8x7b") is True
        assert config.is_moe_model("deepseek-v3") is True
        assert config.is_moe_model("phi-3.5-moe") is True
        assert config.is_moe_model("qwen2.5-7b-instruct") is False
        assert config.is_moe_model("gpt-4o") is False
        assert config.is_moe_model("") is False


# ─── Strategy Factory & MoELocalStrategy ────────────────────────────────────

class TestMoEStrategy:
    def test_routes_moe_model_to_moe_strategy(self):
        s = get_model_strategy("ORNITH1.5:35A3B", "ollama")
        assert isinstance(s, MoELocalStrategy)

    def test_routes_dense_model_to_standard_strategy(self):
        s = get_model_strategy("qwen2.5-7b-instruct", "ollama")
        assert isinstance(s, StandardLocalStrategy)

    def test_manual_moe_override_routes_to_moe_strategy(self, monkeypatch):
        monkeypatch.setattr(config, "get_model_category_override", lambda: "moe")
        s = get_model_strategy("some-unknown-dense-model", "ollama")
        assert isinstance(s, MoELocalStrategy)

    def test_moe_strategy_properties(self):
        s = MoELocalStrategy()
        assert s.get_max_history_messages("small") == 28
        assert s.supports_native_tools() is True
        assert s.wants_objective_anchor() is True
        assert s.wants_factual_anchor_only() is True
        assert s.wants_minified_schemas() is True
        assert s.is_loop_guard_enabled() is True
        assert s.is_auto_healing_enabled() is False


# ─── MemoryManager Error Resolution ──────────────────────────────────────────

class TestMemoryErrorResolution:
    def test_resolves_exact_command_error(self):
        manager = MemoryManager()
        manager.add_error("Command 'python _verify3.py' failed (exit=1)")
        assert len(manager.data["errors_encountered"]) == 1

        manager.resolve_command_error("python _verify3.py")
        assert len(manager.data["errors_encountered"]) == 0

    def test_resolves_compound_command_target(self):
        manager = MemoryManager()
        manager.add_error("Command 'python _verify3.py' failed (exit=1)")
        assert len(manager.data["errors_encountered"]) == 1

        # Command ran with cd or extra arguments
        manager.resolve_command_error('cd "C:/Users/test/Engine" ; python _verify3.py')
        assert len(manager.data["errors_encountered"]) == 0

    def test_clear_current_task(self):
        manager = MemoryManager()
        manager.set_current_task("temporary task")
        assert manager.data["current_task"] == "temporary task"
        manager.clear_current_task()
        assert manager.data["current_task"] == ""


# ─── Objective Anchor Cleanliness ───────────────────────────────────────────

class TestObjectiveAnchorCleanliness:
    def _agent(self, monkeypatch, strategy=None):
        import agent as agent_module
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: MagicMock())
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        ag = agent_module.ArgentAgent()
        if strategy:
            ag.strategy = strategy
        return ag

    def test_standard_anchor_omits_current_task_and_no_retry_phrase(self, monkeypatch):
        from memory_manager import memory
        memory.clear()
        memory.add_completed("Created renderer")
        memory.add_error("syntax error in math.py")
        memory.add_file_modified("pyforge/core.py")

        ag = self._agent(monkeypatch, strategy=StandardLocalStrategy())
        anchor = ag._build_objective_anchor()
        assert "OBJECTIVE" not in anchor
        assert "CURRENT TASK" not in anchor
        assert "RECENT ERRORS" in anchor
        assert "FILES ALREADY EDITED" in anchor

    def test_moe_factual_only_anchor_omits_errors_and_completed(self, monkeypatch):
        from memory_manager import memory
        memory.clear()
        memory.add_completed("Created renderer")
        memory.add_error("some test failure")
        memory.add_file_modified("pyforge/core.py")

        ag = self._agent(monkeypatch, strategy=MoELocalStrategy())
        anchor = ag._build_objective_anchor()
        assert "OBJECTIVE" not in anchor
        assert "FILES ALREADY EDITED" in anchor
        assert "pyforge/core.py" in anchor
        assert "RECENT ERRORS" not in anchor
        assert "COMPLETED" not in anchor

        # When no files were touched, factual-only anchor must be completely omitted (None)
        memory.clear()
        memory.add_completed("Some completed work")
        memory.add_error("Some error")
        assert ag._build_objective_anchor() is None


# ─── Adaptive LoopGuard & Auto-Healing ───────────────────────────────────────

class TestAdaptiveGuards:
    def test_loop_guard_smart_tier_for_moe_and_cloud(self):
        g = LoopGuard()
        # For MoE: smart tier protection (warn at 2, nudge at 4, stop at 6)
        assert g.record("run_command", {"cmd": "test"}, "err", category="moe") is None
        assert g.record("run_command", {"cmd": "test"}, "err", category="moe") == "warn"
        assert g.record("run_command", {"cmd": "test"}, "err", category="moe") == "warn"
        assert g.record("run_command", {"cmd": "test"}, "err", category="moe") == "nudge"
        assert g.record("run_command", {"cmd": "test"}, "err", category="moe") == "nudge"
        assert g.record("run_command", {"cmd": "test"}, "err", category="moe") == "stop"

        g.reset()
        # For Cloud: same smart tier
        assert g.record("run_command", {"cmd": "test"}, "err", category="cloud") is None
        assert g.record("run_command", {"cmd": "test"}, "err", category="cloud") == "warn"
        assert g.record("run_command", {"cmd": "test"}, "err", category="cloud") == "warn"
        assert g.record("run_command", {"cmd": "test"}, "err", category="cloud") == "nudge"
        assert g.record("run_command", {"cmd": "test"}, "err", category="cloud") == "nudge"
        assert g.record("run_command", {"cmd": "test"}, "err", category="cloud") == "stop"

    def test_auto_healing_bypassed_for_moe_and_cloud(self):
        assert detect_tool_failure("run_command", "Exit code: 1", category="moe") is False
        assert detect_tool_failure("run_command", "Exit code: 1", category="cloud") is False
        assert detect_tool_failure("run_command", "Exit code: 1", category="small") is True
