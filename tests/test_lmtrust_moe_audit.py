"""LMTrust Deep Blind-Spot Audit & Regression Test Suite for MoE support in Argent.

Covers:
- L1 (Contract): wants_factual_anchor_only contract honored by agent._build_objective_anchor
- L2 (Boundary & Specs): regex parsing correctly calculates total_b when only active_b is specified (a3b)
- L2b (Modern MoE Detection): detection of DeepSeek-V3, DeepSeek-R1, Phi-3.5-MoE, DBRX, Arctic, Jamba
- L4 (Error & Logging): prompt_compressor accepts valid 'moe' category without warning
- L6 (Cross-System Consistency): MoE preserves full toolset and full operational protocol
- L7 (Strategy Routing): User category override has strict priority over provider and model heuristics
"""

import pytest
from unittest.mock import MagicMock

import config
from src.agent.strategy import (
    get_model_strategy, MoELocalStrategy, StandardLocalStrategy,
    TinyLocalStrategy, CloudStrategy
)
import prompt_compressor


class TestL1FactualAnchorContract:
    def test_moe_factual_only_anchor_omits_errors_and_completed(self, monkeypatch):
        """Verify FIX for BUG 1: wants_factual_anchor_only is strictly honored.
        MoELocalStrategy omits COMPLETED and RECENT ERRORS, keeping only factual working context.
        """
        import agent as agent_module
        from memory_manager import memory

        memory.clear()
        memory.add_completed("Created payment gateway")
        memory.add_error("NullReferenceException in billing.cs")
        memory.add_file_modified("src/billing.cs")

        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: MagicMock())
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)

        ag = agent_module.ArgentAgent()
        ag.strategy = MoELocalStrategy()
        assert ag.strategy.wants_factual_anchor_only() is True

        anchor = ag._build_objective_anchor()
        assert anchor is not None
        assert "src/billing.cs" in anchor
        assert "FILES ALREADY EDITED" in anchor
        assert "RECENT ERRORS" not in anchor
        assert "COMPLETED" not in anchor

    def test_moe_factual_anchor_returns_none_when_no_files_touched(self, monkeypatch):
        """When no files were touched, factual-only anchor must be completely omitted (None)."""
        import agent as agent_module
        from memory_manager import memory

        memory.clear()
        memory.add_completed("Read architecture docs")
        memory.add_error("404 response from server")

        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: MagicMock())
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)

        ag = agent_module.ArgentAgent()
        ag.strategy = MoELocalStrategy()
        assert ag._build_objective_anchor() is None


class TestL2SpecsParsing:
    def test_a3b_total_params_collision_resolved(self):
        """Verify FIX for BUG 2: When model has no total params prefix ('a3b'),
        total_b is correctly defaulted to active_b * 8 (24.0, 3.0), not (3.0, 3.0).
        """
        specs = config.get_moe_specs("a3b")
        assert specs == (24.0, 3.0)

    def test_modern_moe_models_recognized(self):
        """Verify FIX for INC 2: Leading modern MoE architectures are recognized."""
        assert config.is_moe_model("deepseek-v3") is True
        assert config.get_moe_specs("deepseek-v3") == (671.0, 37.0)

        assert config.is_moe_model("deepseek-r1") is True
        assert config.get_moe_specs("deepseek-r1") == (671.0, 37.0)

        assert config.is_moe_model("phi-3.5-moe") is True
        assert config.get_moe_specs("phi-3.5-moe") == (42.0, 6.6)

        assert config.is_moe_model("dbrx-instruct") is True
        assert config.get_moe_specs("dbrx-instruct") == (132.0, 36.0)

        assert config.is_moe_model("snowflake-arctic") is True
        assert config.get_moe_specs("snowflake-arctic") == (480.0, 17.0)


class TestL4PromptCompressor:
    def test_compress_system_prompt_supports_moe_without_warning(self, caplog):
        """Verify FIX for BUG 3: 'moe' category does not produce spurious warnings
        and returns the full prompt cleanly.
        """
        import logging
        caplog.set_level(logging.WARNING, logger="agent")

        prompt = "## ROLE\nArgent Coder"
        res = prompt_compressor.compress_system_prompt(prompt, "ORNITH1.5:35A3B", category="moe")
        assert res == prompt
        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert not any("Unknown model category 'moe'" in w for w in warnings)


class TestL6ToolPreservationAndProtocol:
    def test_moe_preserves_full_toolset_and_protocol(self, monkeypatch):
        """Verify FIX for INC 1: Auto-detected MoE models keep all available tools
        and receive the complete operational protocol.
        """
        import agent as agent_module

        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: MagicMock())
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        monkeypatch.setattr(config, "get_model_category_override", lambda: None)
        monkeypatch.setattr(agent_module, "get_current_model", lambda: "ORNITH1.5:35A3B")

        ag = agent_module.ArgentAgent()
        assert config.is_moe_model(ag.model_name) is True

        allowed = ag.effective_tool_names()
        # Non-core tools must NOT be stripped for MoE
        assert "create_plugin" in allowed or "browser_navigate" in allowed or len(allowed) > 30

        # System prompt protocol must be full (not slimmed)
        sys_prompt = ag.build_system_prompt()
        assert "Batch Reads" in sys_prompt


class TestL7StrategyRoutingPriorities:
    def test_override_takes_strict_priority(self, monkeypatch):
        """Verify FIX for INC 3: User manual override wins unconditionally."""
        # Case A: User forces 'moe' on OpenRouter
        monkeypatch.setattr(config, "get_model_category_override", lambda: "moe")
        strat = get_model_strategy("deepseek/deepseek-chat", "openrouter")
        assert isinstance(strat, MoELocalStrategy)

        # Case B: User forces 'tiny' on a 35B MoE model
        monkeypatch.setattr(config, "get_model_category_override", lambda: "tiny")
        strat = get_model_strategy("ORNITH1.5:35A3B", "ollama")
        assert isinstance(strat, TinyLocalStrategy)

        # Case C: Auto-detection routes MoE to MoELocalStrategy
        monkeypatch.setattr(config, "get_model_category_override", lambda: None)
        strat = get_model_strategy("ORNITH1.5:35A3B", "ollama")
        assert isinstance(strat, MoELocalStrategy)
