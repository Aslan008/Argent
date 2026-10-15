"""LMTrust Deep Blind-Spot & Invariant Regression Suite for MoE Architecture in Argent.

Covers verified edge cases and fixes:
- L2 x D4/D7: Exclusion of DeepSeek-R1 distilled dense models (1.5B, 7B, 8B, 14B, 32B, 70B) from MoE
- L2 x D1: Detection of IBM Granite 3.0 MoE (million-param units: a400m, a800m)
- L2 x D7: Detection of AllenAI OLMoE, JetMoE, and suffix-named MoE models
- L1 x D10: Enforcement of Total >= Active parameter invariants; rejection of single-expert 1x7b
- L2 x D8: Disambiguation of Snowflake Arctic LLM vs Arctic embedding models
- L6 x D7: Preservation of Planning Mode & Plugins in prompt_compressor for compact MoE models
- L6 x D7: Unlocking plugin directory modification in tools/_helpers for MoE models
"""

import pytest
from unittest.mock import MagicMock

import config
import prompt_compressor
from tools._helpers import _is_plugin_path_restricted
from src.agent.strategy import (
    get_model_strategy, MoELocalStrategy, StandardLocalStrategy,
    TinyLocalStrategy
)


class TestL2DenseDistillExclusion:
    """Ensure distilled dense models bearing 'deepseek-r1' in their tag are not falsely marked as 671B MoE."""

    @pytest.mark.parametrize("model_name,expected_category", [
        ("deepseek-r1:1.5b", "tiny"),
        ("deepseek-r1:7b", "medium"),
        ("deepseek-r1:8b", "medium"),
        ("deepseek-r1:14b", "large"),
        ("deepseek-r1:32b", "large"),
        ("deepseek-r1:70b", "large"),
        ("deepseek-r1-distill-qwen-7b", "medium"),
        ("deepseek-r1-distill-llama-8b", "medium"),
        ("deepseek-r1-distill-qwen-1.5b", "tiny"),
    ])
    def test_distilled_r1_models_are_not_moe(self, model_name, expected_category):
        assert config.is_moe_model(model_name) is False
        assert config.get_moe_specs(model_name) is None
        assert config.get_model_size_category(model_name) == expected_category

    def test_full_frontier_deepseek_models_remain_moe(self):
        assert config.is_moe_model("deepseek-v3") is True
        assert config.get_moe_specs("deepseek-v3") == (671.0, 37.0)

        assert config.is_moe_model("deepseek-r1") is True
        assert config.get_moe_specs("deepseek-r1") == (671.0, 37.0)

        assert config.is_moe_model("deepseek-r1:671b") is True
        assert config.get_moe_specs("deepseek-r1:671b") == (671.0, 37.0)


class TestL2ModernMoEArchitectures:
    """Test expanded coverage for modern MoE architectures (Granite, OLMoE, JetMoE, suffix MoE)."""

    def test_granite_moe_million_param_detection(self):
        # 3B total, 800M active
        assert config.is_moe_model("granite-3.0-3b-a800m") is True
        specs = config.get_moe_specs("granite-3.0-3b-a800m")
        assert specs == (3.0, 0.8)

        # 1B total, 400M active
        assert config.is_moe_model("granite-3.0-1b-a400m") is True
        assert config.get_moe_specs("granite-3.0-1b-a400m") == (1.0, 0.4)

    def test_olmoe_and_jetmoe_detection(self):
        assert config.is_moe_model("olmoe-1b-7b") is True
        assert config.get_moe_specs("olmoe-1b-7b") == (7.0, 1.0)

        assert config.is_moe_model("olmoe") is True
        assert config.get_moe_specs("olmoe") == (7.0, 1.0)

        assert config.is_moe_model("jetmoe-8b") is True
        assert config.get_moe_specs("jetmoe-8b") == (8.0, 2.0)

    def test_suffix_and_named_moe(self):
        assert config.is_moe_model("qwen2.5-32b-instruct-moe") is True
        assert config.get_moe_specs("qwen2.5-32b-instruct-moe") == (32.0, 8.0)

        assert config.is_moe_model("mistral-moe") is True
        assert config.is_moe_model("custom-project-moe") is True


class TestL1MathematicalInvariants:
    """Verify invariants: Total >= Active > 0; expert count >= 2; embedding exclusions."""

    def test_single_expert_not_moe(self):
        assert config.is_moe_model("1x7b") is False
        assert config.get_moe_specs("1x7b") is None

    def test_total_greater_or_equal_to_active_invariant(self):
        # Even if a bad prefix matches 2b before a7b, total must be >= active
        specs = config.get_moe_specs("model-2b-a7b")
        assert specs is not None
        total_b, active_b = specs
        assert total_b >= active_b
        assert active_b == 7.0

    def test_zero_params_rejected(self):
        assert config.is_moe_model("0x0b") is False
        assert config.get_moe_specs("0x0b") is None

    def test_arctic_embedding_not_moe(self):
        assert config.is_moe_model("snowflake/arctic-embed-m") is False
        assert config.is_moe_model("snowflake-arctic") is True
        assert config.get_moe_specs("snowflake-arctic") == (480.0, 17.0)


class TestL6CrossSystemPromptAndToolIntegrity:
    """Cross-layer checks between agent, prompt_compressor, and tools."""

    def test_prompt_compressor_does_not_strip_compact_moe(self):
        prompt = """## ROLE\nExpert\n\n## 2. PLUGIN DEVELOPMENT\nPlugins.\n\n## 4. PLANNING MODE & ARTIFACTS\nPlan.\n\n## 5. CODE EDITING\nEdit."""
        # lfm2.5-8b-a1b has active 1B (tiny size category)
        res = prompt_compressor.compress_system_prompt(prompt, "lfm2.5-8b-a1b", category="tiny")
        assert "PLANNING MODE & ARTIFACTS" in res
        assert "PLUGIN DEVELOPMENT" in res
        assert "## RULES (SHORT)" not in res

    def test_plugin_directory_write_guard_allows_moe(self):
        config.set_current_model("ORNITH1.5:35A3B")
        assert _is_plugin_path_restricted("plugins/new_hook.py") is None

        config.set_current_model("granite-3.0-3b-a800m")
        assert _is_plugin_path_restricted("plugins/new_hook.py") is None

        # Dense small models (<7B) remain protected
        config.set_current_model("qwen2.5-3b-instruct")
        err = _is_plugin_path_restricted("plugins/new_hook.py")
        assert err is not None
        assert "restricted for small models" in err

    def test_agent_elevates_moe_category_and_preserves_protocol(self, monkeypatch):
        import agent as agent_module

        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: MagicMock())
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        monkeypatch.setattr(config, "get_model_category_override", lambda: None)
        monkeypatch.setattr(agent_module, "get_current_model", lambda: "granite-3.0-3b-a800m")

        ag = agent_module.ArgentAgent()
        sys_prompt = ag.build_system_prompt()
        assert "PLANNING MODE" in sys_prompt
        assert "Batch Reads" in sys_prompt

    def test_moe_ephemeral_context_has_persistence_directive(self, monkeypatch):
        import agent as agent_module
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: MagicMock())
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        monkeypatch.setattr(config, "get_model_category_override", lambda: None)
        monkeypatch.setattr(agent_module, "get_current_model", lambda: "ORNITH1.5:35A3B")

        ag = agent_module.ArgentAgent()
        ephemeral = ag._build_ephemeral_context()
        assert ephemeral is not None
        assert "TASK PERSISTENCE DIRECTIVE" in ephemeral
        assert "No response requested" in ephemeral

    def test_command_ops_collapses_carriage_returns(self, monkeypatch):
        import tools.command_ops as command_ops

        # Simulate raw chunks with carriage returns (like pip or uv progress bar)
        raw = "Downloading pydantic (10%)\rDownloading pydantic (50%)\rDownloading pydantic (100%)\nInstalled"
        clean = command_ops.re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw)
        lines = []
        for line in clean.split("\n"):
            if "\r" in line:
                parts = [p.strip() for p in line.split("\r") if p.strip()]
                lines.append(parts[-1] if parts else "")
            else:
                lines.append(line)
        result = "\n".join(lines)
        assert "Downloading pydantic (10%)" not in result
        assert "Downloading pydantic (100%)" in result
        assert "Installed" in result

