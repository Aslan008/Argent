"""Unit tests for local inference, delegation, Cloud tier, and /local_model command."""

import json
from unittest.mock import MagicMock, patch
import pytest

import config
from local_inference import (
    SystemResourceAuditor,
    delegate_local,
    DELEGATE_LOCAL_SCHEMA,
    _run_cloud_tier,
)
from tools.schemas import get_tool_schemas, get_available_tools
from src.cli.commands.base import CommandContext
from src.cli.commands.config_commands import handle_local_model


# ---------------------------------------------------------------------------
# 1. SystemResourceAuditor Tests
# ---------------------------------------------------------------------------

class TestSystemResourceAuditor:
    def test_available_tier_cloud_always_safe(self, monkeypatch):
        assert SystemResourceAuditor.available_tier(max_user_tier=0) == 0

    def test_available_tier_high_ram(self, monkeypatch):
        monkeypatch.setattr(SystemResourceAuditor, "get_free_ram_gb", lambda: 8.0)
        monkeypatch.setattr(SystemResourceAuditor, "get_cpu_percent", lambda: 10.0)
        assert SystemResourceAuditor.available_tier(max_user_tier=3) == 3
        assert SystemResourceAuditor.available_tier(max_user_tier=1) == 1

    def test_available_tier_medium_ram(self, monkeypatch):
        monkeypatch.setattr(SystemResourceAuditor, "get_free_ram_gb", lambda: 2.2)
        monkeypatch.setattr(SystemResourceAuditor, "get_cpu_percent", lambda: 10.0)
        assert SystemResourceAuditor.available_tier(max_user_tier=3) == 2

    def test_available_tier_low_ram(self, monkeypatch):
        monkeypatch.setattr(SystemResourceAuditor, "get_free_ram_gb", lambda: 1.4)
        monkeypatch.setattr(SystemResourceAuditor, "get_cpu_percent", lambda: 10.0)
        assert SystemResourceAuditor.available_tier(max_user_tier=3) == 1

    def test_available_tier_exhausted_fallback_enabled(self, monkeypatch):
        monkeypatch.setattr(SystemResourceAuditor, "get_free_ram_gb", lambda: 0.5)
        monkeypatch.setattr(config, "is_local_agent_cloud_fallback", lambda: True)
        assert SystemResourceAuditor.available_tier(max_user_tier=3) == 0

    def test_available_tier_exhausted_fallback_disabled(self, monkeypatch):
        monkeypatch.setattr(SystemResourceAuditor, "get_free_ram_gb", lambda: 0.5)
        monkeypatch.setattr(config, "is_local_agent_cloud_fallback", lambda: False)
        assert SystemResourceAuditor.available_tier(max_user_tier=3) is None

    def test_max_concurrent_agents(self, monkeypatch):
        assert SystemResourceAuditor.max_concurrent_agents(tier=0) == 4
        monkeypatch.setattr(SystemResourceAuditor, "get_free_ram_gb", lambda: 5.0)
        assert SystemResourceAuditor.max_concurrent_agents(tier=1) >= 2


# ---------------------------------------------------------------------------
# 2. Cloud Tier & Fallback Tests
# ---------------------------------------------------------------------------

class TestCloudTierAndFallback:
    def test_run_cloud_tier_success(self, monkeypatch):
        mock_provider = MagicMock()
        mock_provider.sync_chat.return_value = "Result from cloud model"

        monkeypatch.setattr("providers.create_provider", lambda prov: mock_provider)
        monkeypatch.setattr(config, "get_local_agent_cloud_provider", lambda: "openrouter")
        monkeypatch.setattr(config, "get_local_agent_cloud_model", lambda: "google/gemini-2.0-flash-exp:free")

        res = _run_cloud_tier("Extract regex", context="some text")
        assert res == "Result from cloud model"
        assert mock_provider.sync_chat.called

    def test_delegate_local_tier_0_invokes_cloud(self, monkeypatch):
        monkeypatch.setattr("local_inference._run_cloud_tier", lambda t, c, timeout: "Cloud answer")
        res = delegate_local(task="Summarize text", tier=0)
        assert res == "Cloud answer"

    def test_delegate_local_auto_fallback_when_offline(self, monkeypatch):
        monkeypatch.setattr(SystemResourceAuditor, "available_tier", lambda t: 1)
        monkeypatch.setattr("local_inference._is_ollama_online", lambda: False)
        monkeypatch.setattr(config, "is_local_agent_cloud_fallback", lambda: True)
        monkeypatch.setattr("local_inference._run_cloud_tier", lambda t, c, timeout: "Fallback result")

        res = delegate_local(task="Write test")
        assert "[Auto-Cloud Fallback" in res
        assert "Fallback result" in res

    def test_delegate_local_no_fallback_when_offline(self, monkeypatch):
        monkeypatch.setattr(SystemResourceAuditor, "available_tier", lambda t: 1)
        monkeypatch.setattr("local_inference._is_ollama_online", lambda: False)
        monkeypatch.setattr(config, "is_local_agent_cloud_fallback", lambda: False)

        res = delegate_local(task="Write test")
        assert "[LOCAL INFERENCE UNAVAILABLE]" in res


# ---------------------------------------------------------------------------
# 3. Schema & Prefix Stability Tests
# ---------------------------------------------------------------------------

class TestSchemaPrefixStability:
    def test_delegate_local_not_present_when_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "is_local_agent_enabled", lambda: False)
        schemas = get_tool_schemas()
        assert not any(s["function"]["name"] == "delegate_local" for s in schemas)
        tools = get_available_tools()
        assert "delegate_local" not in tools

    def test_delegate_local_strictly_last_when_enabled(self, monkeypatch):
        monkeypatch.setattr(config, "is_local_agent_enabled", lambda: False)
        schemas_disabled = get_tool_schemas()

        monkeypatch.setattr(config, "is_local_agent_enabled", lambda: True)
        schemas_enabled = get_tool_schemas()

        # delegate_local is added
        assert any(s["function"]["name"] == "delegate_local" for s in schemas_enabled)
        # delegate_local is at the very end (prefix stability)
        assert schemas_enabled[-1]["function"]["name"] == "delegate_local"
        # Everything prior to the last item is 100% identical
        assert [s["function"]["name"] for s in schemas_enabled[:-1]] == [s["function"]["name"] for s in schemas_disabled]

        tools = get_available_tools()
        assert "delegate_local" in tools


# ---------------------------------------------------------------------------
# 4. Command /local_model Tests
# ---------------------------------------------------------------------------

class TestLocalModelCommand:
    def test_status_display(self, monkeypatch):
        test_conf = {}
        monkeypatch.setattr(config, "load_config", lambda: test_conf)
        monkeypatch.setattr(config, "save_config", lambda c: test_conf.update(c))
        printed = []
        monkeypatch.setattr("src.cli.commands.config_commands.print_system", lambda m: printed.append(m))

        ctx = CommandContext(agent=MagicMock(), user_input="/local_model")
        handle_local_model(ctx)
        out = "\n".join(printed)
        assert "Агенты делегирования" in out
        assert "Cloud-модель" in out

    def test_subcommand_on_and_off(self, monkeypatch):
        test_conf = {}
        monkeypatch.setattr(config, "load_config", lambda: test_conf)
        monkeypatch.setattr(config, "save_config", lambda c: test_conf.update(c))
        printed = []
        monkeypatch.setattr("src.cli.commands.config_commands.print_system", lambda m: printed.append(m))

        handle_local_model(CommandContext(agent=MagicMock(), user_input="/local_model on"))
        assert config.is_local_agent_enabled() is True

        handle_local_model(CommandContext(agent=MagicMock(), user_input="/local_model off"))
        assert config.is_local_agent_enabled() is False

    def test_subcommand_tier(self, monkeypatch):
        test_conf = {}
        monkeypatch.setattr(config, "load_config", lambda: test_conf)
        monkeypatch.setattr(config, "save_config", lambda c: test_conf.update(c))
        monkeypatch.setattr("src.cli.commands.config_commands.print_system", lambda m: None)

        handle_local_model(CommandContext(agent=MagicMock(), user_input="/local_model tier cloud"))
        assert config.get_local_model_tier() == 0

        handle_local_model(CommandContext(agent=MagicMock(), user_input="/local_model tier 1"))
        assert config.get_local_model_tier() == 1

        handle_local_model(CommandContext(agent=MagicMock(), user_input="/local_model tier 3"))
        assert config.get_local_model_tier() == 3

    def test_subcommand_cloud_config(self, monkeypatch):
        test_conf = {}
        monkeypatch.setattr(config, "load_config", lambda: test_conf)
        monkeypatch.setattr(config, "save_config", lambda c: test_conf.update(c))
        monkeypatch.setattr("src.cli.commands.config_commands.print_system", lambda m: None)

        handle_local_model(CommandContext(agent=MagicMock(), user_input="/local_model cloud openrouter deepseek/deepseek-chat:free"))
        assert config.get_local_agent_cloud_provider() == "openrouter"
        assert config.get_local_agent_cloud_model() == "deepseek/deepseek-chat:free"

    def test_subcommand_fallback(self, monkeypatch):
        test_conf = {}
        monkeypatch.setattr(config, "load_config", lambda: test_conf)
        monkeypatch.setattr(config, "save_config", lambda c: test_conf.update(c))
        monkeypatch.setattr("src.cli.commands.config_commands.print_system", lambda m: None)

        handle_local_model(CommandContext(agent=MagicMock(), user_input="/local_model fallback off"))
        assert config.is_local_agent_cloud_fallback() is False

        handle_local_model(CommandContext(agent=MagicMock(), user_input="/local_model fallback on"))
        assert config.is_local_agent_cloud_fallback() is True
