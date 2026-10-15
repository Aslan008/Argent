"""Tests for sampling parameters (temperature, top_p, top_k, min_p, repeat_penalty)
including configuration, provider propagation, and CLI commands.
"""

from unittest.mock import MagicMock, patch
import pytest

import config
from providers import OllamaProvider, OpenAICompatibleProvider
from src.cli.commands import dispatch_command, CommandContext


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    yield store


class TestConfigSamplingParams:
    def test_default_values(self):
        assert config.get_temperature() == 0.3
        assert config.get_top_p() is None
        assert config.get_top_k() is None
        assert config.get_min_p() is None
        assert config.get_repeat_penalty() is None

        params = config.get_sampling_params()
        assert params == {
            "temperature": 0.3,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "repeat_penalty": None,
        }

    def test_set_and_get_all_params(self):
        config.set_temperature(0.5)
        config.set_top_p(0.85)
        config.set_top_k(50)
        config.set_min_p(0.05)
        config.set_repeat_penalty(1.15)

        assert config.get_temperature() == 0.5
        assert config.get_top_p() == 0.85
        assert config.get_top_k() == 50
        assert config.get_min_p() == 0.05
        assert config.get_repeat_penalty() == 1.15

        params = config.get_sampling_params()
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.85
        assert params["top_k"] == 50
        assert params["min_p"] == 0.05
        assert params["repeat_penalty"] == 1.15

    def test_reset_sampling_params(self):
        config.set_temperature(1.2)
        config.set_top_p(0.9)
        config.set_top_k(20)
        config.set_min_p(0.1)
        config.set_repeat_penalty(1.2)

        config.reset_sampling_params()

        assert config.get_temperature() == 0.3
        assert config.get_top_p() is None
        assert config.get_top_k() is None
        assert config.get_min_p() is None
        assert config.get_repeat_penalty() is None


class TestProviderPropagation:
    def test_ollama_provider_propagates_sampling_options(self, monkeypatch):
        config.set_top_p(0.9)
        config.set_min_p(0.08)
        config.set_top_k(30)
        config.set_repeat_penalty(1.1)

        provider = OllamaProvider()
        captured_kwargs = {}

        def fake_chat(**kwargs):
            captured_kwargs.update(kwargs)
            return iter([])

        monkeypatch.setattr(provider._ollama, "chat", fake_chat)

        list(provider.stream_chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.4,
            context_window=8192,
        ))

        options = captured_kwargs.get("options", {})
        assert options.get("temperature") == 0.4
        assert options.get("num_ctx") == 8192
        assert options.get("top_p") == 0.9
        assert options.get("min_p") == 0.08
        assert options.get("top_k") == 30
        assert options.get("repeat_penalty") == 1.1

    def test_ollama_sync_chat_propagates_sampling_params(self, monkeypatch):
        config.set_top_p(0.88)
        config.set_min_p(0.04)
        config.set_top_k(25)
        config.set_repeat_penalty(1.15)

        from providers import OllamaProvider
        provider = OllamaProvider()
        captured_kwargs = {}

        def fake_chat(**kwargs):
            captured_kwargs.update(kwargs)
            return {"message": {"content": "ok"}}

        monkeypatch.setattr(provider._ollama, "chat", fake_chat)
        res = provider.sync_chat("qwen2.5:7b", [{"role": "user", "content": "hi"}], temperature=0.25)
        assert res == "ok"
        options = captured_kwargs.get("options", {})
        assert options.get("temperature") == 0.25
        assert options.get("top_p") == 0.88
        assert options.get("min_p") == 0.04
        assert options.get("top_k") == 25
        assert options.get("repeat_penalty") == 1.15

    def test_openai_compatible_provider_propagates_sampling_params(self):
        config.set_top_p(0.95)
        config.set_min_p(0.05)
        config.set_top_k(40)
        config.set_repeat_penalty(1.12)

        from providers import XKiroProvider
        provider = XKiroProvider(api_key="sk-test", base_url="https://api.xkiro.com/v1")
        mock_client = MagicMock()
        mock_create = MagicMock(return_value=iter([]))
        mock_client.chat.completions.create = mock_create

        with patch.object(provider, "_get_client", return_value=mock_client):
            list(provider.stream_chat(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.6,
            ))

        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args

        # top_p is standard top-level OpenAI parameter
        assert kwargs.get("temperature") == 0.6
        assert kwargs.get("top_p") == 0.95

        # min_p, top_k, repetition_penalty are in extra_body for OpenRouter/vLLM/KoboldCPP
        extra_body = kwargs.get("extra_body", {})
        assert extra_body.get("min_p") == 0.05
        assert extra_body.get("top_k") == 40
        assert extra_body.get("repetition_penalty") == 1.12


class TestCLICommands:
    def test_handle_temp_command_arg(self):
        ctx = CommandContext(agent=MagicMock(), user_input="/temp 0.2")
        res = dispatch_command(ctx.user_input, ctx)
        assert res.handled is True
        assert config.get_temperature() == 0.2

    def test_handle_temp_command_invalid(self):
        ctx = CommandContext(agent=MagicMock(), user_input="/temp 5.0")
        res = dispatch_command(ctx.user_input, ctx)
        assert res.handled is True
        # Temperature remains default
        assert config.get_temperature() == 0.3

    def test_handle_sampling_set_subcommands(self):
        ctx = CommandContext(agent=MagicMock(), user_input="/sampling min_p 0.05")
        res = dispatch_command(ctx.user_input, ctx)
        assert res.handled is True
        assert config.get_min_p() == 0.05

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling top_p 0.9")
        dispatch_command(ctx.user_input, ctx)
        assert config.get_top_p() == 0.9

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling top_k 40")
        dispatch_command(ctx.user_input, ctx)
        assert config.get_top_k() == 40

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling repeat_penalty 1.1")
        dispatch_command(ctx.user_input, ctx)
        assert config.get_repeat_penalty() == 1.1

    def test_handle_sampling_reset(self):
        config.set_top_p(0.8)
        config.set_min_p(0.1)
        config.set_temperature(1.5)

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling reset")
        res = dispatch_command(ctx.user_input, ctx)
        assert res.handled is True

        assert config.get_temperature() == 0.3
        assert config.get_top_p() is None
        assert config.get_min_p() is None

    def test_handle_individual_param_resets(self):
        config.set_top_p(0.8)
        config.set_min_p(0.1)
        config.set_top_k(50)
        config.set_repeat_penalty(1.2)
        config.set_temperature(0.9)

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling top_p reset")
        dispatch_command(ctx.user_input, ctx)
        assert config.get_top_p() is None

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling min_p none")
        dispatch_command(ctx.user_input, ctx)
        assert config.get_min_p() is None

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling top_k off")
        dispatch_command(ctx.user_input, ctx)
        assert config.get_top_k() is None

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling repeat_penalty null")
        dispatch_command(ctx.user_input, ctx)
        assert config.get_repeat_penalty() is None

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling temp default")
        dispatch_command(ctx.user_input, ctx)
        assert config.get_temperature() == 0.3

        config.set_temperature(1.2)
        ctx = CommandContext(agent=MagicMock(), user_input="/temp default")
        dispatch_command(ctx.user_input, ctx)
        assert config.get_temperature() == 0.3
