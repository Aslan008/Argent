"""
LMTrust Blind Spot & Contract Verification Suite for LLM Sampling Parameters.
Tests layers:
- L0: Smoke
- L1 x D7: API Contract (OpenAI reasoning models o1/o3-mini, Claude Thinking, Ollama)
- L2 x D1: Boundary Conditions (0.0 greedy, clamping, negative/overflow)
- L4 x D4: Adversarial & Invalid CLI Inputs
- L5 x D2: State Transitions & Presets
- L10 x D10: Invariants (Dictionary structure, idempotency)
"""

import pytest
from unittest.mock import MagicMock, patch
import config
from providers import (
    _is_openai_reasoning_model,
    _is_claude_model,
    OllamaProvider,
    OpenAICompatibleProvider,
    XKiroProvider,
)


@pytest.fixture(autouse=True)
def clean_sampling_env(tmp_path, monkeypatch):
    """Isolate config file for each test."""
    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_FILE", cfg_file)
    config._CONFIG_CACHE = None
    config._CONFIG_MTIME = None
    config.reset_sampling_params()
    yield
    config._CONFIG_CACHE = None
    config._CONFIG_MTIME = None


# =============================================================================
# L0: Smoke Tests
# =============================================================================
class TestSamplingSmoke:
    def test_get_and_set_all_params(self):
        config.set_temperature(0.7)
        assert config.get_temperature() == 0.7

        config.set_top_p(0.85)
        assert config.get_top_p() == 0.85

        config.set_top_k(30)
        assert config.get_top_k() == 30

        config.set_min_p(0.08)
        assert config.get_min_p() == 0.08

        config.set_repeat_penalty(1.15)
        assert config.get_repeat_penalty() == 1.15

        params = config.get_sampling_params()
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.85
        assert params["top_k"] == 30
        assert params["min_p"] == 0.08
        assert params["repeat_penalty"] == 1.15


# =============================================================================
# L1 x D7: API Contract & Architecture Specialization
# =============================================================================
class TestSamplingAPIContract:
    def test_reasoning_model_detector(self):
        assert _is_openai_reasoning_model("o1") is True
        assert _is_openai_reasoning_model("o1-mini") is True
        assert _is_openai_reasoning_model("o1-preview") is True
        assert _is_openai_reasoning_model("o3") is True
        assert _is_openai_reasoning_model("o3-mini") is True
        assert _is_openai_reasoning_model("openai/o3-mini") is True
        assert _is_openai_reasoning_model("gpt-4o") is False
        assert _is_openai_reasoning_model("deepseek/deepseek-r1") is False
        assert _is_openai_reasoning_model("") is False
        assert _is_openai_reasoning_model(None) is False

    def test_claude_model_detector(self):
        assert _is_claude_model("claude-3-7-sonnet") is True
        assert _is_claude_model("anthropic/claude-3.5-haiku") is True
        assert _is_claude_model("gpt-4o") is False
        assert _is_claude_model("") is False

    def test_openai_reasoning_models_omit_temperature_and_top_p(self):
        """o1/o3-mini reject temperature and top_p with HTTP 400, and need max_completion_tokens."""
        config.set_temperature(0.2)
        config.set_top_p(0.9)
        config.set_top_k(50)
        config.set_min_p(0.05)
        config.set_max_generation_tokens(4096)

        provider = XKiroProvider(api_key="sk-test", base_url="https://api.xkiro.com/v1")
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__iter__.return_value = []
        mock_client.chat.completions.create.return_value = mock_stream
        provider._get_client = MagicMock(return_value=mock_client)

        list(provider.stream_chat(
            model="o3-mini",
            messages=[{"role": "user", "content": "solve math"}],
            temperature=0.2,
        ))

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        # Contract assertions:
        assert "temperature" not in call_kwargs, "o3-mini must never receive temperature"
        assert "top_p" not in call_kwargs, "o3-mini must never receive top_p"
        assert "max_tokens" not in call_kwargs, "o3-mini must never receive max_tokens"
        assert call_kwargs.get("max_completion_tokens") == 4096
        # extra_body non-standard samplers should not be sent to o1/o3
        assert "extra_body" not in call_kwargs or "top_k" not in call_kwargs.get("extra_body", {})

    def test_claude_enforces_temperature_1_when_thinking(self):
        """Claude 3.7 with extended thinking strictly requires temperature=1.0."""
        config.set_temperature(0.2)
        config.set_thinking_mode("on")
        config.set_thinking_budget(2048)

        provider = XKiroProvider(api_key="sk-test", base_url="https://api.xkiro.com/v1")
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__iter__.return_value = []
        mock_client.chat.completions.create.return_value = mock_stream
        provider._get_client = MagicMock(return_value=mock_client)

        list(provider.stream_chat(
            model="anthropic/claude-3-7-sonnet",
            messages=[{"role": "user", "content": "think and write code"}],
            temperature=0.2,
        ))

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("temperature") == 1.0

    def test_sync_chat_omits_temperature_for_reasoning_models(self):
        """sync_chat must also omit temperature for o1/o3-mini to prevent 400 crashes."""
        provider = XKiroProvider(api_key="sk-test", base_url="https://api.xkiro.com/v1")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="answer"))]
        mock_client.chat.completions.create.return_value = mock_response
        provider._get_client = MagicMock(return_value=mock_client)

        result = provider.sync_chat("o1-preview", [{"role": "user", "content": "hi"}], temperature=0.5)
        assert result == "answer"
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "temperature" not in call_kwargs


# =============================================================================
# L2 x D1: Boundary & Range Hardening
# =============================================================================
class TestSamplingBoundaries:
    def test_greedy_decoding_temperature_zero_not_falsy(self):
        """temperature=0.0 must be preserved as 0.0, not evaluated as falsy."""
        config.set_temperature(0.0)
        assert config.get_temperature() == 0.0

        provider = OllamaProvider.__new__(OllamaProvider)
        mock_client = MagicMock()
        mock_client.chat.return_value = []
        provider._ollama = mock_client

        list(provider.stream_chat(
            model="qwen2.5-coder",
            messages=[{"role": "user", "content": "code"}],
            temperature=0.0,
        ))

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["options"]["temperature"] == 0.0

    def test_temperature_clamping(self):
        config.set_temperature(-1.5)
        assert config.get_temperature() == 0.0

        config.set_temperature(3.5)
        assert config.get_temperature() == 2.0

    def test_top_p_clamping(self):
        config.set_top_p(0.0)
        assert config.get_top_p() == 0.0001

        config.set_top_p(1.5)
        assert config.get_top_p() == 1.0

        config.set_top_p(None)
        assert config.get_top_p() is None

    def test_min_p_clamping(self):
        config.set_min_p(-0.5)
        assert config.get_min_p() == 0.0

        config.set_min_p(2.0)
        assert config.get_min_p() == 1.0

        config.set_min_p(None)
        assert config.get_min_p() is None

    def test_top_k_clamping(self):
        config.set_top_k(0)
        assert config.get_top_k() is None

        config.set_top_k(-10)
        assert config.get_top_k() is None

        config.set_top_k(40)
        assert config.get_top_k() == 40

    def test_repeat_penalty_clamping(self):
        config.set_repeat_penalty(0.1)
        assert config.get_repeat_penalty() == 0.5

        config.set_repeat_penalty(5.0)
        assert config.get_repeat_penalty() == 2.5


# =============================================================================
# L4 x D4: Adversarial & Invalid CLI Inputs
# =============================================================================
class TestSamplingCLIAdversarial:
    def test_cli_preset_subcommand(self):
        from src.cli.commands.config_commands import handle_sampling
        from src.cli.commands.base import CommandContext

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling preset code")
        res = handle_sampling(ctx)
        assert res.should_exit is False

        assert config.get_temperature() == 0.2
        assert config.get_min_p() == 0.05
        assert config.get_repeat_penalty() == 1.05

    def test_cli_preset_precise(self):
        from src.cli.commands.config_commands import handle_sampling
        from src.cli.commands.base import CommandContext

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling preset precise")
        res = handle_sampling(ctx)
        assert res.should_exit is False

        assert config.get_temperature() == 0.0
        assert config.get_min_p() is None

    def test_cli_invalid_preset_does_not_crash(self):
        from src.cli.commands.config_commands import handle_sampling
        from src.cli.commands.base import CommandContext

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling preset non_existent_preset_xyz")
        res = handle_sampling(ctx)
        assert res.should_exit is False

    def test_cli_corrupted_number_handled(self):
        from src.cli.commands.config_commands import handle_sampling
        from src.cli.commands.base import CommandContext

        ctx = CommandContext(agent=MagicMock(), user_input="/sampling min_p not_a_number")
        res = handle_sampling(ctx)
        assert res.should_exit is False
        assert config.get_min_p() is None


# =============================================================================
# L5 x D2: State Transitions & Presets
# =============================================================================
class TestSamplingStateTransitions:
    def test_apply_and_switch_presets(self):
        p1 = config.apply_sampling_preset("creative")
        assert p1["temperature"] == 0.8
        assert config.get_top_p() == 0.95
        assert config.get_top_k() == 50

        p2 = config.apply_sampling_preset("code")
        assert p2["temperature"] == 0.2
        assert config.get_min_p() == 0.05
        assert config.get_top_p() is None

        config.reset_sampling_params()
        assert config.get_temperature() == 0.3
        assert config.get_min_p() is None
        assert config.get_top_p() is None


# =============================================================================
# L10 x D10: Invariants
# =============================================================================
class TestSamplingInvariants:
    def test_sampling_params_dict_invariant(self):
        expected_keys = {"temperature", "top_p", "top_k", "min_p", "repeat_penalty"}
        params = config.get_sampling_params()
        assert set(params.keys()) == expected_keys

        config.apply_sampling_preset("code")
        params2 = config.get_sampling_params()
        assert set(params2.keys()) == expected_keys
