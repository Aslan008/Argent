import json
import pytest
from unittest.mock import patch, MagicMock

import config
import providers
from providers import ClaudeProvider
from src.agent.strategy import get_model_strategy, CloudStrategy
from src.agent.critic import parse_critic_model


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    yield store


class TestClaudeConfig:
    def test_defaults(self):
        assert config.get_claude_url() == "https://api.anthropic.com"
        assert config.get_claude_api_key() is None

    def test_set_and_get_key(self):
        config.set_claude_api_key("sk-ant-test-key")
        assert config.get_claude_api_key() == "sk-ant-test-key"

    def test_custom_url(self):
        config.set_claude_url("https://custom.anthropic.proxy")
        assert config.get_claude_url() == "https://custom.anthropic.proxy"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        assert config.get_claude_api_key() == "sk-ant-from-env"


class TestClaudeFactory:
    def test_create_claude_provider_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "get_provider", lambda: "claude")
        config.set_claude_api_key("sk-ant-factory")
        provider = providers.create_provider()
        assert isinstance(provider, ClaudeProvider)
        assert provider.name == "claude"
        assert provider._api_key == "sk-ant-factory"
        assert provider._base_url == "https://api.anthropic.com"

    def test_create_claude_provider_explicit_name(self):
        config.set_claude_api_key("sk-ant-explicit")
        provider = providers.create_provider("claude")
        assert isinstance(provider, ClaudeProvider)
        assert provider.name == "claude"
        assert provider._api_key == "sk-ant-explicit"

    def test_create_claude_provider_anthropic_alias(self):
        config.set_claude_api_key("sk-ant-alias")
        provider = providers.create_provider("anthropic")
        assert isinstance(provider, ClaudeProvider)
        assert provider.name == "claude"
        assert provider._api_key == "sk-ant-alias"


class TestClaudeValidation:
    def test_missing_key_reports_actionable_error(self):
        p = ClaudeProvider(api_key=None, base_url="https://api.anthropic.com")
        msg = p.validate_config()
        assert msg and "API key" in msg and "console.anthropic.com" in msg

    def test_present_key_validates(self):
        p = ClaudeProvider(api_key="sk-ant-valid", base_url="https://api.anthropic.com")
        assert p.validate_config() is None


class TestClaudeListModels:
    def test_without_a_key_or_a_saved_copy_only_the_small_seed_is_offered(self):
        p = ClaudeProvider(api_key=None, base_url="https://api.anthropic.com")
        models = p.list_models()
        assert {"claude-sonnet-5-5", "claude-opus-5-5"} <= set(models)
        assert not [m for m in models if m.endswith("-latest")], "the old list held ids that never existed"
        assert p.catalog().source == "seed" and not p.catalog().authoritative

    def test_fetches_models_from_api_endpoint_newest_first(self):
        from datetime import datetime, timezone
        from types import SimpleNamespace

        def model(mid, year):
            return SimpleNamespace(id=mid, created_at=datetime(year, 1, 1, tzinfo=timezone.utc), display_name=mid,
                                   max_input_tokens=200000, max_tokens=64000, capabilities=None)

        p = ClaudeProvider(api_key="sk-ant-test", base_url="https://api.anthropic.com")
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(
            auto_paging_iter=lambda: iter([model("claude-3-5-sonnet-20241022", 2024),
                                           model("claude-3-7-sonnet-20250219", 2025)]))

        with patch.object(p, "_get_client", return_value=mock_client):
            assert p.list_models() == ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022"]
            assert p.catalog().authoritative


class TestClaudeErrorHandling:
    def test_handles_401_invalid_key(self):
        p = ClaudeProvider(api_key="sk-ant-bad", base_url="https://api.anthropic.com")
        err = MagicMock()
        err.status_code = 401
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Invalid API key" in str(excinfo.value)
        assert "console.anthropic.com" in str(excinfo.value)

    def test_handles_402_insufficient_credits(self):
        p = ClaudeProvider(api_key="sk-ant-empty", base_url="https://api.anthropic.com")
        err = MagicMock()
        err.status_code = 402
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Insufficient balance" in str(excinfo.value) or "credits" in str(excinfo.value)

    def test_handles_429_rate_limit(self):
        p = ClaudeProvider(api_key="sk-ant-valid", base_url="https://api.anthropic.com")
        err = MagicMock()
        err.status_code = 429
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Rate limited" in str(excinfo.value)
        assert excinfo.value.retryable is True

    def test_handles_529_overloaded(self):
        p = ClaudeProvider(api_key="sk-ant-valid", base_url="https://api.anthropic.com")
        err = MagicMock()
        err.status_code = 529
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "overloaded" in str(excinfo.value)
        assert excinfo.value.retryable is True

    def test_handles_500_server_error(self):
        p = ClaudeProvider(api_key="sk-ant-valid", base_url="https://api.anthropic.com")
        err = MagicMock()
        err.status_code = 500
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Server error" in str(excinfo.value)
        assert excinfo.value.retryable is True


class TestClaudeStrategy:
    def test_claude_always_uses_cloud_strategy(self):
        for model in [
            "claude-sonnet-5-5",
            "claude-opus-5-5",
            "claude-haiku-5-5",
            "claude-5-5-sonnet-latest",
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
        ]:
            assert isinstance(get_model_strategy(model, "claude"), CloudStrategy)
            assert isinstance(get_model_strategy(model, "anthropic"), CloudStrategy)

    def test_cloud_strategy_properties(self):
        s = get_model_strategy("claude-3-7-sonnet-latest", "claude")
        assert s.supports_native_tools() is True
        assert s.wants_constrained_decoding() is False
        assert s.wants_objective_anchor() is False


class TestClaudeCriticParser:
    def test_critic_claude_prefix(self):
        provider, model = parse_critic_model("claude:claude-3-7-sonnet-latest")
        assert provider == "claude"
        assert model == "claude-3-7-sonnet-latest"

    def test_critic_anthropic_prefix(self):
        provider, model = parse_critic_model("anthropic:claude-3-5-sonnet-latest")
        assert provider == "anthropic"
        assert model == "claude-3-5-sonnet-latest"


class TestClaudeMessageConversion:
    def test_converts_system_message(self):
        p = ClaudeProvider(api_key="sk-test")
        messages = [
            {"role": "system", "content": "You are Argent."},
            {"role": "user", "content": "Hello"},
        ]
        system_param, conv = p._convert_messages(messages)
        assert system_param == "You are Argent."
        assert len(conv) == 1
        assert conv[0]["role"] == "user"
        assert conv[0]["content"] == [{"type": "text", "text": "Hello"}]

    def test_caches_large_system_prompt(self):
        p = ClaudeProvider(api_key="sk-test")
        large_prompt = "A" * 2500
        messages = [
            {"role": "system", "content": large_prompt},
            {"role": "user", "content": "Hi"},
        ]
        system_param, conv = p._convert_messages(messages)
        assert isinstance(system_param, list)
        assert system_param[0]["type"] == "text"
        assert system_param[0]["cache_control"] == {"type": "ephemeral"}

    def test_alternating_roles_with_multiple_tool_results(self):
        p = ClaudeProvider(api_key="sk-test")
        messages = [
            {"role": "user", "content": "Check files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "read_file", "arguments": "{\"path\":\"a.txt\"}"}},
                    {"id": "call_2", "function": {"name": "list_dir", "arguments": "{\"path\":\".\"}"}},
                ]
            },
            p.format_tool_result("file contents", "call_1"),
            p.format_tool_result("dir contents", "call_2"),
            {"role": "assistant", "content": "Done checking"},
        ]
        system_param, conv = p._convert_messages(messages)
        assert system_param is None
        # Roles must strictly alternate: user -> assistant -> user (merged tool results) -> assistant
        assert [m["role"] for m in conv] == ["user", "assistant", "user", "assistant"]

        tool_user_msg = conv[2]
        assert len(tool_user_msg["content"]) == 2
        assert tool_user_msg["content"][0]["type"] == "tool_result"
        assert tool_user_msg["content"][0]["tool_use_id"] == "call_1"
        assert tool_user_msg["content"][0]["content"] == "file contents"
        assert tool_user_msg["content"][1]["type"] == "tool_result"
        assert tool_user_msg["content"][1]["tool_use_id"] == "call_2"
        assert tool_user_msg["content"][1]["content"] == "dir contents"

    def test_user_image_attachment(self):
        p = ClaudeProvider(api_key="sk-test")
        messages = [
            {
                "role": "user",
                "content": "Look at this",
                "images": [{"data": "base64data", "media_type": "image/jpeg"}]
            }
        ]
        _, conv = p._convert_messages(messages)
        blocks = conv[0]["content"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "image"
        assert blocks[0]["source"]["data"] == "base64data"
        assert blocks[0]["source"]["media_type"] == "image/jpeg"
        assert blocks[1]["type"] == "text"
        assert blocks[1]["text"] == "Look at this"

    def test_converts_tools_schema(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            }
        ]
        conv_tools = ClaudeProvider._convert_tools(tools)
        assert len(conv_tools) == 1
        assert conv_tools[0]["name"] == "search"
        assert conv_tools[0]["description"] == "Search web"
        assert conv_tools[0]["input_schema"] == {"type": "object", "properties": {"q": {"type": "string"}}}


class TestClaudeStreaming:
    def test_stream_chat_emits_text_and_usage(self):
        p = ClaudeProvider(api_key="sk-test")

        # Mock events
        start_event = MagicMock()
        start_event.type = "message_start"
        start_event.message.usage.input_tokens = 50
        start_event.message.usage.cache_read_input_tokens = 10

        cb_delta = MagicMock()
        cb_delta.type = "content_block_delta"
        cb_delta.delta.type = "text_delta"
        cb_delta.delta.text = "Hello world"

        msg_delta = MagicMock()
        msg_delta.type = "message_delta"
        msg_delta.delta.stop_reason = "end_turn"
        msg_delta.usage.output_tokens = 25

        mock_client = MagicMock()
        mock_client.messages.create.return_value = [start_event, cb_delta, msg_delta]

        with patch.object(p, "_get_client", return_value=mock_client):
            chunks = list(p.stream_chat(
                model="claude-3-5-sonnet-latest",
                messages=[{"role": "user", "content": "hi"}]
            ))

        text_chunks = [c for c in chunks if c["content"]]
        assert len(text_chunks) == 1
        assert text_chunks[0]["content"] == "Hello world"

        usage_chunks = [c for c in chunks if "usage" in c]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["usage"]["prompt"] == 50
        assert usage_chunks[0]["usage"]["completion"] == 25
        assert usage_chunks[0]["usage"]["total"] == 75
        assert usage_chunks[0]["usage"]["cached"] == 10
        assert usage_chunks[0]["truncated"] is False

    def test_stream_chat_tool_use_and_thinking(self):
        p = ClaudeProvider(api_key="sk-test")

        # Thinking delta
        think_event = MagicMock()
        think_event.type = "content_block_delta"
        think_event.delta.type = "thinking_delta"
        think_event.delta.thinking = "Let me search..."

        # Tool start
        tool_start = MagicMock()
        tool_start.type = "content_block_start"
        tool_start.index = 1
        tool_start.content_block.type = "tool_use"
        tool_start.content_block.id = "tool_123"
        tool_start.content_block.name = "web_search"

        # Tool args delta
        tool_delta = MagicMock()
        tool_delta.type = "content_block_delta"
        tool_delta.index = 1
        tool_delta.delta.type = "input_json_delta"
        tool_delta.delta.partial_json = '{"query": "python"}'

        mock_client = MagicMock()
        mock_client.messages.create.return_value = [think_event, tool_start, tool_delta]

        with patch.object(p, "_get_client", return_value=mock_client):
            chunks = list(p.stream_chat(
                model="claude-3-7-sonnet-latest",
                messages=[{"role": "user", "content": "Search"}]
            ))

        thinking_chunks = [c for c in chunks if c["thinking"]]
        assert len(thinking_chunks) == 1
        assert thinking_chunks[0]["thinking"] == "Let me search..."

        tc_deltas = [c["tool_call_deltas"] for c in chunks if c["tool_call_deltas"]]
        assert len(tc_deltas) == 2
        # First: name & id
        assert tc_deltas[0][0]["id"] == "tool_123"
        assert tc_deltas[0][0]["function_name_delta"] == "web_search"
        # Second: args
        assert tc_deltas[1][0]["function_arguments_delta"] == '{"query": "python"}'
 
    def test_stream_chat_claude_5_5_thinking(self):
        p = ClaudeProvider(api_key="sk-test")
        import config
        config.set_thinking_mode("auto")
        config.set_thinking_budget(4096)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = []

        with patch.object(p, "_get_client", return_value=mock_client):
            list(p.stream_chat(
                model="claude-sonnet-5-5",
                messages=[{"role": "user", "content": "analyze"}]
            ))

        create_kwargs = mock_client.messages.create.call_args.kwargs
        assert "thinking" in create_kwargs
        assert create_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
        # The installed SDK no longer takes sampling parameters at all; sending one is a TypeError.
        assert not {"temperature", "top_p", "top_k"} & set(create_kwargs)


class TestClaudeSyncChat:
    def test_sync_chat_success(self):
        p = ClaudeProvider(api_key="sk-test")
        mock_resp = MagicMock()
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Commit message: Fix bug"
        mock_resp.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp

        with patch.object(p, "_get_client", return_value=mock_client):
            res = p.sync_chat("claude-3-5-haiku-latest", [{"role": "user", "content": "generate commit"}])
            assert res == "Commit message: Fix bug"
