"""Anthropic prompt caching: the stable system prefix is marked cacheable."""

from providers import OpenAICompatibleProvider

apply = OpenAICompatibleProvider._apply_prompt_cache

BIG = "You are Argent. " + ("rules and project memory. " * 200)   # > 2000 chars


def _msgs(system=BIG):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "hi"},
    ]


class TestAnthropicModels:
    def test_system_prompt_gets_cache_control(self):
        out = apply(_msgs(), "anthropic/claude-sonnet-4.6")
        block = out[0]["content"]
        assert isinstance(block, list)
        assert block[0]["type"] == "text"
        assert block[0]["text"] == BIG
        assert block[0]["cache_control"] == {"type": "ephemeral"}

    def test_non_system_messages_untouched(self):
        out = apply(_msgs(), "claude-opus-4-8")
        assert out[1] == {"role": "user", "content": "hi"}

    def test_only_one_breakpoint_is_set(self):
        msgs = [
            {"role": "system", "content": BIG},
            {"role": "user", "content": "a"},
            {"role": "system", "content": BIG},      # ephemeral tail context
        ]
        out = apply(msgs, "anthropic/claude-3.7-sonnet")
        assert isinstance(out[0]["content"], list)
        assert isinstance(out[2]["content"], str)     # second one left alone

    def test_short_system_prompt_is_left_alone(self):
        out = apply(_msgs("short prompt"), "anthropic/claude-sonnet-4.6")
        assert out[0]["content"] == "short prompt"    # below the cacheable minimum


class TestOtherModels:
    def test_openai_model_untouched(self):
        # OpenAI caches automatically; the field would just be noise.
        msgs = _msgs()
        assert apply(msgs, "gpt-4o") == msgs

    def test_local_model_untouched(self):
        msgs = _msgs()
        assert apply(msgs, "qwen3.6:27b") == msgs

    def test_missing_model_name_is_safe(self):
        msgs = _msgs()
        assert apply(msgs, None) == msgs


class TestPipelineIntegration:
    def test_stream_chat_sends_cache_control(self, monkeypatch):
        captured = {}

        class _Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        captured.update(kwargs)
                        return iter(())

        class _Concrete(OpenAICompatibleProvider):
            name = "test"

            def list_models(self):
                return []

            def validate_config(self):
                return None

        provider = _Concrete.__new__(_Concrete)
        provider._get_client = lambda: _Client()
        provider._extra_create_kwargs = lambda: {}
        import openai
        provider._openai = openai

        list(provider.stream_chat("anthropic/claude-sonnet-4.6", _msgs(), tools=None))

        system_content = captured["messages"][0]["content"]
        assert isinstance(system_content, list)
        assert system_content[0]["cache_control"] == {"type": "ephemeral"}
