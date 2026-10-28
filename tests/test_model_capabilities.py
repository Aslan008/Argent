"""The toolbar's `n/a` note: only settings that provably never reach the model."""
from argent import model_capabilities as caps


class TestKoboldCpp:
    def test_neither_thinking_nor_depth_is_sent(self):
        assert caps.inert_settings("koboldcpp", "any-model") == ["thinking", "effort"]

    def test_the_label_names_both(self):
        assert caps.inert_label("koboldcpp", "any-model") == "n/a: thinking, depth"


class TestOllama:
    def test_an_ordinary_model_ignores_the_depth_level(self):
        assert caps.inert_settings("ollama", "qwen3:8b") == ["effort"]

    def test_the_gpt_oss_family_accepts_a_level(self):
        assert caps.inert_settings("ollama", "gpt-oss:20b") == []

    def test_a_probed_model_that_cannot_think_marks_the_toggle(self, monkeypatch):
        from argent import providers

        monkeypatch.setitem(providers.OllamaProvider._think_capable, "no-think:1b", False)
        assert caps.inert_settings("ollama", "no-think:1b") == ["thinking", "effort"]
        assert caps.inert_label("ollama", "no-think:1b") == "n/a: thinking, depth"

    def test_an_unprobed_model_is_not_accused(self, monkeypatch):
        from argent import providers

        monkeypatch.delitem(providers.OllamaProvider._think_capable, "unknown:7b", raising=False)
        assert caps.inert_settings("ollama", "unknown:7b") == ["effort"]


class TestCloudProviders:
    def test_nothing_is_claimed_without_offline_metadata(self):
        for provider in ("claude", "openrouter", "zai", "deepseek", "wormsoft", "xkiro"):
            assert caps.inert_settings(provider, "some-model") == []
            assert caps.inert_label(provider, "some-model") == ""

    def test_an_unknown_provider_is_silent(self):
        assert caps.inert_label("", "") == ""
