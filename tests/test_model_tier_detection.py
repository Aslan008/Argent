"""Model tier detection. A misread tier is expensive and SILENT: the model
gets the wrong system prompt, a slimmed toolset and the wrong history budget,
with nothing in the UI saying so."""

import pytest

import config
import prompt_compressor


@pytest.fixture(autouse=True)
def no_override(monkeypatch):
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)


class TestCloudTagging:
    @pytest.mark.parametrize("name", [
        "minimax-m3:cloud", "qwen3-max:cloud", "kimi-k2-cloud",
    ])
    def test_explicit_cloud_tag_wins(self, name):
        # Aggregators tag hosted models this way; such a model must never fall
        # through to the small-model keyword heuristic.
        assert config.get_model_size_category(name) == "cloud"

    @pytest.mark.parametrize("name", ["glm-4.6", "gpt-4o", "claude-sonnet-4.6", "gemini-2.5-pro"])
    def test_known_cloud_families(self, name):
        assert config.get_model_size_category(name) == "cloud"


class TestKeywordFalsePositives:
    def test_minimax_is_not_mini(self):
        """The regression this test exists for: 'mini' inside 'minimax' made a
        large cloud model read as tiny."""
        assert config.get_model_size_category("minimax-m2") != "tiny"

    def test_ministral_is_not_mini(self):
        assert config.get_model_size_category("ministral-8b") != "tiny"

    @pytest.mark.parametrize("name,expected", [
        ("phi-3-mini", "tiny"),        # genuine bounded keyword
        ("tinyllama", "tiny"),         # 'tiny' prefix is reliable
        ("tinydolphin", "tiny"),
        ("nemotron-nano-9b", "medium"),  # explicit size beats the marketing word
    ])
    def test_keyword_boundaries(self, name, expected):
        assert config.get_model_size_category(name) == expected


class TestSizeParsing:
    @pytest.mark.parametrize("name,expected", [
        ("qwen2.5-1.5b", "tiny"),
        ("gemma:2b", "tiny"),
        ("phi:3.8b", "small"),
        ("mistral:7b", "medium"),      # boundary: small is 3–7B, 7 lands in medium
        ("qwen3.6:27b", "large"),
        ("llama3.1:70b", "large"),
        ("lfm2.5-8b-a1b", "tiny"),     # MoE: classified by ACTIVE params
    ])
    def test_parses_parameter_counts(self, name, expected):
        assert config.get_model_size_category(name) == expected

    def test_unknown_name_defaults_to_medium(self):
        assert config.get_model_size_category("some-unknown-model") == "medium"

    def test_empty_name(self):
        assert config.get_model_size_category("") == "medium"


class TestSingleTierResolution:
    """The prompt must be COMPRESSED for the same tier it was BUILT for.
    Deriving the tier twice let the two disagree, silently stripping sections
    the prompt was written to contain."""

    def test_explicit_category_overrides_the_lookup(self, monkeypatch):
        monkeypatch.setattr(prompt_compressor, "get_model_size_category",
                            lambda name: "tiny")
        prompt = "## 4. PLANNING MODE & ARTIFACTS\nPlan first.\n\n## 7. COMMUNICATION\nBe clear.\n"

        # Without the explicit tier the section is stripped as 'tiny'...
        assert "PLANNING MODE" not in prompt_compressor.compress_system_prompt(prompt, "whatever")
        # ...and with it, the caller's tier decides.
        kept = prompt_compressor.compress_system_prompt(prompt, "whatever", category="cloud")
        assert "PLANNING MODE" in kept

    def test_agent_passes_its_resolved_tier(self, monkeypatch):
        """A stubbed tier in the agent must reach the compressor, otherwise the
        prompt depends on the user's live config instead of the test's."""
        from unittest.mock import MagicMock
        import agent as agent_module
        from agent import ArgentAgent

        mem = MagicMock()
        mem.data = {}
        monkeypatch.setattr(agent_module, "memory", mem)
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "cloud")
        # The compressor's own lookup says tiny — it must NOT win.
        monkeypatch.setattr(prompt_compressor, "get_model_size_category", lambda name: "tiny")

        prompt = ArgentAgent().build_system_prompt()
        assert "PLANNING MODE" in prompt and "Blind Spot Pass" in prompt
