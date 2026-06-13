from types import SimpleNamespace

from prompt_toolkit.document import Document

from src.agent.strategy import TinyLocalStrategy, CloudStrategy
from src.cli.cli_prompt import (
    ArgentCommandCompleter, _describe, _tier_label, build_bottom_toolbar,
)


COMMANDS = ["/help", "/model", "/mcp", "/mcp list", "/mcp add", "/doctor"]


def completions(text):
    comp = ArgentCommandCompleter(lambda: COMMANDS)
    doc = Document(text, len(text))
    return list(comp.get_completions(doc, None))


class TestCommandCompleter:
    def test_completes_prefix(self):
        texts = [c.text for c in completions("/mo")]
        assert texts == ["/model"]

    def test_completes_all_on_bare_slash(self):
        texts = [c.text for c in completions("/")]
        assert set(texts) == set(COMMANDS)

    def test_subcommands_match(self):
        texts = [c.text for c in completions("/mcp ")]
        # "/mcp " has a trailing space -> not a single token, no completion.
        assert texts == []

    def test_completion_carries_description(self):
        comp = next(c for c in completions("/doc"))
        assert "иагностика" in comp.display_meta_text  # самодиагностика

    def test_no_completion_for_plain_text(self):
        assert completions("напиши функцию") == []

    def test_no_completion_after_command_with_args(self):
        assert completions("/model gpt-4o") == []

    def test_case_insensitive(self):
        assert [c.text for c in completions("/MO")] == ["/model"]


class TestDescribe:
    def test_base_command(self):
        assert _describe("/doctor")

    def test_subcommand_inherits_base(self):
        assert _describe("/mcp list") == _describe("/mcp")

    def test_unknown_is_empty(self):
        assert _describe("/nonsense") == ""


class TestTierLabel:
    def test_tiny(self):
        agent = SimpleNamespace(strategy=TinyLocalStrategy())
        assert _tier_label(agent) == "tiny"

    def test_cloud(self):
        agent = SimpleNamespace(strategy=CloudStrategy())
        assert _tier_label(agent) == "cloud"


class TestBottomToolbar:
    def test_renders_state_fields(self):
        agent = SimpleNamespace(provider="openrouter", model_name="anthropic/claude-3.7-sonnet",
                                strategy=CloudStrategy())
        toolbar = build_bottom_toolbar(agent, {"mode": "AUTO"})
        text = toolbar().value  # HTML.value is the raw markup string
        assert "AUTO" in text
        assert "openrouter" in text
        assert "claude-3.7-sonnet" in text
        assert "cloud" in text

    def test_survives_missing_fields(self):
        toolbar = build_bottom_toolbar(SimpleNamespace(), {})
        # Must not raise even if agent has no attributes / mode missing.
        result = toolbar()
        assert result is not None
