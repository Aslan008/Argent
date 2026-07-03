from types import SimpleNamespace

from prompt_toolkit.document import Document

from src.agent.strategy import TinyLocalStrategy, CloudStrategy
from src.cli.cli_prompt import (
    ArgentCommandCompleter, _describe, _tier_label, build_bottom_toolbar,
    _fmt_tokens, _context_segment,
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

    def test_context_meter_shown_when_cached(self):
        agent = SimpleNamespace(
            provider="ollama", model_name="qwen2.5-coder", strategy=CloudStrategy(),
            _last_context_usage={"percent": 30, "tokens": 6000, "max": 20000},
        )
        text = build_bottom_toolbar(agent, {"mode": "CHAT"})().value
        assert "ctx:" in text and "30%" in text and "6.0k/20.0k" in text

    def test_context_meter_absent_without_cache(self):
        agent = SimpleNamespace(provider="ollama", model_name="m", strategy=CloudStrategy())
        text = build_bottom_toolbar(agent, {"mode": "CHAT"})().value
        assert "ctx:" not in text  # nothing cached yet -> no meter


class TestFormatTokens:
    def test_thousands_suffix(self):
        assert _fmt_tokens(8600) == "8.6k"
        assert _fmt_tokens(20000) == "20.0k"

    def test_below_thousand_plain(self):
        assert _fmt_tokens(900) == "900"
        assert _fmt_tokens(0) == "0"

    def test_non_numeric_safe(self):
        assert _fmt_tokens(None) == "?"


class TestContextSegment:
    def test_empty_when_no_usage(self):
        assert _context_segment(None) == ""
        assert _context_segment({}) == ""

    def test_green_below_50(self):
        seg = _context_segment({"percent": 42, "tokens": 8600, "max": 20000})
        assert "42%" in seg and "#4caf50" in seg and "8.6k/20.0k" in seg

    def test_amber_between_50_and_80(self):
        assert "#ffb300" in _context_segment({"percent": 65, "tokens": 1, "max": 2})

    def test_red_at_or_above_80(self):
        assert "#e53935" in _context_segment({"percent": 90, "tokens": 1, "max": 2})

    def test_no_token_ratio_when_max_unknown(self):
        seg = _context_segment({"percent": 10, "tokens": 5, "max": 0})
        # No "tokens/max" ratio when the window size is unknown: the fragment
        # ends right after the percentage (the closing </style> tag).
        assert "ctx:" in seg and "10%" in seg and seg.endswith("</style>")
