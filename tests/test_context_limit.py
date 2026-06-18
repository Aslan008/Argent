"""Context-window overflow detection/parsing + history budget reservation."""

import pytest

from src.agent.context_limit import (
    is_context_overflow,
    parse_context_limit,
    effective_history_budget,
)

# The real KoboldCPP error a user hit.
KOBOLD_ERR = (
    "KOBOLDCPP error (HTTP 400): Error code: 400 - {'error': {'code': 400, "
    "'message': 'request (42722 tokens) exceeds the available context size "
    "(42240 tokens), try increasing it', 'type': 'exceed_context_size_error', "
    "'n_prompt_tokens': 42722, 'n_ctx': 42240}}"
)


class TestDetection:
    def test_detects_kobold_overflow(self):
        assert is_context_overflow(KOBOLD_ERR)

    @pytest.mark.parametrize("msg", [
        "This model's maximum context length is 8192 tokens",
        "Requested 5000 tokens but context window is 4096",
        "prompt is too many tokens",
    ])
    def test_detects_other_providers(self, msg):
        assert is_context_overflow(msg)

    def test_ignores_unrelated_errors(self):
        assert not is_context_overflow("Connection refused")
        assert not is_context_overflow("HTTP 500 internal server error")
        assert not is_context_overflow("")


class TestParsing:
    def test_parses_kobold_window(self):
        assert parse_context_limit(KOBOLD_ERR) == 42240

    def test_parses_openai_style(self):
        assert parse_context_limit("maximum context length is 8192") == 8192

    def test_returns_none_without_a_number(self):
        assert parse_context_limit("context size exceeded") is None
        assert parse_context_limit("") is None


class TestBudget:
    def test_leaves_room_for_everything_else(self):
        window, sys = 42240, 3000
        budget = effective_history_budget(window, sys, response_reserve=1024, tools_reserve=2048)
        # system + tools + response + history must fit under the real window
        assert budget + sys + 2048 + 1024 < window

    def test_never_below_floor(self):
        assert effective_history_budget(1000, 5000, floor=2048) == 2048

    def test_smaller_window_yields_smaller_budget(self):
        assert effective_history_budget(8000, 1000) < effective_history_budget(32000, 1000)
