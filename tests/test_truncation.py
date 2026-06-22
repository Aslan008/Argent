"""Regression tests for the truncation auto-continue loop.

Old behaviour: a model writing a file bigger than the generation limit got
truncated, was asked to "continue where you left off", restarted the whole
tool call from scratch, got truncated again — an infinite loop. These tests
drive process_user_input with a provider that always truncates and assert
the turn now terminates with bounded retries and chunked-write guidance.
"""

from itertools import islice
from unittest.mock import MagicMock

import pytest

import agent as agent_module
from agent import ArgentAgent, MAX_TRUNCATE_CONTINUES


class FakeProvider:
    """Scripted provider: yields one chunk list per stream_chat call."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0

    def validate_config(self):
        return None

    def supports_constrained_decoding(self):
        return False

    def stream_chat(self, **kwargs):
        script = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        yield from script

    def format_tool_result(self, content, tool_call_id=None):
        return {"role": "tool", "content": content}


HANG_CAP = 500  # consuming more chunks than this means the old infinite loop


@pytest.fixture
def make_agent(monkeypatch):
    def _make(provider):
        fake_memory = MagicMock()
        fake_memory.data = {}
        monkeypatch.setattr(agent_module, "memory", fake_memory)
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: provider)
        # Hermetic token estimation: no HTTP to a live provider from unit tests.
        monkeypatch.setattr(agent_module, "estimate_tokens",
                            lambda text, model, prov: len(text) // 4)
        return ArgentAgent()
    return _make


def consume(gen):
    chunks = list(islice(gen, HANG_CAP))
    assert next(gen, None) is None, "generator did not terminate (infinite loop)"
    return chunks


PROSE_TRUNCATED = [{"content": "very long file content...", "truncated": True}]
# A non-file tool call: not salvageable, so it exercises the "smaller chunks"
# path. Salvageable file writes are covered in test_salvage.py.
TOOL_TRUNCATED = [{
    "tool_call_deltas": [{
        "index": 0, "id": "x",
        "function_name_delta": "run_command",
        "function_arguments_delta": '{"command": "echo incomplete...',
    }],
    "truncated": True,
}]


class TestProseTruncation:
    def test_always_truncating_prose_terminates_with_bounded_retries(self, make_agent):
        agent = make_agent(FakeProvider([PROSE_TRUNCATED]))
        chunks = consume(agent.process_user_input("напиши длинный текст"))
        errors = [c["content"] for c in chunks if c["type"] == "error"]
        continues = [e for e in errors if "Auto-continuing" in e]
        assert len(continues) == MAX_TRUNCATE_CONTINUES
        assert any("auto-continue stopped" in e for e in errors)

    def test_single_truncation_then_completion_continues_normally(self, make_agent):
        provider = FakeProvider([
            PROSE_TRUNCATED,
            [{"content": " the rest of the answer"}],
        ])
        agent = make_agent(provider)
        chunks = consume(agent.process_user_input("вопрос"))
        errors = [c["content"] for c in chunks if c["type"] == "error"]
        assert sum("Auto-continuing" in e for e in errors) == 1
        assert not any("auto-continue stopped" in e for e in errors)
        # The continue instruction was injected and the partial prose kept.
        roles = [(m.get("role"), m.get("content", "")) for m in agent.messages]
        assert any(r == "user" and "continue exactly where you left off" in c for r, c in roles)
        assert any(r == "assistant" and "very long file content" in c for r, c in roles)


class TestToolTruncation:
    def test_truncated_tool_call_gets_chunked_write_guidance(self, make_agent):
        agent = make_agent(FakeProvider([TOOL_TRUNCATED]))
        chunks = consume(agent.process_user_input("создай большой файл"))
        # Guidance to write in parts is injected instead of "continue".
        user_msgs = [m["content"] for m in agent.messages if m.get("role") == "user"]
        assert any("SMALLER STEPS" in m and "append_to_file" in m for m in user_msgs)
        assert not any("continue exactly where you left off" in m for m in user_msgs)

    def test_partial_tool_json_is_not_kept_in_history(self, make_agent):
        agent = make_agent(FakeProvider([TOOL_TRUNCATED]))
        consume(agent.process_user_input("создай большой файл"))
        assistant_msgs = [m for m in agent.messages if m.get("role") == "assistant"]
        assert not any("incomplete..." in str(m) for m in assistant_msgs)
        assert not any(m.get("tool_calls") for m in assistant_msgs)

    def test_always_truncating_tool_call_terminates(self, make_agent):
        agent = make_agent(FakeProvider([TOOL_TRUNCATED]))
        chunks = consume(agent.process_user_input("создай большой файл"))
        errors = [c["content"] for c in chunks if c["type"] == "error"]
        assert any("auto-continue stopped" in e for e in errors)

    def test_counter_resets_between_turns(self, make_agent):
        provider = FakeProvider([
            TOOL_TRUNCATED, TOOL_TRUNCATED, TOOL_TRUNCATED,  # turn 1: exhausts cap
            [{"content": "ok"}],                              # turn 2: clean
        ])
        agent = make_agent(provider)
        consume(agent.process_user_input("создай большой файл"))
        chunks = consume(agent.process_user_input("что-нибудь простое"))
        errors = [c["content"] for c in chunks if c["type"] == "error"]
        assert not errors
