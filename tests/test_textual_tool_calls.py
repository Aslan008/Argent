"""A tool call written as text must not end the turn.

The failure this covers is the worst one an unattended agent can have: the model
prints `<create_artifact><parameter name="filename">…</parameter></create_artifact>`
instead of making a real call, the loop sees text and no tool calls, calls it a final
answer — and the turn ends with nothing done and the run looking finished.

Two defences: read the tag back into a call where possible, and refuse to treat an
unreadable one as an answer.
"""

import pytest

import argent.agent as agent_module
from argent.agent import ArgentAgent
from src.agent.parser import (
    looks_like_tool_call_text, parse_raw_tool_call, parse_xml_tool_call,
)

# The exact shape that ended a real turn.
BAD = (
    "<create_artifact>\n"
    '<parameter name="filename">tmp_ignore.md</parameter>\n'
    '<parameter name="content">placeholder</parameter>\n'
    "</create_artifact>"
)


# ── reading the tag back ──────────────────────────────────────────────────────

def test_the_shape_that_ended_a_turn_is_read_back_into_a_call():
    call = parse_raw_tool_call(BAD)
    assert call is not None
    name, args = call["parsed"]["name"], call["parsed"]["arguments"]
    assert name == "create_artifact"
    assert args == {"filename": "tmp_ignore.md", "content": "placeholder"}
    assert call["match_str"].strip().startswith("<create_artifact>")


def test_attributes_carry_the_arguments_too():
    call = parse_xml_tool_call('<read_file path="a.py"/>')
    assert call["parsed"] == {"name": "read_file", "arguments": {"path": "a.py"}}


def test_ordinary_markup_is_not_a_tool_call():
    assert parse_xml_tool_call('<div class="note">привет</div>') is None
    assert parse_xml_tool_call("Ответ: <b>готово</b>") is None


def test_a_known_word_in_prose_is_not_a_call():
    assert parse_xml_tool_call("Я вызвал read_file и всё прочитал.") is None


# ── refusing it as an answer ──────────────────────────────────────────────────

def test_a_textual_call_is_recognised_as_such():
    assert looks_like_tool_call_text(BAD) is True
    assert looks_like_tool_call_text("<create_artifact>") is True      # broken, still a call


def test_a_normal_answer_is_not():
    assert looks_like_tool_call_text("Готово, тесты зелёные.") is False
    assert looks_like_tool_call_text("") is False
    assert looks_like_tool_call_text("Use <div> for blocks.") is False


# ── the loop ─────────────────────────────────────────────────────────────────

class _Strategy:
    def wants_constrained_decoding(self): return False
    def supports_native_tools(self): return True
    def wants_objective_anchor(self): return False
    def wants_tool_catalog(self): return False
    def is_loop_guard_enabled(self): return False
    def is_auto_healing_enabled(self): return False


@pytest.fixture
def bare(monkeypatch):
    a = ArgentAgent.__new__(ArgentAgent)
    a.messages = [{"role": "system", "content": "s"}]
    a.model_name = "test-model"
    a.provider = "ollama"
    a.max_context_tokens = 8192
    a.strategy = _Strategy()
    a._native_tools_unsupported = False
    a.loop_guard = agent_module.LoopGuard()
    a.semantic_loop_guard = agent_module.SemanticLoopGuard()
    a._read_cache = {}
    a._turn_started = 0.0
    a._turn_records = []
    monkeypatch.setattr(ArgentAgent, "_refresh_system_prompt", lambda self, t=None: False)
    monkeypatch.setattr(ArgentAgent, "_trim_history", lambda self: None)
    monkeypatch.setattr(ArgentAgent, "_persist_progress", lambda self, **kw: None)
    monkeypatch.setattr(agent_module.memory, "data", {"objective": "x"}, raising=False)
    return a


def _provider(monkeypatch, turns):
    """Answers each request with the next turn: ('text', str) or ('call', name, args)."""
    queue = list(turns)

    class _P:
        def validate_config(self): return None
        def supports_constrained_decoding(self): return False
        def format_tool_result(self, result, call_id=None):
            return {"role": "tool", "content": result, "tool_call_id": call_id or "c1"}

        def stream_chat(self, messages, **kw):
            text = queue.pop(0) if queue else "готово"
            if text.startswith("<"):
                yield {"content": text}
            else:
                yield {"content": text}

    monkeypatch.setattr(agent_module, "create_provider", lambda p: _P())
    ran = []

    def calculate(expression):
        ran.append(expression)
        return "4"

    monkeypatch.setattr(agent_module, "get_available_tools", lambda: {"calculate": calculate})
    return ran


def test_a_readable_textual_call_is_executed_instead_of_ending_the_turn(bare, monkeypatch):
    ran = _provider(monkeypatch, [
        '<calculate><parameter name="expression">2+2</parameter></calculate>',
        "Ответ: 4",
    ])
    chunks = list(bare.process_user_input("посчитай"))

    assert ran == ["2+2"], "инструмент так и не выполнился"
    assert any(c.get("type") == "tool_end" for c in chunks)
    assert not any("пришёл тег в тексте" in str(c.get("content", "")) for c in chunks)


def test_an_unreadable_textual_call_is_not_accepted_as_an_answer(bare, monkeypatch):
    # No arguments inside the tag: nothing can be executed from it.
    _provider(monkeypatch, [
        "<calculate></calculate>",
        "Тогда просто отвечу: 4",
    ])
    chunks = list(bare.process_user_input("посчитай"))
    notes = [c for c in chunks if "пришёл тег в тексте" in str(c.get("content", ""))]

    assert notes, "ход закончился молча, как будто работа сделана"
    assert bare.messages[-1]["role"] == "assistant" or bare.messages[-1]["role"] == "user"


def test_the_nudges_are_bounded(bare, monkeypatch):
    # A model that only ever writes unreadable tags must stop, not spin forever.
    _provider(monkeypatch, ["<calculate></calculate>"] * 10)
    chunks = list(bare.process_user_input("посчитай"))
    notes = [c for c in chunks if "пришёл тег в тексте" in str(c.get("content", ""))]

    assert len(notes) <= agent_module.MAX_TEXTUAL_CALL_CONTINUES
