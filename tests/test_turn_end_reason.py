"""Why a turn stopped is written down.

A turn can end nine different ways and none of them used to leave a line on
disk. When a real run visibly cut short, agent.log held nothing at all for that
window — the only honest answer to "why did it stop?" was "unknowable". These
tests exist so that answer is never correct again.
"""

import logging

import pytest

import agent as agent_module
from agent import ArgentAgent
from providers import ProviderError


class _Strategy:
    def wants_constrained_decoding(self): return False
    def supports_native_tools(self): return True
    def wants_objective_anchor(self): return False
    def wants_tool_catalog(self): return False


@pytest.fixture
def bare(monkeypatch):
    """An agent with the network and the prompt machinery stubbed out."""
    a = ArgentAgent.__new__(ArgentAgent)
    a.messages = [{"role": "system", "content": "s"}]
    a.model_name = "test-model"
    a.provider = "ollama"
    a.max_context_tokens = 8192
    a.strategy = _Strategy()
    a._native_tools_unsupported = False
    monkeypatch.setattr(ArgentAgent, "_refresh_system_prompt", lambda self, t=None: False)
    monkeypatch.setattr(ArgentAgent, "_trim_history", lambda self: None)
    monkeypatch.setattr(agent_module.memory, "data", {"objective": "x"}, raising=False)
    return a


def _reasons(caplog):
    return [r.message for r in caplog.records if "turn ended:" in r.message]


class TestReasonIsRecorded:
    def test_a_provider_that_cannot_be_created(self, bare, monkeypatch, caplog):
        monkeypatch.setattr(agent_module, "create_provider",
                            lambda p: (_ for _ in ()).throw(RuntimeError("no key")))
        with caplog.at_level(logging.INFO, logger=agent_module.log.name):
            list(bare.process_user_input("hi"))
        assert "provider could not be created" in " ".join(_reasons(caplog))
        assert "no key" in " ".join(_reasons(caplog))

    def test_an_invalid_provider_config(self, bare, monkeypatch, caplog):
        class _P:
            def validate_config(self): return "API key missing"

        monkeypatch.setattr(agent_module, "create_provider", lambda p: _P())
        with caplog.at_level(logging.INFO, logger=agent_module.log.name):
            list(bare.process_user_input("hi"))
        assert "provider config invalid" in " ".join(_reasons(caplog))

    def test_a_provider_error_mid_stream(self, bare, monkeypatch, caplog):
        """This is the silent one: the turn stops, the user sees a red line, and
        nothing anywhere records that it happened."""
        class _P:
            def validate_config(self): return None
            def supports_constrained_decoding(self): return False
            def stream_chat(self, **kw): raise ProviderError("HTTP 502 upstream")

        monkeypatch.setattr(agent_module, "create_provider", lambda p: _P())
        with caplog.at_level(logging.INFO, logger=agent_module.log.name):
            list(bare.process_user_input("hi"))
        joined = " ".join(_reasons(caplog))
        assert "provider error" in joined and "502" in joined

    def test_a_final_answer_is_recorded_as_such(self, bare, monkeypatch, caplog):
        """Needed to tell 'it finished' apart from 'it was cut short' — without
        it, a healthy turn and a broken one leave the same empty log."""
        class _P:
            def validate_config(self): return None
            def supports_constrained_decoding(self): return False
            def stream_chat(self, **kw):
                yield {"content": "Here is the answer."}

        monkeypatch.setattr(agent_module, "create_provider", lambda p: _P())
        with caplog.at_level(logging.INFO, logger=agent_module.log.name):
            list(bare.process_user_input("hi"))
        assert "final answer" in " ".join(_reasons(caplog))

    def test_the_caller_walking_away_is_recorded(self, bare, monkeypatch, caplog):
        """Ctrl+C or a closing UI closes the generator, so every break site is
        skipped and the turn vanishes — exactly what 'it just cut off' looks
        like from outside."""
        class _P:
            def validate_config(self): return None
            def supports_constrained_decoding(self): return False
            def stream_chat(self, **kw):
                yield {"content": "partial"}
                yield {"content": " more"}

        monkeypatch.setattr(agent_module, "create_provider", lambda p: _P())
        with caplog.at_level(logging.INFO, logger=agent_module.log.name):
            gen = bare.process_user_input("hi")
            next(gen)
            gen.close()
        assert "interrupted" in " ".join(_reasons(caplog))


class TestExactlyOneLine:
    def test_the_reason_is_not_logged_twice(self, bare, monkeypatch, caplog):
        """The wrapper's finally is a backstop, not a second opinion."""
        class _P:
            def validate_config(self): return None
            def supports_constrained_decoding(self): return False
            def stream_chat(self, **kw):
                yield {"content": "done"}

        monkeypatch.setattr(agent_module, "create_provider", lambda p: _P())
        with caplog.at_level(logging.INFO, logger=agent_module.log.name):
            list(bare.process_user_input("hi"))
        assert len(_reasons(caplog)) == 1

    def test_a_second_turn_gets_its_own_line(self, bare, monkeypatch, caplog):
        class _P:
            def validate_config(self): return None
            def supports_constrained_decoding(self): return False
            def stream_chat(self, **kw):
                yield {"content": "done"}

        monkeypatch.setattr(agent_module, "create_provider", lambda p: _P())
        with caplog.at_level(logging.INFO, logger=agent_module.log.name):
            list(bare.process_user_input("one"))
            list(bare.process_user_input("two"))
        assert len(_reasons(caplog)) == 2


class TestAccounting:
    def test_the_line_carries_tool_count_and_duration(self, bare, monkeypatch, caplog):
        """'Ended after 0 tool calls in 0.3s' and 'after 41 in 12 minutes' are
        different failures; the reason alone does not separate them."""
        class _P:
            def validate_config(self): return None
            def supports_constrained_decoding(self): return False
            def stream_chat(self, **kw):
                yield {"content": "done"}

        monkeypatch.setattr(agent_module, "create_provider", lambda p: _P())
        with caplog.at_level(logging.INFO, logger=agent_module.log.name):
            list(bare.process_user_input("hi"))
        line = _reasons(caplog)[0]
        assert "tool call(s) in" in line and line.rstrip().endswith(("s", "s]"))
