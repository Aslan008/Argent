"""Every gated or interactive tool must say WHAT HAPPENED on its first line.

The three failures this pins, all of them observed while working inside Argent:

* `request_user_approval` answered "User REJECTED" for a person who pressed No and for
  one who simply walked away from the prompt. The agent asked "what should I change?"
  of somebody who was not there, and — worse — a `None` used to be one branch away from
  counting as consent.
* `ask_user_questions` returned "Skipped" and "No answer" as the same kind of prose, so
  "the person dismissed every question" had to be inferred from strings.
* `call_mcp_tool` took only a JSON *string*; a model that passed the object it had just
  built, or encoded it twice, got a failure from the server that said nothing about the
  shape of its arguments.
"""

import json

import pytest

import argent.tools.misc_tools as mt


class TestApprovalOutcomes:
    @pytest.fixture(autouse=True)
    def _fresh_backend(self, monkeypatch):
        from argent.approval import reset_approval_backend

        monkeypatch.setattr(mt, "memory", _Memory())
        yield
        reset_approval_backend()

    def _answer_with(self, decision):
        from argent.approval import set_approval_backend

        set_approval_backend(lambda action, destructive, grant_key: decision)

    def test_approval_says_approved(self):
        self._answer_with("once")
        out = mt.request_user_approval("ship it")
        assert out.splitlines()[0].startswith("APPROVED")
        assert "proceed" in out.lower()

    def test_refusal_keeps_the_reason_when_the_channel_has_one(self):
        self._answer_with({"decision": "deny", "reason": "too risky before the release"})
        out = mt.request_user_approval("ship it")
        first = out.splitlines()[0]
        assert first.startswith("REJECTED")
        assert "too risky before the release" in first

    def test_refusal_without_a_reason_is_still_a_refusal(self):
        self._answer_with("deny")
        out = mt.request_user_approval("ship it")
        assert out.splitlines()[0].startswith("REJECTED")
        assert "no reason given" in out

    def test_a_dismissed_prompt_is_cancelled_not_rejected(self):
        """Nobody said no — and silence must never read as yes."""
        self._answer_with("cancel")
        out = mt.request_user_approval("ship it")
        assert out.splitlines()[0].startswith("CANCELLED")
        assert "do not proceed" in out.lower()

    def test_cancelled_and_rejected_are_different_strings(self):
        self._answer_with("deny")
        rejected = mt.request_user_approval("x")
        self._answer_with("cancel")
        cancelled = mt.request_user_approval("x")
        assert rejected != cancelled


class TestQuestionStatus:
    def test_all_answered(self, monkeypatch):
        monkeypatch.setattr(mt, "_ask_text", lambda prompt, default=None: "yes")
        out = mt.ask_user_questions([{"type": "text", "question": "Proceed?"}])
        assert out.splitlines()[0].startswith("ANSWERED")
        assert "- Proceed?: yes" in out

    def test_dismissed_questions_say_skipped(self, monkeypatch):
        def _interrupt(prompt, default=None):
            raise KeyboardInterrupt

        monkeypatch.setattr(mt, "_ask_text", _interrupt)
        out = mt.ask_user_questions([{"type": "text", "question": "Proceed?"}])
        assert out.splitlines()[0].startswith("SKIPPED")
        assert "SKIPPED (dismissed" in out

    def test_empty_answer_is_not_the_same_as_a_skip(self, monkeypatch):
        monkeypatch.setattr(mt, "_ask_text", lambda prompt, default=None: "   ")
        out = mt.ask_user_questions([{"type": "text", "question": "Name?"}])
        assert out.splitlines()[0].startswith("ANSWERED")
        assert "no preference given" in out

    def test_partly_answered_says_so(self, monkeypatch):
        calls = {"n": 0}

        def _first_answers_then_leaves(prompt, default=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return "a"
            raise KeyboardInterrupt

        monkeypatch.setattr(mt, "_ask_text", _first_answers_then_leaves)
        out = mt.ask_user_questions([
            {"type": "text", "question": "One?"},
            {"type": "text", "question": "Two?"},
        ])
        assert out.splitlines()[0].startswith("PARTLY ANSWERED: 1 of 2")


class TestMcpArguments:
    def test_an_object_passes_through(self):
        args, note, err = mt._decode_mcp_arguments({"path": "a.py"})
        assert args == {"path": "a.py"} and not note and not err

    def test_a_json_string_is_still_accepted(self):
        args, note, err = mt._decode_mcp_arguments(json.dumps({"path": "a.py"}))
        assert args == {"path": "a.py"} and not err
        assert note == ""       # one level is the old contract, not a mistake

    def test_a_string_encoded_twice_is_unwrapped_and_reported(self):
        twice = json.dumps(json.dumps({"path": "a.py"}))
        args, note, err = mt._decode_mcp_arguments(twice)
        assert args == {"path": "a.py"} and not err
        assert "twice" in note

    def test_an_object_wrapped_in_an_object_is_unwrapped(self):
        args, note, err = mt._decode_mcp_arguments({"arguments": json.dumps({"path": "a.py"})})
        assert args == {"path": "a.py"} and not err
        assert "unwrapped" in note

    def test_unparseable_text_names_the_expected_shape(self):
        _args, _note, err = mt._decode_mcp_arguments("{path: a.py}")
        assert err.startswith("Error:")
        assert "OBJECT" in err

    def test_a_list_is_refused_with_a_reason(self):
        _args, _note, err = mt._decode_mcp_arguments(json.dumps([1, 2]))
        assert err.startswith("Error:") and "object" in err.lower()

    def test_call_mcp_tool_sends_decoded_args_and_reports_the_unwrap(self, monkeypatch):
        seen = {}

        class _Client:
            def call_tool(self, server, tool, args):
                seen.update(server=server, tool=tool, args=args)
                return "OK"

        monkeypatch.setattr(mt, "mcp_client", _Client())
        out = mt.call_mcp_tool("srv", "tool", {"arguments": json.dumps({"x": 1})})
        assert seen["args"] == {"x": 1}
        assert out.startswith("OK") and "unwrapped" in out

    def test_the_old_parameter_name_still_works(self, monkeypatch):
        """Saved traces call `arguments_json`; the alias must not become a mystery."""
        class _Client:
            def call_tool(self, server, tool, args):
                return f"got {args}"

        monkeypatch.setattr(mt, "mcp_client", _Client())
        out = mt.call_mcp_tool("srv", "tool", arguments_json='{"x": 1}')
        assert out == "got {'x': 1}"


class _Memory:
    """Minimal stand-in: these tools record what happened for later turns."""

    def __getattr__(self, _name):
        return lambda *a, **k: None
