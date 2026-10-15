import json

from providers import OpenAICompatibleProvider, ZAIProvider


norm = OpenAICompatibleProvider._normalize_outgoing_messages


class TestNormalizeOutgoingMessages:
    def test_dict_arguments_become_json_string(self):
        msgs = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "search_web", "arguments": {"query": "погода"}},
            }],
        }]
        out = norm(msgs)
        args = out[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        # Round-trips and preserves non-ASCII without escaping.
        assert json.loads(args) == {"query": "погода"}
        assert "погода" in args

    def test_string_arguments_left_untouched(self):
        msgs = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "c", "type": "function",
                "function": {"name": "t", "arguments": '{"a": 1}'},
            }],
        }]
        out = norm(msgs)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'

    def test_none_arguments_become_empty_object(self):
        msgs = [{"role": "assistant", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "t", "arguments": None}},
        ]}]
        out = norm(msgs)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_does_not_mutate_input_history(self):
        original_args = {"query": "x"}
        msgs = [{
            "role": "assistant",
            "tool_calls": [{"id": "c", "type": "function",
                            "function": {"name": "t", "arguments": original_args}}],
        }]
        norm(msgs)
        # The agent's in-memory history must stay a dict for tool invocation.
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] is original_args
        assert isinstance(msgs[0]["tool_calls"][0]["function"]["arguments"], dict)

    def test_plain_messages_pass_through(self):
        msgs = [
            {"role": "system", "content": "hi"},
            {"role": "user", "content": "погода?"},
            {"role": "tool", "content": "result", "tool_call_id": "c"},
        ]
        assert norm(msgs) == msgs

    def test_multiple_tool_calls_in_one_message(self):
        msgs = [{"role": "assistant", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}},
            {"id": "b", "type": "function", "function": {"name": "g", "arguments": {"y": 2}}},
        ]}]
        out = norm(msgs)
        assert all(isinstance(tc["function"]["arguments"], str) for tc in out[0]["tool_calls"])

    def test_available_on_subclasses(self):
        # ZAIProvider inherits the same normalization.
        assert ZAIProvider._normalize_outgoing_messages is OpenAICompatibleProvider._normalize_outgoing_messages

    def test_initial_system_message_preserved(self):
        msgs = [
            {"role": "system", "content": "You are Argent."},
            {"role": "user", "content": "Hello"},
        ]
        out = norm(msgs)
        assert out[0]["role"] == "system"
        assert out[0]["content"] == "You are Argent."
        assert out[1]["role"] == "user"

    def test_subsequent_system_messages_converted_to_user(self):
        msgs = [
            {"role": "system", "content": "You are Argent."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "system", "content": "# REPOSITORY CONTEXT\nfiles..."},
            {"role": "system", "content": "[Objective Reminder]\nfocus..."},
        ]
        out = norm(msgs)
        # First remains system
        assert out[0]["role"] == "system"
        assert out[0]["content"] == "You are Argent."
        # Intermediates remain untouched
        assert out[1]["role"] == "user"
        assert out[2]["role"] == "assistant"
        # Non-initial system messages converted to user
        assert out[3]["role"] == "user"
        assert out[3]["content"] == "# REPOSITORY CONTEXT\nfiles..."
        assert out[4]["role"] == "user"
        assert out[4]["content"] == "[Objective Reminder]\nfocus..."
        # Original messages list must not be mutated
        assert msgs[3]["role"] == "system"
        assert msgs[4]["role"] == "system"

    def test_system_message_at_non_zero_index_without_initial_system(self):
        msgs = [
            {"role": "user", "content": "Query without initial system prompt"},
            {"role": "system", "content": "Tail context"},
        ]
        out = norm(msgs)
        assert out[0]["role"] == "user"
        assert out[1]["role"] == "user"
        assert out[1]["content"] == "Tail context"
