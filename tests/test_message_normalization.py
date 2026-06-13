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
