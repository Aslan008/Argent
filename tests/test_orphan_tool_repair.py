"""Orphan tool messages (tool without a preceding matching tool_calls) are dropped.

The image path used to put a user image message between two parallel tool
results; the second tool message then had no preceding assistant tool_calls and
DeepSeek rejected the next request with HTTP 400.
"""

from argent.agent import ArgentAgent


def test_orphan_tool_after_a_user_message_is_dropped_and_repaired():
    agent = ArgentAgent()
    agent.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "view_image", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "delegate_to_leeth", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "image result"},
        {"role": "user", "content": "[image payload]"},
        {"role": "tool", "tool_call_id": "call_2", "content": "late result"},
    ]

    repaired = agent.repair_history(fix_tail=False)

    assert repaired >= 1
    assert [m.get("role") for m in agent.messages] == [
        "system", "user", "assistant", "tool", "tool", "user"
    ]
    tool_ids = [m.get("tool_call_id") for m in agent.messages if m.get("role") == "tool"]
    assert set(tool_ids) == {"call_1", "call_2"}


def test_a_valid_parallel_pair_is_left_alone():
    agent = ArgentAgent()
    agent.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "grep_search", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "one"},
        {"role": "tool", "tool_call_id": "b", "content": "two"},
    ]

    assert agent.repair_history(fix_tail=False) == 0
    assert len(agent.messages) == 5
