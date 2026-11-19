"""A strict provider rejects an assistant tool_calls message with missing results.

DeepSeek returns HTTP 400 "insufficient tool messages following tool_calls" and
then the whole session is stuck. The pair must be repaired before every request,
not only after Ctrl+C or a session restore.
"""

from argent.agent import ArgentAgent


def _agent_with(messages):
    agent = ArgentAgent()
    agent.messages = messages
    return agent


def test_repair_inserts_missing_results_before_the_next_request():
    agent = _agent_with([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "grep_search", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "first result"},
    ])

    repaired = agent.repair_history(fix_tail=False)

    assert repaired == 1
    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_messages} == {"call_1", "call_2"}


def test_repair_before_request_does_not_add_an_assistant_after_the_current_user():
    agent = _agent_with([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "current task"},
    ])

    assert agent.repair_history(fix_tail=False) == 0
    assert agent.messages[-1]["role"] == "user"


def test_repair_after_interrupt_still_fixes_the_tail():
    agent = _agent_with([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "current task"},
    ])

    assert agent.repair_history() == 1
    assert agent.messages[-1]["role"] == "assistant"
