"""compact_tool_history: stub old tool dumps to survive a tight context before
falling back to a full hard reset."""

import src.agent.trimmer as trimmer
from src.agent.trimmer import compact_tool_history, sliding_window_trim


def test_stubs_old_tool_dumps_keeps_recent():
    big = "X" * 5000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {"role": "tool", "content": big},      # old -> stubbed
        {"role": "tool", "content": big},      # old -> stubbed
        {"role": "tool", "content": big},      # recent (keep_recent=2)
        {"role": "tool", "content": big},      # recent
    ]
    out = compact_tool_history(messages, keep_recent=2)
    assert "trimmed" in out[2]["content"] and len(out[2]["content"]) < 300
    assert "trimmed" in out[3]["content"]
    assert out[4]["content"] == big and out[5]["content"] == big   # recent kept full
    assert out[0]["content"] == "sys"                              # system untouched


def test_no_tool_messages_returns_same_object():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    assert compact_tool_history(messages) is messages


def test_small_tool_outputs_not_stubbed():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "tool", "content": "ok"},
        {"role": "tool", "content": "ok2"},
        {"role": "tool", "content": "ok3"},
    ]
    # Old one ("ok") is under the stub threshold -> left intact -> same object.
    assert compact_tool_history(messages, keep_recent=2) is messages


def test_sliding_window_prefers_compaction_over_hard_reset(monkeypatch):
    # Token estimate = raw string length, so a big tool dump dominates the turn.
    monkeypatch.setattr(trimmer, "estimate_tokens", lambda text, *a, **k: len(text))
    monkeypatch.setattr(trimmer, "get_provider", lambda: "test")
    monkeypatch.setattr(trimmer, "get_strip_reasoning", lambda: False)

    big = "X" * 4000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do it"},
        {"role": "tool", "content": big},         # old dump  -> stubbed
        {"role": "tool", "content": big},         # old dump  -> stubbed
        {"role": "tool", "content": big},         # recent    -> kept full
        {"role": "tool", "content": big},         # recent    -> kept full
        {"role": "assistant", "content": "done"},
    ]
    # ~16k of tool output exceeds 0.75*14000; stubbing the two OLD dumps (~8k
    # left) fits, so the turn survives instead of a full hard reset.
    out = sliding_window_trim(messages, "m", max_history_messages=10, max_context_tokens=14000)

    stubbed = [m for m in out if "trimmed" in (m.get("content") or "")]
    assert len(stubbed) == 2                                          # two OLD dumps stubbed
    assert sum(1 for m in out if m.get("content") == big) == 2        # two recent kept full
    assert out[-1]["content"] == "done"                              # real history survived
    assert not any("PERSISTENT MEMORY" in (m.get("content") or "") for m in out)  # no hard reset
