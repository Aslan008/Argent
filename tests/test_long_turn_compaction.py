"""One long autonomous turn must be compressible.

Reported (KoboldCPP, 64k window): after ten minutes of tool calls the request grew past the window,
"trimmed history and retrying" ran three times with the same prompt, the turn died with HTTP 400, and
/compact said "82 → 82 messages". Three defects together:

* the whole last turn was pinned as the "active window", so a single turn bigger than the budget left
  nothing to compress;
* the history was trimmed only when a user turn began, never between the tool rounds of one turn;
* the history budget reserved a flat 2048 tokens for tool schemas that measured ~8k.
"""

import json
from itertools import islice
from unittest.mock import MagicMock

import pytest

import config
from src.agent import trimmer


def _tok(text, model=None, provider=None):
    return len(text) // 4


@pytest.fixture(autouse=True)
def cheap_tokens(monkeypatch):
    monkeypatch.setattr(trimmer, "estimate_tokens", _tok)
    monkeypatch.setattr(trimmer, "_summarize_with_llm", lambda *a, **k: "SUMMARY OF EARLIER STEPS")


def _long_turn(steps=40, output_chars=2000):
    msgs = [{"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "Перенеси обучение на Kaggle и проверь гейты."}]
    for i in range(steps):
        call_id = f"c{i}"
        msgs.append({"role": "assistant", "content": f"step {i}",
                     "tool_calls": [{"id": call_id, "type": "function",
                                     "function": {"name": "run_command",
                                                  "arguments": {"command": f"python step_{i}.py"}}}]})
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": f"output {i} " + "x" * output_chars})
    return msgs


def _no_orphans(msgs):
    """Every tool result follows the assistant message that made the call."""
    open_ids = set()
    for m in msgs:
        if m.get("role") == "assistant":
            open_ids = {tc["id"] for tc in m.get("tool_calls") or []}
        elif m.get("role") == "tool":
            assert m.get("tool_call_id") in open_ids, f"orphan tool result {m.get('tool_call_id')}"


# ── the trimmer ──────────────────────────────────────────────────────────────

def test_a_single_long_turn_is_compressed_under_the_budget():
    msgs = _long_turn()
    before = trimmer._history_tokens(msgs, "m")
    budget = 8000
    assert before > 2 * budget
    out = trimmer.soft_trim_with_summarization(msgs, "m", 28, budget)
    assert trimmer._history_tokens(out, "m") <= budget
    assert out[1]["content"].startswith("Перенеси обучение")          # the request stays verbatim
    assert out[-1] == msgs[-1] and out[-2] == msgs[-2]                  # the latest step is kept whole
    assert any("CONTEXT DISTILLATE" in (m.get("content") or "") for m in out)
    _no_orphans(out)


def test_a_turn_that_fits_is_kept_whole():
    msgs = _long_turn(steps=3, output_chars=100)
    assert trimmer._in_turn_split(msgs, 1, "m", 100_000) == 1


def test_the_split_is_always_at_a_step():
    msgs = _long_turn()
    j = trimmer._in_turn_split(msgs, 1, "m", 5000)
    assert msgs[j]["role"] == "assistant"
    assert j > 1


def test_a_huge_latest_tool_output_is_clipped_to_fit():
    msgs = _long_turn(steps=2, output_chars=200)
    msgs[-1] = dict(msgs[-1], content="y" * 80_000)
    out = trimmer._ensure_fits(msgs, "m", 6000)
    assert trimmer._history_tokens(out, "m") <= 6000
    assert "cut to fit the context" in out[-1]["content"]
    _no_orphans(out)


def test_forced_compaction_compresses_a_history_that_fits():
    msgs = _long_turn(steps=12, output_chars=600)
    plain = trimmer.soft_trim_with_summarization(msgs, "m", 1000, 1_000_000)
    assert plain is msgs
    forced = trimmer.soft_trim_with_summarization(msgs, "m", 1000, 1500, force=True)
    assert trimmer._history_tokens(forced, "m") < trimmer._history_tokens(msgs, "m")


def test_tiny_models_keep_the_latest_steps_instead_of_a_hard_reset():
    """sliding_window_trim (tiny models) used to hard-reset a turn bigger than the budget or longer than
    the message limit: every step of the task was gone and the model started over."""
    msgs = _long_turn(steps=30, output_chars=800)
    out = trimmer.sliding_window_trim(msgs, "m", max_history_messages=20, max_context_tokens=6000)
    assert not any("emergency reset" in (m.get("content") or "") for m in out)
    assert out[1]["content"].startswith("Перенеси обучение")
    assert out[-1] == msgs[-1] and out[-2] == msgs[-2]
    assert len(out) - 1 <= 20
    assert trimmer._history_tokens(out, "m") <= 6000 * 0.75
    assert any("ранних сообщений этой задачи убраны" in (m.get("content") or "") for m in out)
    _no_orphans(out)


def test_the_refused_request_size_is_read_from_the_error():
    from src.agent.context_limit import parse_prompt_tokens
    err = ("KOBOLDCPP error (HTTP 400): {'error': {'code': 400, 'message': 'request (65919 tokens) exceeds "
           "the available context size (65536 tokens), try increasing it', 'n_prompt_tokens': 65919, 'n_ctx': 65536}}")
    assert parse_prompt_tokens(err) == 65919
    assert parse_prompt_tokens("However, your messages resulted in 70000 tokens") == 70000
    assert parse_prompt_tokens("context size exceeded") is None


# ── the agent loop ───────────────────────────────────────────────────────────

class _ReadingProvider:
    """Reads one big file per step for `steps` steps, then answers; records each request's size."""

    def __init__(self, files):
        self.files = files
        self.calls = 0
        self.request_tokens = []

    def validate_config(self):
        return None

    def supports_constrained_decoding(self):
        return False

    def format_tool_result(self, content, tool_call_id=None):
        return {"role": "tool", "content": content, "tool_call_id": tool_call_id}

    def stream_chat(self, model, messages, tools=None, **kw):
        size = sum(_tok(str(m)) for m in messages)
        if tools:
            size += _tok(json.dumps(tools, ensure_ascii=False))
        self.request_tokens.append(size)
        i = self.calls
        self.calls += 1
        if i < len(self.files):
            args = json.dumps({"file_path": str(self.files[i])})
            yield {"content": "", "thinking": "",
                   "tool_call_deltas": [{"index": 0, "id": f"call{i}", "function_name_delta": "read_file",
                                         "function_arguments_delta": args}]}
        else:
            yield {"content": "Готово.", "thinking": "", "tool_call_deltas": []}


def test_a_long_turn_never_sends_a_request_over_the_window(monkeypatch, tmp_path):
    import agent as agent_module
    from agent import ArgentAgent

    files = []
    for i in range(14):
        f = tmp_path / f"part_{i}.txt"
        f.write_text("\n".join(f"line {n} of part {i} " + "z" * 60 for n in range(120)), encoding="utf-8")
        files.append(f)
    monkeypatch.chdir(tmp_path)
    provider = _ReadingProvider(files)

    fake_memory = MagicMock()
    fake_memory.data = {}
    monkeypatch.setattr(agent_module, "memory", fake_memory)
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: provider)
    monkeypatch.setattr(agent_module, "estimate_tokens", _tok)
    monkeypatch.setattr(config, "get_fast_dispatch_backend", lambda: "off")
    window = 24000
    config.set_context_window(window)

    agent = ArgentAgent()
    chunks = list(islice(agent.process_user_input("прочитай все части по очереди"), 2000))
    assert provider.calls >= len(files)
    errors = [c.get("content", "") for c in chunks if c.get("type") == "error"]
    assert not any("still won't fit" in e for e in errors), errors
    assert max(provider.request_tokens) <= window, provider.request_tokens
