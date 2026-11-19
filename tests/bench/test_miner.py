import json

from argent.bench.miner import mine, mine_trace


def _trace(path):
    rows = [
        {"kind": "user_message", "turn": 1, "text": "fix the build"},
        {"kind": "tool_end", "turn": 1, "result": "error: permission denied"},
        {"kind": "checkpoint", "turn": 1, "sha": "abc123"},
        {"kind": "user_message", "turn": 2, "text": "all good"},
        {"kind": "tool_end", "turn": 2, "result": "ok"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_miner_writes_a_draft_without_checks(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _trace(trace)
    drafts = mine_trace(trace, tmp_path / "drafts")
    assert len(drafts) == 1
    text = drafts[0].read_text(encoding="utf-8")
    assert "fix the build" in text
    assert "checks: []" in text
    assert "mined" in text


def test_miner_does_not_duplicate_the_same_draft(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _trace(trace)
    assert len(mine_trace(trace, tmp_path / "drafts")) == 1
    assert mine_trace(trace, tmp_path / "drafts") == []


def test_mine_accepts_a_trace_directory(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _trace(trace)
    drafts = mine(traces_dir=tmp_path, sessions_dir=None, out_dir=tmp_path / "drafts")
    assert len(drafts) == 1
