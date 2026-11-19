import gzip
import json

from argent.replay.reader import load_session_timeline, load_trace
from argent.replay.render import render


def _write_trace(path):
    rows = [
        {"v": 1, "kind": "session_start", "model": "m", "provider": "p", "cwd": "C:/proj", "seq": 1},
        {"v": 1, "kind": "turn_start", "turn": 1, "seq": 2},
        {"v": 1, "kind": "user_message", "turn": 1, "text": "fix it", "seq": 3},
        {"v": 1, "kind": "tool_start", "turn": 1, "name": "read_file", "args": {"file_path": "a.py"}, "seq": 4},
        {"v": 1, "kind": "tool_end", "turn": 1, "name": "read_file", "result_chars": 2, "seq": 5},
        {"v": 1, "kind": "done", "turn": 1, "seq": 6},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_a_trace_renders_its_steps(tmp_path):
    path = tmp_path / "trace.jsonl"
    _write_trace(path)
    rows = load_trace(path)
    assert [r["kind"] for r in rows][0] == "session_start"
    text = render(rows)
    assert "fix it" in text
    assert "read_file" in text
    assert "ход 1" in text
    assert "2 chars" in text


def test_content_is_hidden_unless_requested(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"kind": "content", "text": "visible only with flag"}) + "\n", encoding="utf-8")
    rows = load_trace(path)
    assert "visible only" not in render(rows)
    assert "visible only" in render(rows, content=True)


def test_an_old_session_becomes_a_timeline(tmp_path):
    session = tmp_path / "s.json.gz"
    data = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    }
    with gzip.open(session, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)
    rows = load_session_timeline(session)
    kinds = [r["kind"] for r in rows]
    assert "user_message" in kinds
    assert "assistant_message" in kinds
    assert "hello" in render(rows)
