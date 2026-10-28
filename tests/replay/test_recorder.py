import json

from argent.replay.recorder import TraceRecorder


def test_recorder_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "trace.jsonl"
    rec = TraceRecorder(path=path, meta={"cwd": "C:/proj", "model": "m"})
    rec.record({"type": "tool_start", "name": "read_file", "args": {"file_path": "a.py"}})
    rec.record({"type": "tool_end", "name": "read_file", "result": "ok"})
    rec.link_session("s1")
    rec.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["session_start", "tool_start", "tool_end", "session_link"]
    assert rows[0]["cwd"] == "C:/proj"
    assert rows[-1]["session_id"] == "s1"
    assert [r["seq"] for r in rows] == [1, 2, 3, 4]


def test_a_recorder_that_cannot_open_is_a_noop(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    rec = TraceRecorder(path=blocked / "trace.jsonl")
    assert rec.enabled is False
    rec.record({"type": "content", "text": "hello"})  # must not raise
    rec.close()
