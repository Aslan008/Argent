import gzip
import json

from argent.bench.promote import promote_session, user_turns


def _session(path, messages):
    data = {"id": "s1", "messages": messages}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)


def test_promote_uses_the_last_user_turn(tmp_path):
    session = tmp_path / "s.json.gz"
    _session(session, [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second, fix the import"},
    ])
    path = promote_session(session, out_dir=tmp_path / "cases")
    text = path.read_text(encoding="utf-8")
    assert "fix the import" in text
    assert "checks: []" in text
    assert "promoted" in text


def test_promote_can_choose_a_turn_and_refuses_overwrite(tmp_path):
    session = tmp_path / "s.json.gz"
    _session(session, [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "two"},
    ])
    path = promote_session(session, turn=1, out_dir=tmp_path / "cases")
    assert "one" in path.read_text(encoding="utf-8")
    try:
        promote_session(session, turn=1, out_dir=tmp_path / "cases")
    except Exception as e:
        assert "already exists" in str(e)
    else:
        raise AssertionError("a second promote must not overwrite")


def test_user_turns_ignores_empty_messages():
    assert user_turns([{"role": "user", "content": " "}, {"role": "user", "content": "x"}]) == ["x"]
