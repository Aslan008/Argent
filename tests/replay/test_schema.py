import json

from argent.replay.schema import make_event, normalize_event


def test_a_secret_is_redacted_in_arguments(monkeypatch):
    import src.agent.secret_redactor as redactor

    monkeypatch.setattr(redactor, "enabled", lambda: True)
    event = make_event("tool_start", args={"token": "sk-abcdefghijklmnopqrstuvwxyz012345"})
    dumped = json.dumps(event, ensure_ascii=False)
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in dumped
    assert "token" in dumped


def test_a_tool_result_keeps_its_size_and_hash_but_is_clipped(monkeypatch):
    long = "x" * 10_000
    event = normalize_event({"type": "tool_end", "name": "run_command", "result": long})
    assert event["kind"] == "tool_end"
    assert event["result_chars"] == 10_000
    assert len(event["result_sha256"]) == 64
    assert event["result"].startswith("x" * 100)
    assert "chars)" in event["result"]


def test_unknown_events_are_kept_as_generic_rows():
    event = normalize_event({"type": "custom", "value": 3})
    assert event["kind"] == "custom"
    assert event["value"] == 3
    assert event["v"] == 1
    assert event["ts"].endswith("+00:00")


def test_non_dict_events_are_ignored():
    assert normalize_event("nope") is None
