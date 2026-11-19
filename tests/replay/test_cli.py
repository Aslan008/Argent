import json

from argent.replay.__main__ import main


def test_cli_refuses_when_the_feature_is_off(monkeypatch, capsys):
    monkeypatch.setenv("ARGENT_FEATURE_REPLAY", "0")
    assert main(["--list"]) == 2
    assert "disabled" in capsys.readouterr().out


def test_cli_renders_a_trace_when_the_feature_is_on(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARGENT_FEATURE_REPLAY", "1")
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"kind": "user_message", "turn": 1, "text": "hello"}),
            json.dumps({"kind": "done", "turn": 1}),
        ]) + "\n",
        encoding="utf-8",
    )
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "hello" in out
    assert "ход 1" in out
