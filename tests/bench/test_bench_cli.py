from argent.bench.__main__ import main


def test_bench_is_disabled_by_default(monkeypatch, capsys):
    monkeypatch.setenv("ARGENT_FEATURE_BENCH", "0")
    assert main(["list"]) == 2
    assert "disabled" in capsys.readouterr().out


def test_list_shows_cases(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARGENT_FEATURE_BENCH", "1")
    (tmp_path / "demo.yaml").write_text(
        "id: demo\ntask: do it\nchecks: [echo hi]\n", encoding="utf-8"
    )
    assert main(["list", "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "demo" in out and "1 checks" in out
