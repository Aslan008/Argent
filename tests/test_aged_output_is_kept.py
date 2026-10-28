"""An aged tool result can still be quoted: the full text is on disk.

The stub used to end with "вывод не сохранён — перезапусти команду". That is the worst
of both worlds — the text was already in hand and then thrown away, and the only way
back was to run the command again: a round trip for one quote, and the world may have
changed in between. Now the text is written to `.argent/tool-output/` before the stub
replaces it, and the stub carries the path so `read_file` with a range recovers it.
"""

import pytest

import src.agent.trimmer as trimmer


@pytest.fixture(autouse=True)
def spool(tmp_path, monkeypatch):
    """Keep the tests' output out of the real project folder."""
    directory = tmp_path / "tool-output"
    monkeypatch.setenv("ARGENT_TOOL_OUTPUT_DIR", str(directory))
    return directory


def _cwd_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class TestTheTextSurvives:
    def test_a_large_command_result_is_written_before_it_is_aged(self, tmp_path, monkeypatch):
        _cwd_project(tmp_path, monkeypatch)
        content = "Exit code: 0\n" + "\n".join(f"line {i}" for i in range(200))
        stub = trimmer._build_aged_stub("run_command", "pytest -q", content)

        assert stub.startswith("[aged: run_command(pytest -q)")
        saved = spool_file(stub)
        assert saved.read_text(encoding="utf-8") == content, "текст должен совпасть дословно"
        assert "перезапусти" not in stub.lower()

    def test_the_stub_names_a_path_that_can_be_read_back(self, tmp_path, monkeypatch):
        _cwd_project(tmp_path, monkeypatch)
        content = "\n".join(f"row {i}" for i in range(120))
        stub = trimmer._build_aged_stub("grep_search", "needle", content)
        path = spool_file(stub)
        assert path.exists()
        assert "read_file" in stub

    def test_a_read_file_stub_points_at_the_saved_copy_too(self, tmp_path, monkeypatch):
        _cwd_project(tmp_path, monkeypatch)
        content = "\n".join(f"def f{i}(): pass" for i in range(60))
        stub = trimmer._build_aged_stub("read_file", "big.py", content)
        assert spool_file(stub).read_text(encoding="utf-8") == content

    def test_a_small_result_is_not_spooled(self, tmp_path, monkeypatch):
        _cwd_project(tmp_path, monkeypatch)
        stub = trimmer._build_aged_stub("run_command", "echo hi", "Exit code: 0\nhi")
        assert "Целиком" not in stub
        assert "перезапусти" in stub.lower()      # nothing worth keeping, say so honestly

    def test_a_read_file_fallback_still_says_what_to_do(self, tmp_path, monkeypatch):
        _cwd_project(tmp_path, monkeypatch)
        stub = trimmer._build_aged_stub("read_file", "small.py", "x = 1")
        assert "перечитай файл" in stub
        assert "перезапусти команду" not in stub


class TestTheFolderStaysBounded:
    def test_only_the_newest_files_are_kept(self, tmp_path, monkeypatch, spool):
        monkeypatch.setattr(trimmer, "AGED_OUTPUT_KEEP", 3)
        monkeypatch.chdir(tmp_path)
        spool.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            (spool / f"old{i}.log").write_text("x" * 600, encoding="utf-8")

        trimmer._build_aged_stub("run_command", "cmd", "y" * 600)

        remaining = sorted(p.name for p in spool.iterdir())
        assert len(remaining) <= 3, remaining

    def test_the_total_size_is_capped(self, tmp_path, monkeypatch, spool):
        monkeypatch.setattr(trimmer, "AGED_OUTPUT_MAX_BYTES", 1500)
        monkeypatch.chdir(tmp_path)
        spool.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            (spool / f"big{i}.log").write_text("x" * 900, encoding="utf-8")

        trimmer._build_aged_stub("run_command", "cmd", "y" * 900)

        total = sum(p.stat().st_size for p in spool.iterdir())
        assert total <= 1500 + 900, f"{total} байт в каталоге"


def spool_file(stub: str):
    """The path a stub points at, as a Path."""
    from pathlib import Path

    marker = "'"
    start = stub.find("в '")
    if start < 0:
        raise AssertionError(f"заглушка не называет файл: {stub}")
    begin = start + len("в '")
    end = stub.find(marker, begin)
    return Path(stub[begin:end])
