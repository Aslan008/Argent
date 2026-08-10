"""The truncation notice points at get_file_outline for code files.

The read budget follows the model tier, so it is pinned here — otherwise the
test passes or fails depending on which model the developer happens to have
configured.
"""

import pytest

from tools.file_ops import read_file


@pytest.fixture(autouse=True)
def tiny_model(monkeypatch):
    """400 lines / 12k chars — the budget a 3B model gets."""
    monkeypatch.setattr("config.get_model_size_category", lambda m: "tiny")


def test_py_truncation_suggests_outline(tmp_path):
    f = tmp_path / "big.py"
    f.write_text("".join(f"x{i} = {i}\n" for i in range(600)), encoding="utf-8")
    out = read_file(str(f))
    assert "exceeds 400 lines" in out
    assert "get_file_outline" in out


def test_text_truncation_has_no_outline_hint(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("".join(f"line {i}\n" for i in range(600)), encoding="utf-8")
    out = read_file(str(f))
    assert "exceeds 400 lines" in out
    assert "get_file_outline" not in out   # outline only helps .py/.cs
