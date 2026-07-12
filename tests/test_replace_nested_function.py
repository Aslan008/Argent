"""replace_python_function reaches nested targets, not just the top tier."""

from unittest.mock import MagicMock

import pytest

import tools.file_ops as file_ops


@pytest.fixture(autouse=True)
def _no_side_effects(monkeypatch):
    monkeypatch.setattr(file_ops, "snapshot", MagicMock())
    monkeypatch.setattr(file_ops, "_print_diff", MagicMock())
    monkeypatch.setattr(file_ops, "memory", MagicMock())


def _write(tmp_path, src):
    f = tmp_path / "mod.py"
    f.write_text(src, encoding="utf-8")
    return f


def test_top_level_function(tmp_path):
    f = _write(tmp_path, "def foo():\n    return 1\n")
    res = file_ops.replace_python_function(str(f), "foo", "def foo():\n    return 2\n")
    assert res.startswith("Successfully")
    assert "return 2" in f.read_text(encoding="utf-8")


def test_method_in_top_level_class(tmp_path):
    f = _write(tmp_path, "class A:\n    def m(self):\n        return 1\n")
    res = file_ops.replace_python_function(str(f), "A.m", "def m(self):\n    return 2\n")
    assert res.startswith("Successfully")
    assert "return 2" in f.read_text(encoding="utf-8")


def test_method_in_nested_class(tmp_path):
    src = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def method(self):\n"
        "            return 'old'\n"
    )
    f = _write(tmp_path, src)
    res = file_ops.replace_python_function(
        str(f), "Outer.Inner.method", "def method(self):\n    return 'new'\n")
    assert res.startswith("Successfully")
    out = f.read_text(encoding="utf-8")
    assert "return 'new'" in out and "class Outer" in out and "class Inner" in out


def test_function_nested_in_function(tmp_path):
    src = (
        "def outer():\n"
        "    def inner():\n"
        "        return 1\n"
        "    return inner\n"
    )
    f = _write(tmp_path, src)
    res = file_ops.replace_python_function(
        str(f), "outer.inner", "def inner():\n    return 99\n")
    assert res.startswith("Successfully")
    assert "return 99" in f.read_text(encoding="utf-8")


def test_missing_nested_target_reports_not_found(tmp_path):
    f = _write(tmp_path, "class A:\n    def m(self):\n        return 1\n")
    res = file_ops.replace_python_function(str(f), "A.nope", "def nope(self):\n    pass\n")
    assert "not found" in res


def test_empty_component_is_rejected(tmp_path):
    f = _write(tmp_path, "def foo():\n    return 1\n")
    res = file_ops.replace_python_function(str(f), "A..m", "def m(self):\n    pass\n")
    assert "Invalid function_name" in res
