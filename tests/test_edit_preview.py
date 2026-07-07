"""Edits echo the changed region + exact-match ambiguity lists line numbers +
replace_python_function honours the plugin guard."""

import tools._helpers as helpers
from tools.file_ops import replace_in_file, replace_python_function


def test_replace_in_file_returns_updated_region(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    out = replace_in_file(str(f), "b = 2", "b = 20")
    assert "Successfully replaced" in out
    assert "Updated region:" in out
    assert "b = 20" in out          # the model can verify without re-reading
    assert "→" in out               # the changed line is marked


def test_fuzzy_replace_also_previews(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def f():\n        x = 1\n        return x\n", encoding="utf-8")
    # Indentation mismatch forces the fuzzy path.
    out = replace_in_file(str(f), "x = 1\nreturn x", "x = 2\nreturn x")
    assert "Updated region:" in out
    assert "x = 2" in out


def test_exact_ambiguity_lists_lines(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 0\nx = 0\n", encoding="utf-8")
    out = replace_in_file(str(f), "x = 0", "x = 1")
    assert "appears 2 times" in out
    assert "line 1" in out and "line 2" in out
    assert f.read_text(encoding="utf-8") == "x = 0\nx = 0\n"   # untouched


def test_replace_python_function_previews(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n", encoding="utf-8")
    out = replace_python_function(str(f), "foo", "def foo():\n    return 111\n")
    assert "Updated region:" in out
    assert "return 111" in out


def test_replace_python_function_blocks_plugin_dir(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    f = plugins / "myplugin.py"
    f.write_text("def command_hi():\n    return 0\n", encoding="utf-8")
    monkeypatch.setattr(helpers, "get_hooks_dir", lambda: str(plugins))
    out = replace_python_function(str(f), "command_hi", "def command_hi():\n    return 9\n")
    assert "create_plugin" in out or "restricted" in out.lower()
    assert "return 9" not in f.read_text(encoding="utf-8")   # the edit was refused
