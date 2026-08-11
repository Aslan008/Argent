"""Comprehensive pytest tests for replace_python_function and get_file_outline
from tools/file_ops.py — AST-based surgical editing and structural outlining.
"""

import pytest
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Critical mocking: stub out external dependencies so tests run hermetically.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _stub_deps(monkeypatch):
    import memory_manager
    class FakeMemory:
        def add_completed(self, *a, **k): pass
        def add_file_modified(self, *a, **k): pass
    monkeypatch.setattr(memory_manager, 'memory', FakeMemory())
    import file_tracker
    monkeypatch.setattr(file_tracker, 'snapshot', lambda *a, **k: None)
    import tools._helpers as helpers
    monkeypatch.setattr(helpers, '_validate_code_syntax', lambda fp: None)
    monkeypatch.setattr(helpers, '_print_diff', lambda *a, **k: None)
    yield


from tools.file_ops import replace_python_function, get_file_outline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write(tmp_path, name, src):
    """Write *src* into tmp_path/name and return the Path."""
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


# ===========================================================================
# replace_python_function
# ===========================================================================

class TestReplacePythonFunction:
    """replace_python_function — AST-based surgical replacement."""

    # --- L2: simple top-level function -------------------------------------
    def test_replace_simple_top_level_function(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = _write(tmp_path, "mod.py", "def foo():\n    return 1\n")
        res = replace_python_function(str(f), "foo", "def foo():\n    return 2\n")
        assert res.startswith("Successfully replaced")

    # --- L4×D10: verify new code present, old gone -------------------------
    def test_new_code_present_old_gone(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = _write(tmp_path, "mod.py", "def foo():\n    return 'OLD'\n")
        replace_python_function(str(f), "foo", "def foo():\n    return 'NEW'\n")
        content = f.read_text(encoding="utf-8")
        assert "return 'NEW'" in content
        assert "return 'OLD'" not in content

    # --- L2: replace class method ------------------------------------------
    def test_replace_class_method(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "class A:\n    def m(self):\n        return 1\n"
        f = _write(tmp_path, "mod.py", src)
        res = replace_python_function(str(f), "A.m", "def m(self):\n    return 2\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "return 2" in out
        assert "return 1" not in out
        assert "class A:" in out

    # --- L2: replace nested function ---------------------------------------
    def test_replace_nested_function(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "def outer():\n"
            "    def inner():\n"
            "        return 1\n"
            "    return inner\n"
        )
        f = _write(tmp_path, "mod.py", src)
        res = replace_python_function(
            str(f), "outer.inner", "def inner():\n    return 99\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "return 99" in out
        assert "return 1" not in out

    # --- L2: replace method in nested class --------------------------------
    def test_replace_method_in_nested_class(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "class Outer:\n"
            "    class Inner:\n"
            "        def method(self):\n"
            "            return 'old'\n"
        )
        f = _write(tmp_path, "mod.py", src)
        res = replace_python_function(
            str(f), "Outer.Inner.method", "def method(self):\n    return 'new'\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "return 'new'" in out
        assert "return 'old'" not in out
        assert "class Outer" in out
        assert "class Inner" in out

    # --- L2: replace async function ----------------------------------------
    def test_replace_async_function(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "async def fetch():\n    return 1\n"
        f = _write(tmp_path, "mod.py", src)
        res = replace_python_function(
            str(f), "fetch", "async def fetch():\n    return 2\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "return 2" in out
        assert "return 1" not in out

    # --- L2: replace async class method -----------------------------------
    def test_replace_async_class_method(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "class Svc:\n    async def run(self):\n        return 1\n"
        f = _write(tmp_path, "mod.py", src)
        res = replace_python_function(
            str(f), "Svc.run", "async def run(self):\n    return 2\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "return 2" in out

    # --- L2: decorators included in replacement range ----------------------
    def test_replace_function_with_decorators(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "@my_decorator\n"
            "def foo():\n"
            "    return 1\n"
        )
        f = _write(tmp_path, "mod.py", src)
        res = replace_python_function(
            str(f), "foo", "def foo():\n    return 2\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "return 2" in out
        assert "return 1" not in out
        # The decorator line should be gone (it was part of the range).
        assert "@my_decorator" not in out

    # --- L2: decorators on class method ------------------------------------
    def test_replace_method_with_decorator(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "class A:\n"
            "    @staticmethod\n"
            "    def m():\n"
            "        return 1\n"
        )
        f = _write(tmp_path, "mod.py", src)
        res = replace_python_function(
            str(f), "A.m", "def m():\n    return 2\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "return 2" in out
        assert "@staticmethod" not in out

    # --- L2: new code with different indentation → reindented --------------
    def test_new_code_different_indentation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Original method is indented 4 spaces (inside class).
        src = "class A:\n    def m(self):\n        return 1\n"
        f = _write(tmp_path, "mod.py", src)
        # New code given at column 0 (no leading indent).
        res = replace_python_function(
            str(f), "A.m", "def m(self):\n    return 2\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        # The replacement should be re-indented to sit inside the class.
        assert "    def m(self):" in out
        assert "        return 2" in out

    # --- L2: blank lines in new code become empty strings ------------------
    def test_new_code_blank_lines_become_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "def foo():\n    return 1\n"
        f = _write(tmp_path, "mod.py", src)
        new_code = "def foo():\n\n    return 2\n"
        res = replace_python_function(str(f), "foo", new_code)
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        # The blank line should be present as an empty line (no trailing spaces).
        lines = out.splitlines()
        # Find the foo definition line
        idx = next(i for i, l in enumerate(lines) if "def foo" in l)
        assert lines[idx + 1] == ""  # blank line is truly empty
        assert "return 2" in lines[idx + 2]

    # --- L1: function not found --------------------------------------------
    def test_function_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = _write(tmp_path, "mod.py", "def foo():\n    return 1\n")
        res = replace_python_function(str(f), "bar", "def bar():\n    return 2\n")
        assert "not found" in res
        assert "bar" in res

    # --- L1: non-existent file ---------------------------------------------
    def test_non_existent_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        res = replace_python_function(
            str(tmp_path / "nope.py"), "foo", "def foo():\n    pass\n")
        assert "does not exist" in res

    # --- L1: file with SyntaxError -----------------------------------------
    def test_file_with_syntax_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = _write(tmp_path, "mod.py", "def foo(:\n    return 1\n")
        res = replace_python_function(str(f), "foo", "def foo():\n    return 2\n")
        assert "SyntaxError" in res

    # --- L1: invalid function_name (empty parts) ----------------------------
    def test_invalid_function_name_empty_parts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = _write(tmp_path, "mod.py", "def foo():\n    return 1\n")
        res = replace_python_function(str(f), "a..b", "def b():\n    pass\n")
        assert "Invalid function_name" in res

    def test_invalid_function_name_trailing_dot(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = _write(tmp_path, "mod.py", "def foo():\n    return 1\n")
        res = replace_python_function(str(f), "foo.", "def foo():\n    pass\n")
        assert "Invalid function_name" in res

    def test_invalid_function_name_leading_dot(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = _write(tmp_path, "mod.py", "def foo():\n    return 1\n")
        res = replace_python_function(str(f), ".foo", "def foo():\n    pass\n")
        assert "Invalid function_name" in res

    # --- L1: function_name points to a class (not a function) --------------
    def test_function_name_points_to_class(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "class A:\n    def m(self):\n        return 1\n"
        f = _write(tmp_path, "mod.py", src)
        res = replace_python_function(str(f), "A", "class A:\n    pass\n")
        assert "not found" in res

    # --- L4×D10: prefix and suffix preserved exactly -----------------------
    def test_preserves_prefix_and_suffix(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "# header comment\n"
            "import os\n"
            "\n"
            "def foo():\n"
            "    return 1\n"
            "\n"
            "def bar():\n"
            "    return 2\n"
        )
        f = _write(tmp_path, "mod.py", src)
        original_lines = src.splitlines()
        res = replace_python_function(str(f), "foo", "def foo():\n    return 99\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        out_lines = out.splitlines()
        # Prefix: first 3 lines unchanged
        assert out_lines[:3] == original_lines[:3]
        # Suffix: bar() and its body preserved
        assert "def bar():" in out
        assert "return 2" in out
        # The replaced function
        assert "return 99" in out
        assert "return 1" not in out

    # --- L2: function is last in file (no suffix) --------------------------
    def test_function_last_in_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "def bar():\n"
            "    return 2\n"
            "\n"
            "def foo():\n"
            "    return 1\n"
        )
        f = _write(tmp_path, "mod.py", src)
        res = replace_python_function(str(f), "foo", "def foo():\n    return 99\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "return 99" in out
        assert "return 1" not in out
        assert "def bar():" in out

    # --- L2: function is first in file (no prefix) --------------------------
    def test_function_first_in_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "def foo():\n"
            "    return 1\n"
            "\n"
            "def bar():\n"
            "    return 2\n"
        )
        f = _write(tmp_path, "mod.py", src)
        res = replace_python_function(str(f), "foo", "def foo():\n    return 99\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "return 99" in out
        assert "return 1" not in out
        assert "def bar():" in out

    # --- L3: new_code with leading/trailing newlines → stripped ------------
    def test_new_code_leading_trailing_newlines_stripped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "def foo():\n    return 1\n"
        f = _write(tmp_path, "mod.py", src)
        new_code = "\n\n\ndef foo():\n    return 2\n\n\n"
        res = replace_python_function(str(f), "foo", new_code)
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "return 2" in out
        assert "return 1" not in out

    # --- L1: syntax validation fails → edit rejected, file reverted -------
    def test_syntax_validation_fails_reverted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import tools.file_ops as file_ops_mod
        original = "def foo():\n    return 1\n"
        f = _write(tmp_path, "mod.py", original)

        call_count = [0]

        def fake_validate(fp):
            call_count[0] += 1
            # Only fail for the target file.
            if str(fp) == str(f):
                return "SyntaxError: invalid syntax"
            return None

        # _validate_code_syntax is imported into file_ops namespace, so
        # patch the bound reference there, not the helpers module.
        monkeypatch.setattr(file_ops_mod, '_validate_code_syntax', fake_validate)
        res = replace_python_function(str(f), "foo", "def foo():\n    return 2\n")
        assert "Edit REJECTED" in res or "REJECTED" in res
        # File should be reverted to original.
        assert f.read_text(encoding="utf-8") == original

    # --- D4: UTF-8 strings preserved ---------------------------------------
    def test_utf8_strings_preserved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "# -*- coding: utf-8 -*-\n"
            "def greet():\n"
            "    return 'Привет мир 🌍'\n"
        )
        f = _write(tmp_path, "mod.py", src)
        new_code = "def greet():\n    return 'Здравствуйте мир 🌍'\n"
        res = replace_python_function(str(f), "greet", new_code)
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "Здравствуйте мир 🌍" in out
        assert "Привет мир" not in out

    # --- L2: multi-line function replacement with body ---------------------
    def test_multiline_replacement(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "def compute(x):\n"
            "    a = 1\n"
            "    b = 2\n"
            "    return a + b\n"
        )
        f = _write(tmp_path, "mod.py", src)
        new_code = (
            "def compute(x):\n"
            "    a = 10\n"
            "    b = 20\n"
            "    c = a * b\n"
            "    return c\n"
        )
        res = replace_python_function(str(f), "compute", new_code)
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "c = a * b" in out
        assert "return c" in out
        assert "return a + b" not in out

    # --- L2: replace method in class with multiple methods -----------------
    def test_replace_one_of_many_methods(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "class Svc:\n"
            "    def a(self):\n"
            "        return 1\n"
            "    def b(self):\n"
            "        return 2\n"
            "    def c(self):\n"
            "        return 3\n"
        )
        f = _write(tmp_path, "mod.py", src)
        res = replace_python_function(
            str(f), "Svc.b", "def b(self):\n    return 99\n")
        assert res.startswith("Successfully replaced")
        out = f.read_text(encoding="utf-8")
        assert "return 99" in out
        assert "return 2" not in out
        # Other methods untouched.
        assert "return 1" in out
        assert "return 3" in out


# ===========================================================================
# get_file_outline
# ===========================================================================

class TestGetFileOutlinePython:
    """get_file_outline for .py files."""

    # --- L2: functions -----------------------------------------------------
    def test_outline_functions(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "def alpha():\n    pass\n\ndef beta():\n    return 1\n"
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        assert "def alpha(...): (line 1)" in res
        assert "def beta(...): (line 4)" in res

    # --- L2: classes -------------------------------------------------------
    def test_outline_classes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "class Foo:\n    pass\n\nclass Bar:\n    pass\n"
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        assert "class Foo(): (line 1)" in res
        assert "class Bar(): (line 4)" in res

    # --- L2: class methods shown with extra indent -------------------------
    def test_outline_class_methods_indented(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "class Foo:\n"
            "    def method_a(self):\n"
            "        pass\n"
            "    def method_b(self):\n"
            "        pass\n"
        )
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        assert "class Foo(): (line 1)" in res
        assert "    def method_a(...): (line 2)" in res
        assert "    def method_b(...): (line 4)" in res

    # --- L2: async functions -----------------------------------------------
    def test_outline_async_functions(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "async def fetch():\n    return 1\n"
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        assert "async def fetch(...): (line 1)" in res

    # --- L2: async class methods -------------------------------------------
    def test_outline_async_class_method(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "class Svc:\n    async def run(self):\n        return 1\n"
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        assert "class Svc(): (line 1)" in res
        assert "    async def run(...): (line 2)" in res

    # --- L2: imports --------------------------------------------------------
    def test_outline_imports(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "import os\nfrom pathlib import Path\n"
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        # ImportFrom: "import <module> from <names>"
        assert "import pathlib from Path (line 2)" in res
        # The first import line should reference line 1.
        assert "import os (line 1)" in res

    def test_outline_import_from(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "from collections import OrderedDict, defaultdict\n"
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        assert "import collections from OrderedDict, defaultdict (line 1)" in res

    # --- L2: variable assignments ------------------------------------------
    def test_outline_variables(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "MAX_SIZE = 100\nMIN_SIZE = 10\n"
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        assert "Variable: MAX_SIZE (line 1)" in res
        assert "Variable: MIN_SIZE (line 2)" in res

    # --- L2: inheritance (bases shown) --------------------------------------
    def test_outline_inheritance(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "class Dog(Animal):\n    pass\n"
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        assert "class Dog(Animal): (line 1)" in res

    def test_outline_multiple_inheritance(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "class C(A, B):\n    pass\n"
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        assert "class C(A, B): (line 1)" in res

    # --- L1: non-existent file --------------------------------------------
    def test_outline_non_existent_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        res = get_file_outline(str(tmp_path / "nope.py"))
        assert "does not exist" in res

    # --- L1: path is directory ---------------------------------------------
    def test_outline_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        res = get_file_outline(str(tmp_path))
        assert "is not a file" in res

    # --- L1: non-Python non-C# file ---------------------------------------
    def test_outline_unsupported_extension(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = _write(tmp_path, "notes.txt", "some text\n")
        res = get_file_outline(str(f))
        assert "only available for Python" in res

    # --- L1: Python file with SyntaxError ----------------------------------
    def test_outline_syntax_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = _write(tmp_path, "mod.py", "def foo(:\n    return 1\n")
        res = get_file_outline(str(f))
        assert "SyntaxError" in res

    # --- L3: empty Python file ---------------------------------------------
    def test_outline_empty_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = _write(tmp_path, "mod.py", "")
        res = get_file_outline(str(f))
        assert "No significant structures found" in res

    # --- L3: only comments -------------------------------------------------
    def test_outline_only_comments(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "# this is a comment\n# another comment\n"
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        assert "No significant structures found" in res

    # --- L3: mixed content outline -----------------------------------------
    def test_outline_mixed_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "import os\n"
            "\n"
            "CONSTANT = 42\n"
            "\n"
            "class MyClass(Base):\n"
            "    def method(self):\n"
            "        pass\n"
            "\n"
            "def standalone():\n"
            "    return True\n"
        )
        f = _write(tmp_path, "mod.py", src)
        res = get_file_outline(str(f))
        assert "import" in res and "(line 1)" in res
        assert "Variable: CONSTANT (line 3)" in res
        assert "class MyClass(Base): (line 5)" in res
        assert "    def method(...): (line 6)" in res
        assert "def standalone(...): (line 9)" in res


class TestGetFileOutlineCSharp:
    """get_file_outline for .cs files (regex-based _csharp_outline)."""

    # --- L2: C# classes -----------------------------------------------------
    def test_outline_csharp_class(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "public class Player : MonoBehaviour {\n}\n"
        f = _write(tmp_path, "Player.cs", src)
        res = get_file_outline(str(f))
        assert "class Player (line 1)" in res

    # --- L2: C# methods -----------------------------------------------------
    def test_outline_csharp_methods(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "public class Player {\n"
            "    void Start() {\n"
            "    }\n"
            "    void Update() {\n"
            "    }\n"
            "}\n"
        )
        f = _write(tmp_path, "Player.cs", src)
        res = get_file_outline(str(f))
        assert "class Player (line 1)" in res
        assert "    void Start(...) (line 2)" in res
        assert "    void Update(...) (line 4)" in res

    # --- L2: C# method with return type and params -------------------------
    def test_outline_csharp_method_with_params(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "public class Calc {\n"
            "    public int Add(int a, int b) {\n"
            "        return a + b;\n"
            "    }\n"
            "}\n"
        )
        f = _write(tmp_path, "Calc.cs", src)
        res = get_file_outline(str(f))
        assert "class Calc (line 1)" in res
        # The C# regex captures only the return type, not the modifiers.
        assert "    int Add(...) (line 2)" in res

    # --- L3: C# file with no classes/methods -------------------------------
    def test_outline_csharp_no_classes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "// just a comment\n// nothing else\n"
        f = _write(tmp_path, "Empty.cs", src)
        res = get_file_outline(str(f))
        assert "No classes or methods found" in res

    # --- L2: C# struct and interface --------------------------------------
    def test_outline_csharp_struct_and_interface(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = (
            "public struct Point {\n"
            "}\n"
            "public interface IMovable {\n"
            "}\n"
        )
        f = _write(tmp_path, "Types.cs", src)
        res = get_file_outline(str(f))
        assert "struct Point (line 1)" in res
        assert "interface IMovable (line 3)" in res

    # --- L2: C# enum --------------------------------------------------------
    def test_outline_csharp_enum(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = "public enum Color {\n    Red, Green, Blue\n}\n"
        f = _write(tmp_path, "Color.cs", src)
        res = get_file_outline(str(f))
        assert "enum Color (line 1)" in res