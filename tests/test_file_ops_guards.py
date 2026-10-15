"""Unity .meta write-guard + actionable fuzzy-ambiguity error in replace_in_file,
plus skipping the slow/unreliable dotnet build for Unity project files."""

import tools._helpers as helpers
from tools._helpers import _is_unity_project_file, _validate_code_syntax
from tools.file_ops import (
    append_to_file, multi_replace_in_file_chunk, replace_in_file, write_file,
)


class TestMetaGuard:
    def test_write_refused(self, tmp_path):
        p = tmp_path / "Foo.cs.meta"
        out = write_file(str(p), "x")
        assert "Refusing to modify a Unity .meta" in out
        assert not p.exists()

    def test_append_refused(self, tmp_path):
        p = tmp_path / "Foo.cs.meta"
        p.write_text("orig", encoding="utf-8")
        out = append_to_file(str(p), "more")
        assert "Refusing to modify a Unity .meta" in out
        assert p.read_text(encoding="utf-8") == "orig"

    def test_replace_refused(self, tmp_path):
        p = tmp_path / "a.meta"
        p.write_text("orig", encoding="utf-8")
        out = replace_in_file(str(p), "orig", "new")
        assert "Refusing to modify a Unity .meta" in out
        assert p.read_text(encoding="utf-8") == "orig"

    def test_multi_replace_refused(self, tmp_path):
        p = tmp_path / "b.meta"
        p.write_text("x", encoding="utf-8")
        out = multi_replace_in_file_chunk(str(p), "[]")
        assert "Refusing to modify a Unity .meta" in out

    def test_normal_cs_not_blocked(self, tmp_path):
        p = tmp_path / "Foo.cs"
        out = write_file(str(p), "class Foo {}")
        assert "Refusing" not in out and p.exists()


class TestAmbiguityError:
    def test_reports_locations_and_advice(self, tmp_path):
        p = tmp_path / "T.cs"
        p.write_text(
            "void A() {\n    Foo();\n    Bar();\n}\n\nvoid B() {\n    Foo();\n    Bar();\n}\n",
            encoding="utf-8",
        )
        # Indentation differs -> exact fails -> fuzzy finds both blocks.
        out = replace_in_file(str(p), "Foo();\nBar();", "Baz();")
        assert "matches 2 places" in out
        assert "lines" in out and "Foo();" in out          # tells the model where
        assert "write_file" in out                          # and what to do
        assert "Baz();" not in p.read_text(encoding="utf-8")  # file untouched


def _boom(*a, **k):
    raise AssertionError("dotnet build must not run here")


class TestUnityValidationSkip:
    def test_meta_sibling_marks_unity(self, tmp_path):
        cs = tmp_path / "Foo.cs"
        cs.write_text("class Foo {}", encoding="utf-8")
        (tmp_path / "Foo.cs.meta").write_text("guid: 1", encoding="utf-8")
        assert _is_unity_project_file(str(cs)) is True

    def test_assets_ancestor_marks_unity(self, tmp_path):
        d = tmp_path / "Assets" / "Scripts"
        d.mkdir(parents=True)
        cs = d / "Foo.cs"
        cs.write_text("class Foo {}", encoding="utf-8")
        assert _is_unity_project_file(str(cs)) is True

    def test_plain_cs_not_unity(self, tmp_path):
        cs = tmp_path / "Foo.cs"
        cs.write_text("class Foo {}", encoding="utf-8")
        assert _is_unity_project_file(str(cs)) is False

    def test_validate_skips_dotnet_build_for_unity(self, tmp_path, monkeypatch):
        assets = tmp_path / "Assets"
        assets.mkdir()
        (tmp_path / "Game.csproj").write_text("<Project/>", encoding="utf-8")  # would trigger build
        cs = assets / "Plain.cs"
        cs.write_text("class C {}", encoding="utf-8")  # no UnityEngine reference
        monkeypatch.setattr(helpers, "run_text", _boom)  # explodes if a build is attempted
        assert _validate_code_syntax(str(cs)) is None     # skipped -> no build

    def test_validate_still_builds_for_non_unity(self, tmp_path, monkeypatch):
        (tmp_path / "App.csproj").write_text("<Project/>", encoding="utf-8")
        cs = tmp_path / "Plain.cs"
        cs.write_text("class C {}", encoding="utf-8")  # no Unity, no Assets, no .meta
        called = {}

        class R:
            returncode = 0
            stdout = ""

        monkeypatch.setattr(helpers, "run_text", lambda *a, **k: called.setdefault("built", True) or R())
        _validate_code_syntax(str(cs))
        assert called.get("built")   # a genuine non-Unity C# project still gets checked


class TestPluginGuard:
    def test_small_model_blocked_from_direct_plugin_edit(self, tmp_path, monkeypatch):
        import config
        from hook_manager import hook_manager

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        monkeypatch.setattr(config, "get_hooks_dir", lambda: str(plugins_dir))
        monkeypatch.setattr(config, "get_current_model", lambda: "qwen2.5:3b")  # small model
        monkeypatch.setattr(config, "get_model_category_override", lambda: "small")

        plugin_file = plugins_dir / "my_plugin.py"
        plugin_file.write_text("print('hello')", encoding="utf-8")

        # replace_in_file should be blocked
        out = replace_in_file(str(plugin_file), "print('hello')", "print('world')")
        assert "restricted for small models" in out
        assert plugin_file.read_text(encoding="utf-8") == "print('hello')"

    def test_medium_model_allowed_and_auto_reloads(self, tmp_path, monkeypatch):
        import config
        from hook_manager import hook_manager

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        monkeypatch.setattr(config, "get_hooks_dir", lambda: str(plugins_dir))
        monkeypatch.setattr(config, "get_current_model", lambda: "qwen2.5-coder:14b")  # large/medium model
        monkeypatch.setattr(config, "get_model_category_override", lambda: "large")

        reloaded = []
        monkeypatch.setattr(hook_manager, "reload_plugins", lambda: reloaded.append(True))

        plugin_file = plugins_dir / "my_plugin.py"
        plugin_file.write_text("print('hello')", encoding="utf-8")

        # replace_in_file should succeed and auto-reload
        out = replace_in_file(str(plugin_file), "print('hello')", "print('world')")
        assert "Successfully replaced" in out
        assert plugin_file.read_text(encoding="utf-8") == "print('world')"
        assert len(reloaded) == 1

    def test_non_python_file_in_plugins_blocked(self, tmp_path, monkeypatch):
        import config
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        monkeypatch.setattr(config, "get_hooks_dir", lambda: str(plugins_dir))

        doc_file = plugins_dir / "notes.md"
        out = write_file(str(doc_file), "Notes")
        assert "is for Python plugins only" in out

