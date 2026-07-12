"""Theme is found outside the CWD; page->markdown runs off the event loop."""

import asyncio
import importlib

import pytest


class TestThemeLookup:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        theme = tmp_path / "custom.yaml"
        theme.write_text("colors: {}\n", encoding="utf-8")
        monkeypatch.setenv("ARGENT_THEME", str(theme))
        import ui
        importlib.reload(ui)
        assert ui.THEME_FILE == theme

    def test_theme_found_from_foreign_cwd(self, tmp_path, monkeypatch):
        """The actual regression: launched from an unrelated directory, a theme
        is still resolved (previously the CWD-only lookup returned nothing)."""
        elsewhere = tmp_path / "some" / "other" / "dir"
        elsewhere.mkdir(parents=True)
        monkeypatch.delenv("ARGENT_THEME", raising=False)
        monkeypatch.chdir(elsewhere)
        import ui
        importlib.reload(ui)
        # The repo ships theme.yaml next to ui.py, so it resolves regardless of CWD.
        assert ui.THEME_FILE is not None
        assert ui.THEME_FILE.name == "theme.yaml"

    def test_home_dir_is_a_candidate(self, tmp_path, monkeypatch):
        """_find_theme_file falls through to ~/.argent/theme.yaml."""
        import ui
        home = tmp_path / "home"
        (home / ".argent").mkdir(parents=True)
        expected = home / ".argent" / "theme.yaml"
        expected.write_text("colors: {}\n", encoding="utf-8")

        monkeypatch.delenv("ARGENT_THEME", raising=False)
        monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: home))
        # Neutralize the earlier-priority candidates (cwd + package dir).
        real_is_file = ui.Path.is_file

        def only_home(self):
            return self == expected and real_is_file(self)

        monkeypatch.setattr("pathlib.Path.is_file", only_home)
        assert ui._find_theme_file() == expected


class TestMarkdownOffLoop:
    def test_get_markdown_uses_to_thread(self, monkeypatch):
        import browser_engine

        calls = {"threaded": False}
        real_to_thread = asyncio.to_thread

        async def spy_to_thread(fn, *a, **k):
            calls["threaded"] = True
            return await real_to_thread(fn, *a, **k)

        monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)

        class FakePage:
            async def content(self):
                return "<html><body><h1>Hi</h1><p>world</p></body></html>"

        class FakeSC:
            page = FakePage()

        engine = browser_engine.BrowserEngine.__new__(browser_engine.BrowserEngine)

        async def fake_get_session(session="default"):
            return FakeSC()

        monkeypatch.setattr(engine, "_get_session", fake_get_session)

        md = asyncio.run(engine.get_markdown())
        assert calls["threaded"] is True
        assert "Hi" in md
