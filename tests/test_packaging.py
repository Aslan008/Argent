import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestVersion:
    def test_version_importable(self):
        from version import __version__
        assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


class TestPyproject:
    def test_parses_and_has_entry_point(self):
        import tomllib
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert data["project"]["name"] == "argent-coder"
        assert data["project"]["scripts"]["argent"] == "main:main"
        # version is sourced dynamically from version.py
        assert "version" in data["project"]["dynamic"]
        assert data["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "version.__version__"

    def test_main_module_is_listed(self):
        import tomllib
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert "main" in data["tool"]["setuptools"]["py-modules"]
