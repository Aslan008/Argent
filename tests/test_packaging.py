import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestVersion:
    def test_version_importable(self):
        from argent.version import __version__
        assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


class TestPyproject:
    def test_parses_and_has_entry_point(self):
        import tomllib
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert data["project"]["name"] == "argent-coder"
        assert data["project"]["scripts"]["argent"] == "argent.main:main"
        # version is sourced dynamically from argent/version.py
        assert "version" in data["project"]["dynamic"]
        assert data["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "argent.version.__version__"

    def test_the_package_is_found_by_name_not_by_a_module_list(self):
        import tomllib
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        # The flat `py-modules` list was the reason a module could be in the tree,
        # work from source, and be missing from an installed copy.
        assert "py-modules" not in data["tool"].get("setuptools", {})
        include = data["tool"]["setuptools"]["packages"]["find"]["include"]
        assert "argent*" in include


class TestPackageLayout:
    def test_the_license_is_declared_as_spdx_and_the_file_exists(self):
        import tomllib
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert data["project"]["license"] == "MIT"
        assert (ROOT / "LICENSE").is_file()

    def test_importing_the_package_does_not_import_the_agent_loop(self):
        """`argent/__init__.py` must stay a signpost.

        If it imported its submodules, `import argent.config` would drag in the
        model providers and the SDK clients through the package root.
        """
        module = __import__("argent")
        assert module.__doc__ and "Argent" in module.__doc__
        assert "argent.agent" not in module.__dict__
