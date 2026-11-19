"""One list of dependencies, not three that disagree.

requirements.txt was a hand-copied second list and had drifted from
pyproject.toml in both directions:

  * pillow was missing — pyproject requires it so view_image can downscale a
    screenshot before it is sent; without it the tool degrades quietly;
  * chromadb was missing entirely while sentence-transformers was present, so
    `pip install -r requirements.txt` installed half of RAG;
  * crawl4ai, playwright and sentence-transformers sat among the base deps
    while pyproject deliberately makes them extras.

None of that fails at install time. It fails later, as a feature doing less
than it says.
"""

import importlib
import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ARGENT = _ROOT / "argent"


def _pyproject() -> dict:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _requirement_lines(name: str) -> list:
    text = (_ROOT / name).read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


class TestRequirementsDelegates:
    def test_it_names_no_packages_of_its_own(self):
        """A second list is a list that drifts."""
        lines = _requirement_lines("requirements.txt")
        loose = [ln for ln in lines if not ln.startswith("-")]
        assert not loose, (
            f"requirements.txt снова перечисляет пакеты вручную: {loose}")

    def test_it_installs_the_project(self):
        assert any(ln.startswith("-e .") for ln in _requirement_lines("requirements.txt"))

    def test_the_extras_it_asks_for_exist(self):
        line = next(ln for ln in _requirement_lines("requirements.txt")
                    if ln.startswith("-e ."))
        asked = set(re.findall(r"[\w-]+", line.partition("[")[2].partition("]")[0]))
        declared = set(_pyproject()["project"]["optional-dependencies"])
        assert asked <= declared, f"неизвестные extras: {sorted(asked - declared)}"


class TestTheOnesThatWereMissing:
    def test_pillow_is_required_not_optional(self):
        """view_image downscales with it; without it a full-resolution
        screenshot goes to the model."""
        base = " ".join(_pyproject()["project"]["dependencies"])
        assert "pillow" in base.lower()

    def test_psutil_is_required_not_optional(self):
        """Process-tree kill and lock liveness have fallbacks that are worse:
        an MCP server started through a launcher can outlive Argent."""
        base = " ".join(_pyproject()["project"]["dependencies"])
        assert "psutil" in base.lower()
        assert "psutil" in " ".join(_requirement_lines("requirements-ci.txt")).lower()

    def test_rag_ships_its_store_and_its_encoder_together(self):
        """Half of RAG installed is a semantic search that cannot start."""
        rag = " ".join(_pyproject()["project"]["optional-dependencies"]["rag"]).lower()
        assert "chromadb" in rag and "sentence-transformers" in rag


# Root-level files that are deliberately not part of the installed package:
# test plumbing and the self-contained "Leeth plays Snake" demo (a game, its Qt
# front end and a benchmark), which nothing in Argent imports.
_NOT_SHIPPED = {"conftest", "leeth_player", "run_snake", "snake_benchmark",
                "snake_engine", "snake_ui_qt"}


class TestPackagedModules:
    """The install is one package; a module missing from it breaks only the CLI.

    This is the test that would have caught `host_os`: the file existed, `python
    main.py` worked from the source tree, and the installed `argent` command
    died with ModuleNotFoundError because the flat `py-modules` list did not
    name it. Moving everything under `argent/` removes the list entirely — and
    these assertions keep the next module from being left behind at the root.
    """

    def test_the_entry_point_module_exists(self):
        assert (_ARGENT / "__init__.py").is_file()
        assert (_ARGENT / "main.py").is_file()
        assert _pyproject()["project"]["scripts"]["argent"] == "argent.main:main"

    def test_every_module_in_the_package_imports(self):
        failed = {}
        for path in _ARGENT.glob("*.py"):
            if path.stem == "__init__":
                continue
            try:
                importlib.import_module(f"argent.{path.stem}")
            except Exception as e:  # noqa: BLE001 - report every broken module
                failed[path.stem] = f"{type(e).__name__}: {e}"
        assert failed == {}

    def test_nothing_packaged_is_left_at_the_root(self):
        on_disk = {p.stem for p in _ROOT.glob("*.py")
                   if not p.stem.startswith("test_")
                   and p.stem not in _NOT_SHIPPED}
        assert on_disk == set(), (
            f"these modules are at the root and would not be installed: {sorted(on_disk)}")
