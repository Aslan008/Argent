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

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


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

    def test_rag_ships_its_store_and_its_encoder_together(self):
        """Half of RAG installed is a semantic search that cannot start."""
        rag = " ".join(_pyproject()["project"]["optional-dependencies"]["rag"]).lower()
        assert "chromadb" in rag and "sentence-transformers" in rag


class TestPackagedModules:
    def test_every_packaged_module_exists(self):
        listed = _pyproject()["tool"]["setuptools"]["py-modules"]
        missing = [m for m in listed if not (_ROOT / f"{m}.py").exists()]
        assert not missing, f"py-modules перечисляет несуществующие модули: {missing}"

    def test_every_root_module_is_packaged(self):
        """A module left out of py-modules is absent from an installed copy and
        fails on import only for people who installed rather than cloned."""
        listed = set(_pyproject()["tool"]["setuptools"]["py-modules"])
        on_disk = {p.stem for p in _ROOT.glob("*.py")
                   if not p.stem.startswith("test_") and p.stem != "argent_server"}
        assert not (on_disk - listed), f"не попадут в установку: {sorted(on_disk - listed)}"
