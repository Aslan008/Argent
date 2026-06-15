"""
Deterministic project analysis for onboarding.

analyze_project() gathers the hard facts about a codebase in one call —
directory tree, detected languages/frameworks (from manifests), entry points,
build/test commands and file statistics — so the model can write an accurate
AGENTS.md without blindly probing the filesystem step by step. Collecting the
facts in code (not "by eye") is what makes this reliable even for weak models.
"""

import json
import os
from collections import Counter
from pathlib import Path

from logger import get_logger

log = get_logger("tools")

_IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".idea",
    ".vscode", ".argent", "dist", "build", "obj", "bin", ".pytest_cache",
    ".mypy_cache", "target", ".next", ".nuxt", "coverage", "Library", "Temp",
}

# Manifest file -> (language/ecosystem label, key to read deps from)
_MANIFESTS = {
    "requirements.txt": "Python (pip)",
    "pyproject.toml": "Python (pyproject)",
    "setup.py": "Python (setuptools)",
    "Pipfile": "Python (pipenv)",
    "package.json": "JavaScript/Node",
    "Cargo.toml": "Rust (cargo)",
    "go.mod": "Go (modules)",
    "pom.xml": "Java (Maven)",
    "build.gradle": "Java/Kotlin (Gradle)",
    "Gemfile": "Ruby (bundler)",
    "composer.json": "PHP (composer)",
    "CMakeLists.txt": "C/C++ (CMake)",
}

def _should_skip_dir(name: str) -> bool:
    """Skip noise dirs: hidden, ignore-listed, and any venv-like folder
    (venv, .venv, venv_pyqt6, ...) which would otherwise dominate the stats."""
    return (name in _IGNORE_DIRS or name.startswith(".")
            or name.lower().startswith("venv"))


_ENTRY_CANDIDATES = [
    "main.py", "app.py", "manage.py", "__main__.py", "run.py", "server.py",
    "index.js", "index.ts", "server.js", "app.js", "main.js", "main.ts",
    "Program.cs", "main.go", "main.rs", "index.html", "Main.java",
]


def _detect_frameworks(root: Path) -> list[str]:
    """Best-effort framework detection from package.json / requirements."""
    found = set()
    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            for name in ("react", "vue", "next", "nuxt", "svelte", "angular",
                         "express", "fastify", "electron", "playwright", "vite"):
                if any(name in d.lower() for d in deps):
                    found.add(name)
        except Exception:
            pass
    for req in ("requirements.txt", "pyproject.toml"):
        f = root / req
        if f.exists():
            try:
                text = f.read_text(encoding="utf-8", errors="ignore").lower()
                for name in ("django", "flask", "fastapi", "pyqt", "pyside",
                             "streamlit", "torch", "tensorflow", "transformers"):
                    if name in text:
                        found.add(name)
            except Exception:
                pass
    return sorted(found)


def _build_tree(root: Path, max_depth: int = 2, max_entries: int = 60) -> str:
    lines = []
    count = 0
    for current, dirs, files in os.walk(root):
        rel = Path(current).relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        dirs[:] = sorted(d for d in dirs if not _should_skip_dir(d))
        if depth >= max_depth:
            dirs[:] = []
        indent = "  " * depth
        if str(rel) != ".":
            lines.append(f"{indent}{rel.parts[-1]}/")
            count += 1
        for fname in sorted(files)[:12]:
            lines.append(f"{indent}  {fname}")
            count += 1
            if count >= max_entries:
                lines.append(f"{indent}  ... (truncated)")
                return "\n".join(lines)
    return "\n".join(lines)


def _file_stats(root: Path) -> str:
    ext_counts = Counter()
    total = 0
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not _should_skip_dir(d)]
        for f in files:
            ext = Path(f).suffix.lower() or "(no ext)"
            ext_counts[ext] += 1
            total += 1
    top = ext_counts.most_common(8)
    parts = ", ".join(f"{ext}: {n}" for ext, n in top)
    return f"{total} files. By type: {parts}"


def analyze_project(path: str = ".") -> str:
    """Analyze the project structure: directory tree, languages/frameworks,
    entry points, build/test commands and file statistics. Call this first
    when onboarding to a codebase or writing AGENTS.md."""
    try:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return f"Error: '{path}' is not a directory."

        sections = [f"# PROJECT ANALYSIS: {root}"]

        # Manifests / languages
        manifests = [f"{name} -> {label}" for name, label in _MANIFESTS.items()
                     if (root / name).exists()]
        # *.csproj / *.sln are globbed (variable names).
        if list(root.glob("*.csproj")) or list(root.glob("*.sln")):
            manifests.append("*.csproj/*.sln -> C#/.NET")
        sections.append("## Languages & manifests\n" +
                        ("\n".join(f"- {m}" for m in manifests) if manifests
                         else "- (no standard manifest detected)"))

        frameworks = _detect_frameworks(root)
        if frameworks:
            sections.append("## Detected frameworks/libraries\n" +
                            ", ".join(frameworks))

        # Entry points
        entries = [c for c in _ENTRY_CANDIDATES if (root / c).exists()]
        if entries:
            sections.append("## Likely entry points\n" +
                            "\n".join(f"- {e}" for e in entries))

        # Build/test commands from package.json scripts
        cmds = []
        pkg = root / "package.json"
        if pkg.exists():
            try:
                scripts = json.loads(pkg.read_text(encoding="utf-8")).get("scripts", {})
                cmds += [f"npm run {k}  # {v}" for k, v in list(scripts.items())[:8]]
            except Exception:
                pass
        if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists():
            cmds.append("pytest  # run Python tests")
        if (root / "requirements.txt").exists():
            cmds.append("pip install -r requirements.txt")
        if cmds:
            sections.append("## Build/test commands\n" +
                            "\n".join(f"- {c}" for c in cmds))

        # Tree + stats
        sections.append("## Directory tree (depth 2)\n" + _build_tree(root))
        sections.append("## File statistics\n" + _file_stats(root))

        sections.append(
            "## Next step\nUse this to write a concise .argent/AGENTS.md: what the "
            "project is, its architecture, how to build/run/test it, and key "
            "conventions. Keep it dense — it loads into context every turn."
        )
        log.info("analyze_project: %s", root)
        return "\n\n".join(sections)
    except Exception as e:
        return f"Error analyzing project: {e}"
