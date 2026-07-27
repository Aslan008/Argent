"""Locate the project root instead of trusting the current working directory.

Argent's per-project state (memory, project brain, AGENTS.md) used to be
addressed relative to $PWD. Launching from a subfolder — `cd src` — therefore
found nothing, silently started a "new project", and dropped the accumulated
context.

find_project_root walks UP from the CWD looking for a project marker, the same
way git locates .git. The walk stops AT the home directory so ~/.argent
(Argent's own global state) is never mistaken for a project, and so an
unrelated ancestor can't capture the session.
"""

from pathlib import Path

# Ordered by strength of signal: an .argent dir is Argent's own project state,
# .argent_project.json is an autonomous run, .git marks a repo root.
PROJECT_MARKERS = (".argent", ".argent_project.json", ".git")


def find_project_root(start: Path | None = None) -> Path | None:
    """Nearest ancestor (including start) holding a project marker, else None."""
    try:
        current = Path(start).resolve() if start else Path.cwd().resolve()
        stop = Path.home().resolve()
    except OSError:
        return None

    for d in [current, *current.parents]:
        # A directory at or above the home directory is never a project root:
        # ~/.argent is Argent's own global state, and anything containing the
        # whole home tree is an unrelated ancestor. Without the second check the
        # walk can escape past home entirely (when the CWD lives outside it) and
        # latch onto a far-away .argent.
        if d == stop or d in stop.parents:
            break
        for marker in PROJECT_MARKERS:
            try:
                if (d / marker).exists():
                    return d
            except OSError:
                continue
    return None


def project_root_or_cwd(start: Path | None = None) -> Path:
    """find_project_root, falling back to the CWD for a brand-new project."""
    return find_project_root(start) or (Path(start).resolve() if start else Path.cwd().resolve())
