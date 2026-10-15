"""A private copy of the working tree for a sub-agent to change freely.

Two sub-agents editing the same files at once overwrite each other; one sub-agent
experimenting in your tree leaves you to sort out what it broke. A git worktree is a
second checkout of the same repository in another folder: the sub-agent edits there,
you keep working here, and what it did comes back as a patch you can read and apply
— or throw away.

The copy starts from your tree *as it is now*, not from the last commit: a worktree
made from HEAD would lack the uncommitted work the sub-agent is supposed to build on.
`git stash create` captures tracked changes as a commit without touching the tree or
the stash list; new (untracked, not ignored) files are copied over by hand.

Worktrees live outside the repository, so they never show up in your `git status`,
and are always removed through git so its bookkeeping (.git/worktrees) stays clean.
"""

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from logger import get_logger

log = get_logger("worktrees")

MAX_COPIED_FILES = 2000
MAX_COPIED_BYTES = 50 * 1024 * 1024


class WorktreeError(RuntimeError):
    pass


@dataclass
class Worktree:
    repo: Path
    path: Path
    base: str                 # the commit the copy started from

    def exists(self) -> bool:
        return self.path.is_dir()


def _git(args: List[str], cwd: Path, check: bool = True) -> str:
    from src.agent.shell import run_text
    done = run_text(["git", *args], capture_output=True, cwd=str(cwd))
    if check and done.returncode != 0:
        raise WorktreeError((done.stderr or done.stdout or f"git {args[0]} failed").strip())
    return done.stdout if done.returncode == 0 else ""


def repo_root(start: Optional[Path] = None) -> Path:
    """The top of the git repository that contains `start`."""
    start = Path(start) if start else Path.cwd()
    try:
        top = _git(["rev-parse", "--show-toplevel"], start).strip()
    except WorktreeError:
        raise WorktreeError("Isolation needs a git repository: this folder is not one.")
    return Path(top)


def snapshot(repo: Path) -> str:
    """A commit that is HEAD plus every tracked change in the tree right now."""
    try:
        head = _git(["rev-parse", "--verify", "HEAD"], repo).strip()
    except WorktreeError:
        raise WorktreeError("Isolation needs at least one commit in the repository.")
    stash = _git(["stash", "create"], repo, check=False).strip()
    return stash or head


def _copy_untracked(repo: Path, dest: Path) -> int:
    names = [n for n in _git(["ls-files", "--others", "--exclude-standard"], repo).splitlines() if n.strip()]
    copied, size = 0, 0
    for name in names:
        src = repo / name
        try:
            if not src.is_file() or src.is_symlink():
                continue
            size += src.stat().st_size
            if copied >= MAX_COPIED_FILES or size > MAX_COPIED_BYTES:
                log.warning("worktree: stopped copying new files at %d files / %d bytes", copied, size)
                break
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied += 1
        except OSError as e:
            log.warning("worktree: could not copy %s: %s", name, e)
    return copied


def create(label: str, start: Optional[Path] = None, root: Optional[Path] = None) -> Worktree:
    repo = repo_root(start)
    base = snapshot(repo)
    parent = Path(root) if root else Path(tempfile.gettempdir()) / "argent-worktrees"
    parent.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:40] or "agent"
    path = Path(tempfile.mkdtemp(prefix=f"{repo.name}-{safe}-", dir=str(parent)))
    path.rmdir()                                  # git wants to create the folder itself
    _git(["worktree", "add", "--detach", str(path), base], repo)
    if _copy_untracked(repo, path):
        # The copied new files are part of where the sub-agent STARTS. Commit them (an unnamed commit,
        # no branch) so the patch it produces contains only its own work and applies back cleanly.
        _git(["add", "-A"], path)
        _git(["-c", "user.name=Argent", "-c", "user.email=argent@localhost", "commit", "-q", "--no-verify",
              "--no-gpg-sign", "-m", "argent: starting point of the sub-agent"], path)
        base = _git(["rev-parse", "HEAD"], path).strip()
    return Worktree(repo=repo, path=path, base=base)


def changes(wt: Worktree) -> str:
    """Everything the sub-agent changed in its copy, as a patch against where it started."""
    if not wt.exists():
        return ""
    _git(["add", "-A"], wt.path)
    return _git(["diff", "--cached", "--binary", wt.base], wt.path, check=False)


def changed_files(wt: Worktree) -> List[str]:
    if not wt.exists():
        return []
    _git(["add", "-A"], wt.path)
    out = _git(["diff", "--cached", "--name-only", wt.base], wt.path, check=False)
    return [line for line in out.splitlines() if line.strip()]


def remove(wt: Worktree) -> None:
    """Delete the copy and git's record of it."""
    try:
        _git(["worktree", "remove", "--force", str(wt.path)], wt.repo, check=False)
    finally:
        if wt.path.exists():
            shutil.rmtree(wt.path, ignore_errors=True)
        _git(["worktree", "prune"], wt.repo, check=False)


def apply_patch(repo: Path, patch_file: Path) -> str:
    """Apply a saved patch to the real tree. Returns "" on success, else why not (nothing is changed then)."""
    from src.agent.shell import run_text
    args = ["git", "apply", "--whitespace=nowarn"]
    check = run_text([*args, "--check", str(patch_file)], capture_output=True, cwd=str(repo))
    if check.returncode != 0:
        return (check.stderr or check.stdout or "the patch does not apply").strip()
    done = run_text([*args, str(patch_file)], capture_output=True, cwd=str(repo))
    return "" if done.returncode == 0 else (done.stderr or done.stdout or "git apply failed").strip()
