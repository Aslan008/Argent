"""Turn-level time machine: automatic git checkpoints and safe rewind.

One implementation for every rollback path in Argent:

- auto_checkpoint(): called by the agent before the FIRST file edit of a turn,
  so every turn that changes the tree gets a "before" snapshot for free.
- git_checkpoint / git_rollback (the model-facing tools) and the /rewind
  terminal command all delegate here instead of duplicating git plumbing.

Checkpoints are ordinary commits prefixed "Argent Checkpoint:". Rewind is a
hard reset with two mechanical guarantees enforced HERE (consent is the
caller's job — terminal prompt, tool approval, or GUI dialog):

1. Only checkpoint commits may be discarded: a real user commit between HEAD
   and the target aborts the rewind.
2. Nothing is lost silently: uncommitted + untracked changes are stashed
   before the reset, and discarded checkpoint shas stay recoverable via
   `git reflog`.
"""

import subprocess
from typing import NamedTuple

from logger import get_logger

log = get_logger("checkpoints")

CHECKPOINT_PREFIX = "Argent Checkpoint:"

# Flipped by /vibe and tests; ON by default — fearless rewind is the point.
_auto_enabled = True


class CheckpointError(RuntimeError):
    """A rewind/checkpoint operation that could not proceed safely."""


def set_auto_checkpoint(enabled: bool) -> None:
    global _auto_enabled
    _auto_enabled = bool(enabled)


def auto_checkpoint_enabled() -> bool:
    return _auto_enabled


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def is_git_repo() -> bool:
    try:
        return _git("rev-parse", "--is-inside-work-tree").returncode == 0
    except Exception:
        return False


def create_checkpoint(message: str) -> str | None:
    """Commit the whole tree as an Argent Checkpoint.

    Returns the short sha, or None when the tree is clean. Raises
    CheckpointError when git itself fails (e.g. user.email not configured).
    """
    if not _git("status", "--porcelain").stdout.strip():
        return None  # clean tree — and the index stays untouched
    _git("add", "-A")
    if _git("diff", "--cached", "--quiet").returncode == 0:
        return None
    res = _git("commit", "-m", f"{CHECKPOINT_PREFIX} {message}")
    if res.returncode != 0:
        raise CheckpointError(f"git commit failed: {res.stderr.strip() or res.stdout.strip()}")
    return _git("rev-parse", "--short", "HEAD").stdout.strip()


class CheckpointResult(NamedTuple):
    """sha of the checkpoint, or why none was made. warn=True marks the one
    case the user must hear about: edits are about to happen with no safety net
    even though checkpointing is switched ON."""
    sha: str | None
    reason: str | None = None
    warn: bool = False


def auto_checkpoint_detailed(label: str) -> CheckpointResult:
    """Best-effort per-turn checkpoint, with the reason when it doesn't happen.
    Never raises — a failed snapshot must not block the edit it was insuring."""
    if not _auto_enabled:
        return CheckpointResult(None, "auto-checkpoints are off")
    try:
        if not is_git_repo():
            return CheckpointResult(None, "not a git repository")
        status = _git("status", "--porcelain").stdout
        if any(line and line[0] not in (" ", "?") for line in status.splitlines()):
            # Committing `git add -A` over a hand-crafted index would destroy
            # the user's in-progress commit — worse than missing a checkpoint.
            # But staying SILENT is its own trap: they'd believe /rewind has
            # them covered while the agent edits unprotected.
            log.info("auto-checkpoint skipped: user has staged changes")
            return CheckpointResult(
                None,
                "у вас есть файлы в git-индексе (staged) — авто-чекпоинт пропущен, "
                "чтобы не разрушить ваш подготовленный коммит. Правки этого хода "
                "НЕ покрыты /rewind: закоммитьте или снимите индекс (git reset), "
                "либо откатывайте по файлу через /undo.",
                warn=True,
            )
        sha = create_checkpoint(f"before: {label}")
        if sha is None:
            return CheckpointResult(None, "nothing to checkpoint (clean tree)")
        return CheckpointResult(sha)
    except Exception as e:
        log.warning("auto-checkpoint failed: %s", e)
        return CheckpointResult(None, f"checkpoint failed: {e}", warn=True)


def auto_checkpoint(label: str) -> str | None:
    """Per-turn checkpoint; returns the sha, or None when none was made."""
    return auto_checkpoint_detailed(label).sha


def list_checkpoints(limit: int = 15) -> list[dict]:
    """Argent checkpoints reachable from HEAD, newest first:
    [{"sha", "label", "age"}]."""
    res = _git("log", f"--grep=^{CHECKPOINT_PREFIX}", f"-{limit}",
               "--pretty=%h%x09%cr%x09%s")
    out = []
    for line in res.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3 or not parts[2].startswith(CHECKPOINT_PREFIX):
            continue
        sha, age, subject = parts
        out.append({"sha": sha, "age": age,
                    "label": subject[len(CHECKPOINT_PREFIX):].strip()})
    return out


def blocking_commits(target_sha: str) -> list[str]:
    """Non-checkpoint commits in target..HEAD that a rewind would discard."""
    res = _git("log", f"{target_sha}..HEAD", "--pretty=%h%x09%s")
    bad = []
    for line in res.stdout.splitlines():
        sha, _, subject = line.partition("\t")
        if not subject.startswith(CHECKPOINT_PREFIX):
            bad.append(f"{sha} {subject}")
    return bad


def rewind_to(target_sha: str) -> str:
    """Hard-reset the tree to a checkpoint commit. Consent is the caller's
    responsibility; this function enforces only the mechanical safety rules."""
    if not is_git_repo():
        raise CheckpointError("Not a git repository.")

    subject = _git("log", "-1", "--pretty=%s", target_sha)
    if subject.returncode != 0:
        raise CheckpointError(f"Unknown commit '{target_sha}'.")
    if not subject.stdout.strip().startswith(CHECKPOINT_PREFIX):
        raise CheckpointError(
            f"Commit {target_sha} is not an Argent Checkpoint — rewind refused for safety.")
    if _git("merge-base", "--is-ancestor", target_sha, "HEAD").returncode != 0:
        raise CheckpointError(
            f"Checkpoint {target_sha} is not on the current branch history.")

    bad = blocking_commits(target_sha)
    if bad:
        listing = "\n".join(f"  - {b}" for b in bad)
        raise CheckpointError(
            "Rewind would discard real (non-checkpoint) commits:\n"
            f"{listing}\nAborted for safety.")

    stashed = False
    if _git("status", "--porcelain").stdout.strip():
        _git("stash", "push", "-u", "-m",
             f"Argent Auto-Save before Rewind to {target_sha}")
        stashed = True

    discarded = _git("rev-list", "--count", f"{target_sha}..HEAD").stdout.strip()
    res = _git("reset", "--hard", target_sha)
    if res.returncode != 0:
        raise CheckpointError(f"git reset failed: {res.stderr.strip()}")

    label = subject.stdout.strip()[len(CHECKPOINT_PREFIX):].strip()
    msg = f"Rewound to checkpoint {target_sha} — '{label}'."
    if discarded and discarded != "0":
        msg += f" Discarded {discarded} newer checkpoint(s) (recoverable via `git reflog`)."
    if stashed:
        msg += " Uncommitted changes were stashed (`git stash list`)."
    return msg
