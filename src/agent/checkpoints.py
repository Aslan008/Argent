"""Turn-level time machine: automatic git checkpoints and safe rewind.

One implementation for every rollback path in Argent:

- auto_checkpoint(): called by the agent before the FIRST file edit of a turn,
  so every turn that changes the tree gets a "before" snapshot for free.
- git_checkpoint / git_rollback (the model-facing tools) and the /rewind
  terminal command all delegate here instead of duplicating git plumbing.

A checkpoint is a snapshot commit that lives OFF the branch, under
refs/argent/checkpoints/ (2026-10-20).  It is built in a temporary index
(read the real index for its stat cache, `add -A`, `write-tree`, `commit-tree`
with HEAD as parent), so the branch, HEAD and the user's own index are never
touched, nothing shows up in `git log`, and `git push` does not send it.
Until then a checkpoint was a real `git add -A && git commit --no-verify` on the
current branch: it silently swept every untracked file (datasets, a downloaded
npm package) and the user's uncommitted work into a commit the user never made.

Rewind restores the snapshot's files into the working tree as uncommitted
changes, with two mechanical guarantees enforced HERE (consent is the caller's
job — terminal prompt, tool approval, or GUI dialog):

1. No real commit is thrown away: if HEAD moved since the snapshot, the rewind
   is refused and the commits in the way are named.
2. Nothing is lost silently: uncommitted + untracked changes are stashed
   before the restore.

Checkpoints made the old way (commits titled "Argent Checkpoint:" on the
branch) are still listed and rewound by the old hard-reset path.
"""

import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import NamedTuple

from argent.logger import get_logger

log = get_logger("checkpoints")

CHECKPOINT_PREFIX = "Argent Checkpoint:"
CHECKPOINT_REFS = "refs/argent/checkpoints"
KEEP_CHECKPOINTS = 50      # older snapshot refs are dropped (their objects go with the next git gc)

# Flipped by /vibe and tests; ON by default — fearless rewind is the point.
_auto_enabled = True


class CheckpointError(RuntimeError):
    """A rewind/checkpoint operation that could not proceed safely."""


def set_auto_checkpoint(enabled: bool) -> None:
    global _auto_enabled
    _auto_enabled = bool(enabled)


def auto_checkpoint_enabled() -> bool:
    return _auto_enabled


def _git(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              env={**os.environ, **env} if env else None)
    except FileNotFoundError:
        return subprocess.CompletedProcess(args=["git", *args], returncode=127,
                                          stdout="", stderr="git binary not found")


def is_git_repo() -> bool:
    try:
        return _git("rev-parse", "--is-inside-work-tree").returncode == 0
    except Exception:
        return False


def _checked(res: subprocess.CompletedProcess, what: str) -> str:
    if res.returncode != 0:
        raise CheckpointError(f"git {what} failed: {res.stderr.strip() or res.stdout.strip()}")
    return res.stdout.strip()


def create_checkpoint(message: str) -> str | None:
    """Snapshot the working tree (tracked + untracked, .gitignore respected) as a
    commit under refs/argent/checkpoints/ — not on the branch.

    Returns the short sha, or None when the tree is clean. Raises
    CheckpointError when git itself fails (e.g. user.email not configured).
    The branch, HEAD and the user's index are left exactly as they were.
    """
    if not _git("status", "--porcelain").stdout.strip():
        return None  # clean tree: HEAD itself is the state
    head = current_head()
    if head is None:
        raise CheckpointError("the repository has no commits yet — nothing to anchor a checkpoint to")
    tmp_index = Path(_checked(_git("rev-parse", "--git-path", f"argent-checkpoint-{os.getpid()}.index"),
                              "rev-parse")).resolve()
    real_index = Path(_checked(_git("rev-parse", "--git-path", "index"), "rev-parse")).resolve()
    env = {"GIT_INDEX_FILE": str(tmp_index)}
    try:
        if real_index.exists():      # the stat cache: `add -A` then rehashes only the files that changed
            tmp_index.write_bytes(real_index.read_bytes())
        else:
            _checked(_git("read-tree", head, env=env), "read-tree")
        _checked(_git("add", "-A", env=env), "add")
        tree = _checked(_git("write-tree", env=env), "write-tree")
    finally:
        tmp_index.unlink(missing_ok=True)
    if tree == _git("rev-parse", f"{head}^{{tree}}").stdout.strip():
        return None  # only the index differed from HEAD, the files did not
    sha = _checked(_git("commit-tree", tree, "-p", head, "-m", f"{CHECKPOINT_PREFIX} {message}"), "commit-tree")
    _checked(_git("update-ref", f"{CHECKPOINT_REFS}/{time.time_ns():020d}", sha), "update-ref")
    _prune_checkpoints()
    invalidate_checkpoint_cache()
    return _git("rev-parse", "--short", sha).stdout.strip()


def _snapshot_refs() -> list[str]:
    """Snapshot refs, newest first (the ref name is a zero-padded nanosecond clock)."""
    res = _git("for-each-ref", "--sort=-refname", "--format=%(refname)", CHECKPOINT_REFS)
    return [r for r in res.stdout.splitlines() if r.strip()]


def _prune_checkpoints() -> None:
    for ref in _snapshot_refs()[KEEP_CHECKPOINTS:]:
        _git("update-ref", "-d", ref)
    # Pruning removes entries, so a list cached before it is no longer true.
    invalidate_checkpoint_cache()


def _is_snapshot(full_sha: str) -> bool:
    return bool(_git("for-each-ref", f"--points-at={full_sha}", "--format=%(refname)", CHECKPOINT_REFS).stdout.strip())


class CheckpointResult(NamedTuple):
    """sha of the checkpoint, or why none was made. warn=True marks the one
    case the user must hear about: edits are about to happen with no safety net
    even though checkpointing is switched ON."""
    sha: str | None
    reason: str | None = None
    warn: bool = False
    clean: bool = False     # no checkpoint because the tree was clean: HEAD itself is the pre-turn state


def auto_checkpoint_detailed(label: str) -> CheckpointResult:
    """Best-effort per-turn checkpoint, with the reason when it doesn't happen.
    Never raises — a failed snapshot must not block the edit it was insuring."""
    if not _auto_enabled:
        return CheckpointResult(None, "auto-checkpoints are off")
    try:
        if not is_git_repo():
            return CheckpointResult(None, "not a git repository")
        # Staged files are no reason to skip any more: the snapshot is built in a
        # temporary index, so the user's prepared commit is never touched.
        sha = create_checkpoint(f"before: {label}")
        if sha is None:
            return CheckpointResult(None, "nothing to checkpoint (clean tree)", clean=True)
        return CheckpointResult(sha)
    except Exception as e:
        log.warning("auto-checkpoint failed: %s", e)
        return CheckpointResult(None, f"checkpoint failed: {e}", warn=True)


def auto_checkpoint(label: str) -> str | None:
    """Per-turn checkpoint; returns the sha, or None when none was made."""
    return auto_checkpoint_detailed(label).sha


#: TTL of the cached checkpoint list. Long enough that the toolbar's polling costs
#: nothing, short enough that a checkpoint made elsewhere shows up by itself.
_CACHE_TTL_S = 10.0
_cache_lock = threading.Lock()
_cache: dict = {"at": 0.0, "limit": 0, "key": "", "valid": False, "entries": []}


def list_checkpoints(limit: int = 15) -> list[dict]:
    """Argent checkpoints of the current history, newest first:
    [{"sha", "label", "age"}].  Snapshots whose base commit is HEAD or one of its
    ancestors, plus old-style checkpoint commits on the branch.

    Cached for a few seconds. Everything here is git runs — a `for-each-ref` plus a
    `merge-base --is-ancestor` per entry — and the toolbar asks for this list every
    second. Measured on a real repository: ~340 ms of subprocess per call, on the
    thread that draws the interface. The list only changes when a checkpoint is made
    or a rewind happens, and both call `invalidate_checkpoint_cache()`.
    """
    limit = max(1, int(limit))
    now = time.monotonic()
    # The key is the working tree this list belongs to: two repositories in one
    # process (the tests do exactly that) must never share an answer.
    key = os.getcwd()
    with _cache_lock:
        if (_cache["valid"] and _cache["key"] == key and _cache["limit"] >= limit
                and now - _cache["at"] < _CACHE_TTL_S):
            return [dict(e) for e in _cache["entries"][:limit]]
    entries = _scan_checkpoints(limit)
    with _cache_lock:
        if _cache["key"] != key or limit >= _cache["limit"] or now - _cache["at"] >= _CACHE_TTL_S:
            _cache["at"], _cache["limit"], _cache["key"] = now, limit, key
            _cache["entries"] = [dict(e) for e in entries]
            _cache["valid"] = True
    return entries


def invalidate_checkpoint_cache() -> None:
    """Forget the cached list: a checkpoint was made, or the branch moved under it."""
    with _cache_lock:
        _cache["at"] = 0.0
        _cache["key"] = ""
        _cache["valid"] = False
        _cache["entries"] = []


def _scan_checkpoints(limit: int) -> list[dict]:
    """The git work behind list_checkpoints."""
    found = []      # (unix time, entry)
    fmt = "%(objectname:short)%09%(committerdate:relative)%09%(committerdate:unix)%09%(objectname)%09%(subject)"
    res = _git("for-each-ref", "--sort=-refname", f"--format={fmt}", CHECKPOINT_REFS)
    for line in res.stdout.splitlines():
        parts = line.split("\t", 4)
        if len(parts) != 5 or not parts[4].startswith(CHECKPOINT_PREFIX):
            continue
        short, age, ts, full, subject = parts
        if _git("merge-base", "--is-ancestor", f"{full}^", "HEAD").returncode != 0:
            continue    # made on another branch or before a rewrite: not a state of this history
        found.append((int(ts or 0), {"sha": short, "age": age, "label": subject[len(CHECKPOINT_PREFIX):].strip()}))
        if len(found) >= limit:
            break
    res = _git("log", f"--grep=^{CHECKPOINT_PREFIX}", f"-{limit}", "--pretty=%h%x09%cr%x09%ct%x09%s")
    for line in res.stdout.splitlines():
        parts = line.split("\t", 3)
        if len(parts) != 4 or not parts[3].startswith(CHECKPOINT_PREFIX):
            continue
        sha, age, ts, subject = parts
        found.append((int(ts or 0), {"sha": sha, "age": age, "label": subject[len(CHECKPOINT_PREFIX):].strip()}))
    found.sort(key=lambda x: x[0], reverse=True)    # stable: same-second snapshots keep their ref order
    return [e for _, e in found[:limit]]


def blocking_commits(target_sha: str) -> list[str]:
    """Real commits a rewind to ``target_sha`` would have to throw away.
    A snapshot: every commit made since it (its base must still be HEAD).
    An old-style checkpoint commit: the non-checkpoint commits in target..HEAD."""
    full = _git("rev-parse", "--verify", "-q", f"{target_sha}^{{commit}}").stdout.strip()
    snapshot = bool(full) and _is_snapshot(full)
    res = _git("log", f"{target_sha}^..HEAD" if snapshot else f"{target_sha}..HEAD", "--pretty=%h%x09%s")
    bad = []
    for line in res.stdout.splitlines():
        sha, _, subject = line.partition("\t")
        if snapshot or not subject.startswith(CHECKPOINT_PREFIX):
            bad.append(f"{sha} {subject}")
    return bad


def current_head() -> str | None:
    """Full sha of HEAD, or None outside a repository / in one without commits."""
    try:
        res = _git("rev-parse", "--verify", "-q", "HEAD")
    except Exception:
        return None
    sha = res.stdout.strip()
    return sha if res.returncode == 0 and sha else None


def rewind_to(target_sha: str) -> str:
    """Return the files to a checkpoint: a snapshot is restored on top of its base
    commit, an old-style checkpoint commit is hard-reset to. Consent is the caller's
    responsibility; this function enforces only the mechanical safety rules."""
    return _rewind(target_sha, require_checkpoint=True)


def rewind_to_base(base_sha: str) -> str:
    """Return the tree to ``base_sha`` — the commit HEAD pointed at when a turn began
    with a clean tree, so that commit IS the state before the turn.

    Same rules as ``rewind_to`` (no real commit may be discarded, nothing is lost
    silently: the turn's edits go to ``git stash``), minus the requirement that the
    target is an Argent checkpoint."""
    return _rewind(base_sha, require_checkpoint=False)


def _rewind(target_sha: str, require_checkpoint: bool) -> str:
    if not is_git_repo():
        raise CheckpointError("Not a git repository.")

    target_sha = (target_sha or "").strip()
    if not target_sha or target_sha.startswith("-") or not re.match(r"^[0-9a-zA-Z_\.\^~]+$", target_sha):
        raise CheckpointError(f"Invalid commit reference '{target_sha}'.")

    # Guard against active merge / rebase / cherry-pick (worktree-safe)
    for head_name, desc in [
        ("MERGE_HEAD", "слияние (git merge)"),
        ("rebase-merge", "rebase"),
        ("rebase-apply", "rebase"),
        ("CHERRY_PICK_HEAD", "cherry-pick"),
        ("REVERT_HEAD", "revert"),
    ]:
        git_path_res = _git("rev-parse", "--git-path", head_name)
        if git_path_res.returncode == 0:
            p = Path(git_path_res.stdout.strip())
            if p.exists():
                raise CheckpointError(
                    f"Обнаружен активный незавершённый процесс: {desc} ({head_name} существует). "
                    f"Завершите или отмените его перед откатом."
                )

    subject = _git("log", "-1", "--pretty=%s", target_sha)
    if subject.returncode != 0:
        raise CheckpointError(f"Unknown commit '{target_sha}'.")
    if require_checkpoint and not subject.stdout.strip().startswith(CHECKPOINT_PREFIX):
        raise CheckpointError(
            f"Commit {target_sha} is not an Argent Checkpoint — rewind refused for safety.")
    full = _git("rev-parse", "--verify", "-q", f"{target_sha}^{{commit}}").stdout.strip()
    if require_checkpoint and _is_snapshot(full):
        return _restore_snapshot(target_sha, full, subject.stdout.strip())
    if _git("merge-base", "--is-ancestor", target_sha, "HEAD").returncode != 0:
        raise CheckpointError(
            f"Checkpoint {target_sha} is not on the current branch history.")

    bad = blocking_commits(target_sha)
    if bad:
        listing = "\n".join(f"  - {b}" for b in bad)
        raise CheckpointError(
            "Rewind would discard real (non-checkpoint) commits:\n"
            f"{listing}\nAborted for safety.")

    stashed = _stash_if_dirty(target_sha)

    discarded = _git("rev-list", "--count", f"{target_sha}..HEAD").stdout.strip()
    res = _git("reset", "--hard", target_sha)
    if res.returncode != 0:
        reset_err = res.stderr.strip() or res.stdout.strip()
        raise CheckpointError(f"git reset failed: {reset_err}.{_unstash_note(stashed, target_sha)}")

    label = subject.stdout.strip()[len(CHECKPOINT_PREFIX):].strip()
    if subject.stdout.strip().startswith(CHECKPOINT_PREFIX):
        msg = f"Rewound to checkpoint {target_sha} — '{label}'."
    else:
        msg = f"Rewound to {target_sha[:8]} — '{subject.stdout.strip()[:60]}'."
    if discarded and discarded != "0":
        msg += f" Discarded {discarded} newer checkpoint(s) (recoverable via `git reflog`)."
    if stashed:
        msg += " Uncommitted changes were stashed (`git stash list`)."
    return msg


def _unstash_note(stashed: bool, target_sha: str) -> str:
    """After a failed rewind: put the user's stashed changes back and say how it went."""
    if not stashed:
        return ""
    pop_res = _git("stash", "pop")
    if pop_res.returncode == 0:
        return "\nНезакоммиченные изменения автоматически восстановлены из stash."
    return (f"\nАвтоматический возврат изменений из stash не удался: {pop_res.stderr.strip()}. "
            f"Изменения сохранены в `git stash` ('Argent Auto-Save before Rewind to {target_sha}'). "
            f"Восстановите их вручную: `git stash pop`.")


def _stash_if_dirty(target_sha: str) -> bool:
    stash_msg = f"Argent Auto-Save before Rewind to {target_sha}"
    if not _git("status", "--porcelain").stdout.strip():
        return False
    stash_res = _git("stash", "push", "-u", "-m", stash_msg)
    if stash_res.returncode != 0:
        err = stash_res.stderr.strip() or stash_res.stdout.strip()
        raise CheckpointError(
            f"Не удалось безопасно сохранить незакоммиченные изменения в git stash: {err}.\n"
            f"Откат прерван для защиты ваших данных."
        )
    return True


def _restore_snapshot(target_sha: str, full: str, subject: str) -> str:
    """Put the snapshot's files back as uncommitted changes on top of its base commit.

    The base must still be HEAD — a commit made since would be a state the snapshot
    knows nothing about, so the rewind is refused and the commits are named.  The
    current changes go to `git stash` first; then `read-tree -u --reset` writes the
    snapshot's tree into the working tree (deleting what the snapshot did not have),
    and `reset` puts the index back to HEAD, so every restored change is unstaged,
    as it was when the snapshot was taken.  Newer snapshots stay listed: going back
    is not a reason to forget them."""
    base = _git("rev-parse", "--verify", "-q", f"{full}^").stdout.strip()
    if base != current_head():
        bad = blocking_commits(full)
        if bad and _git("merge-base", "--is-ancestor", base, "HEAD").returncode == 0:
            listing = "\n".join(f"  - {b}" for b in bad)
            raise CheckpointError(
                "Rewind would discard real (non-checkpoint) commits:\n"
                f"{listing}\nAborted for safety.")
        raise CheckpointError(
            f"Checkpoint {target_sha} was made on another commit or branch — rewind refused for safety.")
    stashed = _stash_if_dirty(target_sha)
    res = _git("read-tree", "-u", "--reset", full)
    if res.returncode == 0:
        res = _git("reset", "-q")
    if res.returncode != 0:
        err = res.stderr.strip() or res.stdout.strip()
        _git("reset", "-q", "--hard", "HEAD")   # undo a half-written restore: the base commit, then the user's changes
        raise CheckpointError(f"git read-tree failed: {err}.{_unstash_note(stashed, target_sha)}")
    label = subject[len(CHECKPOINT_PREFIX):].strip()
    msg = f"Rewound to checkpoint {target_sha} — '{label}'."
    if stashed:
        msg += " Uncommitted changes were stashed (`git stash list`)."
    return msg
