"""The `git` node: look at, record and share the project's changes without a shell and without a model.

A person who builds a room should not have to know that "what changed?" is `git status --porcelain -b`
and that its second column is the work tree. Each mode here answers one plain question and puts the
answer into fields a later step (or an arrow's condition) can read:

    status    branch, clean, changed, files, staged, untracked, ahead, behind
    diff      files, count, added, removed, changed
    log       commits, count, last, last_hash
    add       files, count
    commit    committed, commit, branch
    branch    branch, branches
    checkout  branch
    stash     stashed, count
    pull      updated, branch
    push      pushed, branch

The fields are stored as `<id>.<name>` (`checks.clean`); `save_as` keeps the text report under a name.

Git is started with an argument list, never through a shell, so a commit message or a branch name can
not be taken for a command, and the working directory is the project (or `args.repo`). There is no force
option: a room never rewrites published history or discards work, `pull` only fast-forwards, and git is
told not to ask for a password (an unattended run must fail, not hang).
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from src.rooms.models import GIT_MODES

#: Seconds a git call may take: local ones are quick, the ones that talk to a server get longer.
LOCAL_TIMEOUT = 60
REMOTE_TIMEOUT = 180
#: How much of a report goes into the node's output (the rest is counted, not lost silently).
OUTPUT_CHARS = 6000
FULL_DIFF_CHARS = 12000
LOG_DEFAULT, LOG_MAX = 10, 100


class GitResult(str):
    """The report text of a git node, with the fields it produced in `.values`."""

    values: dict

    def __new__(cls, text: str, values: Optional[dict] = None):
        obj = super().__new__(cls, text)
        obj.values = dict(values or {})
        return obj


def _fail(text: str, **values) -> GitResult:
    return GitResult("Error: " + text, {"ok": False, **values})


def _env() -> dict:
    env = dict(os.environ)
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_EDITOR": "true", "GIT_PAGER": "cat", "LC_ALL": "C"})
    env.pop("GIT_ASKPASS", None)
    return env


def _git(repo: Path, *argv: str, timeout: int = LOCAL_TIMEOUT) -> tuple:
    """(exit code, stdout, stderr) of `git <argv>` in `repo`; a missing git or a timeout is a code of -1."""
    try:
        done = subprocess.run(["git", "-c", "core.quotepath=off", "--no-pager", *argv], cwd=str(repo), env=_env(),
                              capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        return -1, "", "git is not installed"
    except subprocess.TimeoutExpired:
        return -1, "", f"git {argv[0] if argv else ''} did not finish in {timeout} s"
    except OSError as e:
        return -1, "", str(e)
    return (done.returncode, done.stdout.decode("utf-8", "replace"), done.stderr.decode("utf-8", "replace"))


def _clip(text: str, limit: int = OUTPUT_CHARS) -> str:
    text = text.rstrip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more characters]"


def _why(stderr: str, stdout: str = "") -> str:
    text = (stderr or stdout or "").strip()
    return " ".join(text.split())[:400] or "no message from git"


def _safe_name(value: str) -> bool:
    """A branch, remote or ref that cannot be read as an option."""
    value = str(value or "").strip()
    return bool(value) and not value.startswith("-") and "\n" not in value and "\x00" not in value


def _as_files(value) -> list:
    if value in (None, "", []):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).replace(",", "\n").splitlines() if part.strip()]


def _porcelain(text: str) -> dict:
    """The fields of `git status --porcelain=v1 --branch`."""
    branch, ahead, behind, staged, changed, untracked = "", 0, 0, [], [], []
    for line in text.splitlines():
        if line.startswith("## "):
            head = line[3:]
            if head.startswith("No commits yet on "):
                branch = head[len("No commits yet on "):].strip()
                continue
            ahead_behind = ""
            if " [" in head and head.endswith("]"):
                head, ahead_behind = head[:-1].split(" [", 1)
            branch = head.split("...", 1)[0].strip()
            for part in ahead_behind.split(","):
                part = part.strip()
                if part.startswith("ahead "):
                    ahead = int(part[6:] or 0)
                elif part.startswith("behind "):
                    behind = int(part[7:] or 0)
            continue
        if len(line) < 4:
            continue
        x, y, path = line[0], line[1], line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if x == "?" and y == "?":
            untracked.append(path)
            continue
        if x not in (" ", "?"):
            staged.append(path)
        changed.append(path)
    files = list(dict.fromkeys(changed + untracked))
    return {"branch": branch, "ahead": ahead, "behind": behind, "staged": staged, "untracked": untracked,
            "files": files, "changed": len(files), "clean": not files}


def _status(repo: Path):
    code, out, err = _git(repo, "status", "--porcelain=v1", "--branch")
    if code != 0:
        return None, _why(err, out)
    return _porcelain(out), ""


def _status_text(s: dict) -> str:
    line = f"On branch {s['branch'] or '(detached)'}"
    if s["ahead"] or s["behind"]:
        line += f" (ahead {s['ahead']}, behind {s['behind']})"
    if s["clean"]:
        return line + ": nothing to commit, the working tree is clean."
    parts = []
    if s["staged"]:
        parts.append(f"{len(s['staged'])} staged")
    modified = [f for f in s["files"] if f not in s["untracked"]]
    if modified:
        parts.append(f"{len(modified)} changed")
    if s["untracked"]:
        parts.append(f"{len(s['untracked'])} new")
    return line + f": {', '.join(parts)}.\n" + "\n".join("  " + f for f in s["files"][:40]) + (
        f"\n  … and {len(s['files']) - 40} more" if len(s["files"]) > 40 else "")


def run(node, cwd: Optional[str] = None) -> GitResult:
    """Do what a git node asks and report it. Never raises for an expected failure: the text starts
    with `Error:` and the node's `ok` is then False."""
    mode = node.mode or "status"
    args = node.args or {}
    if mode not in GIT_MODES:
        return _fail(f"unknown git mode '{mode}' (use {list(GIT_MODES)})")
    repo = Path(str(args.get("repo") or cwd or os.getcwd())).expanduser()
    if not repo.is_dir():
        return _fail(f"the folder '{repo}' does not exist")
    code, out, err = _git(repo, "rev-parse", "--is-inside-work-tree")
    if code != 0 or out.strip() != "true":
        return _fail(f"'{repo}' is not inside a git repository", branch="", clean=False)
    return _MODES[mode](node, repo, args)


# ── the modes ─────────────────────────────────────────────────────────────────────────────────────
def _do_status(node, repo, args) -> GitResult:
    s, why = _status(repo)
    if s is None:
        return _fail(f"git status failed: {why}")
    return GitResult(_status_text(s), s)


def _do_diff(node, repo, args) -> GitResult:
    argv = ["diff"]
    if args.get("staged"):
        argv.append("--cached")
    against = str(args.get("against") or "").strip()
    if against:
        if not _safe_name(against):
            return _fail(f"'{against}' is not a branch, tag or commit")
        argv.append(against)
    paths = _as_files(args.get("files"))
    tail = ["--", *paths] if paths else []
    code, numstat, err = _git(repo, *argv, "--numstat", *tail)
    if code != 0:
        return _fail(f"git diff failed: {_why(err)}")
    files, added, removed = [], 0, 0
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            files.append(parts[2])
            added += int(parts[0]) if parts[0].isdigit() else 0
            removed += int(parts[1]) if parts[1].isdigit() else 0
    values = {"files": files, "count": len(files), "added": added, "removed": removed, "changed": bool(files)}
    if not files:
        return GitResult("No differences.", values)
    if args.get("full"):
        code, text, err = _git(repo, *argv, *tail)
        return GitResult(_clip(text, FULL_DIFF_CHARS), values)
    code, text, err = _git(repo, *argv, "--stat", *tail)
    return GitResult(_clip(text), values)


def _do_log(node, repo, args) -> GitResult:
    try:
        count = max(1, min(int(args.get("count") or LOG_DEFAULT), LOG_MAX))
    except (TypeError, ValueError):
        count = LOG_DEFAULT
    argv = ["log", f"-{count}", "--date=short", "--format=%h%x09%ad%x09%an%x09%s"]
    since = str(args.get("since") or "").strip()
    if since:
        if not _safe_name(since):
            return _fail(f"'{since}' is not a branch, tag or commit")
        argv.insert(1, f"{since}..HEAD")
    code, out, err = _git(repo, *argv)
    if code != 0:
        text = _why(err, out)
        if "does not have any commits" in text or "bad default revision" in text:
            return GitResult("No commits yet.", {"commits": [], "count": 0, "last": "", "last_hash": ""})
        return _fail(f"git log failed: {text}")
    rows = [line.split("\t") for line in out.splitlines() if line.strip()]
    commits = [f"{r[0]} {r[3]}" for r in rows if len(r) == 4]
    lines = [f"{r[0]}  {r[1]}  {r[2]}: {r[3]}" for r in rows if len(r) == 4]
    values = {"commits": commits, "count": len(commits), "last": rows[0][3] if rows and len(rows[0]) == 4 else "",
              "last_hash": rows[0][0] if rows else ""}
    return GitResult("\n".join(lines) or "No commits.", values)


def _do_add(node, repo, args) -> GitResult:
    files = _as_files(args.get("files")) or ["."]
    code, out, err = _git(repo, "add", "--", *files)
    if code != 0:
        return _fail(f"git add failed: {_why(err, out)}")
    code, out, err = _git(repo, "diff", "--cached", "--name-only")
    staged = [ln for ln in out.splitlines() if ln.strip()]
    return GitResult(f"Staged {len(staged)} file(s)." + ("\n" + "\n".join("  " + f for f in staged[:40]) if staged else ""),
                     {"files": staged, "count": len(staged)})


def _do_commit(node, repo, args) -> GitResult:
    message = (node.message or "").strip()
    only_staged = bool(args.get("only_staged"))
    if not only_staged:
        files = _as_files(args.get("files")) or ["-A"]
        code, out, err = _git(repo, "add", *(["--", *files] if files != ["-A"] else ["-A"]))
        if code != 0:
            return _fail(f"git add failed: {_why(err, out)}", committed=False)
    code, out, err = _git(repo, "diff", "--cached", "--quiet")
    if code == 0:
        return GitResult("Nothing to commit: no changes are staged.", {"committed": False, "commit": ""})
    if code != 1:
        return _fail(f"git diff failed: {_why(err, out)}", committed=False)
    code, out, err = _git(repo, "commit", "-m", message)
    if code != 0:
        return _fail(f"git commit failed: {_why(err, out)}", committed=False)
    code, h, _ = _git(repo, "rev-parse", "--short", "HEAD")
    code, b, _ = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    first = message.splitlines()[0] if message else ""
    return GitResult(f"Committed {h.strip()}: {first}", {"committed": True, "commit": h.strip(), "branch": b.strip()})


def _branches(repo: Path) -> list:
    code, out, _ = _git(repo, "branch", "--format=%(refname:short)")
    return [ln.strip() for ln in out.splitlines() if ln.strip()] if code == 0 else []


def _current(repo: Path) -> str:
    code, out, _ = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    return out.strip() if code == 0 else ""


def _do_branch(node, repo, args) -> GitResult:
    name = str(args.get("name") or args.get("branch") or "").strip()
    if not name:
        current, listed = _current(repo), _branches(repo)
        text = f"On branch {current}.\nBranches: " + (", ".join(listed) or "none")
        return GitResult(text, {"branch": current, "branches": listed})
    if not _safe_name(name):
        return _fail(f"'{name}' is not a valid branch name")
    exists = name in _branches(repo)
    argv = ["checkout", name] if exists else ["checkout", "-b", name]
    start = str(args.get("from") or "").strip()
    if start and not exists:
        if not _safe_name(start):
            return _fail(f"'{start}' is not a branch, tag or commit")
        argv.append(start)
    code, out, err = _git(repo, *argv)
    if code != 0:
        return _fail(f"git checkout failed: {_why(err, out)}")
    return GitResult(f"{'Switched to' if exists else 'Created and switched to'} branch {name}.",
                     {"branch": name, "branches": _branches(repo)})


def _do_checkout(node, repo, args) -> GitResult:
    name = str(args.get("branch") or "").strip()
    if not _safe_name(name):
        return _fail(f"'{name}' is not a valid branch name")
    argv = ["checkout", "-b", name] if args.get("create") else ["checkout", name]
    code, out, err = _git(repo, *argv)
    if code != 0:
        return _fail(f"git checkout failed: {_why(err, out)}")
    return GitResult(f"Switched to branch {name}.", {"branch": name})


def _do_stash(node, repo, args) -> GitResult:
    action = str(args.get("action") or "save").strip().lower()
    if action == "list":
        code, out, err = _git(repo, "stash", "list")
        rows = [ln for ln in out.splitlines() if ln.strip()]
        return GitResult("\n".join(rows) or "No stashes.", {"count": len(rows), "stashed": False})
    if action == "pop":
        code, out, err = _git(repo, "stash", "pop")
        if code != 0:
            return _fail(f"git stash pop failed: {_why(err, out)}")
        return GitResult("Restored the latest stash.", {"stashed": False})
    if action != "save":
        return _fail(f"unknown stash action '{action}' (use save, pop or list)")
    argv = ["stash", "push", "--include-untracked"]
    if (node.message or "").strip():
        argv += ["-m", node.message.strip()]
    code, out, err = _git(repo, *argv)
    if code != 0:
        return _fail(f"git stash failed: {_why(err, out)}")
    nothing = "No local changes" in out
    return GitResult("Nothing to stash." if nothing else "Stashed the local changes.", {"stashed": not nothing})


def _remote_args(args, repo) -> tuple:
    remote = str(args.get("remote") or "origin").strip()
    branch = str(args.get("branch") or "").strip()
    if not _safe_name(remote) or (branch and not _safe_name(branch)):
        return None, None, "the remote or branch name is not valid"
    return remote, branch or _current(repo), ""


def _do_pull(node, repo, args) -> GitResult:
    remote, branch, bad = _remote_args(args, repo)
    if bad:
        return _fail(bad)
    code, out, err = _git(repo, "pull", "--ff-only", remote, branch, timeout=REMOTE_TIMEOUT)
    if code != 0:
        return _fail(f"git pull failed: {_why(err, out)}", updated=False)
    return GitResult(_clip(out) or "Done.", {"updated": "Already up to date" not in out, "branch": branch})


def _do_push(node, repo, args) -> GitResult:
    remote, branch, bad = _remote_args(args, repo)
    if bad:
        return _fail(bad)
    code, out, err = _git(repo, "push", "--set-upstream", remote, branch, timeout=REMOTE_TIMEOUT)
    if code != 0:
        return _fail(f"git push failed: {_why(err, out)}", pushed=False)
    return GitResult(f"Pushed {branch} to {remote}.", {"pushed": True, "branch": branch})


_MODES = {"status": _do_status, "diff": _do_diff, "log": _do_log, "add": _do_add, "commit": _do_commit,
          "branch": _do_branch, "checkout": _do_checkout, "stash": _do_stash, "pull": _do_pull, "push": _do_push}
