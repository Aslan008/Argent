"""/review: an independent read of the changes, with findings you can act on.

`/critic` judges a plan; `/commit` can run the critic over a staged diff. Neither
answers the question people actually ask before pushing: "what did I (or the agent)
break?" — as a list of places, each with a file, a line, how bad it is and a concrete
way it goes wrong, so it can be fixed or dismissed in a minute.

The review is done by a sub-agent that did not write the code, in a clean context,
with read-only tools: the diff shows what changed, but whether a changed function
still agrees with its callers is only visible in the rest of the file. It may use the
critic's model when one is configured (a different model tends to notice what the
author's model does not).

The pieces are separate on purpose: collecting the diff (git, budget, noise), building
the task, and reading the answer back are plain functions, tested without a model.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from logger import get_logger

log = get_logger("review")

#: What the reviewer may use: look at code, never change it.
READONLY_TOOLS = [
    "read_file", "grep_search", "search_files", "list_directory", "get_file_outline",
    "read_git_diff", "semantic_search",
]

#: The diff is sent whole up to this size; beyond it the smaller files go in full and the
#: biggest are named, with the tools to read them.
DIFF_BUDGET_CHARS = 60000
MAX_UNTRACKED_FILES = 20
MAX_UNTRACKED_CHARS = 16000
SEVERITIES = ("critical", "high", "medium", "low")


class ReviewError(RuntimeError):
    """Nothing to review, or git cannot say what changed."""


@dataclass
class DiffBundle:
    text: str
    description: str
    files: List[str] = field(default_factory=list)
    omitted: List[str] = field(default_factory=list)


@dataclass
class Finding:
    file: str
    line: int
    severity: str
    summary: str
    scenario: str = ""

    def where(self) -> str:
        return f"{self.file}:{self.line}" if self.line else self.file


# -- the diff ---------------------------------------------------------------------------

def _git(args: List[str], cwd=None) -> str:
    from src.agent.shell import run_text
    done = run_text(["git", *args], capture_output=True, cwd=str(cwd) if cwd else None)
    if done.returncode != 0:
        raise ReviewError((done.stderr or done.stdout or f"git {args[0]} failed").strip())
    return done.stdout


def _untracked_as_diff(cwd: Path, under: str = "") -> str:
    names = [n for n in _git(["ls-files", "--others", "--exclude-standard"], cwd).splitlines() if n.strip()]
    if under:
        prefix = under.replace("\\", "/").strip("/")
        names = [n for n in names if n == prefix or n.startswith(prefix + "/")]
    parts = []
    for name in names[:MAX_UNTRACKED_FILES]:
        path = cwd / name
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:4096]:
            continue
        text = data[:MAX_UNTRACKED_CHARS].decode("utf-8", errors="replace")
        lines = text.splitlines()
        body = "\n".join("+" + line for line in lines)
        cut = "\n+[... file continues ...]" if len(data) > MAX_UNTRACKED_CHARS else ""
        parts.append(f"diff --git a/{name} b/{name}\nnew file mode 100644\n--- /dev/null\n+++ b/{name}\n"
                     f"@@ -0,0 +1,{len(lines)} @@\n{body}{cut}\n")
    if len(names) > MAX_UNTRACKED_FILES:
        parts.append(f"[{len(names) - MAX_UNTRACKED_FILES} more new files not shown]\n")
    return "".join(parts)


def _split_files(diff: str) -> List[tuple]:
    """[(path, chunk)] for a unified diff."""
    chunks, current, name = [], [], None
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                chunks.append((name, "".join(current)))
            current = [line]
            m = re.match(r"diff --git a/(.+?) b/(.+)$", line.strip())
            name = m.group(2) if m else line.strip()
        else:
            current.append(line)
    if current:
        chunks.append((name, "".join(current)))
    return chunks


def _fit(diff: str, budget: int) -> tuple:
    """(text within budget, [files left out]): small files whole, the biggest named instead."""
    from src.agent.critic import sanitize_and_truncate_diff
    diff = sanitize_and_truncate_diff(diff, max_chars=10 ** 9)           # drops lockfiles / minified output
    chunks = _split_files(diff)
    if len(diff) <= budget:
        return diff, []
    kept, used = set(), 0
    for index, (name, chunk) in sorted(enumerate(chunks), key=lambda item: len(item[1][1])):
        if used + len(chunk) <= budget:
            kept.add(index)
            used += len(chunk)
    text = "".join(chunk for i, (name, chunk) in enumerate(chunks) if i in kept)
    omitted = [name for i, (name, chunk) in enumerate(chunks) if i not in kept and name]
    return text, omitted


def collect_diff(target: str = "", cwd=None, budget: int = DIFF_BUDGET_CHARS) -> DiffBundle:
    """The changes to review.

    target: ""            → everything uncommitted (staged, unstaged, new files)
            "staged"      → what is staged for commit
            "A..B" "A...B"→ a range;  "<rev>" → that one commit's changes
            "<path>"      → uncommitted changes under an existing file or folder
    """
    cwd = Path(cwd) if cwd else Path.cwd()
    try:
        _git(["rev-parse", "--is-inside-work-tree"], cwd)
    except ReviewError:
        raise ReviewError("This folder is not a git repository, so there is no diff to review.")

    target = (target or "").strip()
    if not target:
        description = "uncommitted changes (staged, unstaged and new files)"
        try:
            diff = _git(["diff", "HEAD"], cwd)
        except ReviewError:                                    # no commits yet
            diff = _git(["diff", "--cached"], cwd) + _git(["diff"], cwd)
        diff += _untracked_as_diff(cwd)
    elif target.lower() == "staged":
        description, diff = "staged changes", _git(["diff", "--cached"], cwd)
    elif ".." in target:
        description, diff = f"changes {target}", _git(["diff", target], cwd)
    elif (cwd / target).exists():
        description = f"uncommitted changes in {target}"
        try:
            diff = _git(["diff", "HEAD", "--", target], cwd)
        except ReviewError:
            diff = ""
        diff += _untracked_as_diff(cwd, under=target)
    else:
        description, diff = f"commit {target}", _git(["show", "--format=%h %s%n%n%b", target], cwd)

    if not diff.strip():
        raise ReviewError(f"Nothing to review: no {description.split(' (')[0]}.")
    text, omitted = _fit(diff, budget)
    files = [name for name, _ in _split_files(diff) if name]       # `git show` prefixes the message
    return DiffBundle(text=text, description=description, files=files, omitted=omitted)


# -- the task and the answer ------------------------------------------------------------

_CONTRACT = """\
Review the changes below as a careful maintainer who did NOT write them.

Find real problems: bugs, broken edge cases, missing error handling that matters, security holes,
races, resource leaks, API/contract breaks (a changed function that a caller still uses the old way),
tests that no longer test what they claim. Do NOT report style, naming, formatting or taste, and do
not report things that are fine.

For every candidate, VERIFY it with the tools before reporting it: open the file around the change,
find the callers (grep_search), check that the thing you think is missing really is missing. A finding
you did not verify is a guess; leave it out.

Answer with ONE JSON array and nothing else (no prose before or after). Each element:
  {"file": "path/relative/to/repo", "line": <line number in the NEW file>, "severity": "critical|high|medium|low",
   "summary": "one sentence: what is wrong", "scenario": "the concrete input or state that makes it fail, and what happens"}
At most {max_findings} findings, worst first. If you found nothing worth reporting, answer [] — an honest
empty list is a good review; invented findings are a bad one.
Write "summary" and "scenario" in {language}."""


def build_task(bundle: DiffBundle, focus: str = "", max_findings: int = 12, language: str = "Russian") -> str:
    parts = [_CONTRACT.replace("{max_findings}", str(max_findings)).replace("{language}", language),
             f"\nWhat is under review: {bundle.description}."]
    if focus.strip():
        parts.append(f"The person asked you to pay particular attention to: {focus.strip()}")
    if bundle.omitted:
        parts.append("These files changed too but their diffs are too large to include; read them yourself "
                     "with read_git_diff / read_file if they matter: " + ", ".join(bundle.omitted))
    parts.append("--- DIFF ---\n" + bundle.text + "\n--- END OF DIFF ---")
    return "\n".join(parts)


def _json_candidates(text: str):
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)
    yield from fenced
    start = (text or "").find("[")
    end = (text or "").rfind("]")
    if 0 <= start < end:
        yield text[start:end + 1]


def parse_findings(text: str) -> Optional[List[Finding]]:
    """Findings from the reviewer's answer; None when it contains no readable list at all."""
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            data = data.get("findings", [])
        if not isinstance(data, list):
            continue
        found = []
        for item in data:
            if not isinstance(item, dict) or not str(item.get("summary", "")).strip():
                continue
            try:
                line = int(item.get("line") or 0)
            except (TypeError, ValueError):
                line = 0
            severity = str(item.get("severity", "medium")).strip().lower()
            found.append(Finding(
                file=str(item.get("file", "")).strip().replace("\\", "/"), line=max(0, line),
                severity=severity if severity in SEVERITIES else "medium",
                summary=" ".join(str(item["summary"]).split()),
                scenario=" ".join(str(item.get("scenario", "")).split())))
        found.sort(key=lambda f: SEVERITIES.index(f.severity))
        return found
    return None


def drop_unknown_files(findings: List[Finding], files: List[str]) -> tuple:
    """(findings about files that exist in the review, how many were about nothing real).

    A reviewer that cites a file that is not in the repository invented the finding."""
    known = {f.replace("\\", "/") for f in files}
    kept = [f for f in findings if not f.file or f.file in known or Path(f.file).exists()]
    return kept, len(findings) - len(kept)


# -- running it -------------------------------------------------------------------------

@dataclass
class ReviewResult:
    findings: List[Finding]
    bundle: DiffBundle
    raw: str
    readable: bool = True
    dropped: int = 0


def default_runner(task: str) -> str:
    """Run the reviewer sub-agent (the critic's model when one is configured)."""
    from agent import ArgentSubAgent
    from config import get_critic_model, get_critic_provider

    model, provider = get_critic_model() or None, get_critic_provider() or None
    try:
        return ArgentSubAgent("Reviewer", task, tools_override=list(READONLY_TOOLS),
                              model_override=model, provider_override=provider).execute()
    except Exception as e:
        if model or provider:
            log.warning("review on %s/%s failed (%s); using the current model", provider, model, e)
            return ArgentSubAgent("Reviewer", task, tools_override=list(READONLY_TOOLS)).execute()
        raise


def run_review(target: str = "", focus: str = "", cwd=None, runner: Optional[Callable[[str], str]] = None,
               max_findings: int = 12) -> ReviewResult:
    bundle = collect_diff(target, cwd)
    raw = (runner or default_runner)(build_task(bundle, focus, max_findings))
    findings = parse_findings(raw)
    if findings is None:
        return ReviewResult([], bundle, raw, readable=False)
    findings, dropped = drop_unknown_files(findings, bundle.files)
    return ReviewResult(findings, bundle, raw, True, dropped)


_MARK = {"critical": "‼", "high": "●", "medium": "◐", "low": "○"}


def render(result: ReviewResult) -> str:
    """Plain Markdown: what the terminal prints and what is saved."""
    lines = [f"# Ревью: {result.bundle.description}", ""]
    if not result.readable:
        lines += ["Ревьюер не вернул список в ожидаемом формате. Его ответ:", "", result.raw.strip()]
        return "\n".join(lines)
    if not result.findings:
        lines.append("Существенных проблем не найдено.")
    for i, f in enumerate(result.findings, 1):
        lines.append(f"{i}. {_MARK[f.severity]} **{f.severity}** `{f.where()}` — {f.summary}")
        if f.scenario:
            lines.append(f"   Сценарий: {f.scenario}")
    if result.dropped:
        lines.append(f"\n(Отброшено находок про несуществующие файлы: {result.dropped}.)")
    if result.bundle.omitted:
        lines.append("\nНе вошли в ревью целиком (слишком большие диффы): " + ", ".join(result.bundle.omitted))
    return "\n".join(lines)


def save(result: ReviewResult, root=None) -> str:
    path = Path(root or ".") / ".argent" / "artifacts" / f"review-{datetime.now():%Y%m%d-%H%M%S}.md"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(result) + "\n", encoding="utf-8")
        return path.as_posix()
    except OSError as e:
        log.warning("could not save the review: %s", e)
        return ""


def split_target(arg: str, cwd=None) -> tuple:
    """("what to review", "what to look at") from the text after /review.

    The first word is a target when it is `staged`, a range, an existing path, or a git
    revision; otherwise the whole text is the focus ("/review the SQL escaping")."""
    from src.agent.shell import run_text
    cwd = Path(cwd) if cwd else Path.cwd()
    words = (arg or "").strip().split(None, 1)
    if not words:
        return "", ""
    first, rest = words[0], (words[1] if len(words) > 1 else "")
    if first.lower() == "staged" or ".." in first:
        return first, rest
    try:
        if (cwd / first).exists():
            return first, rest
    except OSError:
        pass
    probe = run_text(["git", "rev-parse", "--verify", "-q", f"{first}^{{commit}}"], capture_output=True, cwd=str(cwd))
    if probe.returncode == 0:
        return first, rest
    return "", arg.strip()


def fix_request(result: ReviewResult) -> str:
    """The message that hands the findings to the working agent."""
    lines = ["An independent review of the changes found the problems below. Fix the ones that are real "
             "(check each against the code first; say so if one is a false alarm), then re-run the relevant tests.", ""]
    for i, f in enumerate(result.findings, 1):
        lines.append(f"{i}. [{f.severity}] {f.where()} — {f.summary}" + (f" (fails when: {f.scenario})" if f.scenario else ""))
    return "\n".join(lines)
