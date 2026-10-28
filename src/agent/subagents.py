"""Sub-agents that run beside the supervisor: in the background, several at once, each optionally in a private git worktree.

`run_subagent` used to be a function call: the supervisor stopped, the sub-agent ran in
the same process on the same files, and the supervisor resumed with its answer. Fine
for one question; useless for "have three agents look at three modules" or "try this
refactor without touching my tree".

Here a sub-agent is a *process* (see subagent_worker) tracked by `SubagentManager`:

* **Background.** `spawn` returns at once. The supervisor goes on, and finds out when it
  ends the way it finds out about a build: the condition `subagent_finished("N")` for
  `wait_heartbeat`, or — in the TUI — an automatic wake-up. `result(wait=True)` blocks.
* **Parallel.** Up to MAX_RUNNING at a time; a model that wants ten is told to wait.
* **Isolated.** With isolation="worktree" the sub-agent edits a copy of the tree (see
  worktrees). When it ends, its work is a patch on disk; the supervisor — or you with
  /agents — applies it or throws it away. No patch, no leftover copy.
* **Bounded.** A sub-agent has only the tools of its role (never the ones that need a
  person, start more agents, or change schedules) and only the authority the person had
  already given: their allow rules and session grants, their deny rules. Everything else
  it asks for is refused and listed in its report.
"""

import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from argent.logger import get_logger
from src.agent import governor, lanes, worktrees
from src.agent.subagent_worker import RESULT_MARKER

log = get_logger("subagents")

MAX_RUNNING = 4
MAX_KEPT = 50
#: How much of a child's console output is kept for the "it died" case.
LOG_TAIL_CHARS = 6000
DEFAULT_WAIT_SECONDS = 1800

#: A sub-agent never gets these, whatever the supervisor asks for: they need a person, spawn more
#: agents, or rewrite what runs later.
FORBIDDEN_TOOLS = frozenset({
    "ask_user_questions", "request_user_approval", "exit_plan_mode",
    "wait_heartbeat", "manage_automations", "run_subagent", "subagents", "ultracode", "rooms", "memory", "desktop_inspect", "desktop_act",
})

_READ = ["read_file", "grep_search", "search_files", "list_directory", "get_file_outline", "read_git_diff",
         "semantic_search", "analyze_project"]
_WEB = ["search_web", "read_webpage"]
_EDIT = ["write_file", "replace_in_file", "create_directory", "move_file", "copy_file", "delete_file"]

#: Tools by role. A role the supervisor invents gets the read-only set.
ROLE_TOOLS = {
    "researcher": _READ + _WEB,
    "reviewer": _READ,
    "critic": [],
    "docwriter": _READ + ["write_file", "replace_in_file", "create_directory"],
    "coder": _READ + _EDIT + ["run_command", "run_deep_linter", "todo_write"],
    "browserworker": ["browser_open", "browser_action", "browser_inspect", "browser_close",
                      "read_webpage", "view_image",
                      "leeth_map", "leeth_scan_page", "leeth_click_many", "leeth_fill_form"],
}
READONLY_ROLES = frozenset({"researcher", "reviewer", "critic"})


def tools_for(role: str, requested: Optional[List[str]] = None) -> List[str]:
    base = list(requested) if requested is not None else list(ROLE_TOOLS.get((role or "").strip().lower(), _READ + _WEB))
    return [t for t in dict.fromkeys(base) if t not in FORBIDDEN_TOOLS]


def inherited_authority() -> dict:
    """What the supervisor's person has already allowed, as a child can use it."""
    import argent.approval as approval
    from src.agent.permissions import PERMISSIONS

    allowed = []
    for rule in PERMISSIONS.rules("allow"):
        if rule.covers_tool("run_command"):
            allowed.append(rule.pattern if rule.pattern is not None else "*")
    for key in sorted(approval.get_session_grants()):
        allowed += [key, f"{key} *"]
    return {
        "allowed_commands": allowed,
        "deny": [r.raw for r in PERMISSIONS.rules("deny")],
        "ask": [r.raw for r in PERMISSIONS.rules("ask")],
    }


@dataclass
class Run:
    id: int
    role: str
    task: str
    isolation: str
    cwd: str
    status: str = "running"          # running | done | failed | stopped
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    report: str = ""
    error: str = ""
    denied: list = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    patch: Optional[Path] = None
    worktree: Optional[worktrees.Worktree] = None
    repo: Optional[Path] = None      # the real tree a patch goes back to
    # Where this worker's conversation is kept so a retry can continue it instead of
    # starting over (the `resume` of a worker, by run id).
    resume_from: Optional[str] = None
    disposition: str = ""            # "" | applied | discarded
    # What this worker ran on. Kept on the run so the report (and the person) can see
    # that a cheap model or a low thinking level was chosen, not guessed.
    model: str = ""
    provider: str = ""
    thinking: str = ""
    proc: Optional[subprocess.Popen] = None
    log_tail: str = ""
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def running(self) -> bool:
        return self.status == "running"

    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started


def _child_python() -> str:
    from src.automation.runner import _child_python as pick
    return pick()


class SubagentManager:
    #: Every line a child prints, as it prints it: `(run, line)`. The supervisor uses it
    #: to show a worker's work while it happens instead of one line at the end.
    on_line: Optional[Callable[["Run", str], None]] = None

    def __init__(self, argv_builder: Optional[Callable[[Path], List[str]]] = None):
        self._lock = threading.RLock()
        self._runs: Dict[int, Run] = {}
        self._next = 1
        self._argv_builder = argv_builder          # tests substitute a stand-in child
        atexit.register(self.stop_all)

    # -- starting -----------------------------------------------------------------------

    def _argv(self, spec_path: Path) -> List[str]:
        if self._argv_builder is not None:
            return self._argv_builder(spec_path)
        worker = Path(__file__).resolve().with_name("subagent_worker.py")
        return [_child_python(), str(worker), str(spec_path)]

    def running(self) -> List[Run]:
        with self._lock:
            return [r for r in self._runs.values() if r.running]

    def spawn(self, role: str, task: str, tools: Optional[List[str]] = None, isolation: str = "none",
              model: str = "", provider: str = "", thinking: str = "", cwd: Optional[Path] = None,
              resume_from: Optional[Path] = None) -> Run:
        from src.agent.plan_mode import PLAN

        if isolation not in ("none", "worktree"):
            raise ValueError("isolation must be 'none' or 'worktree'")
        # Rejected here rather than in the child: a typo in the level must not start a
        # process that then dies with a traceback nobody reads.
        from argent.config import thinking_settings
        thinking = (thinking or "").strip().lower()
        thinking_settings(thinking)
        role = (role or "Researcher").strip()
        with self._lock:
            if len(self.running()) >= MAX_RUNNING:
                raise ValueError(f"{MAX_RUNNING} sub-agents are already running — wait for one to finish "
                                 f"(subagents action=list) before starting another")
            run_id = self._next
            self._next += 1

        work = Path(cwd) if cwd else Path.cwd()
        wt = None
        if isolation == "worktree":
            wt = worktrees.create(f"{role}-{run_id}", work)
            work = wt.path
        run = Run(id=run_id, role=role, task=task, isolation=isolation, cwd=str(work), worktree=wt,
                  repo=wt.repo if wt else None,
                  resume_from=str(resume_from) if resume_from else None,
                  model=model or "", provider=provider or "", thinking=thinking)

        spec = {
            "role": role, "task": task, "tools": tools_for(role, tools), "cwd": str(work),
            "model": model or "", "provider": provider or "", "thinking": thinking,
            "plan_mode": PLAN.active,
            "resume_from": run.resume_from or "",
            **inherited_authority(),
        }
        spec_dir = Path(tempfile.mkdtemp(prefix="argent-subagent-"))
        spec_path = spec_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["ARGENT_LANE_PRIORITY"] = str(lanes.P_WORKER)      # its model requests queue behind the supervisor's
        try:
            run.proc = subprocess.Popen(
                self._argv(spec_path), cwd=str(work), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            if wt:
                worktrees.remove(wt)
            raise ValueError(f"could not start the sub-agent: {e}")
        governor.lower_process_priority(run.proc.pid)

        with self._lock:
            self._runs[run.id] = run
            self._forget_old()
        threading.Thread(target=self._reap, args=(run, spec_dir), daemon=True, name=f"ArgentSubagent{run.id}").start()
        log.info("sub-agent #%d (%s) started in %s", run.id, role, work)
        return run

    def _forget_old(self) -> None:
        finished = [r for r in self._runs.values() if not r.running and r.worktree is None]
        for run in sorted(finished, key=lambda r: r.started)[:max(0, len(self._runs) - MAX_KEPT)]:
            self._runs.pop(run.id, None)

    # -- ending -------------------------------------------------------------------------

    def _reap(self, run: Run, spec_dir: Path) -> None:
        """Wait for the child, read its verdict, turn its work into a patch.

        Output is read line by line as it arrives, not in one go at the end: the child's
        own dialogue (what it is reading, what it is running) is the only window a
        person has into a worker, and `communicate()` kept it shut until the work was
        already done.
        """
        lines: List[str] = []
        verdict = None
        try:
            assert run.proc.stdout is not None
            for raw in iter(run.proc.stdout.readline, b""):
                text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not text:
                    continue
                if text.startswith(RESULT_MARKER):
                    try:
                        verdict = json.loads(text[len(RESULT_MARKER):])
                    except ValueError:
                        pass
                    continue
                lines.append(text)
                run.log_tail = "\n".join(lines[-80:])[-LOG_TAIL_CHARS:]
                hook = self.on_line
                if hook is not None:
                    try:
                        hook(run, text)
                    except Exception as e:                # a screen must never kill a worker
                        log.debug("sub-agent #%d: on_line failed: %s", run.id, e)
            run.proc.wait()
        except Exception as e:                               # the pipe broke: still finish the run
            log.warning("sub-agent #%d: reading its output failed: %s", run.id, e)
        if verdict is None:
            for line in reversed(lines):
                if line.startswith(RESULT_MARKER):
                    try:
                        verdict = json.loads(line[len(RESULT_MARKER):])
                    except ValueError:
                        pass
                    break
        try:
            self._collect_work(run)
        except Exception as e:
            log.warning("sub-agent #%d: collecting its changes failed: %s", run.id, e)
            run.error = f"{run.error}\ncould not collect the changes: {e}".strip()
        with self._lock:
            if run.status == "stopped":
                pass
            elif verdict is None:
                run.status = "failed"
                run.error = run.error or ("the sub-agent ended without a report (exit code "
                                          f"{run.proc.returncode}). Last output:\n{run.log_tail[-1500:]}")
            elif verdict.get("status") == "ok":
                run.status = "done"
            else:
                run.status = "failed"
                run.error = verdict.get("error") or "the sub-agent failed"
            if verdict:
                run.report = (verdict.get("report") or "").strip()
                run.denied = verdict.get("denied") or []
            run.finished = time.time()
        run.done.set()
        try:
            for leftover in spec_dir.iterdir():
                leftover.unlink()
            spec_dir.rmdir()
        except OSError:
            pass
        log.info("sub-agent #%d ended: %s after %.0fs", run.id, run.status, run.elapsed())

    def _collect_work(self, run: Run) -> None:
        wt = run.worktree
        if wt is None:
            return
        patch = worktrees.changes(wt)
        if not patch.strip():
            worktrees.remove(wt)
            run.worktree = None
            return
        run.files = worktrees.changed_files(wt)
        folder = wt.repo / ".argent" / "subagents"
        folder.mkdir(parents=True, exist_ok=True)
        run.patch = folder / f"{run.id}-{run.role.lower()}.patch"
        run.patch.write_text(patch, encoding="utf-8", newline="\n")

    # -- looking ------------------------------------------------------------------------

    def get(self, run_id) -> Optional[Run]:
        try:
            return self._runs.get(int(str(run_id).strip().lstrip("#")))
        except (TypeError, ValueError):
            return None

    def all(self) -> List[Run]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.id)

    def is_finished(self, run_id) -> bool:
        """For wait conditions: an id nobody knows counts as finished, or the wait would hang for ever."""
        run = self.get(run_id)
        return run is None or not run.running

    def wait(self, run: Run, timeout: float = DEFAULT_WAIT_SECONDS) -> bool:
        return run.done.wait(timeout)

    # -- acting -------------------------------------------------------------------------

    def stop(self, run: Run) -> bool:
        if not run.running or run.proc is None:
            return False
        from src.agent.shell import kill_process_tree
        with self._lock:
            run.status = "stopped"
        kill_process_tree(run.proc)
        return True

    def stop_all(self) -> None:
        for run in self.running():
            try:
                self.stop(run)
            except Exception:
                pass

    def apply(self, run: Run) -> str:
        """Put the sub-agent's patch into the real tree. "" on success, else why not (nothing changed)."""
        if run.patch is None or not run.patch.is_file():
            return "this sub-agent left no changes to apply"
        if run.disposition == "applied":
            return "already applied"
        problem = worktrees.apply_patch(run.repo or Path.cwd(), run.patch)
        if problem:
            return problem
        run.disposition = "applied"
        self._drop_copy(run)
        return ""

    def discard(self, run: Run) -> None:
        self._drop_copy(run)
        if run.patch is not None:
            try:
                run.patch.unlink()
            except OSError:
                pass
        run.disposition = run.disposition or "discarded"

    def _drop_copy(self, run: Run) -> None:
        if run.worktree is not None:
            worktrees.remove(run.worktree)
            run.worktree = None


MANAGER = SubagentManager()


# -- what a worker may run on ---------------------------------------------------------------

PROVIDERS = ("ollama", "openrouter", "zai", "deepseek", "claude", "xkiro", "wormsoft", "koboldcpp")

_KEY_GETTERS = {
    "openrouter": "get_openrouter_api_key",
    "zai": "get_zai_api_key",
    "deepseek": "get_deepseek_api_key",
    "claude": "get_claude_api_key",
    "xkiro": "get_xkiro_api_key",
    "wormsoft": "get_wormsoft_api_key",
}


def provider_ready(provider: str) -> tuple[bool, str]:
    """Whether a worker could actually run on this provider now, and why not.

    Checked before a process is started: a cloud provider with no key would fail
    halfway through an unattended run, where nobody reads the traceback.
    """
    name = (provider or "").strip().lower()
    if name == "anthropic":
        name = "claude"
    if name not in PROVIDERS:
        return False, f"unknown provider {provider!r}; known: {', '.join(PROVIDERS)}"
    if name in ("ollama", "koboldcpp"):
        return True, ""                        # a local server is its own configuration
    from argent import config
    getter = getattr(config, _KEY_GETTERS.get(name, ""), None)
    if getter is not None and not getter():
        return False, f"{name} has no API key configured"
    return True, ""


def resolve_target(model: str = "", provider: str = "", thinking: str = "",
                   strict: bool | None = None) -> tuple[str, str, str]:
    """Model, provider and thinking level for a sub-agent, with the defaults applied.

    Precedence: what the caller asked for, then the person's `subagent_*` keys,
    then "as the main agent". A provider without a model uses that provider's
    saved model; a model without a provider keeps the main provider.

    `strict` decides whether an unconfigured provider is an error. A target the
    caller picked explicitly is checked before a process starts — a 401 halfway
    through an unattended run is the worst kind of surprise. A target that came
    from the person's own config is attempted as written: that path already has
    a fallback, and refusing it would take the choice away from them.
    """
    from argent import config

    explicit_provider = bool((provider or "").strip())
    if strict is None:
        strict = explicit_provider

    thinking = (thinking or "").strip().lower() or config.get_subagent_thinking()
    config.thinking_settings(thinking)         # a typo fails here, before anything runs

    model = (model or "").strip() or config.get_subagent_model()
    provider = (provider or "").strip() or config.get_subagent_provider()

    if provider:
        if strict:
            ok, why = provider_ready(provider)
            if not ok:
                raise ValueError(why)
        if provider.lower() == "anthropic":
            provider = "claude"
        if not model:
            model = config.get_current_model(provider) or ""
    elif model:
        provider = config.get_provider()
    return model, provider, thinking


def available_targets() -> str:
    """What a sub-agent may be told to run on, and where each one runs."""
    from argent import config
    from src.agent.local_backend import classify

    main_provider, main_model = config.get_provider(), config.get_current_model()
    saved = config.get_provider_models() or {}
    lines = [f"Main agent: {main_provider}:{main_model} "
             f"[{'local' if classify(main_provider, main_model).local else 'cloud'}]"]
    for name in PROVIDERS:
        ok, _why = provider_ready(name)
        if not ok:
            continue
        model = saved.get(name) or (main_model if name == main_provider else "")
        where = "local" if classify(name, model).local else "cloud"
        lines.append(f"  {name}: {model or '(pass model= explicitly)'} [{where}]")
    lines.append("thinking: off | low | medium | high | xhigh | auto "
                 "(auto or empty = as the main agent).")
    lines.append("Defaults: config subagent_model / subagent_provider / subagent_thinking; "
                 "unset = the main agent's own model and thinking.")
    return "\n".join(lines)


# -- how a run is told to the model / the person -------------------------------------------

def describe(run: Run, full: bool = True) -> str:
    state = {"running": "running", "done": "finished", "failed": "FAILED", "stopped": "stopped"}[run.status]
    head = f"Sub-agent #{run.id} ({run.role}) {state}"
    head += f" after {run.elapsed():.0f}s" if not run.running else f" for {run.elapsed():.0f}s"
    if run.isolation == "worktree":
        head += " [isolated copy]"
    if run.model or run.provider or run.thinking:
        target = f"{run.provider}:" if run.provider else ""
        target += run.model or "main model"
        if run.thinking:
            target += f", thinking {run.thinking}"
        head += f" on {target}"
    lines = [head + "."]
    if run.running:
        lines.append(f"Task: {run.task[:200]}")
        return "\n".join(lines)
    if run.error:
        lines.append(f"Problem: {run.error}")
    if run.report and full:
        lines += ["Report:", run.report]
    if run.denied:
        lines.append("Actions it needed a person for and was REFUSED (do them yourself if they matter): "
                     + "; ".join(d.get("action", "?") for d in run.denied[:8]))
    if run.patch is not None and run.disposition != "applied":
        shown = ", ".join(run.files[:8]) + (f" and {len(run.files) - 8} more" if len(run.files) > 8 else "")
        lines.append(f"It changed {len(run.files)} file(s) in its isolated copy: {shown}. The patch is saved at "
                     f"{run.patch.as_posix()}. Read it, then apply it to the real tree with "
                     f"subagents(action=\"apply\", id={run.id}) or drop it with subagents(action=\"discard\", id={run.id}).")
    elif run.isolation == "worktree" and run.worktree is None and run.disposition == "":
        lines.append("It made no changes in its isolated copy.")
    if run.disposition == "applied":
        lines.append("Its patch has been applied to the real tree.")
    elif run.disposition == "discarded":
        lines.append("Its changes were discarded.")
    return "\n".join(lines)


def table(runs: List[Run]) -> str:
    if not runs:
        return "No sub-agents in this session."
    rows = []
    for r in runs:
        note = ""
        if r.patch is not None and r.disposition == "":
            note = f" — {len(r.files)} changed file(s), patch not applied"
        rows.append(f"#{r.id} {r.role:<12} {r.status:<8} {r.elapsed():>5.0f}s  {r.task[:60]!r}{note}")
    return "\n".join(rows)
