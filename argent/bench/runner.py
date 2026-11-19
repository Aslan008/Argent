"""Run a bench case in an isolated worktree and decide pass/fail from checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from argent.bench.case import Case, CaseError
from argent.logger import get_logger

log = get_logger("bench")

_IGNORE_COPY = {".git", ".venv", "venv", "node_modules", "__pycache__", "target", ".argent"}


@dataclass
class CheckResult:
    cmd: str
    passed: bool
    exit_code: Optional[int] = None
    output: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cmd": self.cmd,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "output": self.output[-2000:],
            "error": self.error,
        }


@dataclass
class CaseResult:
    case_id: str
    passed: bool = False
    checks: List[CheckResult] = field(default_factory=list)
    setup_error: str = ""
    agent_error: str = ""
    timed_out: bool = False
    duration_s: float = 0.0
    events: int = 0
    diff: str = ""
    worktree: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case_id,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "setup_error": self.setup_error,
            "agent_error": self.agent_error,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "events": self.events,
            "diff": self.diff,
            "worktree": self.worktree,
        }


def _run_command(cmd: str, cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd),
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def run_checks(case: Case, cwd: Path) -> List[CheckResult]:
    results: List[CheckResult] = []
    for check in case.checks:
        try:
            proc = _run_command(check.cmd, cwd, check.timeout_s)
            out = ((proc.stdout or "") + (proc.stderr or "")).strip()
            ok = proc.returncode == check.expect_exit
            if ok and check.stdout_contains is not None:
                ok = check.stdout_contains in out
            if ok and check.stdout_not_contains is not None:
                ok = check.stdout_not_contains not in out
            results.append(CheckResult(
                cmd=check.cmd,
                passed=ok,
                exit_code=proc.returncode,
                output=out,
            ))
        except subprocess.TimeoutExpired:
            results.append(CheckResult(cmd=check.cmd, passed=False, error=f"timeout after {check.timeout_s}s"))
        except Exception as e:
            results.append(CheckResult(cmd=check.cmd, passed=False, error=str(e)))
    return results


def create_worktree(repo: Path, start: str = "HEAD") -> Path:
    repo = Path(repo).resolve()
    probe = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if probe.returncode != 0:
        # A plain directory is still usable for checks; the agent just runs in a copy.
        dest = Path(tempfile.mkdtemp(prefix="argent_bench_copy_"))
        for item in repo.iterdir():
            if item.name in _IGNORE_COPY:
                continue
            try:
                if item.is_dir():
                    shutil.copytree(item, dest / item.name, symlinks=True)
                else:
                    shutil.copy2(item, dest / item.name)
            except OSError:
                pass
        return dest

    dest = Path(tempfile.mkdtemp(prefix="argent_bench_"))
    dest.rmdir()
    proc = subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(dest), start],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise CaseError(f"git worktree add failed: {(proc.stderr or proc.stdout).strip()}")
    return dest


def remove_worktree(path: Path, repo: Optional[Path] = None) -> None:
    path = Path(path)
    try:
        if repo is not None:
            subprocess.run(
                ["git", "-C", str(Path(repo).resolve()), "worktree", "remove", "--force", str(path)],
                capture_output=True, text=True, timeout=30,
            )
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        shutil.rmtree(path, ignore_errors=True)


def _safe_approval(action: str, destructive: bool, grant_key) -> str:
    # Bench never grants a destructive action; everything else is allowed so
    # the task can actually edit files and run commands.
    return "deny" if destructive else "once"


def _run_agent(case: Case, cwd: Path, timeout_s: float,
               agent_factory: Optional[Callable[[], Any]] = None) -> tuple[List[dict], str, bool]:
    from argent import approval
    from src.bridge.session import AgentSession

    events: List[dict] = []
    error: List[str] = []
    done = threading.Event()
    session_box: List[AgentSession] = []

    def worker() -> None:
        try:
            approval.set_approval_backend(_safe_approval)
            agent = agent_factory() if agent_factory is not None else _default_agent()
            session = AgentSession(agent=agent)
            session_box.append(session)
            for event in session.handle(case.task):
                events.append(event)
        except Exception as e:
            error.append(str(e))
        finally:
            try:
                approval.reset_approval_backend()
            except Exception:
                pass
            if session_box:
                try:
                    session_box[0].close_trace()
                except Exception:
                    pass
            done.set()

    thread = threading.Thread(target=worker, name="bench-agent", daemon=True)
    thread.start()
    if not done.wait(timeout_s):
        if session_box:
            try:
                session_box[0].cancel()
            except Exception:
                pass
        return events, "timeout", True
    return events, (error[0] if error else ""), False


def _default_agent():
    from argent.agent import ArgentAgent

    return ArgentAgent()


def _diff(worktree: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(worktree), "diff", "--stat"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def _run_agent_subprocess(case: Case, worktree: Path, timeout_s: float,
                          repo: Path) -> tuple[int, str, bool]:
    """Run the agent in a child process, so a timeout can actually kill it."""
    tmp = Path(tempfile.mkdtemp(prefix="argent_bench_run_"))
    try:
        task_file = tmp / "task.txt"
        events_file = tmp / "events.jsonl"
        result_file = tmp / "result.json"
        task_file.write_text(case.task, encoding="utf-8")
        command = [
            sys.executable, "-m", "argent.bench.worker",
            "--task-file", str(task_file),
            "--events", str(events_file),
            "--result", str(result_file),
        ]
        if case.allowed_tools:
            command += ["--allowed-tools", ",".join(case.allowed_tools)]
        env = os.environ.copy()
        root = str(Path(repo).resolve())
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(
                command, cwd=str(worktree), env=env, timeout=timeout_s,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return 0, "timeout", True
        except Exception as e:
            return 0, f"worker failed to start: {e}", False
        events = 0
        if events_file.exists():
            events = sum(1 for _ in events_file.open("r", encoding="utf-8", errors="replace"))
        if result_file.exists():
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
                if data.get("ok"):
                    return events, "", False
                return events, str(data.get("error") or "worker failed"), False
            except ValueError:
                pass
        detail = ((proc.stderr or "") + (proc.stdout or "")).strip()[-2000:]
        return events, f"worker exit {proc.returncode}: {detail}", False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_case(case: Case, repo: Path, *, worktree: Optional[Path] = None,
             agent_factory: Optional[Callable[[], Any]] = None,
             keep: bool = False) -> CaseResult:
    case.validate_runnable()
    started = time.time()
    owns = worktree is None
    wt = Path(worktree) if worktree is not None else None
    result = CaseResult(case_id=case.id)
    old_cwd = Path.cwd()
    try:
        if wt is None:
            wt = create_worktree(repo, case.start)
        result.worktree = str(wt)
        os.chdir(wt)

        for setup in case.setup:
            try:
                proc = _run_command(setup, wt, case.timeout_s)
            except subprocess.TimeoutExpired:
                result.setup_error = f"setup timeout: {setup}"
                break
            except Exception as e:
                result.setup_error = f"setup failed: {setup}: {e}"
                break
            if proc.returncode != 0:
                result.setup_error = (
                    f"setup failed (exit {proc.returncode}): {setup}\n"
                    f"{((proc.stdout or '') + (proc.stderr or ''))[-1500:]}"
                )
                break

        if not result.setup_error:
            if agent_factory is None:
                count, err, timed_out = _run_agent_subprocess(case, wt, case.timeout_s, repo)
                result.events = count
            else:
                events, err, timed_out = _run_agent(case, wt, case.timeout_s, agent_factory)
                result.events = len(events)
            result.timed_out = timed_out
            result.agent_error = err
            if not timed_out:
                result.checks = run_checks(case, wt)
                result.passed = not err and all(c.passed for c in result.checks)
            result.diff = _diff(wt)
    finally:
        os.chdir(old_cwd)
        if owns and wt is not None and not keep:
            remove_worktree(wt, repo)
        result.duration_s = time.time() - started
    return result
