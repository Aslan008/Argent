"""Run one automation with nobody watching.

The interactive agent assumes a human is present: gated actions raise a prompt
and wait. Neither half of that works unattended — waiting hangs the scheduler
forever, and auto-approving hands an unsupervised model the authority to delete
files or publish things.

So an unattended run REFUSES every gated action and records what it wanted to
do. The refusal is not a failure of the run: the model gets "Execution aborted"
back as a normal tool result and can finish the safe part of its job, and the
attempt shows up in the run log for you to read afterwards. Anything that
genuinely needs to act on the world belongs behind an explicit review step, not
behind a policy flag.

There is exactly one pre-approval, and it is the human's, written down when the
automation is created: ``allowed_commands``. Each entry is a shell command the
model may run as-is (``*`` stays inside one command and never matches across
``; & | < > ` $``), and a command the risk-gate does not grade "safe" is refused
whatever the list says. Everything else still needs a person.

Three more bounds, all of which exist because nobody will notice a runaway:
  * allowed_tools — the task gets only the tools it needs;
  * max_turns — a hard cap on tool round-trips;
  * a single run at a time per automation (the scheduler enforces overlap).

Two of the three modes need no model at all (see store.Automation.mode): a
`command` run executes the command, a `remind` run delivers the message.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

_ARGENT_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__" and __package__ in (None, ""):
    # Started by path (the OS scheduler does that), not with `python -m`.
    sys.path.insert(0, str(_ARGENT_ROOT))

from logger import get_logger  # noqa: E402

log = get_logger("automation")

# Tools an unattended run may never use, whatever the automation declares.
# These either need a human (they ask questions), hand control to something
# that outlives the run (background processes, sub-agents with their own loops),
# or would let a run rewrite its own schedule and permissions.
UNATTENDED_TOOL_DENYLIST = {
    "ask_user_questions", "request_user_approval",
    "run_admin_command",
    "wait_heartbeat",
    "manage_automations",
}

# Reply with exactly this when a run found nothing worth reporting; the runner
# turns it into a quiet "nothing new" instead of a message that says nothing.
SILENT_MARKER = "NOTHING_NEW"

DEFAULT_COMMAND_TIMEOUT_MINUTES = 30


# --- the one human pre-approval: exact commands ---------------------------------------

# A '*' in an allowed command matches within ONE command: it can never swallow a
# shell operator, so "git log*" cannot be stretched into "git log; rm -rf x".
_WILDCARD = r"[^;&|<>`$\r\n]*"


_CONTROL = re.compile(r"[\r\n\x00]")


def _literal(text: str) -> str:
    """Regex for literal text in which any run of blanks matches any other run."""
    return "".join(r"[ \t]+" if chunk.strip(" \t") == "" else re.escape(chunk)
                   for chunk in re.split(r"([ \t]+)", text) if chunk)


def command_pattern(pattern: str):
    """Compile an allowed-command pattern. Everything but '*' is literal."""
    return re.compile(_WILDCARD.join(_literal(p) for p in pattern.strip().split("*")))


def command_allowed(command: str, patterns) -> bool:
    """Whether the command line is covered by one of the owner's patterns.

    Matched against the command exactly as it would be executed: a line break is
    a second command to a shell, so a command containing one is never allowed —
    normalising it away first would approve text that is not what runs.
    """
    command = (command or "").strip()
    if not command or _CONTROL.search(command):
        return False
    return any(command_pattern(p).fullmatch(command) for p in patterns or [] if p and p.strip())


class UnattendedGate:
    """Approval backend for a run with no human: deny, and remember.

    The single exception is a shell command the automation's owner listed in
    ``allowed_commands`` — and then only when it is not destructive.
    """

    # Read by approval.request_approval: suppresses the session grants and the
    # POLICY_AUTO shortcut, both of which are consent a present human gave for
    # their own work. Without it a user who had run /vibe, or who had once
    # answered "always allow git", handed every later scheduled run the same
    # authority — and the refusal log stayed empty because nothing was refused.
    unattended = True

    def __init__(self, allowed_commands=()):
        self.denied = []
        self.ran = []
        self._allowed_commands = list(allowed_commands or [])

    def __call__(self, action: str, destructive: bool, grant_key) -> str:
        if not destructive and self._allowed_commands:
            from approval import command_from_action
            command = command_from_action(action)
            if command is not None and command_allowed(command, self._allowed_commands):
                self.ran.append(command.strip())
                log.info("automation ran a pre-approved command: %s", command)
                return "once"
        self.denied.append({"action": action, "destructive": bool(destructive)})
        log.info("automation refused a gated action: %s", action)
        return "deny"


# Granted to every run whatever it declared. Remembering what you already
# reported is not authority over anything, and a narrow toolset that leaves it
# out turns a monitoring job into a machine for repeating itself.
UNATTENDED_ALWAYS_ALLOWED = ("filter_new_items",)


def resolve_tools(allowed_tools: list) -> list | None:
    """The toolset for a run: the declared list minus the denylist.

    None means "the agent's default set" and is only returned when the
    automation declared nothing — the denylist still applies at the approval
    layer, since those tools all route through it or need a human.
    """
    if not allowed_tools:
        return None
    tools = [t for t in allowed_tools if t not in UNATTENDED_TOOL_DENYLIST]
    return tools + [t for t in UNATTENDED_ALWAYS_ALLOWED if t not in tools]


# --- what the model is told ----------------------------------------------------------

def build_prompt(automation, now: datetime, previous: dict = None) -> str:
    """The task, wrapped in what a model with no human in the room needs to know.

    A bare task string leaves the model guessing: is anyone there to ask, what
    happened last time, what should the answer look like. Without the last report
    a "what changed?" job cannot say what changed.
    """
    lines = [
        f'[Scheduled run] You are the automation "{automation.name}" '
        f'({automation.schedule}), started {now:%Y-%m-%d %H:%M}, {now:%A}.',
        "Nobody is watching this run: you cannot ask questions, and any action that "
        "needs approval is refused (the refusal is logged — carry on with what you can do).",
    ]
    if automation.allowed_commands:
        listed = "\n".join(f"  {c}" for c in automation.allowed_commands)
        lines.append("You may run these shell commands without approval, exactly as written "
                     f"(`*` stands for any arguments):\n{listed}")
    if previous and previous.get("summary"):
        lines.append(f"Previous run ({str(previous.get('finished', ''))[:16].replace('T', ' ')}, "
                     f"{previous.get('status', '?')}): {' '.join(previous['summary'].split())[:600]}")
    lines.append(
        "Finish with a short report: what you found, what changed, what needs attention. "
        "Write it in the language of the task. "
        f"If there is nothing new or nothing worth the user's attention, reply with exactly {SILENT_MARKER} "
        "and nothing else. To report only what is new, pass what you found to filter_new_items.")
    lines.append(f"\nTASK:\n{automation.task}")
    return "\n".join(lines)


def split_silent(text: str):
    """(silent, report) — whether the model answered with the quiet marker."""
    stripped = (text or "").strip()
    if stripped.upper().startswith(SILENT_MARKER):
        return True, stripped[len(SILENT_MARKER):].lstrip(" :.-—\n").strip()
    return False, stripped


def digest_of(text: str) -> str:
    folded = " ".join((text or "").lower().split())
    return hashlib.sha1(folded.encode("utf-8")).hexdigest()[:16] if folded else ""


# --- the three modes -----------------------------------------------------------------

def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # cmd.exe on Russian Windows still answers in the OEM code page.
        return data.decode("cp866" if os.name == "nt" else "utf-8", errors="replace")


def _run_shell(automation, started: datetime) -> dict:
    """mode=command: run the stored command line, exit code 0 = ok."""
    from src.agent.shell import kill_process_tree
    from src.automation.store import project_root

    limit = (automation.timeout_minutes or DEFAULT_COMMAND_TIMEOUT_MINUTES) * 60
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    status, text = "ok", ""
    try:
        proc = subprocess.Popen(
            automation.task, shell=True, cwd=str(project_root()), env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as e:
        return {"status": "error", "text": f"could not start: {e}", "returncode": None}
    try:
        out, _ = proc.communicate(timeout=limit)
        text = _decode(out or b"").strip()
        if proc.returncode != 0:
            status = "error"
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        try:
            out, _ = proc.communicate(timeout=5)
            text = _decode(out or b"").strip()
        except Exception:
            pass
        status = "error"
        text = (text + f"\n[stopped: no result within {limit // 60} min]").strip()
    return {"status": status, "text": text, "returncode": proc.returncode}


def _run_room(automation, started: datetime, mode: str) -> dict:
    """mode=room: run the room named first in `task` with nobody to ask. The room's own run record is
    written like any other; this one adds the schedule's record and notification."""
    from src.rooms.headless import run_headless, split_task
    room, text, params = split_task(automation.task)
    outcome = run_headless(room, text, params)
    status = {"ok": "ok", "denied": "denied"}.get(outcome["status"], "error")
    summary = outcome["summary"]
    if outcome.get("needs_person"):
        summary = (summary + "\n[комната остановилась: ей нужен ответ человека]").strip()
    denied = list(outcome.get("declined") or [])
    return _finish(automation, started, status, summary, denied, 0, False, [], mode, digest_text=summary)


def _tail(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else "…" + text[-limit:]


def _collect(agent, prompt: str, tools, max_turns: int):
    """Drive the agent and return (final report, notes, exception or None).

    Only the text after the last tool call is the report: everything before it is
    the model thinking out loud ("let me look at the page…"), and a report made
    of that is unreadable.
    """
    segments, current, notes = [], [], []
    turns = 0
    error = None
    try:
        for chunk in agent.process_user_input(prompt, allowed_tools=tools):
            kind = chunk.get("type")
            if kind in ("content_stream", "content"):
                current.append(chunk.get("content", ""))
            elif kind == "content_replace":
                current = [chunk.get("content", "")]
            elif kind == "tool_end":
                turns += 1
                text = "".join(current).strip()
                if text:
                    segments.append(text)
                current = []
                if turns >= max(1, max_turns):
                    notes.append(f"[stopped: reached the {max_turns}-turn budget]")
                    break
            elif kind == "error":
                notes.append(f"[{chunk.get('content', '')}]")
    except Exception as e:
        error = e
        log.warning("automation run failed: %s", e)
    final = "".join(current).strip() or (segments[-1] if segments else "")
    return final, notes, error


def run_automation(automation, agent=None, now=None) -> dict:
    """Execute one automation. Never raises — a scheduler must survive its jobs.

    Returns {status, summary, denied_actions, new_items, started, finished,
    silent, changed, digest, commands, mode} where status is ok | error | denied.
    'denied' means the task could not do its job without an action that requires
    a human.
    """
    import approval
    from src.automation import memory
    from src.automation.store import load_runs

    started = now or datetime.now()
    mode = getattr(automation, "mode", "agent") or "agent"

    if mode == "remind":
        text = (automation.task or "").strip()
        return _finish(automation, started, "ok", text, [], 0, False, [], mode)

    if mode == "room":
        return _run_room(automation, started, mode)

    if mode == "command":
        outcome = _run_shell(automation, started)
        text = outcome["text"]
        body = _tail(text) if text else ("(пустой вывод)" if outcome["status"] == "ok" else "")
        if outcome["status"] == "error":
            code = outcome["returncode"]
            head = "команда завершилась с ошибкой" + (f" (код {code})" if code is not None else "")
            body = f"{head}\n{body}".strip()
        return _finish(automation, started, outcome["status"], body, [], 0,
                       outcome["status"] == "ok" and not text, [], mode,
                       digest_text=text)

    gate = UnattendedGate(getattr(automation, "allowed_commands", None))
    tools = resolve_tools(automation.allowed_tools)
    status = "ok"
    new_items = 0

    if agent is None:
        from agent import ArgentAgent
        agent = ArgentAgent()
        agent.autosave_enabled = False      # a scheduled run must not become "the last session" offered at startup

    previous = None
    try:
        history = load_runs(limit=1, name=automation.name)
        previous = history[-1] if history else None
    except Exception:
        pass

    approval.set_approval_backend(gate)
    memory.set_scope(automation.name)
    try:
        final, notes, error = _collect(agent, build_prompt(automation, started, previous),
                                       tools, automation.max_turns)
        if error is not None:
            status = "error"
            notes.append(f"[run failed: {error}]")
    finally:
        approval.reset_approval_backend()
        # A run that crashed never delivered its findings, so releasing them
        # keeps the items unseen and they come back next time. Committing here
        # would lose exactly what the automation exists to catch.
        if status == "ok":
            new_items = memory.commit()
        else:
            memory.rollback()
            new_items = 0       # nothing was delivered, so nothing was news
        memory.reset_scope()

    silent, report = split_silent(final)
    if silent and not report:
        report = "Ничего нового."
    summary = "\n".join(part for part in [report, *notes] if part).strip()
    if status == "ok" and gate.denied and not summary:
        # It needed a human and produced nothing else — that is the whole story
        # of the run, so don't report it as a success.
        status = "denied"

    return _finish(automation, started, status, summary, gate.denied, new_items,
                   silent and not new_items, gate.ran, mode,
                   digest_text=report if not silent else "")


def _finish(automation, started, status, summary, denied, new_items, silent, commands,
            mode, digest_text=None) -> dict:
    digest = digest_of(summary if digest_text is None else digest_text)
    previous_digest = getattr(automation, "last_digest", "") or ""
    return {
        "status": status,
        "summary": summary,
        "denied_actions": denied,
        "new_items": new_items,
        "started": started,
        "finished": datetime.now(),
        "silent": bool(silent),
        "digest": digest,
        "changed": bool(digest) and digest != previous_digest,
        "commands": commands,
        "mode": mode,
    }


def run_and_record(automation, agent=None, notify: bool = True, trigger: str = None) -> dict:
    """run_automation + persist the outcome + tell the user if it matters.

    ``notify`` is off for a manual /tasks run: you are already looking at the
    output, and a toast about something on your screen is pure noise.
    """
    from src.automation.notify import notify_run, should_alert
    from src.automation.store import project_root, record_run

    trigger = trigger or ("schedule" if notify else "manual")
    result = run_automation(automation, agent=agent)
    policy = getattr(automation, "notify", "auto") or "auto"
    result["trigger"] = trigger
    result["notify_policy"] = policy
    if result["mode"] == "remind":
        result["alert"] = policy != "never"
    else:
        result["alert"] = should_alert(policy, result["status"], len(result["denied_actions"]),
                                       result["new_items"], result["silent"], result["changed"])

    result["report"] = _write_report(automation, result, project_root())

    record_run(automation.name, result["status"], result["summary"],
               result["started"], result["finished"], result["denied_actions"],
               result.get("new_items", 0), trigger=trigger, alert=result["alert"],
               silent=result["silent"], digest=result["digest"] or None,
               report=result["report"], commands=result["commands"])
    if notify:
        try:
            notify_run(automation.name, result, getattr(automation, "channels", ()))
        except Exception as e:      # never let delivery break a good run
            log.info("could not notify about %r: %s", automation.name, e)
    return result


def _write_report(automation, result: dict, root: Path) -> str:
    """Append this run to the automation's report file, if it has one."""
    template = getattr(automation, "report_file", "") or ""
    if not template or not result["summary"]:
        return ""
    try:
        from src.automation.report import append_report, resolve_report_path
        path = resolve_report_path(template, automation.name, result["finished"], root)
        append_report(path, automation.name, result["status"], result["finished"],
                      result["summary"], result.get("trigger", "schedule"))
        return str(path)
    except Exception as e:
        log.warning("could not write the report of %r: %s", automation.name, e)
        return ""


# --- running a job in its own interpreter -------------------------------------
# The host that owns the scheduler is an interactive program. The TUI's screen
# and the REPL's prompt are drawn on the very stdout the agent's tools print to
# (live command output, diffs, approval lines), so a job running inside the host
# would scribble over what the user is typing. It would also share the host's
# process-wide state with a model nobody is watching. A child process has neither
# problem, and a crash or a hang in a job costs the job, not the session.

RESULT_MARKER = "ARGENT_AUTOMATION_RESULT "
_RESULT_KEYS = ("status", "summary", "denied_actions", "new_items", "silent", "alert",
                "changed", "trigger", "mode", "report", "commands")

# A run is bounded by its turn budget, but a provider that stops answering never
# reaches the next turn. This is the wall clock under that.
DEFAULT_RUN_TIMEOUT_SECONDS = 30 * 60


def _child_python() -> str:
    """The interpreter for a job. When the OS scheduler started us under
    pythonw.exe (no console, so no stdout of our own), the child still wants the
    console flavour: its verdict travels on stdout."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return sys.executable


class SubprocessRunner:
    """Scheduler runner: executes each automation as `python -m src.automation.runner`.

    Callable like the in-process `run_and_record`: takes an Automation and returns
    the result dict. The child records the run and sends the notification itself;
    only when it dies without reporting does the parent write the error down, so
    a job that never started still shows up in `/tasks runs`.
    """

    def __init__(self, timeout: float = DEFAULT_RUN_TIMEOUT_SECONDS, argv=None):
        self._timeout = timeout
        self._argv = argv                  # tests substitute a stand-in child
        self._lock = threading.Lock()
        self._procs: dict = {}

    def _build_argv(self, automation, trigger: str) -> list:
        if self._argv is not None:
            return list(self._argv(automation))
        return [_child_python(), "-m", "src.automation.runner", automation.name,
                "--trigger", trigger]

    def _environment(self) -> dict:
        env = os.environ.copy()
        # The child is not the TUI, and must not behave as if one were attached
        # (a prompt would wait for a modal nobody can answer).
        env.pop("ARGENT_TUI", None)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_ARGENT_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _limit_for(self, automation) -> float:
        minutes = int(getattr(automation, "timeout_minutes", 0) or 0)
        return minutes * 60 if minutes > 0 else self._timeout

    def __call__(self, automation, trigger: str = "schedule") -> dict:
        from project_paths import project_root_or_cwd
        from src.agent.shell import kill_process_tree

        started = datetime.now()
        limit = self._limit_for(automation)
        try:
            proc = subprocess.Popen(
                self._build_argv(automation, trigger),
                cwd=str(project_root_or_cwd()),     # the project's .argent/ is what it reads
                env=self._environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as e:
            return self._failed(automation, started, f"could not start the run: {e}", trigger)

        with self._lock:
            self._procs[automation.name] = proc
        try:
            try:
                out, err = proc.communicate(timeout=limit)
            except subprocess.TimeoutExpired:
                kill_process_tree(proc)
                try:
                    proc.communicate(timeout=5)
                except Exception:
                    pass
                return self._failed(
                    automation, started,
                    f"timed out after {int(limit)} s and was stopped", trigger)
        finally:
            with self._lock:
                self._procs.pop(automation.name, None)
        if getattr(proc, "_argent_stopped", False):
            return self._failed(automation, started, "stopped by the user", trigger)

        result = self._parse(out)
        if result is not None:
            result.setdefault("started", started)
            result.setdefault("finished", datetime.now())
            return result
        tail = " ".join((err or out or "").split())[-300:]
        return self._failed(
            automation, started,
            f"the run ended without a result (exit code {proc.returncode}): {tail}", trigger)

    @staticmethod
    def _parse(stdout: str):
        for line in reversed((stdout or "").splitlines()):
            if line.startswith(RESULT_MARKER):
                try:
                    payload = json.loads(line[len(RESULT_MARKER):])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
        return None

    @staticmethod
    def _failed(automation, started, reason: str, trigger: str = "schedule") -> dict:
        from src.automation.store import record_run
        finished = datetime.now()
        log.warning("automation %r: %s", automation.name, reason)
        record_run(automation.name, "error", reason, started, finished, [], 0,
                   trigger=trigger, alert=True)
        return {"status": "error", "summary": reason, "denied_actions": [],
                "new_items": 0, "started": started, "finished": finished,
                "alert": True, "silent": False, "trigger": trigger,
                "mode": getattr(automation, "mode", "agent")}

    def is_running(self, name: str) -> bool:
        with self._lock:
            return name in self._procs

    def running(self) -> list:
        with self._lock:
            return list(self._procs)

    def stop(self, name: str) -> bool:
        """Stop one run in flight. False when it is not running here."""
        from src.agent.shell import kill_process_tree
        with self._lock:
            proc = self._procs.get(name)
            if proc is not None:
                proc._argent_stopped = True
        if proc is None:
            return False
        kill_process_tree(proc)
        return True

    def terminate_all(self) -> None:
        """Stop every run in flight. Called when the host exits: a job must not
        outlive the program whose lifetime it was scoped to."""
        from src.agent.shell import kill_process_tree
        with self._lock:
            procs = list(self._procs.values())
        for proc in procs:
            kill_process_tree(proc)


def main(argv=None) -> int:
    """Child entry point.

        python -m src.automation.runner <name> [--trigger schedule|manual|startup|retry]
        python -m src.automation.runner --tick [--project DIR]

    The first runs one automation and reports on stdout; the second is what the
    OS scheduler calls (see service.py) to run whatever is due while Argent is
    not open.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    project = None
    if "--project" in args:
        index = args.index("--project")
        project = args[index + 1] if index + 1 < len(args) else None
        del args[index:index + 2]
    if project:
        os.chdir(project)

    if "--tick" in args:
        from src.automation.scheduler import run_due_once
        return run_due_once()

    trigger = "schedule"
    if "--trigger" in args:
        index = args.index("--trigger")
        trigger = args[index + 1] if index + 1 < len(args) else "schedule"
        del args[index:index + 2]

    if len(args) != 1:
        print("usage: python -m src.automation.runner <automation name> [--trigger T] "
              "| --tick [--project DIR]", file=sys.stderr)
        return 2
    from src.automation.store import get_automation

    automation = get_automation(args[0])
    if automation is None:
        print(f"no automation named {args[0]!r} in this project", file=sys.stderr)
        return 2
    result = run_and_record(automation, notify=(trigger != "manual"), trigger=trigger)
    payload = {key: result.get(key) for key in _RESULT_KEYS}
    # The agent's tools print freely; the marker is how the parent finds the verdict.
    print(RESULT_MARKER + json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
