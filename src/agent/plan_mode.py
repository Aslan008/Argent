"""Plan mode: look, think, write the plan — change nothing until the person says go.

"Make a plan first" in the system prompt is a request. A weak model forgets it
mid-task and starts editing; a strong one sometimes decides the change is small
enough not to bother. Neither is the person's fault, and neither can be undone by
asking nicely afterwards. So plan mode is not an instruction: while it is on, the
tools that change anything are not advertised and, if a model calls one anyway
(from memory, from a stale schema, in raw-JSON mode), the call is refused here
before it runs.

What stays open is everything that only looks: reading and searching files, the
web, the project map, and shell commands from a short list of read-only programs.
That list is deliberately an allowlist — "does not match a dangerous pattern" is
the question the risk gate answers for a run that is about to be approved by a
human, and it is the wrong question for a mode whose whole promise is "nothing
changes".

The way out is the model's `exit_plan_mode(plan)`: it shows the plan and asks. The
person's yes turns plan mode off. Nothing else does, except `/plan off`.
"""

import re
import threading
from typing import Optional, Tuple

from logger import get_logger

log = get_logger("agent")

#: Tools that change files, processes, git history, schedules or the outside world.
MUTATING_TOOLS = frozenset({
    "write_file", "replace_in_file", "delete_file", "move_file", "copy_file",
    "create_directory", "run_admin_command", "git_rollback", "git_checkpoint",
    "start_background_command", "send_background_command", "stop_background_command",
    "browser_action", "call_mcp_tool", "manage_automations", "delegate_to_leeth", "run_deep_linter",
    "ultracode", "desktop_act", "leeth_fill_form", "leeth_click_many",
})

#: Registered only while the local agent is enabled, so it is not in the static registry.
_CONDITIONAL_MUTATING = "delegate_local"

#: Only advertised while planning.
PLAN_ONLY_TOOLS = frozenset({"exit_plan_mode"})

#: Shell programs that only read. `git` and `find` are policed separately below.
_READONLY_PROGRAMS = frozenset({
    "ls", "dir", "cat", "type", "head", "tail", "wc", "grep", "rg", "findstr", "pwd", "echo",
    "tree", "stat", "file", "which", "where", "whoami", "printenv",
    "get-childitem", "get-content", "select-string", "get-location", "get-item", "get-process",
    "get-command", "measure-object", "get-itemproperty", "test-path", "resolve-path",
    "sort", "uniq", "diff", "fc", "cut", "column", "nl", "basename", "dirname", "realpath",
    "python --version", "node --version", "cargo --version", "git --version",
})
_READONLY_GIT = frozenset({
    "status", "diff", "log", "show", "blame", "ls-files", "rev-parse", "describe", "shortlog",
    "grep", "cat-file", "ls-tree", "rev-list", "name-rev", "reflog", "whatchanged",
    "branch", "tag", "remote", "config", "stash",
})
# Flags that make `git branch` / `git tag` write something.
_GIT_WRITING_FLAGS = frozenset({
    "-d", "-D", "-m", "-M", "-c", "-C", "-f", "-F", "-s", "-u", "--delete", "--move", "--copy",
    "--force", "--annotate", "--sign", "--set-upstream-to", "--unset-upstream", "--edit-description",
    "--set-upstream", "--create-reflog",
})
# With one of these a positional argument is a query value, not a name to create.
_GIT_QUERY_FLAGS = frozenset({"-l", "--list", "--contains", "--no-contains", "--merged",
                              "--no-merged", "--points-at"})
_GIT_CONFIG_READS = frozenset({"--get", "--get-all", "--list", "-l", "--get-regexp", "--show-origin"})
# Options placed before the subcommand; the first two take a separate value.
_GIT_GLOBAL_WITH_VALUE = frozenset({"-C", "-c", "--git-dir", "--work-tree"})
_GIT_GLOBAL_FLAGS = frozenset({"--no-pager", "-P", "--paginate", "--no-optional-locks", "--literal-pathspecs"})
# PowerShell pipeline cmdlets that only reshape what flows through them — until they are handed a
# script block ({ ... }), which can run anything.
_PIPELINE_CMDLETS = frozenset({
    "select-object", "sort-object", "format-table", "format-list", "out-string", "group-object",
    "get-date", "get-filehash", "get-help", "write-output", "out-host",
})
# Read-only programs that still have a flag to write their result somewhere.
_PROGRAM_WRITE_FLAGS = {"sort": {"-o", "--output"}, "tree": {"-o"}}
_FIND_WRITES = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprintf", "-fls"})
_WRITING_OUTPUT_ARGS = frozenset({"--output", "-o", "--output-file", "--ext-diff", "--textconv"})


def _unquoted_redirect_or_subshell(command: str) -> Optional[str]:
    """Why the text writes or runs something hidden — an unquoted `>`, `` ` `` or `$(`."""
    in_single = in_double = escape = False
    for i, ch in enumerate(command):
        if escape:
            escape = False
            continue
        if ch == "\\" and not in_single:
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single:
            continue
        if ch == "`":
            return "a backtick substitution"
        if ch == "$" and command[i + 1:i + 2] == "(":
            return "a $( ) substitution"
        if not in_double and ch == ">":
            # `2>&1` and `>&2` only join streams; everything else writes a file.
            if command[i + 1:i + 2] == "&" or command[max(0, i - 1):i] == "&":
                continue
            return "an output redirection"
    return None


def readonly_command(command: str) -> Tuple[bool, str]:
    """(True, "") when the shell command only reads; otherwise (False, why)."""
    from approval import _split_unquoted_chain_commands, _has_unclosed_quotes

    command = (command or "").strip()
    if not command:
        return False, "empty command"
    if _has_unclosed_quotes(command):
        return False, "unclosed quotation mark"
    why = _unquoted_redirect_or_subshell(command)
    if why:
        return False, f"it contains {why}"
    for part in _split_unquoted_chain_commands(command):
        tokens = part.strip().split()
        if tokens and tokens[0] == "&" and len(tokens) > 1:
            tokens = tokens[1:]                      # PowerShell call operator
        if not tokens:
            continue
        program = tokens[0].strip("\"'").lower()
        program = re.sub(r"\.exe$", "", program.replace("\\", "/").rsplit("/", 1)[-1])
        args = [a.strip("\"'") for a in tokens[1:]]
        if program == "git":
            ok, why = _git_readonly(args)
        elif program == "find":
            bad = _FIND_WRITES.intersection(args)
            ok, why = (not bad, f"find {sorted(bad)[0] if bad else ''} changes files")
        elif program in _PIPELINE_CMDLETS:
            ok, why = ("{" not in part, f"'{program}' with a script block can run anything")
        elif program in _READONLY_PROGRAMS:
            bad = _PROGRAM_WRITE_FLAGS.get(program, set()).intersection(a.split("=", 1)[0] for a in args)
            ok, why = (not bad, f"'{program} {sorted(bad)[0] if bad else ''}' writes a file")
        elif f"{program} {' '.join(args)}".strip() in _READONLY_PROGRAMS:
            ok, why = True, ""
        else:
            ok, why = False, f"'{program}' is not on the read-only list"
        if not ok:
            return False, why
    return True, ""


def _git_readonly(args) -> Tuple[bool, str]:
    args = list(args)
    while args:
        head = args[0]
        if head in _GIT_GLOBAL_WITH_VALUE:
            args = args[2:]
        elif head in _GIT_GLOBAL_FLAGS or head.split("=", 1)[0] in ("--git-dir", "--work-tree"):
            args = args[1:]
        else:
            break
    sub = args[0] if args else ""
    rest = args[1:]
    if sub not in _READONLY_GIT:
        return False, f"'git {sub}' is not a read-only git command" if sub else "git needs a subcommand"
    if _WRITING_OUTPUT_ARGS.intersection(a.split("=", 1)[0] for a in rest):
        return False, "it writes its output to a file"
    if sub in ("branch", "tag"):
        flags = {a.split("=", 1)[0] for a in rest if a.startswith("-")}
        # `-a` lists every branch but annotates a tag.
        if flags & _GIT_WRITING_FLAGS or (sub == "tag" and "-a" in flags):
            return False, f"'git {sub}' with that flag changes the repository"
        positional = [a for a in rest if not a.startswith("-")]
        if positional and not (flags & _GIT_QUERY_FLAGS):
            return False, f"'git {sub} <name>' creates one"
    elif sub == "remote":
        if rest and rest[0] not in ("-v", "--verbose", "show", "get-url"):
            return False, f"'git remote {rest[0]}' changes the repository"
    elif sub == "config":
        if not _GIT_CONFIG_READS.intersection(rest):
            return False, "'git config' without --get/--list writes the configuration"
    elif sub == "stash":
        if not rest or rest[0] not in ("list", "show"):
            return False, "'git stash' changes the working tree"
    return True, ""


class PlanMode:
    def __init__(self):
        self._lock = threading.RLock()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def enter(self) -> bool:
        with self._lock:
            changed = not self._active
            self._active = True
        if changed:
            log.info("plan mode on")
        return changed

    def leave(self) -> bool:
        with self._lock:
            changed = self._active
            self._active = False
        if changed:
            log.info("plan mode off")
        return changed


PLAN = PlanMode()


def _mutates(tool_name: str) -> bool:
    return tool_name in MUTATING_TOOLS or tool_name == _CONDITIONAL_MUTATING


def hides(tool_name: str) -> bool:
    """Whether the tool is left out of the toolset the model is shown right now."""
    if PLAN.active:
        return _mutates(tool_name)
    if tool_name in ("desktop_inspect", "desktop_act"):   # only while the person has desktop control on
        from src.automation import desktop_guard
        return not desktop_guard.enabled()
    if tool_name == "memory":                         # nothing to offer while the memory is switched off
        from src.memory import hub
        return not hub.enabled()
    if tool_name == "ultracode":                      # only offered to the model while the person has it on
        from src.agent.workflow import runtime
        return not runtime.enabled()
    return tool_name in PLAN_ONLY_TOOLS


def violation(tool_name: str, args: Optional[dict] = None) -> Optional[str]:
    """What to tell the model when plan mode refuses this call; None when it may run."""
    if not PLAN.active:
        return None
    args = args or {}
    if _mutates(tool_name):
        return (f"Plan mode is ON: '{tool_name}' changes things, so it is not available until the plan is "
                "approved. Finish exploring, write the plan, then call exit_plan_mode(plan).")
    if tool_name == "memory" and str(args.get("action", "")).strip().lower() in ("write", "forget"):
        return ("Plan mode is ON: saving or forgetting a memory note changes things. Put it in the plan and "
                "do it after the plan is approved.")
    if tool_name == "subagents" and str(args.get("action", "")).strip().lower() in ("apply", "discard"):
        return ("Plan mode is ON: applying a sub-agent's patch changes files. Describe it in the plan and "
                "do it after the plan is approved.")
    if tool_name == "rooms" and str(args.get("action", "")).strip().lower() in ("run", "resume", "build", "edit", "patch", "create"):
        return ("Plan mode is ON: running, building or editing a room changes things, so it is not available until "
                "the plan is approved. rooms(action='list'|'show') still works; put the room in the plan.")
    if tool_name == "run_subagent":
        role = str(args.get("role", "")).strip().lower()
        if role in ("coder", "docwriter", "browserworker") or str(args.get("isolation", "")).strip().lower() == "worktree":
            return ("Plan mode is ON: only read-only sub-agents (Researcher, Reviewer, Critic) may run now. "
                    "Put the implementation work in the plan.")
    if tool_name in ("run_command",):
        command = str(args.get("command") or args.get("cmd") or "")
        ok, why = readonly_command(command)
        if not ok:
            return (f"Plan mode is ON: this command is not read-only ({why}). Only look around now — "
                    "git status/diff/log, ls/dir, cat/type, grep/rg, find. Put the command in the plan "
                    "instead and run it after the plan is approved.")
    return None


def plan_instructions() -> str:
    """The tail-of-request reminder while planning (volatile, so never in the cached prefix)."""
    return (
        "## PLAN MODE IS ON\n"
        "The person asked for a PLAN before any change. Until they approve it:\n"
        "- You may only look: read_file, grep_search, search_files, list_directory, get_file_outline, "
        "read_git_diff, search_web, read_webpage, and read-only shell commands (git status/diff/log, ls, cat, grep).\n"
        "- Every tool that edits files, runs builds/tests, starts processes or touches the outside world is OFF; "
        "calling one fails.\n"
        "- Explore until you know enough, ask the person with ask_user_questions if something decides the design, "
        "then call exit_plan_mode(plan) with the COMPLETE plan: goal, files to change and how, the order of steps, "
        "how you will verify it, and the risks. The person approves or sends you back.\n"
        "- Do not start the work in this turn and do not describe the plan only in chat: exit_plan_mode is how "
        "the plan reaches the person."
    )
