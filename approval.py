"""
Centralized user-approval gate for dangerous tool actions.

Every tool that needs explicit consent routes through request_approval().
The active policy decides what happens:

- POLICY_ASK  (default): prompt the user interactively for every gated action.
- POLICY_AUTO (autonomous mode): auto-approve safe actions, still prompt for
  destructive ones.

The user can also grant a session-wide allowance for a command family
(e.g. "always allow 'git' commands this session") directly from the prompt.
Destructive actions are never auto-approved and never covered by grants.

Both shortcuts are consent from a human who is present. A backend marked
`unattended` (see _nobody_is_watching) turns them off: whatever the interactive
session pre-authorised, a scheduled run has to ask its own gate.
"""

import re
import threading

import questionary

from logger import get_logger

log = get_logger("approval")

POLICY_ASK = "ask"
POLICY_AUTO = "auto"

_policy = POLICY_ASK
_session_grants: set[str] = set()

# Patterns that mark a shell command as destructive (case-insensitive).
# Kept deliberately broad: a false positive only costs one extra confirmation,
# a false negative costs user data.
_DESTRUCTIVE_PATTERNS = [
    r"\b(rm|del|erase|rd|rmdir|ri)\b",
    r"\bremove-item\b",
    r"\bclear-(content|item)\b",
    r"\bformat(-volume)?\b",
    r"\b(diskpart|mkfs)\b",
    r"\bgit\s+(reset\s+--hard|clean\b|push\b.*--force|checkout\s+--\s)",
    r"\breg\s+delete\b",
    r"\b(shutdown|taskkill|stop-process|stop-computer|restart-computer)\b",
]
_DESTRUCTIVE_RE = re.compile("|".join(_DESTRUCTIVE_PATTERNS), re.IGNORECASE)

# Interpreters invoked with inline code (python -c, node -e, perl/ruby -e,
# powershell -Command …). The destructive patterns above scan the literal
# command string, so a delete hidden inside `python -c "import os; os.remove(...)"`
# reads as "safe" — and in POLICY_AUTO / vibe mode a "safe" action is
# auto-approved with no prompt. We can't parse arbitrary embedded code, so we
# treat any inline-code interpreter invocation as "warn": it still runs, but
# never silently. A plain `python script.py` (no inline flag) is unaffected.
# Note: only INLINE code (-c / -e / -Command / -EncodedCommand) qualifies.
# `python script.py` and `python -m pytest` name a visible, inspectable target
# and stay "safe" so vibe mode doesn't prompt on every test run.
_INLINE_CODE_INTERPRETER_RE = re.compile(
    r"\b(?:python\d?|py|node|deno|bun|perl|ruby|php|osascript)\b[^\n|&;]*\s-(?:c|e)\b"
    r"|\b(?:powershell|pwsh)\b[^\n|&;]*\s-(?:c|e|enc|encodedcommand)\w*\b"
    r"|\b(?:sh|bash|zsh)\b[^\n|&;]*\s-c\b",
    re.IGNORECASE,
)

# Obfuscation indicators. Whether a command is destructive is UNDECIDABLE when
# the command text is assembled at runtime — `$a="Remove"; $b="-Item"; & "$a$b"`
# defeats any static scanner, regex or AST alike, because the dangerous token
# simply does not exist until the shell builds it. So instead of trying to see
# through the obfuscation, we treat the obfuscation ITSELF as the risk signal:
# a command that builds or evals another command never runs unattended.
# In POLICY_ASK nothing changes (everything already prompts); this closes the
# hole in POLICY_AUTO / vibe, where a "safe" verdict means silent execution.
_OBFUSCATION_PATTERNS = [
    # Invoke-Expression / iex — the PowerShell "eval".
    (r"\b(?:invoke-expression|iex)\b", "evaluates a dynamically built command (Invoke-Expression)"),
    # Call operator on a string/variable: & "$a$b", & $cmd
    (r"&\s*[\"'$]", "invokes a command built from a variable or string"),
    # Bash/POSIX eval.
    (r"\beval\b", "evaluates a dynamically built command (eval)"),
    # Decoding into execution: FromBase64String, base64 -d | sh
    (r"\bfrombase64string\b", "decodes a base64 payload before running it"),
    (r"\bbase64\b[^|&;]*-{1,2}d(?:ecode)?\b", "decodes a base64 payload before running it"),
    # String concatenation inside a command invocation: ("Rem"+"ove-Item")
    (r"[\"'][^\"'\n]*[\"']\s*\+\s*[\"'][^\"'\n]*[\"']", "assembles a command from concatenated strings"),
]
_OBFUSCATION_RE = [(re.compile(p, re.IGNORECASE), why) for p, why in _OBFUSCATION_PATTERNS]

# Catastrophic / irreversible / system-wide commands. Each carries a human
# reason. Broad on purpose: in "warn" mode a false positive is one confirmation;
# in "block" mode it refuses outright, so the gate stays opt-in for that tier.
_CATASTROPHIC_PATTERNS = [
    (r"\brm\s+(?:-\S+\s+)*-\S*[rf]\S*[rf]\S*\s+(?:-\S+\s+)*['\"]?(?:/|~|/\*|\*|\$HOME)(?:\s|/|$|['\"])",
     "recursive force-delete of a root, home, or wildcard path"),
    (r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&?\s*\}\s*;\s*:", "fork bomb"),
    (r"\bmkfs(?:\.\w+)?\b", "filesystem format (mkfs)"),
    (r"\bdd\b[^|&;]*\bof=/dev/", "raw write to a block device (dd of=/dev/…)"),
    (r">\s*/dev/sd[a-z]", "overwrite of a block device"),
    (r"\bformat(?:-volume)?\b[^|&;]*\b[a-zA-Z]:", "drive format"),
    (r"\bdel\b[^|&;]*\s/[sq]\b[^|&;]*\b[a-zA-Z]:\\?(?:\s|$)", "recursive delete at a drive root (del /s /q)"),
    (r"\bremove-item\b[^|&;]*-recurse[^|&;]*-force[^|&;]*\b[a-zA-Z]:\\?(?:\s|\"|$)",
     "recursive force delete at a drive root"),
    (r"\bdiskpart\b", "disk partitioning (diskpart)"),
    (r"\bcipher\b\s+/w", "secure disk wipe (cipher /w)"),
    (r"\b(?:curl|wget|iwr|invoke-webrequest)\b[^|]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|pwsh|powershell|python\d?|iex|invoke-expression)\b",
     "piping downloaded content straight into a shell"),
    (r"\bnetsh\s+advfirewall\s+set\s+\w+\s+state\s+off\b", "disabling the firewall"),
]
_CATASTROPHIC_RE = [(re.compile(p, re.IGNORECASE), why) for p, why in _CATASTROPHIC_PATTERNS]

_ALWAYS_PREFIX = "✅ Да, и всегда разрешать"


def set_policy(policy: str):
    """Switch the active approval policy (POLICY_ASK / POLICY_AUTO)."""
    global _policy
    if policy not in (POLICY_ASK, POLICY_AUTO):
        raise ValueError(f"Unknown approval policy: {policy}")
    if policy != _policy:
        log.info("Approval policy changed: %s -> %s", _policy, policy)
    _policy = policy


def get_policy() -> str:
    return _policy


def clear_session_grants():
    _session_grants.clear()


def get_session_grants() -> set[str]:
    return set(_session_grants)


def assess_command_risk(command: str):
    """Grade a shell command's risk.

    Returns (level, reasons) where level is 'safe' | 'warn' | 'block':
    - 'block': catastrophic / irreversible / system-wide (with specific reasons).
    - 'warn':  destructive but recoverable/scoped (deletes files, kills procs…).
    - 'safe':  everything else.
    """
    if not command or not command.strip():
        return "safe", []
    reasons = [why for rx, why in _CATASTROPHIC_RE if rx.search(command)]
    if reasons:
        return "block", reasons
    if _DESTRUCTIVE_RE.search(command):
        return "warn", ["deletes/overwrites files or stops processes"]
    if _INLINE_CODE_INTERPRETER_RE.search(command):
        return "warn", ["runs inline interpreter code — effects can't be inspected"]
    obfuscated = [why for rx, why in _OBFUSCATION_RE if rx.search(command)]
    if obfuscated:
        return "warn", obfuscated
    return "safe", []


def is_destructive_command(command: str) -> bool:
    """Heuristic check whether a shell command can destroy data or processes."""
    return assess_command_risk(command)[0] != "safe"


def command_grant_key(command: str) -> str | None:
    """Session-grant key for a shell command: its first token (the executable)."""
    if not command or not command.strip():
        return None
    return command.strip().split()[0].lower()


# The interactive decision is pluggable so a non-TTY transport (the GUI's
# WebSocket) can answer approvals without deadlocking on questionary. A backend
# takes (action, destructive, grant_key) and returns one of: 'deny' | 'once' |
# 'always'. It is stored per-thread, so the terminal thread and a server worker
# thread never interfere; unset -> the terminal prompt below.
_local = threading.local()


def set_approval_backend(fn) -> None:
    """Install the interactive-decision backend for the CURRENT thread."""
    _local.backend = fn


def reset_approval_backend() -> None:
    _local.backend = None


def _active_backend():
    return getattr(_local, "backend", None) or _terminal_decision


def _nobody_is_watching() -> bool:
    """True when the installed backend speaks for an unattended run.

    Session grants and POLICY_AUTO are pre-authorisations a PRESENT human gave
    for the work in front of them. They say nothing about what a scheduled job
    may do at 04:00, and a backend that exists to refuse everything is no gate
    at all if those shortcuts answer before it is asked. A backend declares
    itself by carrying `unattended = True`.
    """
    return bool(getattr(_active_backend(), "unattended", False))


def _terminal_decision(action: str, destructive: bool, grant_key: str | None) -> str:
    """Interactive TTY prompt. Returns 'deny' | 'once' | 'always'."""
    from ui import console

    style = "bold red" if destructive else "bold yellow"
    icon = "⚠️ " if destructive else ""
    console.print(f"\n[{style}]{icon}Агент запрашивает разрешение: {action}[/{style}]")

    try:
        if grant_key and not destructive:
            always_label = f"{_ALWAYS_PREFIX} '{grant_key}' в этой сессии"
            choice = questionary.select(
                "Разрешить это действие?",
                choices=["✅ Да (однократно)", always_label, "❌ Нет"],
            ).ask()
            if choice is None or choice.startswith("❌"):
                return "deny"
            return "always" if choice.startswith(_ALWAYS_PREFIX) else "once"

        approved = questionary.confirm("Разрешить это действие?").ask()
        return "once" if approved else "deny"
    except (KeyboardInterrupt, EOFError):
        return "deny"


def request_approval(action: str, *, destructive: bool = False,
                     grant_key: str | None = None) -> bool:
    """Single decision point for every gated tool action.

    Returns True when the action may proceed. Ctrl+C / closed stdin / a client
    denial counts as denial.
    """
    from ui import console

    if not destructive and not _nobody_is_watching():
        if grant_key and grant_key in _session_grants:
            console.print(f"[dim]✓ Авто-одобрено (сессионное разрешение '{grant_key}'): {action}[/dim]")
            log.info("Approved via session grant '%s': %s", grant_key, action)
            return True
        if _policy == POLICY_AUTO:
            console.print(f"[dim]✓ Авто-одобрено (автономный режим): {action}[/dim]")
            log.info("Approved via AUTO policy: %s", action)
            return True

    decision = _active_backend()(action, destructive, grant_key)

    if decision == "always" and grant_key and not destructive:
        _session_grants.add(grant_key)
        log.info("Session grant added: '%s'", grant_key)
        return True
    if decision == "once":
        return True
    log.info("Denied by user: %s", action)
    return False
