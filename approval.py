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
"""

import re

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
    return "safe", []


def is_destructive_command(command: str) -> bool:
    """Heuristic check whether a shell command can destroy data or processes."""
    return assess_command_risk(command)[0] != "safe"


def command_grant_key(command: str) -> str | None:
    """Session-grant key for a shell command: its first token (the executable)."""
    if not command or not command.strip():
        return None
    return command.strip().split()[0].lower()


def request_approval(action: str, *, destructive: bool = False,
                     grant_key: str | None = None) -> bool:
    """Single decision point for every gated tool action.

    Returns True when the action may proceed. Ctrl+C / closed stdin during
    the prompt counts as denial.
    """
    from ui import console

    if not destructive:
        if grant_key and grant_key in _session_grants:
            console.print(f"[dim]✓ Авто-одобрено (сессионное разрешение '{grant_key}'): {action}[/dim]")
            log.info("Approved via session grant '%s': %s", grant_key, action)
            return True
        if _policy == POLICY_AUTO:
            console.print(f"[dim]✓ Авто-одобрено (автономный режим): {action}[/dim]")
            log.info("Approved via AUTO policy: %s", action)
            return True

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
                log.info("Denied by user: %s", action)
                return False
            if choice.startswith(_ALWAYS_PREFIX):
                _session_grants.add(grant_key)
                log.info("Session grant added: '%s'", grant_key)
            return True

        approved = questionary.confirm("Разрешить это действие?").ask()
        if not approved:
            log.info("Denied by user: %s", action)
        return bool(approved)
    except (KeyboardInterrupt, EOFError):
        log.info("Approval prompt interrupted — treating as denial: %s", action)
        return False
