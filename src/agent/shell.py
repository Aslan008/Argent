"""
Adaptive shell selection for command execution on Windows.

Argent's system prompt tells the model to use PowerShell, but commands were run
through cmd.exe (subprocess shell=True), so PowerShell-only constructs like
`Select-Object` failed with "is not recognized as an internal or external
command". This picks the right shell per command:

- default: PowerShell (matches the prompt; a superset for most dev commands)
- cmd-style `&&` / `||` chains -> cmd.exe (they are invalid in Windows
  PowerShell 5.1, the version shipped with Windows)
- explicit PowerShell cmdlets / syntax -> PowerShell, even if a chain is present

100% accurate shell inference is impossible, so this is "sensible default +
explicit-signal routing"; the command-diagnostics layer covers the rest.
"""

import re

# PowerShell cmdlets (Verb-Noun) and syntax that only work in PowerShell.
_POWERSHELL_HINTS = re.compile(
    r"\b(?:Select|Where|ForEach|Sort|Measure|Format|Out|Get|Set|New|Remove|Add|"
    r"Test|Invoke|Start|Stop|Write|Read|ConvertTo|ConvertFrom|Import|Export|"
    r"Copy|Move|Rename|Clear|Resolve)-\w+"
    r"|\$env:"
    r"|\$\w+\s*=",
    re.IGNORECASE,
)
_POWERSHELL_PARAMS = re.compile(
    r"(?:^|\s)-(?:ErrorAction|First|Last|Recurse|Force|Filter|Path|Name|"
    r"Encoding|Raw)\b",
    re.IGNORECASE,
)
# cmd-style conditional chaining; invalid in Windows PowerShell 5.1.
_CMD_CHAIN = re.compile(r"(?<![|&])(?:&&|\|\|)(?![|&])")


def choose_shell(command: str) -> str:
    """Return 'powershell' or 'cmd' for a Windows command."""
    if _POWERSHELL_HINTS.search(command) or _POWERSHELL_PARAMS.search(command):
        return "powershell"
    if _CMD_CHAIN.search(command):
        return "cmd"
    return "powershell"


def build_command_argv(command: str, shell_kind: str) -> list[str]:
    """Build the argv for subprocess (shell=False) for the chosen shell.

    Passing the command as a single argument avoids the double-escaping that a
    wrapped `-Command "..."` string would suffer. PowerShell output is forced to
    UTF-8 so the byte decoder downstream stays consistent.
    """
    if shell_kind == "powershell":
        wrapped = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + command
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", wrapped]
    return ["cmd", "/c", command]
