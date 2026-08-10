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
import subprocess


def run_text(*args, **kwargs) -> "subprocess.CompletedProcess":
    """``subprocess.run`` in text mode, with a timeout that actually fires.

    Crash-proof decoding: many Windows tools (tasklist, git, MSBuild/dotnet,
    node, ...) emit output in the OEM/locale code page, not UTF-8. Under
    Python's UTF-8 mode a plain ``text=True`` capture decodes stdout/stderr as
    strict UTF-8, so the first stray byte (e.g. 0xFF) raises UnicodeDecodeError
    *inside subprocess's reader thread* and crashes it. ``errors='replace'``
    degrades a bad byte instead of throwing.

    Two more guards, both learned from a three-and-a-half-hour hang:

    * **stdin is closed.** A captured child that asks a question — `npx` with
      "Ok to proceed? (y)" — inherits the console and waits for an answer
      nobody can give, because its prompt is inside the captured pipe and
      invisible. With DEVNULL it reads EOF and exits.
    * **The timeout kills the whole tree.** ``subprocess.run(timeout=…)`` kills
      only the direct child; under ``shell=True`` on Windows that is cmd.exe,
      and its grandchild keeps the stdout pipe open, so the communicate() that
      follows the kill blocks forever. A ten-second timeout produced a hang of
      12489 seconds, ended by Ctrl+C.
    """
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    kwargs["text"] = True

    payload = kwargs.pop("input", None)
    if payload is not None:
        kwargs["stdin"] = subprocess.PIPE          # a caller that means to feed it
    elif kwargs.get("capture_output") or kwargs.get("stdout") is not None:
        kwargs.setdefault("stdin", subprocess.DEVNULL)

    timeout = kwargs.pop("timeout", None)
    if timeout is None:
        return subprocess.run(*args, input=payload, **kwargs) if payload is not None \
            else subprocess.run(*args, **kwargs)

    if kwargs.pop("capture_output", False):
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)

    with subprocess.Popen(*args, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input=payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(process)
            # A short second wait: the tree is gone, so this returns promptly —
            # but never wait unbounded again, which is the original bug.
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout = stderr = ""
            raise subprocess.TimeoutExpired(process.args, timeout,
                                            output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(process.args, process.returncode,
                                           stdout, stderr)


def _kill_tree(process) -> None:
    """Kill the process AND its descendants.

    On Windows a shell=True child is cmd.exe; killing it orphans the real
    program, which keeps the inherited stdout pipe open and hangs any further
    read. taskkill /T walks the tree.
    """
    import os

    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, timeout=10)
            return
        except Exception:
            pass
    try:
        process.kill()
    except Exception:
        pass

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


def build_command_argv(command: str, shell_kind: str):
    """What to hand subprocess (shell=False) for the chosen shell.

    A LIST for PowerShell, a raw command-line STRING for cmd.exe — because the
    two disagree about how an embedded quote is written and only one of them can
    be expressed as a list.

    subprocess builds a Windows command line from a list with list2cmdline,
    which escapes an inner `"` as `\\"`. PowerShell accepts that form. cmd.exe
    does not: backslash-escaping is a C-runtime convention, so cmd passes the
    backslash through and the quote reads as a delimiter. Measured, this

        cd C:\\proj && python -c "import sys; print('ok')"

    reached Python as argv[2] == '"import' and died with "unterminated string
    literal" — the model then blamed itself and started writing temp files.

    `/s /c "…"` is the form that survives: with /s cmd strips exactly the first
    and last quote and takes the rest verbatim. Without it a command that BEGINS
    with a quote — `"C:\\Program Files\\...\\python.exe" -c ...` — loses the
    quotes around its own executable path and fails to launch.
    """
    if shell_kind == "powershell":
        wrapped = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + command
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", wrapped]
    return f'cmd /s /c "{command}"'
