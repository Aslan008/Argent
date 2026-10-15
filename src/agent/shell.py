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
            kill_process_tree(process)
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


def descendant_processes(process) -> list:
    """The live descendants of `process`, as psutil handles; [] without psutil.

    Take this BEFORE stopping the parent. Windows never re-parents orphans, but
    `taskkill /T` can only walk a tree whose root still exists, so once the
    parent is gone the grandchildren (the real server behind a `uv`/`npx`/`cmd`
    launcher) can no longer be found from it.
    """
    try:
        import psutil
        return psutil.Process(process.pid).children(recursive=True)
    except Exception:
        return []


def _has_exited(process) -> bool:
    poll = getattr(process, "poll", None)
    if not callable(poll):
        return False
    try:
        return poll() is not None
    except Exception:
        return False


def kill_process_tree(process, descendants=()) -> None:
    """Kill `process` AND everything it started, and reap it. Never blocks long.

    Why the whole tree: on Windows `terminate()` ends only the direct child (a
    launcher, `cmd.exe`, `uv`), leaving the real program alive and holding the
    pipe handles it inherited. Any `read()` pending on those pipes then never
    returns, and - the part that bites - neither does `close()` on them:
    closing a synchronous Windows pipe waits for the read already in progress.
    The symptom is a process that cannot exit, or a stop() that never returns.

    `descendants` are handles taken earlier with `descendant_processes`, for a
    caller that let the parent exit on its own first.

    Order: psutil (descendants, then the root) -> `taskkill /F /T` -> process
    group on POSIX -> plain kill. Each step is best effort; none can raise.
    A root that has already exited costs nothing beyond its stragglers.
    """
    import os

    if process is None:
        return

    try:
        import psutil
    except ImportError:
        psutil = None

    if psutil is not None:
        victims = list(descendants)
        try:
            for child in psutil.Process(process.pid).children(recursive=True):
                if child not in victims:
                    victims.append(child)
        except Exception:
            pass
        for victim in victims:
            try:
                victim.kill()
            except Exception:
                pass

    if _has_exited(process):
        return

    if psutil is not None:
        # A root psutil cannot see (a fake pid, a race with exit) must not stop
        # the fallbacks below from running.
        try:
            psutil.Process(process.pid).kill()
            try:
                process.wait(timeout=2)
            except Exception:
                pass
            return
        except Exception:
            pass

    if os.name == "nt":
        try:
            result = subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                                    capture_output=True, timeout=10)
            if result.returncode == 0:
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass
                return
        except Exception:
            pass
    else:
        # Needs start_new_session=True at spawn. A child that shares OUR group
        # must not be killed this way: killpg would take Argent down with it.
        try:
            import signal
            pgid = os.getpgid(process.pid)
            if pgid != os.getpgid(0):
                os.killpg(pgid, signal.SIGKILL)
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass
                return
        except Exception:
            pass

    try:
        process.kill()
        process.wait(timeout=2)
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


import base64


def _try_wrap_inline_python_ps(cmd: str) -> str | None:
    """If command is a standalone python/py -c invocation, wrap code via Base64 stdin
    to make it immune to PowerShell's parser, quote-stripping, and newline breakage."""
    s = cmd.strip()

    # Check for unquoted shell pipe or redirection operators outside quotes
    in_single = False
    in_double = False
    escaped = False
    for i, c in enumerate(s):
        if escaped:
            escaped = False
            continue
        if c == '\\':
            escaped = True
            continue
        if c == '"' and not in_single:
            in_double = not in_double
            continue
        if c == "'" and not in_double:
            in_single = not in_single
            continue
        if not in_single and not in_double:
            if c in ('|', '>', '<') or s[i:i+2] in ('&&', '||'):
                return None

    m = re.match(
        r'^(?P<prefix>(?:\s*\$env:[\w\.\-]+\s*=\s*(?:\'[^\']*\'|"[^"]*"|[^\s;]+)\s*;\s*)*)'
        r'(?P<amp>&?\s*)'
        r'(?P<py>(?:"[^"]*python(?:\d(?:\.\d+)?)?\.exe")|(?:\'[^\']*python(?:\d(?:\.\d+)?)?\.exe\')|[^\s"\'|&;]*python(?:\d(?:\.\d+)?)?\.exe|python\d*(?:\.\d+)?|py)'
        r'(?P<flags>(?:\s+-(?:[a-zA-Z0-9_-]+(?:\s+(?!--)[^\s-]+)?))*?)'
        r'\s+-c\s+(?P<rest>.*)$',
        s,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None

    prefix = m.group("prefix") or ""
    amp = m.group("amp").strip()
    py_bin = m.group("py").strip()
    flags = m.group("flags").strip()
    flags_str = f" {flags}" if flags else ""
    rest = m.group("rest").strip()
    if not rest:
        return None

    code = None
    extra_args_str = ""

    if rest.startswith('"'):
        end_idx = -1
        i = 1
        n = len(rest)
        while i < n:
            if rest[i] == '\\':
                i += 2
                continue
            if rest[i] == '"':
                end_idx = i
                break
            i += 1
        if end_idx != -1:
            raw_code = rest[1:end_idx]
            clean_chars = []
            ci = 0
            cn = len(raw_code)
            while ci < cn:
                if raw_code[ci] == '\\' and ci + 1 < cn and raw_code[ci+1] in ('"', '\\'):
                    clean_chars.append(raw_code[ci+1])
                    ci += 2
                else:
                    clean_chars.append(raw_code[ci])
                    ci += 1
            code = "".join(clean_chars)
            trailing = rest[end_idx+1:].strip()
            if trailing:
                extra_args_str = " " + trailing
    elif rest.startswith("'"):
        end_idx = rest.find("'", 1)
        if end_idx != -1:
            code = rest[1:end_idx]
            trailing = rest[end_idx+1:].strip()
            if trailing:
                extra_args_str = " " + trailing
    else:
        parts = rest.split(None, 1)
        code = parts[0]
        if len(parts) > 1:
            extra_args_str = " " + parts[1].strip()

    if code is None:
        return None

    # Only wrap when the inline script contains characters that trigger
    # PowerShell 5.1 quote-stripping or parser bugs:
    # 1. Newlines (PS treats each line as a separate statement)
    # 2. Nested double quotes (PS strips unescaped quotes for native apps)
    # 3. Escaped double quotes \"
    # 4. Outer single-quoted code block (PS does not treat single quotes as arg delimiters for native apps)
    # 5. Environment variable prefix or interpreter flags where PS parser would split statements
    needs_wrap = (
        "\n" in code
        or '"' in code
        or '\\"' in rest
        or rest.startswith("'")
        or bool(prefix)
        or bool(flags)
    )
    if not needs_wrap:
        return None

    b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
    needs_amp = bool(amp) or py_bin.startswith(('"', "'"))
    py_exec = f"& {py_bin}" if needs_amp else py_bin
    return f"{prefix}$__code = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{b64}')); $__code | {py_exec}{flags_str} -{extra_args_str}"



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
        cmd = command
        # Inline python -c commands: wrap via Base64 stdin so PowerShell's argument
        # parser never strips quotes, breaks on semicolons, or splits multiline scripts.
        ps_inline = _try_wrap_inline_python_ps(cmd)
        if ps_inline:
            cmd = ps_inline
        else:
            # A command that starts with a quoted path — e.g. `"C:\dir\app.bat" args`
            # — is treated by PowerShell's `-Command` mode as a *string literal*,
            # not something to execute, so PS just prints the path.  The call
            # operator `&` forces execution.  We ONLY prepend it when the command
            # starts with a quote; unconditional `&` breaks statements (`$x=…`,
            # `if …`) and doubles an already-present `& …`.
            stripped = cmd.lstrip()
            if stripped.startswith(('"', "'")):
                cmd = "& " + cmd
        wrapped = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + cmd
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", wrapped]
    return f'cmd /s /c "{command}"'


