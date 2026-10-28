"""
Centralized user-approval gate for dangerous tool actions.

Every tool that needs explicit consent routes through request_approval().
The active policy decides what happens:

- POLICY_ASK  (default): prompt the user interactively for every gated action.
- POLICY_AUTO (autonomous mode, /vibe and /auto): answer by itself. It refuses
  what cannot be put back — a delete outside the project, Argent's own state
  (`.argent/`), a command built at run time, and taking the machine down — and
  allows the rest. A delete inside the project is allowed only after the files
  are copied to `.argent/trash` (see `argent.trash` and `/trash`), so "the agent
  may delete my file" never means "my file is gone". A command whose effect
  merely cannot be read (`python -c`, `node -e`) runs with a line in the
  transcript saying so, because a mode that refuses those refuses most of what
  the agent actually does.

The user can also grant a session-wide allowance for a command family
(e.g. "always allow 'git' commands this session") directly from the prompt.
Destructive actions are never auto-approved and never covered by grants.

Both shortcuts are consent from a human who is present. A backend marked
`unattended` (see _nobody_is_watching) turns them off: whatever the interactive
session pre-authorised, a scheduled run has to ask its own gate.
"""

import os
import re
import sys
import threading
import ast
from dataclasses import dataclass
from pathlib import Path

import questionary

from argent.logger import get_logger

log = get_logger("approval")

POLICY_ASK = "ask"
POLICY_AUTO = "auto"

_policy = POLICY_ASK
_session_grants: set[str] = set()


# --- What a risky command is risky ABOUT ---------------------------------------
#
# One word — "destructive" — was answering two different questions: whether a
# human should be asked, and whether an autonomous run may go ahead. They come
# apart. Deleting the person's files needs a human whatever the mode is; a command
# whose effect merely cannot be read (`python -c`, `node -e`, `powershell
# -Command`) has to run in /vibe, or the mode refuses nearly everything the agent
# does: `python -c "import sys; print(sys.version)"` was auto-denied.
NONE = "none"
DELETES = "deletes"            # destroys or overwrites data on disk
KILLS = "kills"                # stops processes
POWER = "power"                # shuts the machine down / restarts it
UNVETTED = "unvetted"          # code that WAS readable and holds no deletion
UNDECIDABLE = "undecidable"    # assembled or evaluated at run time: nothing can be said
IRREVERSIBLE = "irreversible"  # an action someone declared destructive, with no command to read
CATASTROPHIC = "catastrophic"

# Loudest first: a chain is judged by the worst thing in it.
_KIND_ORDER = (CATASTROPHIC, DELETES, POWER, UNDECIDABLE, KILLS, UNVETTED)

_KIND_LABELS = {
    DELETES: "удаление файлов",
    IRREVERSIBLE: "необратимое действие",
    POWER: "выключение машины",
    UNDECIDABLE: "команда собирается в рантайме",
    CATASTROPHIC: "необратимая операция",
}


@dataclass(frozen=True)
class Risk:
    """A verdict with two dimensions: how loudly (`level`) and about what (`kind`)."""

    level: str        # "safe" | "warn" | "block"
    kind: str         # one of the constants above
    reasons: tuple = ()


def _kind_label(kind: str) -> str:
    return _KIND_LABELS.get(kind, kind)

# --- Inline python (`python -c "..."`) -------------------------------------------
#
# In POLICY_AUTO / vibe mode a command graded "safe" runs with no prompt, so
# "safe" has to mean something. The previous check parsed the code and rejected
# a list of dangerous names. A blacklist of names is a game the snippet can
# always win by spelling the same effect differently, and it did: `subprocess.run`,
# `shutil.copyfile`, `os.makedirs`, `os.startfile`, `getattr(os, "rem" + "ove")`
# and `Path(p).open(mode)` all graded "safe".
#
# So the question is inverted: code is safe only when EVERYTHING in it is
# recognised. A snippet that imports a stdlib module with side effects, makes a
# dynamic lookup, touches a private or dunder attribute, or writes, is "warn" and
# asks once. Common one-liners (version probes, "does X import?", json/path
# inspection, torch checkpoint peeks) keep running unprompted.
#
# Imports are the one place that cannot be a plain allowlist: `import filelock`,
# `import numpy`, `import torch` are everyday probes (and reported annoyances
# when they prompted), and PyPI cannot be enumerated. The stdlib can — so the
# default-deny lives there. A stdlib module not listed below (subprocess, shutil,
# socket, ctypes, tempfile, importlib, pickle, …, and any module a newer Python
# adds) makes the snippet "warn". An installed third-party package is trusted
# like any code the user chose to install, except the ones that exist to act on
# the outside world.

# Stdlib modules that add no capability by themselves. `os`, `sys` and `pathlib`
# are here but are policed attribute by attribute (see below).
_SAFE_STDLIB_IMPORTS = frozenset({
    "abc", "array", "ast", "base64", "binascii", "bisect", "calendar", "cmath",
    "collections", "colorsys", "contextlib", "copy", "csv", "dataclasses",
    "datetime", "decimal", "difflib", "enum", "fnmatch", "fractions", "functools",
    "glob", "hashlib", "heapq", "html", "itertools", "json", "keyword", "math",
    "numbers", "pprint", "random", "re", "secrets", "statistics", "string",
    "struct", "textwrap", "time", "token", "tokenize", "typing", "unicodedata",
    "uuid", "zoneinfo",
    "os", "sys", "platform", "pathlib", "posixpath", "ntpath",
})
# Third-party packages whose purpose is to reach out: network, other processes,
# the desktop, cloud and model APIs.
_CAPABILITY_PACKAGES = frozenset({
    "requests", "httpx", "aiohttp", "urllib3", "websockets", "websocket",
    "paramiko", "fabric", "pexpect", "sh", "plumbum", "psutil", "docker", "boto3",
    "botocore", "pyautogui", "pynput", "keyboard", "mouse", "pywinauto",
    "win32api", "win32com", "win32gui", "win32process", "win32file", "pywintypes",
    "pythoncom", "wmi", "playwright", "selenium", "openai", "anthropic", "ollama",
})
# Dotted stdlib imports allowed although their top-level package is not.
_SAFE_INLINE_SUBMODULES = frozenset({"importlib.metadata"})

# Attribute names that write, delete, spawn or look code up by name. Checked on
# EVERY attribute, whatever it hangs off, so aliasing (`import os as o`,
# `f = os.remove`) changes nothing. Only what an ALLOWED import can reach matters
# here: `subprocess`, `shutil`, `socket`… are refused at the import itself, and
# their names (run, copy, connect…) are left out so that `d.copy()` and
# `session.run(...)` stay unremarkable.
_DANGEROUS_PYTHON_ATTRS = frozenset({
    # delete / move / overwrite
    "remove", "removedirs", "unlink", "rmdir", "rmtree", "truncate", "ftruncate",
    "write", "writelines", "write_text", "write_bytes", "move", "rename", "renames",
    "replace", "drop", "delete", "format",
    # create / change
    "mkdir", "makedirs", "mkfifo", "mknod", "touch", "chmod", "lchmod", "chown",
    "lchown", "utime", "symlink", "symlink_to", "hardlink_to", "link", "copy2",
    "copyfile", "copyfileobj", "copymode", "copystat", "copytree", "mktemp",
    "mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryFile", "putenv", "unsetenv",
    # serialise to disk (numpy / torch / pandas)
    "save", "savez", "savez_compressed", "savetxt", "tofile", "to_csv", "to_pickle",
    "to_json", "to_excel", "to_parquet", "to_feather", "to_hdf", "to_sql",
    # processes
    "system", "popen", "Popen", "check_call", "check_output", "getoutput",
    "getstatusoutput", "startfile", "kill", "killpg", "terminate", "fork", "forkpty",
    "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp",
    "spawnvpe", "setuid", "setgid",
    # network
    "urlopen", "urlretrieve",
    # the back doors to every module refused at import, and to code by name
    "modules", "import_module", "unsafe_load", "attrgetter", "methodcaller",
})
# `platform.system()` is a harmless query; `os.system(cmd)` is not. The call
# with no arguments is the former; anything else about the name is the latter.
_ARGLESS_OK_ATTRS = frozenset({"system"})

# Bare names that reach everything the attribute checks exist to stop.
_DANGEROUS_PYTHON_BUILTINS = frozenset({
    "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "breakpoint", "open",
})

# Dunders that are plain metadata; every other `__x__` (`__class__.__bases__`,
# `__globals__`, `__builtins__`…) is how a sandbox is left.
_SAFE_DUNDERS = frozenset({"__name__", "__version__", "__file__", "__doc__",
                           "__class__", "__module__", "__qualname__"})

_MODE_CHARS = set("rwxabt+U")
_WRITE_MODE_CHARS = set("wax+")


def _is_write_mode(value) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= 4
            and set(value) <= _MODE_CHARS and bool(set(value) & _WRITE_MODE_CHARS))


def _extract_inline_python_code(command: str) -> str | None:
    """Extract the inline code string from a python -c / py -c command."""
    m = re.search(
        r"\b(?:python\d?|py)\b[^\n|&;]*\s-c\s+(.*)$",
        command,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    rest = m.group(1).strip()
    m_quote = re.match(r"^([\"'])(.*?)\1(?:\s+[0-9]?>.*)?$", rest, re.DOTALL)
    if m_quote:
        return m_quote.group(2)
    m_plain = re.match(r"^([^\s\"']+)(?:\s+[0-9]?>.*)?$", rest)
    if m_plain:
        return m_plain.group(1)
    return rest


class _InlinePythonAudit(ast.NodeVisitor):
    """Walks a snippet and records why it is not demonstrably read-only."""

    def __init__(self):
        self.problems: list[str] = []
        self._call_funcs: set[int] = set()     # Attribute / Name nodes that are being called

    def flag(self, why: str) -> None:
        self.problems.append(why)

    # -- imports ----------------------------------------------------------
    @staticmethod
    def _import_ok(module: str) -> bool:
        if module in _SAFE_INLINE_SUBMODULES:
            return True
        top = module.split(".")[0]
        if top in _SAFE_STDLIB_IMPORTS:
            return True
        return top not in sys.stdlib_module_names and top not in _CAPABILITY_PACKAGES

    def visit_Import(self, node):
        for alias in node.names:
            if not self._import_ok(alias.name):
                self.flag(f"imports '{alias.name}'")

    def visit_ImportFrom(self, node):
        module = node.module or ""
        if node.level or not self._import_ok(module):
            self.flag(f"imports from '{'.' * node.level}{module}'")
            return
        for alias in node.names:
            if alias.name == "*":
                self.flag("star import")
            elif alias.name in _DANGEROUS_PYTHON_ATTRS or alias.name.startswith("_"):
                self.flag(f"imports '{alias.name}'")

    # -- names and attributes --------------------------------------------
    def visit_Name(self, node):
        if node.id in _DANGEROUS_PYTHON_BUILTINS and id(node) not in self._call_funcs:
            self.flag(f"refers to '{node.id}'")

    def visit_Attribute(self, node):
        attr = node.attr
        if attr.startswith("_") and attr not in _SAFE_DUNDERS:
            self.flag(f"touches private attribute '{attr}'")
        elif attr in _DANGEROUS_PYTHON_ATTRS and id(node) not in self._call_funcs:
            self.flag(f"uses '.{attr}'")
        elif attr == "open" and id(node) not in self._call_funcs:
            self.flag("takes `.open` without calling it where it can be vetted")
        self.generic_visit(node)

    # -- calls ------------------------------------------------------------
    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _DANGEROUS_PYTHON_BUILTINS:
                if func.id == "open":
                    self._check_open(node, mode_index=1)
                else:
                    self.flag(f"calls '{func.id}'")
                self._call_funcs.add(id(func))
        elif isinstance(func, ast.Attribute):
            if func.attr == "open":
                self._check_open(node, mode_index=None)
                self._call_funcs.add(id(func))
            elif func.attr in _ARGLESS_OK_ATTRS and not node.args and not node.keywords:
                self._call_funcs.add(id(func))
        else:
            # `f()()`, `d["k"](x)`: the callee is computed, so it cannot be vetted.
            self.flag("calls a computed callable")
        self.generic_visit(node)

    def _check_open(self, node, mode_index):
        """`open(path)` reads. A write mode, or a mode that cannot be read off the
        source, does not. `.open(...)` on a path-like keeps positions unknown, so
        any argument that is not a plain string constant refuses it."""
        mode_nodes = [kw.value for kw in node.keywords if kw.arg == "mode"]
        if mode_index is not None:
            if len(node.args) > mode_index:
                mode_nodes.append(node.args[mode_index])
        else:
            mode_nodes.extend(node.args)
            mode_nodes.extend(kw.value for kw in node.keywords if kw.arg != "encoding"
                              and kw.arg != "errors" and kw.arg != "newline")
        for value in mode_nodes:
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                self.flag("opens a file with a mode it cannot read off the source")
            elif _is_write_mode(value.value):
                self.flag("opens a file for writing")

    # -- subscripts of calls etc. are handled by generic_visit -------------


def _python_source_is_safe(code: str) -> bool:
    """True when python source is demonstrably read-only.

    Every import, name, attribute and call is checked against what is known to be
    harmless (see the notes above). Anything not recognised makes it unsafe: the
    caller then treats it as "warn". A syntax error is unsafe too — if it cannot
    be read, it cannot be vetted.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    audit = _InlinePythonAudit()
    audit.visit(tree)
    return not audit.problems


def _is_safe_inline_python(command: str) -> bool:
    """True when a `python -c` command is demonstrably read-only code."""
    code = _extract_inline_python_code(command)
    return code is not None and _python_source_is_safe(code)


# --- Programs fed through stdin ----------------------------------------------
# `python -c` is not the only way to run text the gate never read: a heredoc
# (`python - <<'EOF'`), a PowerShell here-string (`@'…'@ | python -`) and a pipe
# (`echo "…" | python`) all hand an interpreter a program through stdin, and
# each of them graded "safe" because the destructive patterns look at command
# names, not at what a program does. Python heredocs and here-strings are vetted
# like `-c` (the source is right there in the command); any other program on
# stdin is unreadable, so it asks.
_PYTHON_FRONT = r"(?:sudo[ \t]+)?(?:python\d?(?:\.\d+)?|py)(?:\.exe)?(?:[ \t]+-[A-Za-z0-9.]+)*"
_HEREDOC_PYTHON_RE = re.compile(
    r"(?:^|(?<=[;&|\n]))[ \t]*" + _PYTHON_FRONT +
    r"[ \t]+-?[ \t]*<<-?[ \t]*(?P<q>['\"]?)(?P<tag>\w+)(?P=q)[^\n]*\n"
    r"(?P<body>.*?)\n[ \t]*(?P=tag)[ \t]*(?=\n|$)",
    re.IGNORECASE | re.DOTALL,
)
_HERESTRING_PYTHON_RE = re.compile(
    r"@(?P<q>['\"])[ \t]*\n(?P<body>.*?)\n(?P=q)@[ \t]*\|[ \t]*" + _PYTHON_FRONT +
    r"(?:[ \t]+-)?(?=[ \t]*(?:$|[;&|\n]))",
    re.IGNORECASE | re.DOTALL,
)


def _vet_stdin_python(command: str):
    """Cut vetted python heredocs / here-strings out of `command`.

    Returns (what is left to assess, Risk | None). A Risk means a program fed
    through stdin was not demonstrably read-only — unless it plainly deletes
    something, in which case it is graded as deletion, not as "unknown".
    The cut matters: a heredoc body would otherwise be split into "commands" at
    its newlines and judged line by line, and an apostrophe in a comment would
    read as an unclosed quote.
    """
    risk = None

    def _cut(match):
        nonlocal risk
        body = match.group("body")
        if not _python_source_is_safe(body):
            if _code_has_destruction(body):
                risk = Risk("warn", DELETES,
                            ("python fed through stdin that deletes or overwrites files",))
            else:
                risk = Risk("warn", UNVETTED,
                            ("feeds python through stdin — its effect can't be inspected",))
        return "\n"

    command = _HEREDOC_PYTHON_RE.sub(_cut, command)
    command = _HERESTRING_PYTHON_RE.sub(_cut, command)
    return command, risk


_STDIN_INTERPRETER_RE = re.compile(
    r"^(?:python\d?(?:\.\d+)?|py|node|deno|bun|perl|ruby|php|bash|sh|zsh|pwsh|powershell)(?:\.exe)?$",
    re.IGNORECASE,
)
# Flags that ask the interpreter about itself and never read a program.
_INTERPRETER_INFO_FLAGS = frozenset({"--version", "-V", "-VV", "--help", "-h", "-?", "--info"})
# Flags that name what to run (a module, a file) or hand over inline code, which
# the checks above already grade.
_INTERPRETER_TARGET_FLAGS = frozenset({
    "-m", "-c", "-e", "-p", "-f", "-file", "-command", "-encodedcommand", "-ec",
})


def _reads_program_from_stdin(command: str) -> bool:
    """True for an interpreter invoked with nothing to run: bare, flags only, or `-`.

    `python`, `python -u -`, `bash`, `node -` read their program from stdin, which
    is how `echo "…" | python` and a heredoc get code past the gate. `python x.py`,
    `python -m pytest` and `python --version` name their target and are fine.
    """
    tokens = command.strip().split()
    while tokens and (tokens[0].lower() == "sudo" or
                      ("=" in tokens[0] and not tokens[0].startswith("-"))):
        tokens = tokens[1:]                      # `sudo python`, `PYTHONPATH=. python`
    if not tokens:
        return False
    exe = tokens[0].strip("\"'").replace("\\", "/").rsplit("/", 1)[-1]
    if not _STDIN_INTERPRETER_RE.match(exe):
        return False
    for arg in tokens[1:]:
        if arg in _INTERPRETER_INFO_FLAGS:
            return False
    for arg in tokens[1:]:
        if arg == "-":
            return True
        if arg.lower() in _INTERPRETER_TARGET_FLAGS:
            return False
        if not arg.startswith("-"):
            return False                         # a script, a file: a visible target
    return True


# Deletion and overwrite of data. This is the tier /vibe still refuses: it is the
# one kind of damage a checkpoint cannot always undo (a deleted untracked file is
# gone), and it is exactly what "не удалять" means.
_DELETES_PATTERNS = [
    # Match deletion commands only at start of command, after chain operators, or after sudo/cmd
    r"(?:^|[;&|\n]\s*|(?:sudo|cmd(?:\.exe)?\s*/c)\s+)\b(rm|del|erase|rd|rmdir|ri)\b(?:\s+|$)",
    r"\bremove-item\b",
    r"\bclear-(content|item)\b",
    r"\bformat\s+[a-zA-Z]:|\bformat-volume\b",
    r"\b(diskpart|mkfs)\b",
    r"\bgit\s+(reset\s+--hard|clean\b|push\b.*--force|checkout\s+--\s)",
    r"\breg\s+delete\b",
    # Additional high-risk destructive commands (closed False Negatives):
    r"\bfind\b[^\n|&;]*-delete\b",
    r"\bgit\s+branch\s+-[dD]\b",
    r"\bgit\s+stash\s+(?:drop|clear)\b",
    r"\bgit\s+push\b[^\n|&;]*(?:--delete|:\w+)\b",
    r"\btruncate\b[^\n|&;]*-s\s*0\b",
    r"\b(?:srm|shred|wipe)\b",
]
_DELETES_RE = re.compile("|".join(_DELETES_PATTERNS), re.IGNORECASE)

# Stopping a process is recoverable in the ordinary sense — start it again — and
# /vibe often has to kill its own stuck interpreter, so it goes through there.
_KILLS_RE = re.compile(r"\b(taskkill|stop-process)\b", re.IGNORECASE)

# Taking the machine down. Nothing checkpoints that, and an autonomous run cannot
# be asked again afterwards, so it stays refused even though it deletes nothing.
_POWER_RE = re.compile(r"\b(shutdown|stop-computer|restart-computer)\b", re.IGNORECASE)

# Deletion hiding inside a program the gate CAN read (`python -c "…"`, `node -e
# "…"`, `echo '…' | python`). The command-position patterns above cannot see
# `os.remove` or `shutil.rmtree`, which is exactly what a one-liner uses.
_DESTRUCTIVE_API_RE = re.compile(
    r"\bos\.(?:remove|unlink|rmdir|removedirs|rename|replace)\b"
    r"|\bshutil\.(?:rmtree|move|copy\w*)\b"
    r"|\brmtree\b"
    r"|\.unlink\s*\("
    r"|\.rmdir\s*\("
    r"|\.write_text\s*\(|\.write_bytes\s*\("
    r"|\bopen\s*\([^)]*['\"][wax]"
    r"|\bfs\.(?:rm|rmdir|unlink)\w*\b"
    r"|\brmSync\s*\(|\brmdirSync\s*\(|\bunlinkSync\s*\("
    r"|\bunlink\s*\("
    r"|\.delete\s*\(",
    re.IGNORECASE,
)
# A program that shells out, together with a deletion word anywhere in the text,
# says what `subprocess.run(["rm", "-rf", "x"])` does — no command-position
# pattern can see that, because the word is not in command position.
_SHELL_RUNNER_RE = re.compile(
    r"\b(?:subprocess|os\.system|popen|execv?\w*|startfile)\b", re.IGNORECASE)
_SHELL_DELETE_WORD_RE = re.compile(
    r"\b(?:rm|del|erase|rd|rmdir|remove-item|rmtree|unlink)\b", re.IGNORECASE)


def _code_has_destruction(text: str) -> bool:
    """True when a program's own text contains a delete or an overwrite."""
    if not text:
        return False
    if _DELETES_RE.search(text) or _DESTRUCTIVE_API_RE.search(text):
        return True
    return bool(_SHELL_RUNNER_RE.search(text) and _SHELL_DELETE_WORD_RE.search(text))


def _inline_code_has_destruction(command: str) -> bool:
    """True when the inline program in `command` deletes or overwrites something."""
    code = _extract_inline_python_code(command)
    text = code if code is not None else command
    return bool(_code_has_destruction(text) or _SHELL_DELETE_WORD_RE.search(text))


# --- Housekeeping vs damage -----------------------------------------------------
#
# /vibe could not delete the probe file it had just written: `Remove-Item
# .\tmp\x.py` was "destructive", and destructive was auto-denied. Clearing its own
# temp and `.argent/` is not the person's data, so that one deletion goes through.
_SCRATCH_DELETE_RE = re.compile(
    r"(?:^|[;&|\n]\s*|(?:sudo|cmd(?:\.exe)?\s*/c)\s+)"
    r"\b(?:rm|del|erase|rd|rmdir|ri|remove-item|clear-content)\b(?P<rest>[^\n;|&]*)",
    re.IGNORECASE,
)
_PATH_FLAGS = frozenset({"-path", "-literalpath"})       # these take a value
_FLAGS_WITH_VALUE = frozenset({"-filter", "-include", "-exclude", "-stream"})
_TOKEN_RE = re.compile(r"'[^']*'|\"[^\"]*\"|[^\s]+")
_ENV_TEMP_RE = re.compile(r"^(?:\$env:(?:temp|tmp)|%temp%|%tmp%)", re.IGNORECASE)


def _looks_like_flag(token: str) -> bool:
    """`-Recurse` and `/q` are switches; `/tmp/x` and `C:/x` are paths."""
    if re.fullmatch(r"-[A-Za-z][\w-]*", token):
        return True
    return bool(re.fullmatch(r"/[A-Za-z?]{1,3}", token))


def _scratch_roots():
    """Directories whose contents are this run's own scratch, not the person's work.

    Deliberately the working subdirectories and NOT all of `.argent/`: that tree
    also holds sessions, checkpoints and permission rules, and deleting it would
    destroy exactly what makes a mistake recoverable.
    """
    import tempfile

    roots = [Path(tempfile.gettempdir()), Path("/tmp")]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "Temp")
    try:
        from argent.project_paths import project_root_or_cwd
        base = Path(project_root_or_cwd()) / ".argent"
        roots.extend(base / name for name in ("artifacts", "tmp", "scratch"))
    except Exception:
        pass
    return roots


def _under(path: Path, root: Path) -> bool:
    try:
        root = root.resolve()
    except (OSError, ValueError):
        return False
    return path == root or root in path.parents


def _delete_targets(command: str):
    """Paths a plain delete command names, or None when it is not one of those.

    A name that is a variable is filled in from a literal assignment in the same
    command (`$tmp = ".argent/artifacts/x"; Remove-Item $tmp`): otherwise every
    delete written with a variable was an unreadable target, and unreadable meant
    refused — which in an autonomous run is most of them.
    """
    found = False
    targets = []
    for match in _SCRATCH_DELETE_RE.finditer(command or ""):
        found = True
        tokens = [t.strip("\"'") for t in _TOKEN_RE.findall(match.group("rest"))]
        i = 0
        while i < len(tokens):
            token = tokens[i]
            low = token.lower()
            if low in _PATH_FLAGS:
                i += 1
                if i < len(tokens):
                    targets.append(tokens[i])
            elif low in _FLAGS_WITH_VALUE:
                i += 1
            elif not _looks_like_flag(token):
                targets.append(token)
            i += 1
    if not found:
        return None
    values = _literal_assignments(command or "")
    return [_filled_in(token, values) for token in targets]


# --- Naming the target the command actually deletes -----------------------------
#
# `$tmp = ".argent/artifacts/probe"; Remove-Item -Recurse $tmp` is housekeeping and
# was refused: the delete token was `$tmp`, no path was found, and an unreadable
# target is a refused target. The assignment sits in the same command, so read it.
# Only a literal right-hand side is used; anything computed stays unresolved and
# the command keeps being refused, exactly as before.
_PS_ASSIGN_RE = re.compile(
    r"\$([A-Za-z_][\w]*)\s*=\s*(?:\"([^\"$`()]*)\"|'([^'$`()]*)'|([^\s;|&)\]$`\"(]+))")
_SH_ASSIGN_RE = re.compile(
    r"(?:^|[;&|\n]\s*)([A-Za-z_][\w]*)=(\"[^\"$`()]*\"|'[^'$`()]*'|[^\s;|&$`\"]+)",
    re.MULTILINE)


def _literal_assignments(command: str) -> dict[str, str]:
    """`name -> literal value` for assignments written in the command itself."""
    values: dict[str, str] = {}
    for match in _PS_ASSIGN_RE.finditer(command or ""):
        name = match.group(1).lower()
        value = next((g for g in match.groups()[1:] if g), "")
        if value:
            values.setdefault(name, value)
    for match in _SH_ASSIGN_RE.finditer(command or ""):
        name, raw = match.group(1).lower(), match.group(2)
        value = raw.strip("\"'")
        if value:
            values.setdefault(name, value)
    return values


def _filled_in(token: str, values: dict[str, str]) -> str:
    """Replace `$name`, `${name}` and `%name%` with their literal values."""
    if not token or not values:
        return token
    out = token
    for name, value in values.items():
        # The replacement is a Windows path, and re.sub reads `\U` in a replacement
        # string as an escape; a callable hands the path over verbatim.
        out = re.sub(r"\$\{" + re.escape(name) + r"\}",
                     lambda _m, v=value: v, out, flags=re.IGNORECASE)
        out = re.sub(r"\$" + re.escape(name) + r"\b",
                     lambda _m, v=value: v, out, flags=re.IGNORECASE)
        out = out.replace("%" + name + "%", value)
        out = out.replace("%" + name.upper() + "%", value)
    return out


# Derived data: a build produces these again, so removing them is housekeeping
# rather than data loss. Matched by the deleted name only — a source file that
# happens to live in a folder called `build` is still snapshotted.
_REGENERABLE_NAMES = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
    ".ipynb_checkpoints", ".next", ".turbo", ".parcel-cache", "node_modules",
    "target", "dist", "build", "htmlcov", ".coverage",
})

# A wildcard is expanded, but never without a ceiling: `rm *` in a big tree must
# not turn into a million-entry snapshot.
MAX_EXPANDED = 200


def _project_root() -> Path:
    from argent.project_paths import project_root_or_cwd
    return Path(project_root_or_cwd()).resolve()


def _expand(target: str) -> list[Path]:
    """Absolute paths one delete target stands for (wildcards expanded)."""
    import tempfile

    raw = (target or "").strip()
    if not raw:
        return []
    match = _ENV_TEMP_RE.match(raw)
    if match:
        rest = raw[match.end():].lstrip("\\/")
        raw = str(Path(tempfile.gettempdir()) / rest) if rest else tempfile.gettempdir()
    path = Path(raw)
    if not path.is_absolute():
        path = _project_root() / path
    if any(ch in str(path) for ch in "*?["):
        parent, pattern = path.parent, path.name
        if not parent.is_dir():
            return []
        try:
            return sorted(parent.glob(pattern))[:MAX_EXPANDED]
        except (OSError, ValueError):
            return []
    return [path]


def _delete_plan(command: str):
    """What an autonomous run may do with this delete.

    Returns (action, paths, why) where action is:
    - 'allow'    — housekeeping or derived data, nothing to save;
    - 'snapshot' — the person's files inside the project: copy them, then allow;
    - 'deny'     — outside the project, Argent's own state, or an unreadable target.
    """
    raw_targets = _delete_targets(command)
    if not raw_targets:
        return "deny", [], "в команде нет пути, который можно проверить (переменная, git-операция)"
    scratch = _scratch_roots()
    project = _project_root()
    to_save: list[Path] = []
    for raw in raw_targets:
        expanded = _expand(raw)
        if not expanded:
            return "deny", [], f"цель удаления не найдена: {raw}"
        for path in expanded:
            try:
                resolved = path.resolve()
            except (OSError, ValueError):
                return "deny", [], f"путь не разобрать: {raw}"
            if any(_under(resolved, root) for root in scratch):
                continue                      # own scratch, nothing to save
            # Judged by the path, not by what is there right now: `Remove-Item
            # .argent/checkpoints` on a machine where that folder does not exist
            # yet must not read as housekeeping.
            if _under(resolved, project / ".argent"):
                return "deny", [], f"это состояние самого Argent: {resolved}"
            if not _under(resolved, project):
                return "deny", [], f"вне проекта: {resolved}"
            if not path.exists() and not path.is_symlink():
                continue                      # already gone: nothing can be lost
            if resolved.name.lower() in _REGENERABLE_NAMES:
                continue                      # derived data, rebuilt by a build
            to_save.append(resolved)
    return ("snapshot" if to_save else "allow"), to_save, ""


# What the agent (not only the person) can do about a refusal: the line lands in
# the transcript and in the tool's error text, so it has to name the way forward.
_VIBE_REMEDY = {
    CATASTROPHIC: "Такое не откатить: переформулируйте команду уже, без уничтожения корня или диска.",
    POWER: "Машину в автономном режиме не выключаем — это оборвёт саму работу.",
    UNDECIDABLE: "Соберите команду без Invoke-Expression и без вызова из переменной, чтобы её можно было прочитать.",
    IRREVERSIBLE: "У действия нет команды, которую можно проверить; в автономном режиме оно не выполняется.",
}


def _vibe_may_ask() -> bool:
    """Vibe refuses silently unless someone is actually there (see `argent.presence`)."""
    try:
        from argent import presence

        return presence.is_attended()
    except Exception:
        return False


def _vibe_ask(action: str, destructive: bool, why: str) -> bool:
    """Ask instead of refusing — only while the person is at the keyboard.

    The wait is bounded by the transport's own answer timeout, so a question that
    nobody answers ends in a refusal just as it would have without asking.
    """
    from argent.ui import console

    try:
        decision = _active_backend()(action, destructive, None)
    except Exception as e:
        log.info("vibe ask failed, refusing as before: %s", e)
        return False
    if isinstance(decision, dict):
        decision = str(decision.get("decision") or "deny")
    if decision in ("once", "always"):
        console.print(f"[yellow]✓ Одобрено (/vibe, вы разрешили): {action}[/yellow]")
        log.info("Person approved an action vibe would have refused: %s", action)
        return True
    console.print(f"[red]✗ Отклонено (/vibe): {action}{f' — {why}' if why else ''}[/red]")
    return False


def _vibe_delete(command: str, action: str) -> bool:
    """Delete in an autonomous run: allow what is recoverable, refuse the rest."""
    from argent.ui import console

    plan, paths, why = _delete_plan(command)
    if plan == "deny":
        # A deletion outside the project is exactly the kind of thing a person may
        # mean on purpose, so an attended run is asked rather than refused.
        if _vibe_may_ask() and _vibe_ask(action, True, why):
            return True
        console.print(
            f"[red]✗ Авто-отклонено (/vibe, удаление): {action}"
            f"{f' — {why}' if why else ''}. Удалять можно внутри проекта;"
            f" свой мусор держите в .argent/artifacts или во временной папке."
            f" Вне /vibe решается вручную.[/red]")
        log.info("Auto-denied delete in AUTO policy: %s (%s)", action, why)
        return False
    if plan == "allow":
        console.print(f"[yellow]✓ Авто-одобрено (автономный режим, уборка): {action}[/yellow]")
        log.info("Approved housekeeping delete in AUTO policy: %s", action)
        return True

    try:
        from argent import trash
        snap = trash.snapshot(paths, command=command,
                              reason="удаление в автономном режиме")
    except Exception as e:                    # TooBig, TrashError, or worse
        console.print(
            f"[red]✗ Авто-отклонено (/vibe, удаление): {action}"
            f" — копию сохранить не удалось ({e}). Ничего не удалено.[/red]")
        log.info("Auto-denied delete: snapshot failed: %s (%s)", action, e)
        return False

    names = ", ".join(p.name for p in paths[:3])
    more = f" и ещё {len(paths) - 3}" if len(paths) > 3 else ""
    console.print(
        f"[yellow]✓ Авто-одобрено (/vibe, удаление отменяемо): {action}"
        f" — {names}{more} сохранены в .argent/trash/{snap.id}."
        f" Вернуть: /trash restore {snap.id}[/yellow]")
    log.info("Approved delete with snapshot %s in AUTO policy: %s", snap.id, action)
    return True


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
    r"|\b(?:powershell|pwsh)\b[^\n|&;]*\s-(?:c\b|comm?a?n?d?\b|e\b|ec\b|enc(?:odedcomm?a?n?d?)?\b)"
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
    # Call operator on dynamic variable, concatenation, or subexpression: & $cmd, & "$a$b", & ($a + $b)
    # In PowerShell, & "C:\path\app.exe" is the standard way to run a quoted path.
    # We flag invocation of variables, dynamic expressions, or concatenated strings, NOT quoted paths or $env: / $HOME
    (r"&\s*(?:\$(?!env:|HOME[\\/])[a-zA-Z_]|[\"']\s*\$(?!env:|HOME[\\/])[a-zA-Z_]|\(|\"[\"']\s*\+)", "invokes a command built from a variable or dynamic expression"),
    # Bash/POSIX eval.
    (r"\beval\b", "evaluates a dynamically built command (eval)"),
    # Decoding into execution: FromBase64String, base64 -d | sh
    (r"\bfrombase64string\b", "decodes a base64 payload before running it"),
    (r"\bbase64\b[^|&;]*-{1,2}d(?:ecode)?\b", "decodes a base64 payload before running it"),
    # `pwsh -EncodedCommand <base64>`: the program is not in the command at all.
    (r"-enc(?:odedcommand)?\b", "carries its program base64-encoded (EncodedCommand)"),
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

_ALWAYS_PREFIX = "✓ Да, и всегда разрешать"


def approval_prompt_spec(grant_key: str | None, destructive: bool) -> dict:
    """What the terminal asks for a gated action.

    One definition shared by the terminal prompt below and by a frontend that
    draws the question itself (the TUI), so the wording cannot drift.
    `decisions[i]` is the decision the i-th choice stands for.
    """
    question = "Разрешить это действие?"
    if grant_key and not destructive:
        return {
            "kind": "select",
            "question": question,
            "choices": ["✓ Да (однократно)",
                        f"{_ALWAYS_PREFIX} '{grant_key}' в этой сессии",
                        "✕ Нет"],
            "decisions": ["once", "always", "deny"],
        }
    return {"kind": "confirm", "question": question,
            "choices": ["Да", "Нет"], "decisions": ["once", "deny"]}


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


def _split_unquoted_chain_commands(command: str) -> list[str]:
    """Split a compound command into individual commands by unquoted ;, &&, ||, |, & or newline.

    Preserves quoted strings and escapes. Returns a list of trimmed non-empty subcommands.
    """
    in_single = False
    in_double = False
    escape = False
    commands = []
    current = []
    i = 0
    n = len(command)

    while i < n:
        char = command[i]
        if escape:
            escape = False
            current.append(char)
            i += 1
            continue
        if char == "\\" and not in_single:
            escape = True
            current.append(char)
            i += 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            current.append(char)
            i += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            current.append(char)
            i += 1
            continue

        if not in_single and not in_double:
            if char in (";", "\n"):
                sub = "".join(current).strip()
                if sub:
                    commands.append(sub)
                current = []
                i += 1
                continue

            if char == "&":
                prev = "".join(current).rstrip()
                if prev.endswith(">") or prev.endswith("<"):
                    current.append(char)
                    i += 1
                    continue
                if not prev and i + 1 < n and command[i + 1] != "&":
                    current.append(char)
                    i += 1
                    continue
                sub = "".join(current).strip()
                if sub:
                    commands.append(sub)
                current = []
                if i + 1 < n and command[i + 1] == "&":
                    i += 2
                else:
                    i += 1
                continue

            if char == "|":
                sub = "".join(current).strip()
                if sub:
                    commands.append(sub)
                current = []
                if i + 1 < n and command[i + 1] == "|":
                    i += 2
                else:
                    i += 1
                continue

        current.append(char)
        i += 1

    if current:
        sub = "".join(current).strip()
        if sub:
            commands.append(sub)

    return commands


def _has_unclosed_quotes(command: str) -> bool:
    """Check if command has unclosed single or double quotes."""
    in_single = False
    in_double = False
    escape = False
    for char in command:
        if escape:
            escape = False
            continue
        if char == "\\" and not in_single:
            escape = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
    return in_single or in_double


def classify_command_risk(command: str) -> Risk:
    """Grade a shell command: how loudly (`level`) and about what (`kind`).

    level drives the ordinary gate: 'block' is catastrophic, 'warn' means a human
    should look, 'safe' needs no one. kind says WHETHER the risk is deletion of
    data, a stopped process, or merely code the gate cannot read — which is what an
    autonomous mode needs to know, since it must answer by itself.
    """
    command = command or ""
    if not command.strip():
        return Risk("safe", NONE)

    reasons = [why for rx, why in _CATASTROPHIC_RE if rx.search(command)]
    if reasons:
        return Risk("block", CATASTROPHIC, tuple(reasons))

    command, stdin_risk = _vet_stdin_python(command)
    if stdin_risk is not None:
        return stdin_risk
    if not command.strip():
        return Risk("safe", NONE)

    if _has_unclosed_quotes(command):
        return Risk("warn", UNDECIDABLE,
                    ("unclosed quotation mark — the command would fail as written",))

    # For compound commands with unquoted chain operators (; && || | & newline),
    # assess each subcommand individually and keep the worst:
    subcommands = _split_unquoted_chain_commands(command)
    if len(subcommands) > 1:
        risks = [classify_command_risk(sub) for sub in subcommands]
        worst = next((r for kind in _KIND_ORDER for r in risks if r.kind == kind), None)
        if worst is None:
            return Risk("safe", NONE)
        if worst.level == "block":
            return Risk("block", CATASTROPHIC, worst.reasons)
        merged = tuple(dict.fromkeys(w for r in risks for w in r.reasons))
        return Risk("warn", worst.kind, merged)

    # If it is a python -c invocation that is demonstrably safe read-only code:
    if _is_safe_inline_python(command):
        return Risk("safe", NONE)

    if _DELETES_RE.search(command):
        return Risk("warn", DELETES, ("deletes or overwrites files",))
    if _POWER_RE.search(command):
        return Risk("warn", POWER, ("shuts down or restarts the machine",))
    if _KILLS_RE.search(command):
        return Risk("warn", KILLS, ("stops processes",))
    # Before the "can we read the code" branches: a command that builds or decodes
    # itself has no code to read, whatever interpreter would eventually run it.
    obfuscated = [why for rx, why in _OBFUSCATION_RE if rx.search(command)]
    if obfuscated:
        return Risk("warn", UNDECIDABLE, tuple(obfuscated))
    if _INLINE_CODE_INTERPRETER_RE.search(command):
        if _inline_code_has_destruction(command):
            return Risk("warn", DELETES, ("inline code that deletes or overwrites files",))
        return Risk("warn", UNVETTED,
                    ("runs inline interpreter code — effects can't be inspected",))
    if _reads_program_from_stdin(command):
        if _code_has_destruction(command):
            return Risk("warn", DELETES, ("program fed on stdin that deletes or overwrites files",))
        return Risk("warn", UNVETTED,
                    ("runs a program supplied on stdin — effects can't be inspected",))
    return Risk("safe", NONE)


def assess_command_risk(command: str):
    """Grade a shell command's risk.

    Returns (level, reasons) where level is 'safe' | 'warn' | 'block':
    - 'block': catastrophic / irreversible / system-wide (with specific reasons).
    - 'warn':  destructive but recoverable/scoped (deletes files, kills procs…).
    - 'safe':  everything else.

    The kind of risk (see `classify_command_risk`) is what an autonomous mode
    needs; this shape is kept for the callers that only show the level.
    """
    risk = classify_command_risk(command)
    return risk.level, list(risk.reasons)


def is_destructive_command(command: str) -> bool:
    """True when a command can destroy data, the session or the machine.

    Deliberately NOT "anything that is not safe": code the gate merely cannot
    read (`python -c` without a deletion in it) and a stopped process are the
    difference between an autonomous mode that works and one that refuses its own
    work. Both still ask in the ordinary mode — this flag only decides who may
    answer without a human.
    """
    return (classify_command_risk(command or "").kind
            in (DELETES, POWER, UNDECIDABLE, CATASTROPHIC))


def _has_unquoted_chain_operators(command: str) -> bool:
    """Check if command contains unquoted chaining operators (; && || | & newline)."""
    in_single = False
    in_double = False
    escape = False
    i = 0
    n = len(command)
    while i < n:
        char = command[i]
        if escape:
            escape = False
            i += 1
            continue
        if char == "\\" and not in_single:
            escape = True
            i += 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if not in_single and not in_double:
            if char in (";", "\n"):
                return True
            if char == "|":
                return True
            if char == "&":
                # Disregard redirection like 2>&1, >&, <&
                prev = command[:i].rstrip()
                if prev.endswith(">") or prev.endswith("<"):
                    i += 1
                    continue
                # PowerShell call operator at the start of command: & "cmd"
                if not prev and i + 1 < n and command[i+1] != "&":
                    i += 1
                    continue
                return True
        i += 1
    return False



# What the approval line for a shell command starts with. A scheduled run's gate
# (src/automation/runner.py) reads the command back out of the action text, so the
# prefixes live here, next to request_approval, instead of being typed twice.
RUN_COMMAND_ACTION = "выполнить команду: "
BACKGROUND_COMMAND_ACTION = "запустить фоновый процесс: "


def command_from_action(action: str) -> str | None:
    """The shell command an approval request is about, or None when the request
    is about something else (deleting a file, a rollback, an admin command)."""
    for prefix in (RUN_COMMAND_ACTION, BACKGROUND_COMMAND_ACTION):
        if (action or "").startswith(prefix):
            return action[len(prefix):]
    return None


def command_grant_key(command: str) -> str | None:
    """Session-grant key for a shell command: its first token (the executable).

    Compound commands with unquoted separators (;, &&, ||, |, &) cannot be
    safely granted by their first token (e.g. 'cd /tmp; rm -rf /' must NOT
    grant 'cd'), so they return None (prompting every time).
    """
    if not command or not command.strip():
        return None
    cleaned = command.strip()
    if _has_unquoted_chain_operators(cleaned):
        return None
    tokens = cleaned.split()
    if not tokens:
        return None
    first = tokens[0]
    # Skip PowerShell call operator '&' if present at start
    if first == "&" and len(tokens) > 1:
        first = tokens[1]
    return first.strip("\"'").lower()



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


# The three facts a caller must be able to tell apart. "deny" was the backend's single
# word for all of them, so an agent could not know whether the person refused the plan
# or simply walked away from the prompt — and asked "why?" in the second case too.
APPROVED = "approved"
REJECTED = "rejected"
CANCELLED = "cancelled"


def _record_status(ok: bool, status: str, reason: str = "") -> bool:
    """Remember the outcome for `request_approval_status` and pass the flag on."""
    _local.last_status = (status, reason)
    return ok


def request_approval_status(action: str, *, destructive: bool = False,
                            grant_key: str | None = None) -> tuple[str, str]:
    """`request_approval`, but it also says WHY: (status, reason).

    status is one of APPROVED / REJECTED / CANCELLED. A backend may answer with a
    dict `{"decision": "deny", "reason": "..."}` when it can carry the person's
    words; plain `deny`/`once`/`always` still work and leave the reason empty.
    """
    _local.last_status = None
    ok = request_approval(action, destructive=destructive, grant_key=grant_key)
    status, reason = _local.last_status or ((APPROVED if ok else REJECTED), "")
    return status, reason


#: Commands that throw away work the person has not committed. The consent line has to
#: say how MUCH, not just what: `git checkout -- .` against a clean tree is harmless,
#: against forty changed files it is the whole afternoon.
_DESTRUCTIVE_COMMANDS = (
    "git checkout --", "git checkout .", "git checkout -f", "git restore",
    "git reset --hard", "git clean", "git stash drop", "git stash clear",
    "git branch -D",
)


def destructive_scope(command: str) -> str:
    """How much uncommitted work a command would destroy, for the consent text.

    Only for commands whose whole purpose is to discard local changes. Returns "" for
    everything else, and for anything it cannot measure — a guess in a consent prompt is
    worse than no number, because it gets trusted.
    """
    text = (command or "").strip().lower()
    if not text:
        return ""
    hit = any(text.startswith(prefix) or f"; {prefix}" in text or f"&& {prefix}" in text
              for prefix in _DESTRUCTIVE_COMMANDS)
    if not hit:
        return ""
    try:
        import subprocess
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, timeout=5)
    except Exception:
        return ""
    if result.returncode != 0:
        # Not a repository (or git is unavailable): no claim at all. "Nothing to lose"
        # when we simply could not look is the kind of reassurance that gets trusted.
        return ""
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return "рабочее дерево чисто — терять нечего"
    modified = sum(1 for line in lines if not line.startswith("??"))
    untracked = len(lines) - modified
    return (f"уничтожит НЕЗАКОММИЧЕННОЕ: {modified} изменённых файлов, "
            f"{untracked} неотслеживаемых")


def current_scope_note() -> str:
    """The scope note for the approval being answered right now ("" when none).

    Kept beside the decision rather than inside `action`: the automation runner and the
    subagent worker read the command back out of that string, so decorating it would
    send "git checkout -- . [уничтожит 6 файлов]" to the shell.
    """
    return getattr(_local, "scope_note", "") or ""


def _raw_terminal_prompt(action: str, destructive: bool, grant_key: str | None) -> str:
    """Interactive TTY prompt on console."""
    from argent.ui import console
    try:
        from src.cli.sound import play_sound
        play_sound("prompt")
    except Exception:
        pass

    from argent.ui import approval_header_markup
    console.print(approval_header_markup(action, destructive))
    note = current_scope_note()
    if note:
        console.print(f"[yellow]→ {note}[/yellow]")

    try:
        from src.cli.input_queue import pause_active_input_listener, resume_active_input_listener
        pause_active_input_listener()
    except Exception:
        resume_active_input_listener = lambda: None

    try:
        spec = approval_prompt_spec(grant_key, destructive)
        if spec["kind"] == "select":
            choice = questionary.select(
                spec["question"],
                choices=spec["choices"],
            ).ask()
            if choice is None:
                # Escaped out: nobody said no, nobody said yes.
                return "cancel"
            if choice.startswith("✕"):
                return "deny"
            return "always" if choice.startswith(_ALWAYS_PREFIX) else "once"

        approved = questionary.confirm(spec["question"]).ask()
        if approved is None:
            return "cancel"
        return "once" if approved else "deny"
    except (KeyboardInterrupt, EOFError):
        return "cancel"
    finally:
        resume_active_input_listener()


def _terminal_decision(action: str, destructive: bool, grant_key: str | None) -> str:
    """Interactive TTY decision. In shared/companion mode, relays to Telegram simultaneously."""
    try:
        from argent.config import get_telegram_mode
        from src.telegram.bot import telegram_bridge
        if telegram_bridge.is_running and get_telegram_mode() == "shared":
            return telegram_bridge.relay_terminal_approval(action, destructive, grant_key, _raw_terminal_prompt)
    except Exception:
        pass

    return _raw_terminal_prompt(action, destructive, grant_key)



def _action_risk_kind(action: str, destructive: bool):
    """The kind of risk an approval request carries, and why.

    The caller's `destructive` flag only says "someone thought this was dangerous".
    For a shell command the gate can be exact, because the action text carries the
    command — and exactness is what lets /vibe allow a `python -c` probe while
    still refusing `python -c "os.remove(x)"`.
    """
    command = command_from_action(action)
    if command:
        risk = classify_command_risk(command)
        return risk.kind, "; ".join(risk.reasons)
    if destructive:
        return IRREVERSIBLE, "действие, помеченное вызывающим как необратимое"
    return NONE, ""


def request_approval(action: str, *, destructive: bool = False,
                     grant_key: str | None = None) -> bool:
    """Single decision point for every gated tool action.

    Returns True when the action may proceed. Ctrl+C / closed stdin / a client
    denial counts as denial. The outcome is recorded on the current Leeth
    trajectory (Phase 4 telemetry — a denial is the strongest 'bad proposal'
    signal the feedback loop gets).
    """
    approved = _request_approval_impl(action, destructive=destructive, grant_key=grant_key)
    try:
        from src.leeth import trajectory as leeth_trajectory
        if leeth_trajectory.current_trajectory_id():
            leeth_trajectory.record_event(
                "approval", action=action, approved=approved, destructive=destructive)
    except Exception:
        pass
    return approved


def _request_approval_impl(action: str, *, destructive: bool = False,
                           grant_key: str | None = None) -> bool:
    """Original approval logic (wrapped by request_approval for telemetry)."""
    from argent.ui import console

    # The person's standing rules (permissions.json) for shell commands. Like a session grant they
    # are consent from someone present, so an unattended run never reads them. An `ask` rule is
    # the one that takes power away: it switches off every shortcut below, auto mode included.
    rule_allow = rule_ask = None
    command = command_from_action(action)
    # What a destructive command takes with it. Computed here, once, so every transport
    # (terminal, TUI, phone) shows the same number instead of only the command text.
    _local.scope_note = destructive_scope(command) if command else ""
    if command is not None and not _nobody_is_watching():
        try:
            from src.agent.permissions import PERMISSIONS
            rule_ask = PERMISSIONS.command_rule("ask", command)
            if rule_ask is None and not destructive:
                rule_allow = PERMISSIONS.command_rule("allow", command)
        except Exception as e:
            log.warning("permission rules unreadable: %s", e)

    # What this action threatens, read from its own text rather than trusted from
    # the caller's boolean: `run_command` and `start_background_command` carry the
    # command, so the gate can tell "deletes the person's files" (needs a human,
    # in every mode) from "effect we cannot read" (must run in /vibe, or the mode
    # refuses nearly everything the agent does).
    kind, why = _action_risk_kind(action, destructive)
    vibe = _policy == POLICY_AUTO and not _nobody_is_watching() and rule_ask is None

    if not destructive and not _nobody_is_watching() and not vibe:
        if rule_allow is not None:
            console.print(f"[yellow]✓ Авто-одобрено (правило {rule_allow.raw}): {action}[/yellow]")
            log.info("Approved via permission rule %s: %s", rule_allow.raw, action)
            return _record_status(True, APPROVED, f"правило {rule_allow.raw}")
        if rule_ask is None and grant_key and grant_key in _session_grants:
            console.print(f"[yellow]✓ Авто-одобрено (сессионное разрешение '{grant_key}'): {action}[/yellow]")
            log.info("Approved via session grant '%s': %s", grant_key, action)
            return _record_status(True, APPROVED, f"сессионное разрешение '{grant_key}'")

    # Autonomous mode answers by itself, and says why in the same line the person
    # would otherwise have been asked in. Only the plain safe case stays silent.
    if vibe:
        # Things no one can sensibly say "yes" to: refused without asking, even while
        # the person is watching. A question about `rm -rf /` only burns a minute.
        if kind in (CATASTROPHIC, POWER, UNDECIDABLE):
            console.print(
                f"[red]✗ Авто-отклонено (автономный режим, {_kind_label(kind)}): {action}"
                f"{f' — {why}' if why else ''}. {_VIBE_REMEDY.get(kind, '')}"
                f" Выйдите из /vibe, чтобы решить вручную.[/red]")
            log.info("Auto-denied %s action in AUTO policy: %s (%s)", kind, action, why)
            return _record_status(
                False, REJECTED,
                f"автономный режим: {_kind_label(kind)}{f' — {why}' if why else ''}")
        if kind == IRREVERSIBLE:
            # An action whose effect cannot be read is the person's call, not a
            # policy's — but only while they are actually there (see presence.py).
            if _vibe_may_ask() and _vibe_ask(action, destructive, why):
                return _record_status(True, APPROVED, "автономный режим: разрешено человеком")
            console.print(
                f"[red]✗ Авто-отклонено (автономный режим, {_kind_label(kind)}): {action}"
                f"{f' — {why}' if why else ''}. {_VIBE_REMEDY.get(kind, '')}"
                f" Выйдите из /vibe, чтобы решить вручную.[/red]")
            log.info("Auto-denied %s action in AUTO policy: %s (%s)", kind, action, why)
            return _record_status(
                False, REJECTED,
                f"автономный режим: {_kind_label(kind)}{f' — {why}' if why else ''}")
        if kind == DELETES:
            ok = _vibe_delete(command or "", action)
            return _record_status(ok, APPROVED if ok else REJECTED, "автономный режим: удаление")
        if kind == KILLS:
            console.print(f"[yellow]✓ Авто-одобрено (автономный режим, останавливает процесс): {action}[/yellow]")
            log.info("Approved process stop in AUTO policy: %s", action)
            return _record_status(True, APPROVED, "автономный режим")
        if kind == UNVETTED:
            console.print(
                f"[yellow]✓ Авто-одобрено (автономный режим): {action}"
                f" — проверить нельзя{f': {why}' if why else ''}."
                f" Чекпоинт хода позволит откатить /rewind.[/yellow]")
            log.info("Approved unvetted action in AUTO policy: %s (%s)", action, why)
            return _record_status(True, APPROVED, "автономный режим")
        console.print(f"[yellow]✓ Авто-одобрено (автономный режим): {action}[/yellow]")
        log.info("Approved via AUTO policy: %s", action)
        return _record_status(True, APPROVED, "автономный режим")

    decision = _active_backend()(action, destructive, grant_key)

    # A backend that knows the person's words answers with a dict. The bare strings
    # stay supported so every existing transport keeps working unchanged.
    reason = ""
    if isinstance(decision, dict):
        reason = str(decision.get("reason") or "")
        decision = str(decision.get("decision") or "deny")

    # "always" only *grants* when there is a session key to record it under and
    # the action is not destructive. Without a key it is still an approval for
    # this action — falling through to the deny branch made "always approve"
    # silently reject the very action it was given for, which the agent then
    # reported as "Execution aborted by user".
    if decision == "always":
        if grant_key and not destructive:
            _session_grants.add(grant_key)
            log.info("Session grant added: '%s'", grant_key)
            return _record_status(True, APPROVED, reason)
        if destructive:
            log.info(
                "Approved once only: '%s' is destructive and never covered by a session grant",
                action,
            )
        else:
            log.info("Approved once only: no session grant key for '%s'", action)
        return _record_status(True, APPROVED, reason)
    if decision == "once":
        return _record_status(True, APPROVED, reason)
    if decision == "cancel":
        log.info("Approval cancelled (no answer): %s", action)
        return _record_status(False, CANCELLED, "никто не ответил (Esc/обрыв ввода)")
    log.info("Denied by user: %s", action)
    return _record_status(False, REJECTED, reason)
