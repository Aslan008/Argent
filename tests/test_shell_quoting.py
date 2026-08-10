"""Quotes the model wrote must be the quotes the shell sees.

Reported:

    > run_command('cd C:\\Users\\mshat\\Desktop\\Argent && python -c
      "import multilspy; print(\\'multilspy version:\\', ...)"')
      File "<string>", line 1
        "import
        ^
    SyntaxError: unterminated string literal
    Анализ: Странно, кавычки сломались.

The command was correct all the way to the spawn. subprocess builds a Windows
command line from an argv LIST using list2cmdline, which escapes an inner `"`
as `\\"`. PowerShell accepts that spelling; cmd.exe does not — backslash
escaping is a C-runtime convention, so cmd passed the backslash through, the
quote became a delimiter, and Python received argv[2] == '"import'.

The model had no way to see any of that. It concluded its own quoting was
wrong and started writing temp files to work around a bug in the harness.
"""

import os
import subprocess

import pytest

from src.agent.shell import build_command_argv, choose_shell

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows shell routing")


def _run(command):
    args = build_command_argv(command, choose_shell(command))
    return subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=60)


class TestTheShapesThatBroke:
    def test_inline_python_with_quotes_survives(self):
        """The exact report."""
        r = _run('cd C:\\Users && python -c "import sys; print(\'A ok\')"')
        assert r.returncode == 0, r.stderr
        assert "A ok" in r.stdout

    def test_an_executable_path_with_a_space_keeps_its_quotes(self):
        """`cmd /c` without /s strips the quotes around the program itself, so
        it never launches — a second bug the same construction fixes."""
        exe = f'"{os.sys.executable}"'
        r = _run(f'{exe} -c "print(\'B ok\')" && echo done')
        assert r.returncode == 0, r.stderr
        assert "B ok" in r.stdout

    def test_nested_quotes_inside_an_argument(self):
        r = _run('python -c "print(\'D \\"ok\\"\')" && echo done')
        assert r.returncode == 0, r.stderr
        assert "ok" in r.stdout

    def test_an_operator_inside_a_string_is_not_a_chain(self):
        r = _run('python -c "print(\'E && ok\')"')
        assert r.returncode == 0, r.stderr
        assert "E && ok" in r.stdout


class TestTheOrdinaryPathsStillWork:
    def test_a_plain_cmd_chain(self):
        r = _run("echo one && echo two")
        assert r.returncode == 0
        assert "one" in r.stdout and "two" in r.stdout

    def test_powershell_is_still_a_list(self):
        argv = build_command_argv("Get-ChildItem", "powershell")
        assert isinstance(argv, list) and argv[0] == "powershell"

    def test_powershell_still_handles_inner_quotes(self):
        r = _run('Write-Output "PS ok"')
        assert r.returncode == 0, r.stderr
        assert "PS ok" in r.stdout

    def test_cmd_is_a_raw_command_line(self):
        """It cannot be a list: that is the whole bug."""
        built = build_command_argv("echo hi && echo there", "cmd")
        assert isinstance(built, str)
        assert built.startswith('cmd /s /c "')


class TestRouting:
    @pytest.mark.parametrize("command,expected", [
        ("echo a && echo b", "cmd"),
        ("Get-ChildItem -Recurse", "powershell"),
        ("python -c \"print(1)\"", "powershell"),
        ("git status", "powershell"),
    ])
    def test_shell_choice(self, command, expected):
        assert choose_shell(command) == expected
