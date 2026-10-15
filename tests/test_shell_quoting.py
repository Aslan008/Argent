"""Quotes the model wrote must be the quotes the shell sees.

Reported:

    > run_command('cd C:\\Users\\user\\Desktop\\Argent && python -c
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

    def test_powershell_inline_python_with_nested_quotes(self):
        """Pure PowerShell execution (no cmd-chaining &&) with nested quotes in dicts."""
        r = _run('python -c "import sys; d={\\"a\\": 1}; print(d[\\"a\\"])"')
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "1"

    def test_powershell_inline_python_multiline(self):
        """Multiline python -c invocation with newlines and semicolons."""
        cmd = '''python -c "
import sys
sys.path.insert(0, '.')
d = {'key': 'val', \\"nested\\": \\"it works!\\"}
print('Result:', d['key'], d['nested'])
"'''
        r = _run(cmd)
        assert r.returncode == 0, r.stderr
        assert "Result: val it works!" in r.stdout

    def test_powershell_inline_python_single_quoted(self):
        """Single-quoted python -c on Windows."""
        r = _run('python -c \'import sys; print("single_quoted_ok")\'')
        assert r.returncode == 0, r.stderr
        assert "single_quoted_ok" in r.stdout

    def test_powershell_inline_python_with_env_prefix(self):
        """PowerShell env variable prefix before python -c."""
        cmd = '$env:MY_TEST_VAR=\'hello_from_env\'; python -c "import os; print(\'GOT_ENV:\', os.environ.get(\'MY_TEST_VAR\'))"'
        r = _run(cmd)
        assert r.returncode == 0, r.stderr
        assert "GOT_ENV: hello_from_env" in r.stdout

    def test_powershell_inline_python_with_flags_and_nested_quotes(self):
        """Python flags like -X utf8 before -c with nested quotes."""
        cmd = 'python -X utf8 -c "d = {\'a\': \\"b\\"}; print(\'FLAGS_OK:\', d[\'a\'])"'
        r = _run(cmd)
        assert r.returncode == 0, r.stderr
        assert "FLAGS_OK: b" in r.stdout

    def test_powershell_inline_python_with_env_and_flags_multiline(self):
        """Real-world case: env prefix, flags (-X utf8), multiline, and string method calls."""
        cmd = '''$env:PYTHONIOENCODING='utf-8'; python -X utf8 -c "
import os, sys
res = '1+__import__(\\'os\\').system(\\'whoami\\')'
d = {\\"res\\": res}
print('SYSTEM_CALL_STR:', d[\\"res\\"])
"'''
        r = _run(cmd)
        assert r.returncode == 0, r.stderr
        assert "SYSTEM_CALL_STR: 1+__import__('os').system('whoami')" in r.stdout


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
