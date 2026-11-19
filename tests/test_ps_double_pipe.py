"""`||` must survive the trip to Windows PowerShell 5.1.

PS 5.1 has no `||` (only PowerShell 7+): the whole line died with
"The token '||' is not a valid statement separator in this version". `_ps_chain_to_semicolon`
translated `&&` and left `||` untouched, so a cmdlet plus a fallback branch never ran.
"""

import pytest

from src.agent.shell import build_command_argv, choose_shell


def ps_script(command: str) -> str:
    """The script actually handed to powershell.exe, minus the encoding preamble."""
    argv = build_command_argv(command, "powershell")
    assert argv[0] == "powershell"
    return argv[-1].split(";", 1)[1] if "OutputEncoding" in argv[-1] else argv[-1]


class TestDoublePipe:
    def test_a_cmdlet_fallback_becomes_an_if_block(self):
        script = ps_script("Get-ChildItem x || echo missing")
        assert "||" not in script
        assert "if (-not $?)" in script
        assert script.rstrip().endswith("}")

    def test_the_fallback_is_inside_the_block(self):
        script = ps_script("Get-ChildItem x || echo missing")
        assert script.index("if (-not $?)") < script.index("echo missing")

    def test_two_fallbacks_nest(self):
        script = ps_script("Get-ChildItem a || Get-ChildItem b || echo none")
        assert script.count("if (-not $?)") == 2
        assert script.count("}") == 2
        assert script.index("echo none") > script.index("Get-ChildItem b")

    def test_an_and_chain_still_becomes_a_semicolon(self):
        script = ps_script("cd C:/proj && Get-ChildItem")
        assert "&&" not in script
        assert "if (-not $?)" not in script

    def test_a_pipe_inside_quotes_is_untouched(self):
        script = ps_script('Write-Output "a || b"')
        assert '"a || b"' in script
        assert "if (-not $?)" not in script

    def test_a_single_pipe_is_not_a_chain(self):
        script = ps_script("Get-ChildItem | Select-Object -First 3")
        assert "|" in script
        assert "if (-not $?)" not in script


class TestRoutingStillPrefersCmd:
    @pytest.mark.parametrize("cmd", [
        "cd build && cmake ..",
        "test.exe || echo failed",
    ])
    def test_cmd_chains_go_to_cmd(self, cmd):
        assert choose_shell(cmd) == "cmd"

    def test_a_cmdlet_keeps_the_command_in_powershell(self):
        # PowerShell wins over the chain, which is exactly why the translation exists.
        assert choose_shell("Get-ChildItem || echo none") == "powershell"
