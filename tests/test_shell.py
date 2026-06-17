from src.agent.shell import choose_shell, build_command_argv


class TestChooseShell:
    def test_powershell_cmdlet_forces_powershell(self):
        assert choose_shell('dotnet build x.cs 2>&1 | Select-Object -First 20') == "powershell"
        assert choose_shell("Get-ChildItem -Recurse") == "powershell"
        assert choose_shell("echo $env:PATH") == "powershell"

    def test_cmd_chain_routes_to_cmd(self):
        assert choose_shell("cd build && cmake ..") == "cmd"
        assert choose_shell("test.exe || echo failed") == "cmd"

    def test_cmdlet_wins_over_chain(self):
        # A PowerShell cmdlet present -> PowerShell even with && in the string.
        assert choose_shell("Get-Item x && echo y") == "powershell"

    def test_plain_command_defaults_to_powershell(self):
        assert choose_shell("dotnet build") == "powershell"
        assert choose_shell("python -m pytest -q") == "powershell"
        assert choose_shell("git status") == "powershell"

    def test_single_pipe_is_not_a_chain(self):
        # A single | (pipe) is valid in both; not treated as cmd chaining.
        assert choose_shell("type file.txt | findstr x") == "powershell"


class TestBuildArgv:
    def test_powershell_argv_forces_utf8(self):
        argv = build_command_argv("dotnet build", "powershell")
        assert argv[0] == "powershell"
        assert "-NoProfile" in argv and "-Command" in argv
        assert "OutputEncoding" in argv[-1]
        assert argv[-1].endswith("dotnet build")

    def test_cmd_argv(self):
        assert build_command_argv("dir", "cmd") == ["cmd", "/c", "dir"]
