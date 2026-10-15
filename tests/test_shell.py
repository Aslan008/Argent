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

    def test_cmd_gets_a_raw_command_line(self):
        """Not a list: list2cmdline escapes an inner quote as \\", which cmd.exe
        does not understand. See test_shell_quoting.py."""
        assert build_command_argv("dir", "cmd") == 'cmd /s /c "dir"'

    def test_quoted_path_gets_call_operator(self):
        """A quoted path at the start of a command needs & in PowerShell,
        otherwise PS treats it as a string literal and just prints the path."""
        argv = build_command_argv('"C:\\tool\\run.bat"', "powershell")
        assert "& " in argv[-1], "call operator & must be prepended"
        assert argv[-1].endswith('& "C:\\tool\\run.bat"')

    def test_quoted_path_with_args_gets_call_operator(self):
        argv = build_command_argv('"C:\\Program Files\\app.exe" --flag', "powershell")
        assert '& "C:\\Program Files\\app.exe" --flag' in argv[-1]

    def test_unquoted_command_has_no_call_operator(self):
        """Normal commands must NOT get & — it breaks statements and pipelines."""
        argv = build_command_argv("python -c \"print(1)\"", "powershell")
        assert "& python" not in argv[-1]

    def test_variable_assignment_no_call_operator(self):
        argv = build_command_argv("$x = 1; echo $x", "powershell")
        assert "& $x" not in argv[-1]

    def test_already_has_call_operator_no_double(self):
        """If the command already starts with &, don't add another."""
        argv = build_command_argv("& python -c \"print(1)\"", "powershell")
        assert "& & " not in argv[-1]

