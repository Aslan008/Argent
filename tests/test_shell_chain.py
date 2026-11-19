from src.agent.shell import build_command_argv, choose_shell


def test_powershell_gets_semicolon_for_a_cmd_chain():
    assert choose_shell("cd C:/proj && Get-ChildItem") == "powershell"
    argv = build_command_argv("cd C:/proj && Get-ChildItem", "powershell")
    assert "&&" not in argv[-1]
    assert "cd C:/proj ; Get-ChildItem" in argv[-1]


def test_a_chain_inside_quotes_is_not_rewritten():
    argv = build_command_argv('Write-Output "a && b"', "powershell")
    assert '"a && b"' in argv[-1]


def test_cmd_chains_are_left_alone():
    cmd = build_command_argv("cd C:/proj && cargo test", "cmd")
    assert "&&" in cmd


def test_native_tools_stay_out_of_powershell_stderr_rules():
    # cargo clean writes "Removed N files" to stderr; PowerShell 5 wraps that
    # in NativeCommandError and returns 1 even on success.
    assert choose_shell("cargo clean") == "cmd"
    assert choose_shell("cargo check") == "cmd"
    assert choose_shell("git status") == "powershell"
    assert choose_shell("Get-ChildItem") == "powershell"

