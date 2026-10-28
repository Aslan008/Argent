from src.agent.command_diagnostics import diagnose_command_error
from src.agent.healing import build_healing_hint


class TestDiagnose:
    def test_powershell_cmdlet_in_cmd(self):
        # The exact case from the bug report.
        cmd = 'dotnet build Warlord.cs 2>&1 | Select-Object -First 20'
        out = '"Select-Object" не является внутренней или внешней\nкомандой'
        hint = diagnose_command_error(cmd, out, 255)
        assert hint and "Select-Object" in hint
        assert "PowerShell" in hint and "cmd" in hint
        assert "NOT the problem" in hint  # steers away from blaming the C# code

    def test_unknown_program_not_recognized(self):
        hint = diagnose_command_error("foobar --x", "'foobar' is not recognized as an internal or external command", 1)
        assert hint and "not found" in hint.lower()

    def test_access_denied(self):
        hint = diagnose_command_error("net stop x", "Access is denied.", 5)
        assert hint and "admin=true" in hint

    def test_a_taskkill_that_found_nothing_points_at_ownership_not_a_process(self):
        out = 'ERROR: The process "cargo.exe" not found.\nERROR: The process "rustc.exe" not found.'
        hint = diagnose_command_error("taskkill /F /IM cargo.exe /IM rustc.exe", out, 1)
        assert hint and "No cargo/rustc process" in hint
        assert "elevated" in hint and "retry" in hint.lower()

    def test_removing_the_target_is_not_a_build_retry(self):
        out = "Remove-Item : Access to the path 'C:\\proj\\target\\.cargo-lock' is denied."
        hint = diagnose_command_error(
            "Remove-Item -Recurse -Force C:\\proj\\target", out, 1)
        assert hint and "cannot be removed" in hint
        assert "elevated shell" in hint

    def test_a_cargo_lock_is_not_a_reason_to_elevate(self):
        out = (
            "error: failed to open: C:\\proj\\target\\debug\\.cargo-lock\n"
            "Caused by:\n  Access is denied. (os error 5)"
        )
        hint = diagnose_command_error("cargo build", out, 101)
        assert hint and "cargo" in hint.lower() and "lock" in hint.lower()
        assert "do not run cargo as administrator" in hint.lower()
        assert "admin=true" not in hint

    def test_argents_own_state_file_is_not_the_command(self):
        out = (
            "[Errno 13] Permission denied: "
            "'C:\\proj\\.argent\\.memory.json.9832.7336.7cd77319.tmp'"
        )
        hint = diagnose_command_error("cargo clean", out, 101)
        assert hint and "Argent" in hint and "NOT the command's failure" in hint
        assert "admin=true" not in hint

    def test_a_plain_file_denial_does_not_send_the_build_to_admin(self):
        hint = diagnose_command_error("cargo build", "Access is denied. (os error 5)", 101)
        assert hint and "ownership" in hint.lower()
        assert "admin=true" not in hint

    def test_path_not_found(self):
        hint = diagnose_command_error("type nope.txt", "The system cannot find the path specified.", 1)
        assert hint and "path" in hint.lower()

    def test_csharp_compiler_error(self):
        out = "Program.cs(12,5): error CS0103: The name 'x' does not exist"
        hint = diagnose_command_error("dotnet build", out, 1)
        assert hint and "CS0103" in hint

    def test_python_traceback(self):
        out = "Traceback (most recent call last):\n  File 'a.py', line 3\nValueError: bad"
        hint = diagnose_command_error("python a.py", out, 1)
        assert hint and "traceback" in hint.lower()

    def test_success_returns_none(self):
        assert diagnose_command_error("echo hi", "hi", 0) is None

    def test_unrecognized_failure_returns_none(self):
        assert diagnose_command_error("weird", "some unfamiliar output", 1) is None


class TestHealingHintContext:
    def test_command_hint_does_not_assume_code(self):
        hint = build_healing_hint(1, func_name="run_command")
        assert "COMMAND" in hint
        assert "Do not assume it's the code" in hint

    def test_file_tool_hint_targets_code(self):
        hint = build_healing_hint(1, func_name="write_file")
        assert "replace_in_file" in hint

    def test_exhausted(self):
        assert "FAILED" in build_healing_hint(4, func_name="run_command")
