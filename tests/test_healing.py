from src.agent.healing import detect_tool_failure, build_healing_hint


class TestDetectToolFailure:
    def test_nonzero_exit_code_is_failure(self):
        assert detect_tool_failure("run_command", "Exit code: 5\nOUTPUT:\nboom")

    def test_zero_exit_code_with_failed_word_is_not_failure(self):
        # The old substring matcher misfired on legitimate output like this.
        assert not detect_tool_failure("run_command", "Exit code: 0\nOUTPUT:\n0 tests failed, all green")

    def test_zero_exit_code_with_error_word_is_not_failure(self):
        assert not detect_tool_failure("run_command", "Exit code: 0\nOUTPUT:\nerror: see logs (informational)")

    def test_user_denial_is_not_failure(self):
        assert not detect_tool_failure("run_command", "Execution aborted by user. The command 'rm x' was NOT run.")

    def test_execution_error_without_exit_code_is_failure(self):
        assert detect_tool_failure("run_command", "Error running command 'foo': not found")

    def test_fire_and_forget_nonzero_exit_is_not_failure(self):
        assert not detect_tool_failure("run_command", "Exit code: 1", command="explorer .")
        assert not detect_tool_failure("run_command", "Exit code: 1", command="start notepad.exe")

    def test_write_file_error_prefix_is_failure(self):
        assert detect_tool_failure("write_file", "Error writing file 'x.py': denied")

    def test_write_file_compilation_failed_is_failure(self):
        assert detect_tool_failure("write_file", "File 'x.py' written successfully, BUT COMPILATION FAILED:\nSyntaxError")

    def test_successful_write_is_not_failure(self):
        assert not detect_tool_failure("write_file", "Successfully wrote to 'x.py'.")

    def test_non_healing_tool_is_ignored(self):
        assert not detect_tool_failure("read_file", "Error: File 'x' does not exist.")

    def test_replace_in_file_error_is_failure(self):
        assert detect_tool_failure("replace_in_file", "Error: target text not found in 'x.py'.")


class TestBuildHealingHint:
    def test_hint_contains_attempt_counter(self):
        assert "Attempt 2/3" in build_healing_hint(2)

    def test_exhausted_attempts_produce_stop_message(self):
        assert "AUTO-HEALING FAILED" in build_healing_hint(4)
