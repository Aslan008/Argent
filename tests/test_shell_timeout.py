"""A timeout that actually fires, and a child that cannot ask questions.

replace_in_file on a .ts file validated it with `npx tsc`, which offers to
DOWNLOAD tsc and then waits for "Ok to proceed? (y)" — on a captured pipe,
where the question is invisible and unanswerable. The call passed timeout=10.
It hung for 12489 seconds and was ended by Ctrl+C.

Two independent reasons, both fixed here:

* stdin was inherited, so the child waited on a human;
* subprocess.run's timeout kills only the DIRECT child, which under shell=True
  on Windows is cmd.exe. Its grandchild keeps the stdout pipe open, and the
  communicate() that follows the kill waits for an EOF that never comes.
"""

import subprocess
import sys
import time

import pytest

from src.agent.shell import run_text


class TestStdinIsClosed:
    def test_a_child_that_asks_a_question_does_not_wait(self, tmp_path):
        """The reported hang: npx asks, nobody can answer, everything stops."""
        script = tmp_path / "ask.py"
        script.write_text(
            "import sys\n"
            "sys.stdout.write('Ok to proceed? (y) ')\n"
            "sys.stdout.flush()\n"
            "sys.stdin.readline()\n", encoding="utf-8")

        started = time.monotonic()
        result = run_text([sys.executable, str(script)], capture_output=True, timeout=15)
        assert time.monotonic() - started < 10        # not a timeout, an exit
        assert result.returncode == 0
        assert "Ok to proceed?" in result.stdout      # and the text is not lost

    def test_an_explicit_stdin_is_respected(self, tmp_path):
        """Closing stdin is a default, not a policy: a caller that means to feed
        the child must still be able to."""
        script = tmp_path / "echo.py"
        script.write_text("import sys; print('got:', sys.stdin.readline().strip())",
                          encoding="utf-8")
        result = run_text([sys.executable, str(script)], capture_output=True,
                          input="привет\n", timeout=15)
        assert "got: привет" in result.stdout


class TestTimeoutKillsTheTree:
    def test_a_grandchild_holding_the_pipe_cannot_block_the_timeout(self, tmp_path):
        """This is why timeout=10 produced a hang of three and a half hours."""
        script = tmp_path / "spawner.py"
        script.write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "time.sleep(120)\n", encoding="utf-8")

        started = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            run_text([sys.executable, str(script)], capture_output=True, timeout=3)
        assert time.monotonic() - started < 20        # bounded, not forever

    def test_a_plain_slow_command_still_times_out(self, tmp_path):
        script = tmp_path / "slow.py"
        script.write_text("import time; time.sleep(60)", encoding="utf-8")
        with pytest.raises(subprocess.TimeoutExpired):
            run_text([sys.executable, str(script)], capture_output=True, timeout=2)


class TestUnchangedBehaviour:
    def test_output_is_captured_and_decoded(self, tmp_path):
        script = tmp_path / "out.py"
        script.write_text("print('привет мир')", encoding="utf-8")
        result = run_text([sys.executable, str(script)], capture_output=True, timeout=15)
        assert "привет мир" in result.stdout and result.returncode == 0

    def test_a_failing_command_reports_its_code(self, tmp_path):
        script = tmp_path / "fail.py"
        script.write_text("import sys; sys.stderr.write('плохо'); sys.exit(3)",
                          encoding="utf-8")
        result = run_text([sys.executable, str(script)], capture_output=True, timeout=15)
        assert result.returncode == 3 and "плохо" in result.stderr

    def test_undecodable_bytes_do_not_crash_the_reader(self, tmp_path):
        """Windows tools emit the OEM code page; strict UTF-8 would raise inside
        subprocess's reader thread."""
        script = tmp_path / "bad.py"
        script.write_text("import sys; sys.stdout.buffer.write(b'ok\\xff\\xfe')",
                          encoding="utf-8")
        result = run_text([sys.executable, str(script)], capture_output=True, timeout=15)
        assert "ok" in result.stdout

    def test_without_a_timeout_the_simple_path_is_used(self, tmp_path):
        script = tmp_path / "out.py"
        script.write_text("print('без таймаута')", encoding="utf-8")
        result = run_text([sys.executable, str(script)], capture_output=True)
        assert "без таймаута" in result.stdout


class TestValidationNeverInstalls:
    def test_tsc_is_invoked_with_no_install(self):
        """npx without --no-install offers to download the package and blocks on
        the confirmation. Validation must never install anything; a project
        without a local tsc simply goes unchecked."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "tools" / "_helpers.py").read_text(
            encoding="utf-8")
        assert '"npx", "--no-install", "tsc"' in source
