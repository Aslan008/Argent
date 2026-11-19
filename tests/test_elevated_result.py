"""`run_command(admin=True)` must report the finished command, not its banner.

The elevated child redirects output to a temp file, and the old loop returned as
soon as that file had *any* bytes: the model got the first line and no way to tell
that the command was still running, while the rest of the output was thrown away.
The child now writes a completion marker as its last act, and that is what the
wait is for.
"""

import os
import subprocess
import threading
import time

import pytest

import argent.tools.command_ops as co


def paths(tmp_path):
    out = tmp_path / "admin.txt"
    done = tmp_path / "admin.txt.done"
    return out, done


class TestWaitingForTheMarker:
    def test_the_full_output_arrives_not_just_the_first_bytes(self, tmp_path):
        out, done = paths(tmp_path)
        out.write_text("banner\n", encoding="utf-8")

        def finish_later():
            time.sleep(0.35)
            with out.open("a", encoding="utf-8") as fh:
                fh.write("the real result\n")
            done.write_text("0", encoding="utf-8")

        threading.Thread(target=finish_later, daemon=True).start()
        message = co._collect_elevated_result(out, done, timeout=10)

        assert "banner" in message
        assert "the real result" in message, "the wait ended before the command did"
        assert "exit code 0" in message

    def test_a_nonzero_exit_code_is_reported(self, tmp_path):
        out, done = paths(tmp_path)
        out.write_text("boom\n", encoding="utf-8")
        done.write_text("1", encoding="utf-8")
        assert "exit code 1" in co._collect_elevated_result(out, done, timeout=2)

    def test_a_command_that_never_finishes_keeps_what_it_printed(self, tmp_path):
        out, done = paths(tmp_path)
        out.write_text("started\n", encoding="utf-8")
        message = co._collect_elevated_result(out, done, timeout=0.4)
        assert "did not finish" in message
        assert "started" in message, "partial output is still useful"

    def test_an_empty_output_is_not_an_error(self, tmp_path):
        out, done = paths(tmp_path)
        done.write_text("0", encoding="utf-8")
        message = co._collect_elevated_result(out, done, timeout=1)
        assert "completed" in message


class TestTheWrapperScript:
    def test_it_wraps_the_command_and_names_both_files(self, tmp_path):
        out, done = paths(tmp_path)
        script = co._elevated_wrapper("Write-Output one; Write-Output two", out, done)
        # `cmd1; cmd2 > file` redirects only the last statement: the block matters.
        assert script.startswith("& {")
        assert "Write-Output one; Write-Output two" in script
        assert str(out) in script and str(done) in script

    @pytest.mark.skipif(os.name != "nt", reason="PowerShell 5.1 lives on Windows")
    def test_the_script_really_reports_all_output_and_the_exit_code(self, tmp_path):
        out, done = paths(tmp_path)
        script = co._elevated_wrapper(
            "Write-Output one; Write-Output two; cmd /c exit 5", out, done)
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand",
             co._encode_powershell_command(script)],
            capture_output=True, timeout=120)

        assert done.read_text(encoding="utf-8").strip() == "5"
        text = co._read_process_output_file(out)
        assert "one" in text, "the first statement's output was lost"
        assert "two" in text


class TestReadingTheChildsOutput:
    @pytest.mark.parametrize("payload,encoding", [
        ("привет\n", "utf-16"),          # PowerShell's own default with -EncodedCommand
        ("привет\n", "utf-8-sig"),
        ("привет\n", "utf-8"),
        ("привет\n", "cp1251"),
    ])
    def test_every_encoding_the_child_might_pick(self, tmp_path, payload, encoding):
        out, _ = paths(tmp_path)
        out.write_bytes(payload.encode(encoding))
        assert co._read_process_output_file(out) == "привет"
