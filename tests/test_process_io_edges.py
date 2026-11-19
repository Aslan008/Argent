"""Two process-I/O edges: a tail cut mid-character, and a POSIX child's group.

1. Trimming the output tail to its byte budget could slice a multi-byte UTF-8
   symbol in half. The buffer then failed to decode and the fallback turned the
   whole 16 KB tail into mojibake.
2. On POSIX the child shared Argent's process group, so `kill_process_tree` could
   not use `killpg` (it would kill us too) and fell back to killing only /bin/sh —
   the real work survived as an orphan.
"""

from types import SimpleNamespace

import pytest

import argent.tools.command_ops as co


class TestTheTailKeepsItsEncoding:
    def test_a_symbol_cut_in_half_is_dropped_not_the_whole_tail(self):
        data = "привет".encode("utf-8")          # 12 bytes, 2 per symbol
        tail = co._OutputTail(limit=9)           # cuts 3 bytes -> mid-symbol
        tail.add(data)
        assert tail.text() == "ивет", "the tail no longer decodes as UTF-8"

    def test_a_clean_boundary_is_left_alone(self):
        data = "привет".encode("utf-8")
        tail = co._OutputTail(limit=10)          # cuts exactly one symbol ("п")
        tail.add(data)
        assert tail.text() == "ривет"

    def test_the_helper_ignores_ascii(self):
        data = b"hello"
        assert co._drop_partial_leading_char(data) is data

    def test_legacy_codepage_bytes_are_not_dropped(self):
        # In cp866 a 0x80-0xBF byte is an ordinary character, not a continuation.
        data = "Привет мир".encode("cp866")
        assert co._drop_partial_leading_char(data) == data

    def test_a_broken_utf8_tail_is_returned_unchanged(self):
        data = b"\xbf\xbf\xbf"                   # continuation bytes, nothing valid after
        assert co._drop_partial_leading_char(data) == data


class TestASpawnedChildGetsItsOwnSession:
    def test_posix_child_gets_its_own_process_group(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(co, "os", SimpleNamespace(name="posix"))
        monkeypatch.setattr(co, "subprocess", SimpleNamespace(
            Popen=lambda *a, **kw: (captured.update(kw), "proc")[1]))

        assert co._spawn("echo hi") == "proc"
        assert captured["start_new_session"] is True, \
            "the child shares Argent's group, so the tree cannot be killed"

    def test_an_explicit_setting_wins(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(co, "os", SimpleNamespace(name="posix"))
        monkeypatch.setattr(co, "subprocess", SimpleNamespace(
            Popen=lambda *a, **kw: (captured.update(kw), "proc")[1]))

        co._spawn("echo hi", start_new_session=False)
        assert captured["start_new_session"] is False
