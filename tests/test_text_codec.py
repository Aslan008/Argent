"""Text in any common encoding is read as text, and an edit keeps the encoding.

Reported: `read_file` on a log written by PowerShell's `>` said "file appears to be binary" - PowerShell 5.1
writes UTF-16LE, and the tool treated any NUL byte as binary. Worse, the edit tools decoded every file as
UTF-8 with errors="replace" and wrote the result back, so editing a UTF-16 or cp1251 file destroyed it.
"""

import codecs

import pytest

import text_codec
from text_codec import decode, sniff, write_text


TEXT = "строка один\r\nline two: 42\r\n"


# ── sniffing ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("data, codec", [
    (codecs.BOM_UTF16_LE + TEXT.encode("utf-16-le"), "utf-16"),
    (codecs.BOM_UTF16_BE + TEXT.encode("utf-16-be"), "utf-16"),
    (codecs.BOM_UTF8 + TEXT.encode("utf-8"), "utf-8-sig"),
    (codecs.BOM_UTF32_LE + TEXT.encode("utf-32-le"), "utf-32"),
    ("plain ascii\n".encode(), "utf-8"),
    (TEXT.encode("utf-8"), "utf-8"),
    ("Exit code: 0\nhello world\n".encode("utf-16-le"), "utf-16-le"),
    ("Exit code: 0\nhello world\n".encode("utf-16-be"), "utf-16-be"),
])
def test_text_is_recognised_with_its_encoding(data, codec):
    assert sniff(data) == codec


def test_binary_is_still_binary():
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00\x00\x00\x01\x00\x08\x06\x00\x00\x00"
    assert sniff(png) is None
    assert sniff(bytes(range(256)) * 4) is None


def test_a_non_utf8_file_falls_back_to_the_ansi_code_page(monkeypatch):
    monkeypatch.setattr(text_codec, "_legacy_codec", lambda: "cp1251")
    data = "Привет, мир".encode("cp1251")
    assert sniff(data) == "cp1251"
    assert decode(data) == ("Привет, мир", "cp1251")


def test_utf8_is_checked_past_the_head(monkeypatch):
    monkeypatch.setattr(text_codec, "_legacy_codec", lambda: "cp1251")
    data = b"a" * (text_codec.HEAD_BYTES + 10) + "конец".encode("cp1251")
    text, codec = decode(data)
    assert codec == "cp1251" and text.endswith("конец")


def test_a_multibyte_character_cut_by_the_head_is_not_a_reason_to_doubt_utf8():
    data = b"a" * (text_codec.HEAD_BYTES - 1) + "ж".encode("utf-8")
    assert sniff(data[:text_codec.HEAD_BYTES]) == "utf-8"
    assert decode(data)[1] == "utf-8"


# ── round trip ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("codec", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "cp1251"])
def test_a_round_trip_keeps_the_bytes(tmp_path, monkeypatch, codec):
    monkeypatch.setattr(text_codec, "_legacy_codec", lambda: "cp1251")
    path = tmp_path / "f.txt"
    original = TEXT.encode(codec)
    path.write_bytes(original)
    text, found = text_codec.read_text(path)
    assert text == TEXT
    write_text(path, text, found)
    assert path.read_bytes() == original


def test_a_character_the_codec_cannot_hold_writes_utf8_instead_of_failing(tmp_path):
    path = tmp_path / "f.txt"
    write_text(path, "тест ✓", "cp1251")
    assert path.read_bytes().decode("utf-8") == "тест ✓"


# ── the tools ────────────────────────────────────────────────────────────────

def test_read_file_reads_a_powershell_log(tmp_path, monkeypatch):
    from tools.file_ops import read_file
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "miss_diff_v2.log"
    log.write_bytes(codecs.BOM_UTF16_LE + "Бенигн прошли: 3/6\r\nExit code: 0\r\n".encode("utf-16-le"))
    out = read_file(str(log))
    assert "binary" not in out
    assert "Бенигн прошли: 3/6" in out and "Exit code: 0" in out


def test_read_file_still_refuses_binary(tmp_path, monkeypatch):
    from tools.file_ops import read_file
    monkeypatch.chdir(tmp_path)
    blob = tmp_path / "x.bin"
    blob.write_bytes(bytes(range(256)) * 8)
    assert "binary" in read_file(str(blob))


@pytest.mark.parametrize("codec", ["utf-16", "cp1251", "utf-8-sig"])
def test_replace_in_file_keeps_the_encoding(tmp_path, monkeypatch, codec):
    from tools.file_ops import replace_in_file
    monkeypatch.setattr(text_codec, "_legacy_codec", lambda: "cp1251")
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "notes.txt"
    path.write_bytes("Заголовок\r\nстарое значение\r\nконец\r\n".encode(codec))
    out = replace_in_file(str(path), "старое значение", "новое значение")
    assert "Success" in out or "success" in out.lower(), out
    assert path.read_bytes() == "Заголовок\r\nновое значение\r\nконец\r\n".encode(codec)
