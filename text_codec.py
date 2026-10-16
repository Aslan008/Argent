"""Which encoding a text file is in, so it is read as text and an edit writes it back the way it was.

Every file tool opened files as UTF-8, and read_file called anything with a NUL byte in its first 4 KB
"binary". On Windows that is wrong about a whole class of ordinary text: PowerShell 5.1's `>` writes
UTF-16LE, so `script.py > out.log` produced a log the agent could not read ("file appears to be binary"),
and an edit of a UTF-16 or ANSI (cp1251) file decoded it with errors="replace" and wrote the replacement
characters back - the file was silently destroyed.

`sniff` decides from the first bytes; `decode` and `read_text` return the text with the codec that read
it, and `write_text` writes with that codec, so a round trip keeps the encoding, the BOM and the line ends.
"""

from __future__ import annotations

import codecs
import locale
from pathlib import Path
from typing import Optional, Tuple

#: how much of a file `sniff` looks at
HEAD_BYTES = 64 * 1024

_BOMS = (
    # the UTF-32 marks first: the UTF-32LE BOM starts with the UTF-16LE one
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def _legacy_codec() -> str:
    """The ANSI code page text files are written in when they are not UTF-8 (cp1251 on a Russian Windows)."""
    try:
        name = locale.getpreferredencoding(False) or ""
        codecs.lookup(name)
    except (LookupError, TypeError, ValueError):
        return "latin-1"
    # a UTF-8 locale gives no second guess: latin-1 at least decodes every byte
    return "latin-1" if codecs.lookup(name).name == "utf-8" else name


def _utf16_without_bom(head: bytes) -> Optional[str]:
    """'utf-16-le' / 'utf-16-be' when the bytes look like UTF-16 text with no BOM, else None.

    Text in UTF-16 that is mostly ASCII (logs, code) has a zero in every other byte: at the odd
    positions for little endian, at the even ones for big endian. Binary data has zeros anywhere."""
    sample = head[: len(head) - (len(head) % 2)]
    pairs = len(sample) // 2
    if pairs < 2:
        return None
    zeros_even = sum(1 for b in sample[0::2] if b == 0)
    zeros_odd = sum(1 for b in sample[1::2] if b == 0)
    for codec, mostly, rarely in (("utf-16-le", zeros_odd, zeros_even), ("utf-16-be", zeros_even, zeros_odd)):
        if mostly >= pairs * 0.3 and rarely <= pairs * 0.02:
            try:
                codecs.getincrementaldecoder(codec)().decode(sample, final=False)
            except UnicodeDecodeError:
                continue
            return codec
    return None


def _valid_utf8_prefix(head: bytes) -> bool:
    try:
        # incremental and not final: a multibyte character cut by the end of the sample is fine
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
        return True
    except UnicodeDecodeError:
        return False


def sniff(head: bytes) -> Optional[str]:
    """The codec to read a file that starts with `head`, or None when it is binary."""
    for bom, codec in _BOMS:
        if head.startswith(bom):
            return codec
    if b"\x00" in head:
        return _utf16_without_bom(head)
    return "utf-8" if _valid_utf8_prefix(head) else _legacy_codec()


def sniff_file(path) -> Optional[str]:
    with open(path, "rb") as f:
        return sniff(f.read(HEAD_BYTES))


def decode(data: bytes) -> Tuple[str, str]:
    """(text, codec) for the whole content. Raises ValueError for binary data.

    The codec is checked against ALL the bytes, not just the head: a UTF-8 file whose only Cyrillic
    comment is past the first 64 KB is still UTF-8, and one with a stray cp1251 byte there is not."""
    codec = sniff(data[:HEAD_BYTES])
    if codec is None:
        raise ValueError("binary content")
    if codec == "utf-8":
        try:
            return data.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            codec = _legacy_codec()
    return data.decode(codec, errors="replace"), codec


def read_text(path) -> Tuple[str, str]:
    """(text, codec) of a file, line ends untouched (like open(..., newline=""))."""
    return decode(Path(path).read_bytes())


def write_text(path, text: str, codec: str = "utf-8") -> None:
    """Write `text` with `codec`, line ends as they are in the text.

    A codec that cannot hold a character the edit introduced (an emoji into a cp1251 file) would make
    the write fail halfway; such a file is written as UTF-8 instead, which keeps every character."""
    try:
        data = text.encode(codec)
    except (UnicodeEncodeError, LookupError):
        data = text.encode("utf-8")
    Path(path).write_bytes(data)
