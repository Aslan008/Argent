"""Admin commands go through -EncodedCommand: quotes survive verbatim."""

import base64

from tools.command_ops import _encode_powershell_command


def _decode(encoded: str) -> str:
    return base64.b64decode(encoded).decode("utf-16-le")


def test_roundtrips_plain_command():
    assert _decode(_encode_powershell_command("Get-Process")) == "Get-Process"


def test_double_quotes_survive():
    # The exact case that broke -Command "…": embedded double quotes.
    script = 'Write-Host "hello world"'
    assert _decode(_encode_powershell_command(script)) == script


def test_mixed_quotes_and_redirect():
    script = "Write-Output \"a'b\" > 'C:\\Temp\\out.txt' 2>&1"
    assert _decode(_encode_powershell_command(script)) == script


def test_output_is_ascii_base64():
    # -EncodedCommand needs a pure-ASCII base64 token on the command line.
    enc = _encode_powershell_command('Write-Host "юникод тоже"')
    enc.encode("ascii")                       # must not raise
    assert _decode(enc) == 'Write-Host "юникод тоже"'
