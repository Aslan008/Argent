"""One-off recovery: reverse the CP1251 mojibake introduced by a PowerShell
Get-Content/Set-Content round-trip on a UTF-8 (no BOM) file. ASCII code is
untouched; the transform is lossless for CP1251-representable text."""

from pathlib import Path

P = Path("src/leeth/dataset_generator.py")
raw = P.read_bytes()
had_bom = raw.startswith(b"\xef\xbb\xbf")
if had_bom:
    raw = raw[3:]
moji = raw.decode("utf-8")
orig_bytes = moji.encode("cp1251")
P.write_bytes(orig_bytes)
print(f"restored {len(orig_bytes)} bytes (bom was: {had_bom})")

check = P.read_bytes().decode("utf-8")
n = check.count("Оцените критичность")
print(f"cyrillic check: {n} occurrences of 'Оцените критичность'")
assert n >= 5, "recovery failed"
print("OK")