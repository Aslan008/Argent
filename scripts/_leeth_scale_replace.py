"""One-off: switch score question texts from the 0..100 scale to the 1-5
ordinal scale (5 identical occurrences). Python-based to avoid the
PowerShell encoding pitfall (non-ASCII files must never round-trip
through Get-Content/Set-Content)."""

from pathlib import Path

P = Path("src/leeth/dataset_generator.py")
text = P.read_text(encoding="utf-8")
n = text.count("по шкале 0..100")
assert n == 5, f"expected 5 occurrences, found {n}"
text = text.replace("по шкале 0..100", "по шкале 1-5")
P.write_text(text, encoding="utf-8")
print(f"replaced {n} occurrences")

check = P.read_text(encoding="utf-8")
assert check.count("по шкале 1-5") == 5
assert "по шкале 0..100" not in check
assert "Оцените критичность" in check
print("OK: cyrillic intact, scale updated")