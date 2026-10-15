# -*- coding: utf-8 -*-
"""Полный прогон тестов с логом в файл (обход буферизации pytest)."""
import subprocess
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = str(ROOT / ".argent" / "kaggle" / "pytest_full.log")

proc = subprocess.Popen(
    ["python", "-m", "pytest", "tests/", "-q", "--timeout=60", "-p", "no:cacheprovider",
     "--ignore=tests/lmtrust", "--ignore=tests/test_local_inference.py",
     "--ignore=tests/test_needle_coprocessor.py", "--ignore=tests/test_docs_match_reality.py",
     "--tb=no"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace", bufsize=1,
    cwd=str(ROOT),
)
t0 = time.time()
last_log = t0
with open(LOG, "w", encoding="utf-8") as lf:
    while True:
        line = proc.stdout.readline()
        if line:
            last_log = time.time()
            lf.write(line)
            lf.flush()
        if proc.poll() is not None:
            lf.write(f"\nEXIT code={proc.returncode} at {time.time()-t0:.0f}s\n")
            break
        if time.time() - last_log > 600:
            lf.write(f"\nSTALL 600s at {time.time()-t0:.0f}s — kill\n")
            proc.kill()
            break
        if time.time() - t0 > 7200:
            lf.write("\nTIMEOUT 2h\n")
            proc.kill()
            break
print("done", flush=True)