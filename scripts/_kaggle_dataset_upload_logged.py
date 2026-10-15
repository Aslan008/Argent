# -*- coding: utf-8 -*-
"""Загрузка датасета leeth-md8 (версия с фиксом) с логом в файл."""
import subprocess
import os
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT / ".argent" / "kaggle" / "md8_dataset")
LOG = str(ROOT / ".argent" / "kaggle" / "md8_dataset_upload.log")

proc = subprocess.Popen(
    ["kaggle", "datasets", "version", "-p", ".", "-m", "fix Union import v3", "--dir-mode", "zip"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace", bufsize=1,
)
t0 = time.time()
last_log = t0
n100 = 0
with open(LOG, "w", encoding="utf-8") as lf:
    while True:
        line = proc.stdout.readline()
        if line:
            last_log = time.time()
            lf.write(line)
            lf.flush()
            if "100%|" in line:
                n100 += 1
        if proc.poll() is not None:
            lf.write(f"\nEXIT code={proc.returncode} at {time.time()-t0:.0f}s, files100={n100}\n")
            break
        if time.time() - last_log > 300:
            lf.write(f"\nSTALL 300s at {time.time()-t0:.0f}s — kill\n")
            proc.kill()
            break
        if time.time() - t0 > 14400:
            lf.write("\nTIMEOUT 4h\n")
            proc.kill()
            break
print("done", flush=True)