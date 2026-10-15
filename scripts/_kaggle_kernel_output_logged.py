# -*- coding: utf-8 -*-
"""Скачивание вывода kernel с логом в файл (обход буферизации kaggle CLI)."""
import subprocess
import os
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
LOG = str(ROOT / ".argent" / "kaggle" / "kernel_output.log")

proc = subprocess.Popen(
    ["kaggle", "kernels", "output", "hunzlee/leeth-md8-full-ft", "-p", ".argent/kaggle/kernel_md8_out"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace", bufsize=1,
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
        if time.time() - last_log > 180:
            lf.write(f"\nSTALL 180s at {time.time()-t0:.0f}s — kill\n")
            proc.kill()
            break
        if time.time() - t0 > 1800:
            lf.write("\nTIMEOUT 30m\n")
            proc.kill()
            break
print("done", flush=True)