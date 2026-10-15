# -*- coding: utf-8 -*-
import subprocess
import time
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

kernel_name = "hunzlee/leeth-diverse-deep-train"
print(f"Начало ожидания завершения кернела {kernel_name}...")

while True:
    res = subprocess.run(["kaggle", "kernels", "status", kernel_name], capture_output=True, text=True)
    out = res.stdout + res.stderr
    if "KernelWorkerStatus.COMPLETE" in out:
        print(f"Кернел {kernel_name} успешно завершился (COMPLETE)!")
        break
    elif "KernelWorkerStatus.ERROR" in out or "error" in out.lower() and "exception" in out.lower():
        print(f"Ошибка в кернеле {kernel_name}:\n{out}")
        break
    time.sleep(15)
