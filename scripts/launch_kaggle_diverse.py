"""
scripts/launch_kaggle_diverse.py — Pushes and monitors the Leeth Diverse Training kernel on Kaggle.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

KERNEL_DIR = Path(__file__).resolve().parents[1] / ".argent" / "kaggle" / "kernel_diverse"

def main():
    os.chdir(KERNEL_DIR)
    print(f"Working directory: {os.getcwd()}")
    
    cmd = ["kaggle", "kernels", "push", "-p", "."]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    
    if res.returncode == 0 or "Kernel version" in res.stdout:
        print("\nУспешно отправлено на Kaggle!")
        time.sleep(3)
        status_cmd = ["kaggle", "kernels", "status", "hunzlee/leeth-diverse-deep-train"]
        st = subprocess.run(status_cmd, capture_output=True, text=True)
        print("Текущий статус кернела на Kaggle:", st.stdout.strip())

if __name__ == "__main__":
    main()
