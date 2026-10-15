"""
scripts/upload_diverse_dataset.py — Uploads diverse dataset to Kaggle.
"""

import os
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / ".argent" / "kaggle" / "diverse_dataset"

def main():
    os.chdir(DATA_DIR)
    print(f"Working directory: {os.getcwd()}")
    
    # Try create first, if already exists try version
    cmd = ["kaggle", "datasets", "create", "-p", ".", "-r", "zip"]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    
    if "already exists" in res.stdout or "already in use" in res.stdout or "already exists" in res.stderr:
        print("Dataset exists, creating new version...")
        cmd = ["kaggle", "datasets", "version", "-p", ".", "-m", "50k procedural diverse dataset with web ranking", "--dir-mode", "zip"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)

if __name__ == "__main__":
    main()
