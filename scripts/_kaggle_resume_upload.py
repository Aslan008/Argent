# -*- coding: utf-8 -*-
"""Дозагрузка leeth_best.pt в Kaggle dataset через resumable upload (resume с offset)."""
import os
import sys
import time
from pathlib import Path

import requests

# The resumable-upload session URL carries a one-time upload_id (a credential): give it at run time.
URL = os.environ["KAGGLE_UPLOAD_URL"]
SRC = str(Path(__file__).resolve().parents[1] / ".argent" / "kaggle" / "md8_dataset" / "leeth_best.pt")
TOTAL = 1452176998
CHUNK = 8 * 1024 * 1024

session = requests.Session()
# 1) узнать, сколько уже загружено
r = session.put(URL, headers={"Content-Range": "bytes */%d" % TOTAL, "Content-Length": "0"}, timeout=60)
if r.status_code == 308:
    rng = r.headers.get("Range", "bytes=0--1")
    resume = int(rng.split("-")[1]) + 1
elif r.status_code in (200, 201):
    print("ALREADY_COMPLETE")
    sys.exit(0)
else:
    print("UNEXPECTED", r.status_code, r.text[:200])
    sys.exit(1)
print(f"resume from {resume} ({resume / TOTAL * 100:.1f}%)", flush=True)

t0 = time.time()
with open(SRC, "rb") as f:
    f.seek(resume)
    sent = resume
    while sent < TOTAL:
        chunk = f.read(CHUNK)
        if not chunk:
            break
        end = min(sent + len(chunk) - 1, TOTAL - 1)
        cr = f"bytes {sent}-{end}/{TOTAL}"
        for attempt in range(5):
            try:
                r = session.put(URL, data=chunk,
                                headers={"Content-Range": cr, "Content-Length": str(len(chunk))},
                                timeout=600)
                break
            except Exception as e:
                print(f"retry {attempt + 1}: {type(e).__name__}", flush=True)
                time.sleep(5)
        else:
            print("FAILED after retries")
            sys.exit(1)
        if r.status_code in (200, 201):
            print(f"COMPLETE in {time.time() - t0:.0f}s", flush=True)
            sys.exit(0)
        elif r.status_code == 308:
            rng = r.headers.get("Range", "")
            sent = int(rng.split("-")[1]) + 1 if rng else sent + len(chunk)
            el = time.time() - t0
            speed = (sent - resume) / max(1, el) / 1024
            print(f"{sent / TOTAL * 100:.1f}% ({el:.0f}s, ~{speed:.0f} KB/s)", flush=True)
        else:
            print(f"ERROR {r.status_code}: {r.text[:200]}", flush=True)
            sys.exit(1)
print("DONE", flush=True)