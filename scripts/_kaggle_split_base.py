# -*- coding: utf-8 -*-
"""scripts/_kaggle_split_base.py — нарезка model.safetensors на части по 20 MB.

Паттерн md8-трека: канал ~100 KB/s, обрывы DNS — части устойчивы к рестартам.
Склейка в кернеле: mmbert-argent-fp16-parts/part_* -> model.safetensors.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path(".argent/kaggle/du_dataset/_assembled/model.safetensors")
PARTS_DIR = Path(".argent/kaggle/du_dataset")
CHUNK = 20 * 1024 * 1024


def main():
    total = SRC.stat().st_size
    md5 = hashlib.md5()
    PARTS_DIR.mkdir(exist_ok=True)
    parts = []
    with open(SRC, "rb") as f:
        i = 0
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            p = PARTS_DIR / f"part_{i:03d}"
            p.write_bytes(chunk)
            md5.update(chunk)
            parts.append({"name": p.name, "size": len(chunk)})
            i += 1
    manifest = {"total": total, "md5": md5.hexdigest(), "n_parts": len(parts), "parts": parts}
    (PARTS_DIR / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"частей: {len(parts)}, total {total} bytes, md5 {md5.hexdigest()}")


if __name__ == "__main__":
    main()