# -*- coding: utf-8 -*-
"""scripts/bench_mmbert.py — Фаза E1: гейт mmBERT (скачивание + бенчмарк).

Зачем: ДО всех вложений в миграцию проверяем главный гейт — латентность на
CPU. Если mmBERT-base медленнее 100 мс на одиночный вопрос, фаза отменяется
(наш прод-путь — CPU INT8, 44-61 мс; медленная база убьёт нишу System 1).

Скачивание (~600 МБ) — не обучение, можно запускать сразу (идёт в фоне).
Бенчмарк — быстрый инференс.

Запуск: python scripts/bench_mmbert.py [--download] [--seq 36]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_ID = "jhu-clsp/mmBERT-base"
CACHE_DIR = Path(".argent/models/mmbert-base")


def ensure_downloaded() -> bool:
    """Скачивает модель в локальный кэш (I/O, не обучение)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub не установлен: pip install huggingface_hub")
        return False
    marker = CACHE_DIR / ".downloaded"
    if marker.exists():
        print(f"Already downloaded: {CACHE_DIR}")
        return True
    print(f"Downloading {MODEL_ID} -> {CACHE_DIR} ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(CACHE_DIR),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model", "pytorch_model*.bin"],
        )
        marker.write_text("ok", encoding="utf-8")
        print("Download OK")
        return True
    except Exception as exc:
        print(f"Download failed: {exc}")
        return False


def bench(seq_len: int = 36, rounds: int = 20) -> dict:
    import time
    import numpy as np
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(CACHE_DIR))
    model = AutoModelForMaskedLM.from_pretrained(str(CACHE_DIR), dtype=torch.float32)
    model.eval()
    load_s = time.perf_counter() - t0

    # Теплый прогон
    ids = torch.randint(5, 1000, (1, seq_len))
    mask = torch.ones(1, seq_len, dtype=torch.long)
    with torch.no_grad():
        for _ in range(5):
            model(input_ids=ids, attention_mask=mask)

    ts = []
    with torch.no_grad():
        for _ in range(rounds):
            t = time.perf_counter()
            model(input_ids=ids, attention_mask=mask)
            ts.append((time.perf_counter() - t) * 1000)

    # INT8 dynamic quantization — наш прод-путь (как у Leeth INT8 ONNX).
    qmodel = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()
    with torch.no_grad():
        for _ in range(5):
            qmodel(input_ids=ids, attention_mask=mask)
    qts = []
    with torch.no_grad():
        for _ in range(rounds):
            t = time.perf_counter()
            qmodel(input_ids=ids, attention_mask=mask)
            qts.append((time.perf_counter() - t) * 1000)

    return {
        "model": MODEL_ID,
        "seq_len": seq_len,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "fp32_p50_ms": round(float(np.percentile(ts, 50)), 1),
        "int8_p50_ms": round(float(np.percentile(qts, 50)), 1),
        "int8_p95_ms": round(float(np.percentile(qts, 95)), 1),
        "load_s": round(load_s, 1),
        "gate_pass": bool(np.percentile(qts, 50) <= 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="скачать модель (~600 МБ)")
    parser.add_argument("--seq", type=int, default=36)
    args = parser.parse_args()

    if args.download:
        if not ensure_downloaded():
            sys.exit(1)

    if not (CACHE_DIR / ".downloaded").exists():
        print("Модель не скачана. Запусти: python scripts/bench_mmbert.py --download")
        print("Гейт E1: p50 <= 100 мс на CPU (seq=36). Иначе Фаза E отменяется.")
        return

    result = bench(args.seq)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    verdict = "PASS — mmBERT проходит гейт латентности" if result["gate_pass"] else "FAIL — mmBERT слишком медленный для CPU, фаза E отменяется"
    print(verdict)


if __name__ == "__main__":
    main()