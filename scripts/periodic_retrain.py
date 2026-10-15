# -*- coding: utf-8 -*-
"""scripts/periodic_retrain.py — Непрерывный процесс: retrain из feedback-петли (⏳).

Зачем: модель доучивается на реальных ошибках, пока мы работаем. Скрипт
берёт накопленный feedback (393 записи и растёт), дообучает модель с
replay-защитой от катастрофического забывания и валидационным гейтом.

Запуск: вручную или по расписанию (раз в неделю / при +200 записей).
БЕЗ --run: dry-run (сколько записей накопилось, хватит ли на retrain).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FEEDBACK_PATH = Path("src/leeth/data/leeth_feedback.jsonl")
MIN_SAMPLES = 20


def feedback_stats() -> dict:
    if not FEEDBACK_PATH.exists():
        return {"records": 0}
    records = 0
    labeled = 0
    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                rec = json.loads(line)
                # Labeled = есть вердикт ревьюера (как в feedback.py: _labeled_records).
                if rec.get("verdict"):
                    labeled += 1
            except Exception:
                pass
    return {"records": records, "labeled": labeled}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="запустить retrain (долгая операция)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--model-dir", default="src/leeth/data_400m",
                        help="какую модель дообучать (прод = data_400m)")
    args = parser.parse_args()

    stats = feedback_stats()
    enough = stats.get("records", 0) >= MIN_SAMPLES

    if not args.run:
        print("\n=== DRY-RUN (долгая операция НЕ запущена) ===")
        print(f"Feedback:    {stats['records']} записей (labeled: {stats.get('labeled', 0)})")
        print(f"Достаточно:  {'ДА' if enough else 'НЕТ'} (нужно >= {MIN_SAMPLES})")
        print(f"Обучение:    finetune_from_feedback, {args.epochs} эпох, lr={args.lr}")
        print("\nДля запуска: python scripts/periodic_retrain.py --run")
        return

    # ── Долгая операция ⏳ ────────────────────────────────────────────────────
    import os
    os.environ.setdefault("LEETH_FEEDBACK_DISABLED", "1")
    from src.leeth.finetune import finetune_from_feedback

    summary = finetune_from_feedback(epochs=args.epochs, learning_rate=args.lr,
                                     data_dir=args.model_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if summary.get("status") == "not_enough_samples":
        print("Недостаточно размеченных записей — продолжайте работу с review-режимом ON.")


if __name__ == "__main__":
    main()