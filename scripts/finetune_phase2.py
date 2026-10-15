# -*- coding: utf-8 -*-
"""scripts/finetune_phase2.py — Фаза A3: запуск дообучения Noul+MLM (ДОЛГАЯ ОПЕРАЦИЯ ⏳).

Зачем: модель читает маскированные деструктивы как безопасные. Дообучаем
Noul-голову на solver-данных (арбитр = оракул меток). Choice-голова НЕ
трогается — роутинг остаётся как был.

БЕЗ --run: печатает dry-run (что будет обучаться, на чём, сколько времени).
С --run: запускает finetune_from_feedback с extra_replay из сгенерированных
датасетов. Валидационный гейт — после обучения (A4).

Запуск (по команде пользователя): python scripts/finetune_phase2.py --run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG_PATH = Path("scripts/finetune_phase2.json")


def load_extra_samples(paths: list) -> list:
    from src.leeth.dataset_generator import LeethSample
    samples = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            print(f"  [skip] {path} не существует")
            continue
        with open(path, "r", encoding="utf-8") as f:
            n = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    clean = {k: v for k, v in rec.items() if not k.startswith("_")}
                    samples.append(LeethSample(**clean))
                    n += 1
                except Exception:
                    continue
            print(f"  loaded {n:5d} <- {path}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
                        help="ЗАПУСТИТЬ обучение (долгая операция). Без флага — dry-run.")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    extra = load_extra_samples(config["extra_replay_files"])

    if not args.run:
        print("\n=== DRY-RUN (долгая операция НЕ запущена) ===")
        print(f"Модель:      {config['model_dir']}/leeth_best.pt")
        print(f"Обучение:    Noul + MLM головы, {config['epochs']} эпох, lr={config['learning_rate'] if 'learning_rate' in config else config.get('lr', 1e-5)}")
        print(f"Extra replay: {len(extra)} семплов")
        print(f"Гейты:       {json.dumps(config['gates'], ensure_ascii=False)}")
        print(f"Оценка времени: {config['estimated_time']}")
        print("\nДля запуска: python scripts/finetune_phase2.py --run")
        return

    # ── Долгая операция ⏳ ────────────────────────────────────────────────────
    import os
    os.environ.setdefault("LEETH_FEEDBACK_DISABLED", "1")
    from src.leeth.finetune import finetune_from_feedback

    summary = finetune_from_feedback(
        epochs=config["epochs"],
        learning_rate=config.get("learning_rate", 1e-5),
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        replay_ratio=config["replay_ratio"],
        replay_max=config["replay_max"],
        correction_boost=config["correction_boost"],
        extra_replay=extra,
        data_dir=config["model_dir"],  # ПРОД: data_400m, иначе тренируется старый чекпоинт
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()