"""
scripts/generate_leeth_diverse_dataset.py — Generates deep procedural datasets for Leeth.

Generates:
1. .argent/kaggle/diverse_dataset/leeth_diverse_train.jsonl (10,000 samples)
2. .argent/kaggle/diverse_dataset/leeth_diverse_val.jsonl   (1,200 samples)
3. .argent/kaggle/diverse_dataset/browser_ranking_train.jsonl (2,000 samples)
4. .argent/kaggle/diverse_dataset/choice_classes.json
"""

import json
import random
from pathlib import Path
from src.leeth.diverse_generator import DiverseUniversalGenerator, CHOICES_ROUTING


def generate_browser_ranking_samples(count: int, seed: int = 42) -> list[dict]:
    """Generates procedural ranking pairs: goal + target element + distractors for BrowserRouter."""
    rng = random.Random(seed)
    samples = []
    
    pages = [
        ("Checkout", [
            {"role": "button", "text": "Оплатить заказ 1450 ₽", "aria_label": "Оплата", "danger": True},
            {"role": "button", "text": "Вернуться в корзину", "aria_label": "Назад", "danger": False},
            {"role": "textbox", "text": "ул. Ленина 14", "aria_label": "Адрес доставки", "danger": False},
            {"role": "combobox", "text": "Банковская карта", "aria_label": "Способ оплаты", "danger": False},
            {"role": "link", "text": "Условия доставки и возврата", "aria_label": "", "danger": False},
        ], [
            ("нажми вернуться в корзину", 1),
            ("выбери способ оплаты", 3),
            ("открой условия доставки", 4),
            ("оплати покупку", 0),
        ]),
        ("Search", [
            {"role": "searchbox", "text": "", "aria_label": "Поиск по сайту", "danger": False},
            {"role": "button", "text": "Найти", "aria_label": "Отправить поиск", "danger": False},
            {"role": "link", "text": "Все категории", "aria_label": "Каталог", "danger": False},
            {"role": "button", "text": "Очистить фильтры", "aria_label": "", "danger": False},
            {"role": "link", "text": "Служба поддержки", "aria_label": "Помощь", "danger": False},
        ], [
            ("нажми кнопку найти", 1),
            ("введи запрос в строку поиска", 0),
            ("сбрось все фильтры", 3),
            ("перейди в каталог категорий", 2),
        ]),
        ("Settings", [
            {"role": "textbox", "text": "developer_99", "aria_label": "Никнейм", "danger": False},
            {"role": "textbox", "text": "dev@work.com", "aria_label": "Электронная почта", "danger": False},
            {"role": "button", "text": "Сохранить профиль", "aria_label": "Сохранение", "danger": False},
            {"role": "button", "text": "Удалить профиль навсегда", "aria_label": "Удаление", "danger": True},
            {"role": "button", "text": "Выйти из системы", "aria_label": "Выход", "danger": False},
        ], [
            ("сохрани настройки профиля", 2),
            ("удали мой аккаунт", 3),
            ("выйди из системы", 4),
            ("измени никнейм", 0),
        ]),
        ("GitHub Repository", [
            {"role": "button", "text": "Code", "aria_label": "Clone or download code", "danger": False},
            {"role": "link", "text": "Pull requests", "aria_label": "Open pull requests list", "danger": False},
            {"role": "link", "text": "Issues", "aria_label": "Project issue tracker", "danger": False},
            {"role": "button", "text": "Fork", "aria_label": "Fork this repository", "danger": False},
            {"role": "button", "text": "Star", "aria_label": "Star this repo", "danger": False},
            {"role": "button", "text": "Merge pull request", "aria_label": "Merge PR #42", "danger": True},
        ], [
            ("открой вкладку issues", 2),
            ("нажми merge pull request", 5),
            ("поставь star проекту", 4),
            ("склонируй репозиторий через code", 0),
            ("сделай fork", 3),
        ]),
        ("Cloud Infrastructure Console", [
            {"role": "button", "text": "Create Instance", "aria_label": "Provision new VM", "danger": False},
            {"role": "button", "text": "Terminate Instance", "aria_label": "Permanently destroy VM", "danger": True},
            {"role": "combobox", "text": "us-east-1 (N. Virginia)", "aria_label": "Region selector", "danger": False},
            {"role": "link", "text": "CloudWatch Logs", "aria_label": "View telemetry logs", "danger": False},
            {"role": "textbox", "text": "my-k8s-cluster", "aria_label": "Cluster identifier", "danger": False},
        ], [
            ("создай новый инстанс", 0),
            ("уничтожь сервер навсегда", 1),
            ("переключи регион на us-east-1", 2),
            ("посмотри логи кластера", 3),
        ]),
        ("Video Streaming Player", [
            {"role": "button", "text": "Play", "aria_label": "Start playback", "danger": False},
            {"role": "button", "text": "Mute", "aria_label": "Turn off audio", "danger": False},
            {"role": "button", "text": "Skip Ad", "aria_label": "Skip advertisement", "danger": False},
            {"role": "slider", "text": "0:45 / 10:20", "aria_label": "Seek progress bar", "danger": False},
            {"role": "button", "text": "Fullscreen", "aria_label": "Expand player", "danger": False},
            {"role": "button", "text": "Subscribe", "aria_label": "Subscribe to channel", "danger": False},
        ], [
            ("пропусти рекламу", 2),
            ("включи полноэкранный режим", 4),
            ("выключи звук", 1),
            ("подпишись на канал", 5),
            ("нажми play", 0),
        ]),
        ("Developer Documentation", [
            {"role": "searchbox", "text": "Search docs...", "aria_label": "Search API reference", "danger": False},
            {"role": "button", "text": "Copy code", "aria_label": "Copy code block to clipboard", "danger": False},
            {"role": "link", "text": "REST API v2", "aria_label": "API specification", "danger": False},
            {"role": "button", "text": "Dark Mode", "aria_label": "Toggle color theme", "danger": False},
            {"role": "combobox", "text": "v3.12 (latest)", "aria_label": "Language version", "danger": False},
        ], [
            ("скопируй пример кода", 1),
            ("переключи тему на тёмную", 3),
            ("найди раздел REST API", 2),
            ("выбери версию документации", 4),
        ]),
    ]
    
    for _ in range(count):
        title, elements, goals = rng.choice(pages)
        goal_text, target_idx = rng.choice(goals)
        samples.append({
            "page_title": title,
            "goal": goal_text,
            "target_index": target_idx,
            "elements": elements,
            "target_element": elements[target_idx],
        })
    return samples


def main():
    out_dir = Path(".argent/kaggle/diverse_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_count = 50000
    val_count = 5000
    browser_count = 5000

    print(f"Генерация train-выборки ({train_count:,} сэмплов)...")
    gen_train = DiverseUniversalGenerator(seed=101, split="train")
    gen_train.generate_and_save(out_dir / "leeth_diverse_train.jsonl", train_count)
    
    print(f"Генерация val-выборки ({val_count:,} сэмплов с изолированным сидом)...")
    gen_val = DiverseUniversalGenerator(seed=808, split="val")
    gen_val.generate_and_save(out_dir / "leeth_diverse_val.jsonl", val_count)
    
    print(f"Генерация браузерных ranking-сэмплов ({browser_count:,} сэмплов)...")
    ranking_data = generate_browser_ranking_samples(browser_count, seed=555)
    with open(out_dir / "browser_ranking_train.jsonl", "w", encoding="utf-8") as f:
        for r in ranking_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Сохраняем словарь классов для роутинга
    with open(out_dir / "choice_classes.json", "w", encoding="utf-8") as f:
        json.dump(CHOICES_ROUTING, f, ensure_ascii=False, indent=2)
        
    # Сохраняем метаданные Kaggle датасета
    meta = {
        "title": "leeth-diverse-data",
        "id": "hunzlee/leeth-diverse-data",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(out_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"\nДатасет успешно сформирован в {out_dir}:")
    for p in out_dir.iterdir():
        if p.is_file():
            print(f"  - {p.name}: {p.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
