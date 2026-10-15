# -*- coding: utf-8 -*-
"""scripts/generate_browser_pairs.py — Фаза G1: датасет браузерных решений.

Зачем: ranking-голова для браузера учится выбирать верный элемент из списка
DOM против цели. Формат: (цель, список элементов, индекс верного).
Генерируем шаблонами: цель + страница с верным элементом и дистракторами
(похожие кнопки, другие ссылки, поля ввода).

Выход: src/leeth/data/browser_pairs.jsonl
  {"goal", "elements": [{"text","tag","role"}...], "target_index", "split"}

Запуск: python scripts/generate_browser_pairs.py [--n 2000]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_PATH = Path("src/leeth/data/browser_pairs.jsonl")

# (goal template, element-pool key)
GOALS = [
    ("нажми кнопку {item}", "buttons"),
    ("кликни {item}", "buttons"),
    ("найди и нажми {item}", "buttons"),
    ("открой раздел {item}", "links"),
    ("перейди в {item}", "links"),
    ("заполни поле {item}", "inputs"),
    ("введи значение в {item}", "inputs"),
    ("выбери опцию {item} в списке", "options"),
]

BUTTONS = ["Save", "Submit", "Cancel", "Next", "Previous", "Download", "Upload",
           "Search", "Login", "Sign up", "Continue", "Back", "Confirm", "Close",
           "Apply", "Reset", "Delete", "Edit", "Create", "Export"]
LINKS = ["Documentation", "Pricing", "Downloads", "API Reference", "Blog", "Support",
         "Community", "Tutorials", "Changelog", "Releases", "Settings", "Profile"]
FIELDS = ["email", "password", "username", "search", "comment", "phone", "address",
          "card number", "verification code", "file name"]
OPTIONS = ["United States", "Germany", "Russia", "Option A", "Option B", "Option C",
           "Ascending", "Descending", "Dark theme", "Light theme"]

POOLS = {
    "buttons": [(b, "button", "button") for b in BUTTONS],
    "links": [(l, "a", "link") for l in LINKS],
    "inputs": [(f, "input", "textbox") for f in FIELDS],
    "options": [(o, "option", "option") for o in OPTIONS],
}

DANGEROUS = {"delete", "confirm", "pay", "checkout", "purchase", "submit payment",
             "place order", "remove", "erase", "authorize"}


def make_page(rng: random.Random, target: tuple) -> list:
    """Строит список элементов страницы: верный + 4-8 дистракторов."""
    target_text, target_tag, target_role = target
    all_items = [item for pool in POOLS.values() for item in pool]
    distractors = [item for item in all_items if item[0] != target_text]
    # Prefer same-pool distractors (hard negatives: похожие кнопки).
    same_pool = [item for item in distractors if item[1] == target_tag]
    other_pool = [item for item in distractors if item[1] != target_tag]
    rng.shuffle(same_pool)
    rng.shuffle(other_pool)
    n_distr = rng.randint(4, 8)
    chosen = (same_pool[: max(2, n_distr // 2)] + other_pool[: n_distr - max(2, n_distr // 2)])[:n_distr]
    items = chosen + [target]
    rng.shuffle(items)
    return [{"text": t, "tag": tag, "role": role} for (t, tag, role) in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    rng = random.Random(777)
    rows: list = []
    n_dangerous = 0
    for _ in range(args.n):
        goal_tpl, pool_key = rng.choice(GOALS)
        target = rng.choice(POOLS[pool_key])
        goal = goal_tpl.format(item=target[0])
        elements = make_page(rng, target)
        target_index = next(j for j, e in enumerate(elements) if e["text"] == target[0])
        is_dangerous = any(w in elements[target_index]["text"].lower() for w in DANGEROUS)
        n_dangerous += is_dangerous
        rows.append({
            "goal": goal,
            "elements": elements,
            "target_index": target_index,
            "dangerous": is_dangerous,
            "split": "val" if rng.random() < args.val_ratio else "train",
        })

    rng.shuffle(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_val = sum(1 for r in rows if r["split"] == "val")
    print(f"Written {len(rows)} rows ({n_val} val, {n_dangerous} dangerous-target) -> {OUT_PATH}")


if __name__ == "__main__":
    main()