# -*- coding: utf-8 -*-
"""scripts/generate_skill_pairs.py — Фаза F2: датасет «запрос -> верный скилл».

Зачем: ranking-голова (bi-encoder) учится сопоставлять запрос пользователя
и описание скилла. Для обучения нужны тройки (запрос, верный скилл,
дистракторы). Генерируем шаблонно: домен-специфичные фразы + реальные имена
скиллов из skills/. Дистракторы — похожие по словам, но чужие скиллы.

Выход: src/leeth/data/skill_pairs.jsonl
  {"query", "positive", "distractors": [...], "split": "train|val"}

Запуск: python scripts/generate_skill_pairs.py [--n 2000]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_PATH = Path("src/leeth/data/skill_pairs.jsonl")

# Реальные скиллы Argent (skills/) с доменными фразами-триггерами.
SKILL_DOMAINS = {
    "unity-dev": ["работаю над unity проектом", "нужна помощь с unity", "добавь скрипт в unity",
                  "настрой unity сцену", "проблема с unity ассетами"],
    "unity-physics": ["настрой физику в unity", "коллайдеры работают неправильно",
                      "rigidbody дёргается", "настрой пружины и сочленения",
                      "почему объект проваливается сквозь пол",
                      "физика нестабильна при высоких скоростях"],
    "unity-navmesh": ["настрой навигацию npc", "боты не обходят препятствия",
                      "навмеш не запекается", "проложи путь для агента"],
    "unity-animation": ["настрой анимации персонажа", "аниматор не переключает состояния",
                        "сделай blend tree для бега"],
    "unity-build-pipeline": ["собери билд unity", "настрой автоматическую сборку",
                             "билд падает на этапе IL2CPP"],
    "unity-scriptableobjects": ["создай scriptable object", "данные для конфига в SO",
                                "настрой SO-архитектуру"],
    "unity-tilemap-2d": ["создай 2d уровень из тайлов", "настрой tilemap палитру",
                         "тайлмап режет спрайты"],
    "performance-optimization": ["игра тормозит", "оптимизируй производительность",
                                 "просадки fps в бою", "снизь потребление памяти"],
    "level-design": ["спроектируй уровень", "нужен левел-дизайн для арены",
                     "расставь точки интереса на карте"],
    "procedural-gen": ["сгенерируй подземелье", "процедурная генерация мира",
                       "случайная карта для рогалика"],
    "roguelike": ["делаю рогалик", "механика пермадефа", "генерация забегов"],
    "tower-defense": ["делаю tower defense", "баланс волн врагов", "башни стреляют мимо"],
    "fps-shooter": ["делаю шутер", "настрой стрельбу", "отдача оружия слишком резкая"],
    "platformer": ["делаю платформер", "прыжок ощущается ватным", "настрой coyote time"],
    "puzzle": ["делаю головоломку", "механика пазла не сходится",
               "головоломка с механикой", "пазл-уровень"],
    "rpg": ["делаю рпг", "система прокачки персонажа", "инвентарь и лут"],
    "survival-crafting": ["делаю выживалку", "система крафта", "голод и жажда механики"],
    "dialogue-systems": ["система диалогов для npc", "ветвящиеся диалоги", "диалоговое дерево"],
    "save-systems": ["система сохранений", "сейв не загружается", "сериализация прогресса"],
    "game-ui-ux": ["интерфейс для игры", "hud перегружен", "меню паузы"],
    "game-feel": ["управление ощущается ватным", "добавь сочности управлению", "game feel"],
    "game-ai": ["ии врагов", "поведение ботов", "дерево поведения для мобов"],
    "audio-design": ["звуки в игре", "музыка не зацикливается", "аудио-микшер"],
    "shader-programming": ["напиши шейдер", "шейдер воды", "HLSL ошибка компиляции"],
    "physics-tuning": ["подкрути физику", "пружины слишком жёсткие", "гравитация странная"],
    "input-systems": ["настрой управление", "геймпад не определяется", "новый input system"],
    "camera-systems": ["камера дёргается", "сглаживание камеры", "cinemachine настройка"],
    "steam-publish": ["публикация в steam", "стрим страницы игры", "steamworks интеграция"],
    "itch-publish": ["выложить игру на itch.io", "страница игры на itch"],
    "game-jam": ["геймджам через выходные", "успеть прототип за джем"],
    "prototype-fast": ["быстро проверить идею", "сделать прототип механики за вечер"],
    "lmtrust": ["проверь ответы модели на галлюцинации", "валидация выводов LLM"],
    "router": ["маршрутизация запросов агента", "какой инструмент вызвать"],
    "kern-gate": ["безопасность вызовов инструментов", "гейт для опасных команд"],
    "blender-mcp": ["работа с blender", "модели через blender mcp"],
    "threejs-scene-setup": ["сцена в three.js", "настрой threejs рендерер"],
    "phaser-core": ["игра на phaser", "phaser сцена не создаётся"],
    "visual-novel": ["делаю визуальную новеллу", "система vn-диалогов"],
    "card-game": ["делаю карточную игру", "логика колоды и руки"],
    "impeccable": ["проверь качество кода", "аудит качества проекта"],
    "sourced-researcher": ["найди источники по теме", "исследование с цитатами"],
    "ccc": ["проверь покрытие тестами", "code coverage"],
    "unfairgaps": ["проверь несправедливые механики", "баланс сложности"],
}

GENERIC_NEGATIVE_QUERIES = [
    "прочитай файл config.py",
    "выполни git status",
    "посчитай 128 * 64",
    "объясни как работает рекурсия",
    "продолжай",
    "какие у тебя есть возможности?",
]

# Грамматически нейтральные обёртки: одна фраза -> много уникальных запросов.
# (иначе 43 скилла x 4-6 фраз дают ~130 уникальных пар на 2000 строк — дубликаты)
QUERY_WRAPPERS = [
    "{p}",
    "{p} (срочно)",
    "{p}, пошагово",
    "{p}, подробно",
    "текущая задача: {p}",
    "что делать: {p}",
    "как решить: {p}",
    "план: {p}",
    "помоги с этим: {p}",
    "нужно разобраться: {p}",
    "задача на сегодня: {p}",
    "вопрос по работе: {p}",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    rng = random.Random(999)
    skill_names = list(SKILL_DOMAINS.keys())
    rows: list = []
    seen: set = set()
    # Комбинаторика: фраза x обёртка -> уникальные запросы без дублей.
    combos = []
    for skill in skill_names:
        for phrase in SKILL_DOMAINS[skill]:
            for wrapper in QUERY_WRAPPERS:
                combos.append((skill, wrapper.format(p=phrase)))
    rng.shuffle(combos)
    combos = combos[: args.n]

    for skill, query in combos:
        key = (query, skill)
        if key in seen:
            continue
        seen.add(key)
        # Distractors: same-first-token (hard) + random (soft).
        hard = [s for s in skill_names if s != skill and s.split("-")[0] == skill.split("-")[0]]
        soft = [s for s in skill_names if s != skill and s not in hard]
        rng.shuffle(hard)
        rng.shuffle(soft)
        distractors = (hard + soft)[:7]
        rows.append({
            "query": query,
            "positive": skill,
            "distractors": distractors,
            "split": "val" if rng.random() < args.val_ratio else "train",
        })
        # Negative-only rows: queries that need NO skill at all.
        if rng.random() < 0.15:
            neg_q = rng.choice(GENERIC_NEGATIVE_QUERIES)
            nkey = (neg_q, None)
            if nkey not in seen:
                seen.add(nkey)
                rows.append({
                    "query": neg_q,
                    "positive": None,  # no skill needed
                    "distractors": rng.sample(skill_names, 7),
                    "split": "val" if rng.random() < 0.5 else "train",
                })

    rng.shuffle(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_val = sum(1 for r in rows if r["split"] == "val")
    print(f"Written {len(rows)} rows ({n_val} val) -> {OUT_PATH}")
    print(f"Skills covered: {len(skill_names)}")


if __name__ == "__main__":
    main()