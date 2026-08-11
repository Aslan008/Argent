# Интеграция с Kern Gate

Kern Gate — система формальной верификации математической логики. LMTrust
дополняет Kern Gate, покрывая состояния, взаимодействия, композиционные сбои и
негативное пространство.

## Разделение ответственности

| Аспект | Kern Gate | LMTrust |
|--------|-----------|---------|
| Математическая логика | ✅ Контрактная верификация | L3/L4 как дополнение |
| Состояния | ❌ | ✅ L5 |
| Взаимодействие систем | ❌ | ✅ L6 |
| Композиционные сбои | ❌ | ✅ L7 |
| Негативное пространство | ❌ | ✅ L9 |
| Формальное доказательство | ✅ Proved | ❌ |
| Fallback-пути | ❌ | ✅ L4b |

## Мост LMTrust → Kern Gate

1. LMTrust анализирует код и извлекает математические предположения
2. Генерирует `spec_<function>.py` для Kern Gate
3. Kern Gate верифицирует через лестницу рангов
4. Результат ранга записывается в coverage matrix LMTrust

### Формат spec-файла (проверен на реальном gate.py)

Спека — обычный Python-файл. gate.py читает атрибуты модуля: `NAME`, `DOC`,
`REFERENCE`, `EXAMPLES`, `PROPERTIES`, `DOMAIN`, `EXHAUSTIVE`.

```python
# spec_calculate_damage.py
# -*- coding: utf-8 -*-
"""Контракт: расчёт урона с учётом брони и критического множителя."""

NAME = "calculate_damage"
DOC = "урон = max(0, base - armor*0.5), умноженный на crit_multiplier если > 1"


def REFERENCE(base, armor, crit_multiplier):
    """Эталон: медленно, зато очевидно правильно."""
    damage = base - armor * 0.5
    if damage < 0:
        damage = 0
    if crit_multiplier > 1:
        damage *= crit_multiplier
    return float(damage)


EXAMPLES = [
    # ((base, armor, crit_multiplier), expected_result)
    ((100, 20, 1.0), 90.0),       # 100 - 10 = 90, без крита
    ((100, 20, 2.0), 180.0),      # 90 * 2 = 180, с критом
    ((10, 50, 1.0), 0.0),         # 10 - 25 = -15 → 0 (clamped)
    ((100, 0, 1.0), 100.0),       # нет брони
    ((0, 20, 1.0), 0.0),          # нет базы
]

PROPERTIES = [
    # (name, predicate(args_tuple, result) -> bool)
    ("результат всегда >= 0", lambda a, r: r >= 0),
    ("результат конечный (не NaN/Inf)", lambda a, r: r == r and r != float('inf')),
    ("crit=1 не умножает", lambda a, r: True if a[2] != 1.0 else r == a[0] - a[1] * 0.5),
]


def DOMAIN(rng):
    """Случайные входы для фаззинга."""
    base = rng.uniform(0, 1000)
    armor = rng.uniform(0, 500)
    crit = rng.uniform(0, 10)
    return (base, armor, crit)


def EXHAUSTIVE():
    """Все комбинации до границы — для ранга Исчерпание."""
    for base in [0, 50, 100, 200]:
        for armor in [0, 20, 50, 100]:
            for crit in [0.0, 1.0, 2.0]:
                yield (base, armor, crit)
```

### Команда запуска

```bash
python C:\Users\mshat\Desktop\kern\gate.py spec_calculate_damage.py ai_calculate_damage.py
```

Ранги: Черновик → Примеры → Свойства → Дифференциал → Исчерпание → Proved

- **Черновик** — файл импортируется, функция на месте
- **Примеры** — совпал на явных примерах из `EXAMPLES`
- **Свойства** — выдержал случайные входы против `PROPERTIES`
- **Дифференциал** — совпал с `REFERENCE` на случайных входах
- **Исчерпание** — совпал на ВСЕХ входах из `EXHAUSTIVE`
- **Proved** — только через ядро Керна

### Ключевые правила формата

1. **`NAME`** — строка, имя функции-кандидата (gate.py ищет её в файле)
2. **`REFERENCE`** — функция (не dict!), принимает те же аргументы
3. **`EXAMPLES`** — список `((arg1, arg2, ...), expected)`; один аргумент — без скобок: `(value, expected)`
4. **`PROPERTIES`** — список `(name, lambda(args_tuple, result): bool)`; `args_tuple` — кортеж аргументов
5. **`DOMAIN`** — функция `lambda rng: tuple`, возвращает кортеж аргументов
6. **`EXHAUSTIVE`** — генератор, yield'ит кортежи аргументов
7. Все секции кроме `NAME` и `EXAMPLES` опциональны (но без них ранг не растёт)

## Мост Kern Gate → LMTrust

1. Kern Gate находит контрпример → LMTrust использует как тест-кейс для L1/L2
2. Kern Gate даёт ранг «Исчерпание» → LMTrust отмечает L3 (Property) как покрытый