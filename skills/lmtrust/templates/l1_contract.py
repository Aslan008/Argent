# L1: Contract — «Оно делает то, что говорит?»
# Слепая зона: код работает на типичных входах, но нарушает свой контракт.
# Named Pattern — Known Coordinates (D11): проверяй не ЧТО, а ГДЕ.
#
# Запуск: pytest templates/l1_contract.py -v

import pytest


# ── SUT ──────────────────────────────────────────────────────────────────────
def calculate_damage(base, armor, crit_multiplier):
    damage = base - armor * 0.5
    if damage < 0:
        damage = 0
    if crit_multiplier > 1:
        damage *= crit_multiplier
    return damage


# ── Тесты ────────────────────────────────────────────────────────────────────
class TestContract:
    """L1: Contract tests — документированные примеры и типы."""

    def test_documented_examples(self):
        """Каждый документированный пример: input → expected output."""
        assert calculate_damage(100, 20, 1.0) == 90.0
        assert calculate_damage(100, 20, 2.0) == 180.0
        assert calculate_damage(10, 50, 1.0) == 0.0

    def test_return_type(self):
        """Возвращаемый тип соответствует контракту."""
        result = calculate_damage(100, 20, 1.0)
        assert isinstance(result, (int, float))

    def test_no_side_effects(self):
        """Функция чистая — не имеет побочных эффектов."""
        # Для чистой функции: повторный вызов с теми же аргументами = тот же результат
        r1 = calculate_damage(100, 20, 1.0)
        r2 = calculate_damage(100, 20, 1.0)
        assert r1 == r2


class TestCoordinates:
    """L1 × Known Coordinates: операция нацелена в правильное место."""

    def test_damage_applies_to_correct_enemy(self):
        """Урон применяется к нужному врагу, не к соседнему."""
        enemies = [{"hp": 100}, {"hp": 200}, {"hp": 300}]
        target_idx = 1
        dmg = calculate_damage(50, 0, 1.0)

        enemies[target_idx]["hp"] -= dmg
        assert enemies[0]["hp"] == 100, "Wrong enemy damaged!"
        assert enemies[1]["hp"] == 150, "Target enemy not damaged!"
        assert enemies[2]["hp"] == 300, "Wrong enemy damaged!"