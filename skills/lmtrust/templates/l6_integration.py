# L6: Cross-System — «A работает, B работает, но A+B ломается»
# Слепая зона: каждый компонент проходит свои тесты, но взаимодействие проваливается.
#
# ПРИНЦИП No-Mock E2E: хотя бы один тест — без единого мока.
# Исключение: внешние API (мокаем транспорт, тестируем логику).
#
# Named Pattern — Cross-Layer Consistency (D12): проверяй преобразования
# данных между слоями (1-indexed vs 0-indexed, тип данных).
#
# Запуск: pytest templates/l6_integration.py -v

import pytest


# ── SUT: Combat система (damage calc + enemy) ───────────────────────────────
def calculate_damage(base, armor, crit_multiplier):
    damage = base - armor * 0.5
    if damage < 0:
        damage = 0
    if crit_multiplier > 1:
        damage *= crit_multiplier
    return damage


class Enemy:
    def __init__(self, hp, armor):
        self.hp = hp
        self.armor = armor

    def receive_damage(self, base, crit_multiplier):
        dmg = calculate_damage(base, self.armor, crit_multiplier)
        self.hp -= dmg
        if self.hp < 0:
            self.hp = 0
        return dmg


# ── Тесты ────────────────────────────────────────────────────────────────────
class TestIntegration:
    """L6: Cross-system integration — без моков (No-Mock E2E)."""

    def test_damage_calc_plus_enemy_hp(self):
        """calculate_damage + Enemy.receive_damage — полная цепочка без моков."""
        enemy = Enemy(hp=100, armor=20)
        dealt = enemy.receive_damage(100, 1.0)
        assert dealt == 90.0
        assert enemy.hp == 10  # 100 - 90 = 10

    def test_crit_kill_chain(self):
        """Критический удар убивает врага — полная цепочка."""
        enemy = Enemy(hp=100, armor=20)
        dealt = enemy.receive_damage(100, 2.0)  # 180 damage
        assert dealt == 180.0
        assert enemy.hp == 0  # 100 - 180 = -80 → clamped to 0


class TestLayerConsistency:
    """L6 × Cross-Layer: преобразования данных между слоями."""

    def test_indexing_conversion(self):
        """Преобразование индексации: UI (1-indexed) → internal (0-indexed)."""
        ui_line = 1  # 1-indexed (UI слой)
        internal_index = ui_line - 1  # преобразование: 1→0
        assert internal_index == 0, f"Double conversion: expected 0, got {internal_index}"

    def test_round_trip_conversion(self):
        """Обратное преобразование: round-trip == identity."""
        ui_value = 5
        internal = ui_value - 1  # → 0-indexed
        back = internal + 1      # → 1-indexed
        assert back == ui_value, "Round-trip conversion failed"