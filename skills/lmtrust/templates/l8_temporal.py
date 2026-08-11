# L8: Temporal — «Зависит ли порядок от чего-то?»
# Слепая зона: код работает в одном порядке вызовов, но ломается в другом.
#
# Запуск: pytest templates/l8_temporal.py -v

import pytest


# ── SUT: Enemy с heal и take_damage ──────────────────────────────────────────
class Enemy:
    def __init__(self, hp, max_hp=None):
        self.hp = hp
        self.max_hp = max_hp or hp

    def take_damage(self, damage):
        self.hp -= damage
        if self.hp < 0:
            self.hp = 0

    def heal(self, amount):
        self.hp = min(self.hp + amount, self.max_hp)


# ── Тесты ────────────────────────────────────────────────────────────────────
class TestTemporal:
    """L8: Temporal / ordering tests."""

    def test_heal_then_damage_vs_damage_then_heal(self):
        """A→B vs B→A: результат тот же? (коммутативность)"""
        # heal then damage (max_hp=100, чтобы heal не упирался в потолок)
        p1 = Enemy(hp=50, max_hp=100)
        p1.heal(20)    # 70
        p1.take_damage(30)  # 40

        # damage then heal
        p2 = Enemy(hp=50, max_hp=100)
        p2.take_damage(30)  # 20
        p2.heal(20)    # 40

        assert p1.hp == p2.hp, f"Order dependence: {p1.hp} != {p2.hp}"

    def test_idempotent_zero_damage(self):
        """Повторный нулевой урон не накапливает эффект."""
        enemy = Enemy(hp=100)
        for _ in range(1000):
            enemy.take_damage(0)
        assert enemy.hp == 100, "Zero damage accumulated"

    def test_repeated_heal_caps_at_max(self):
        """Повторный heal не превышает max_hp (идемпотентность насыщения)."""
        enemy = Enemy(hp=80, max_hp=100)
        enemy.heal(50)  # 80+50=130 → capped at 100
        assert enemy.hp == 100
        enemy.heal(50)  # already at max → stays 100
        assert enemy.hp == 100, "Heal above max should be idempotent"

    def test_damage_then_heal_to_full(self):
        """Урон → heal до полного — HP восстанавливается корректно."""
        enemy = Enemy(hp=100)
        enemy.take_damage(40)
        assert enemy.hp == 60
        enemy.heal(40)
        assert enemy.hp == 100