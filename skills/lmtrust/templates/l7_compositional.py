# L7: Compositional Failure — «Что если две вещи ломаются одновременно?»
# Слепая зона: код обрабатывает одиночные сбои корректно, но ломается при комбинации.
#
# Запуск: pytest templates/l7_compositional.py -v

import math
import pytest


# ── SUT: Player с shield и heal ──────────────────────────────────────────────
class Player:
    def __init__(self, hp, shield=0):
        self.hp = hp
        self.shield = shield
        self.max_hp = hp

    def take_damage(self, damage):
        if self.shield > 0:
            absorbed = min(self.shield, damage)
            self.shield -= absorbed
            remaining = damage - absorbed
            if remaining > 0:
                self.hp -= remaining
        else:
            self.hp -= damage
        if self.hp < 0:
            self.hp = 0

    def heal(self, amount):
        if not math.isfinite(amount):
            raise ValueError("Heal amount must be finite")
        self.hp = min(self.hp + amount, self.max_hp)


# ── Тесты ────────────────────────────────────────────────────────────────────
class TestCompositional:
    """L7: Compositional failure tests — комбинации сбоев."""

    def test_damage_after_failed_heal(self):
        """Сбой heal (NaN) + последующий take_damage — состояние не повреждено."""
        player = Player(hp=50, shield=0)
        try:
            player.heal(float('nan'))  # heal падает
        except ValueError:
            pass
        # Состояние не повреждено? take_damage должен работать корректно
        player.take_damage(30)
        assert math.isfinite(player.hp), "State corrupted by failed heal"
        assert player.hp == 20, f"HP should be 20, got {player.hp}"

    def test_shield_depleted_and_heal(self):
        """Shield исчерпан + heal — комбинация двух операций."""
        player = Player(hp=100, shield=20)
        player.take_damage(50)  # shield: 20→0, HP: 100→70
        assert player.shield == 0
        assert player.hp == 70
        player.heal(20)  # heal: HP 70→90
        assert player.hp == 90

    def test_damage_exceeding_shield_and_hp(self):
        """Урон больше shield + больше HP — оба пути задействованы."""
        player = Player(hp=30, shield=20)
        player.take_damage(100)  # shield: 20→0, HP: 30→0 (70 remaining, but HP=30)
        assert player.shield == 0
        assert player.hp == 0, f"HP should be 0, got {player.hp}"