# L5: State Machine — «Все ли состояния достижимы?»
# Слепая зона: код работает в нормальной последовательности, но ломается при
# необычных переходах.
#
# Запуск: pytest templates/l5_state.py -v

import pytest


# ── SUT: Enemy с состояниями alive/dead ──────────────────────────────────────
class Enemy:
    def __init__(self, hp):
        self.hp = hp
        self.state = "alive" if hp > 0 else "dead"

    def take_damage(self, damage):
        if self.state == "dead":
            return 0
        self.hp -= damage
        if self.hp <= 0:
            self.hp = 0
            self.state = "dead"
        return damage

    def heal(self, amount):
        if self.state == "dead":
            return 0
        self.hp += amount
        return amount


# ── Тесты ────────────────────────────────────────────────────────────────────
class TestState:
    """L5: State machine tests — состояния и переходы."""

    def test_alive_after_creation(self):
        """Начальное состояние — alive."""
        enemy = Enemy(hp=100)
        assert enemy.state == "alive"

    def test_dead_after_kill(self):
        """Переход alive → dead при HP <= 0."""
        enemy = Enemy(hp=100)
        enemy.take_damage(100)
        assert enemy.state == "dead"
        assert enemy.hp == 0

    def test_dead_not_healable(self):
        """Dead — терминальное: heal не работает."""
        enemy = Enemy(hp=100)
        enemy.take_damage(100)
        assert enemy.state == "dead"
        enemy.heal(50)
        assert enemy.state == "dead", "Dead enemy was healed!"
        assert enemy.hp == 0

    def test_double_death(self):
        """Повторный урон мёртвому врагу — идемпотентен."""
        enemy = Enemy(hp=100)
        enemy.take_damage(100)
        result = enemy.take_damage(50)  # уже мёртв
        assert result == 0, "Dead enemy should return 0 from take_damage"
        assert enemy.hp == 0

    def test_partial_damage_keeps_alive(self):
        """Частичный урон сохраняет alive."""
        enemy = Enemy(hp=100)
        enemy.take_damage(30)
        assert enemy.state == "alive"
        assert enemy.hp == 70