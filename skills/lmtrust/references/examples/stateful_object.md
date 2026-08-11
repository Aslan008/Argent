# Пример 2: Stateful объект — `Player`

## Исходный код

```python
class Player:
    def __init__(self, hp, shield=0, max_hp=None):
        self.hp = hp
        self.shield = shield
        self.max_hp = max_hp or hp
        self.state = "alive" if hp > 0 else "dead"

    def take_damage(self, damage):
        if self.state == "dead":
            return 0
        if self.shield > 0:
            absorbed = min(self.shield, damage)
            self.shield -= absorbed
            remaining = damage - absorbed
            if remaining > 0:
                self.hp -= remaining
        else:
            self.hp -= damage
        if self.hp <= 0:
            self.hp = 0
            self.state = "dead"
        return damage

    def heal(self, amount):
        if self.state == "dead":
            return 0
        self.hp = min(self.hp + amount, self.max_hp)
        return amount
```

## Дерево решений: Stateful объект → L0, L1, L5, L4, L9 × D2, D3, D4

## Tests

```python
import math
import pytest

# L0: Smoke
class TestSmoke:
    def test_instantiation(self):
        player = Player(hp=100)
        assert player is not None
        assert player.state == "alive"

# L1: Contract
class TestContract:
    def test_basic_damage(self):
        player = Player(hp=100)
        player.take_damage(30)
        assert player.hp == 70

    def test_basic_heal(self):
        player = Player(hp=50)
        player.heal(20)
        assert player.hp == 70

    def test_shield_absorption(self):
        player = Player(hp=100, shield=50)
        player.take_damage(30)
        assert player.shield == 20
        assert player.hp == 100  # HP не тронуто

# L5: State Machine
class TestState:
    def test_dead_after_kill(self):
        player = Player(hp=100)
        player.take_damage(100)
        assert player.state == "dead"

    def test_dead_not_healable(self):
        player = Player(hp=100)
        player.take_damage(100)
        player.heal(50)
        assert player.state == "dead"
        assert player.hp == 0

    def test_double_death(self):
        player = Player(hp=100)
        player.take_damage(100)
        player.take_damage(50)  # уже мёртв
        assert player.hp == 0

    def test_shield_gating(self):
        """Shield поглощает весь урон, HP не страдает."""
        player = Player(hp=100, shield=50)
        player.take_damage(30)
        assert player.shield == 20
        assert player.hp == 100

    def test_shield_then_hp(self):
        """Урон больше щита — остаток идёт в HP."""
        player = Player(hp=100, shield=20)
        player.take_damage(50)
        assert player.shield == 0
        assert player.hp == 70  # 100 - (50-20) = 70

# L4: Adversarial
class TestAdversarial:
    def test_nan_damage(self):
        """Предположение: damage is finite. Что если NaN?"""
        player = Player(hp=100)
        player.take_damage(float('nan'))
        assert math.isfinite(player.hp), "NaN damage corrupted HP"

    def test_negative_damage_heals(self):
        """Предположение: damage >= 0. Что если отрицательный?"""
        player = Player(hp=50)
        player.take_damage(-20)
        # Отрицательный урон не должен лечить
        assert player.hp <= 50, "Negative damage should not heal"

    def test_inf_damage(self):
        """Предположение: damage is finite. Что если Inf?"""
        player = Player(hp=100, shield=50)
        player.take_damage(float('inf'))
        assert player.state == "dead", "Inf damage should kill"

# L9: Negative Space
class TestNegativeSpace:
    def test_no_silent_overheal(self):
        """Heal не должен превышать max_hp молча."""
        player = Player(hp=50, max_hp=100)
        player.heal(200)
        assert player.hp <= 100, "Heal must not exceed max HP silently"

    def test_no_negative_hp(self):
        """HP не должен уходить в минус молча."""
        player = Player(hp=100)
        player.take_damage(200)  # больше чем HP
        assert player.hp >= 0, "HP must not go negative silently"

    def test_dead_returns_zero(self):
        """Мёртвый игрок возвращает 0 от take_damage, не крашит."""
        player = Player(hp=100)
        player.take_damage(100)
        result = player.take_damage(50)
        assert result == 0, "Dead player should return 0 from take_damage"
```