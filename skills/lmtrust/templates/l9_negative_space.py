# L9: Negative Space — «Что НЕ должно происходить?»
# Слепая зона: код делает что-то, чего не должен, но никто не проверяет,
# что этого НЕ происходит.
#
# Запуск: pytest templates/l9_negative_space.py -v

import math
import pytest
from hypothesis import given, strategies as st


# ── SUT ──────────────────────────────────────────────────────────────────────
def calculate_damage(base, armor, crit_multiplier):
    damage = base - armor * 0.5
    if damage < 0:
        damage = 0
    if crit_multiplier > 1:
        damage *= crit_multiplier
    return damage


class Player:
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
class TestNegativeSpace:
    """L9: Negative space tests — проверка запретов."""

    def test_no_negative_hp(self):
        """HP НЕ должен уходить в минус молча."""
        player = Player(hp=100)
        player.take_damage(200)  # больше чем HP
        assert player.hp >= 0, f"HP went negative silently: {player.hp}"

    def test_no_overheal(self):
        """Heal НЕ должен превышать max_hp молча."""
        player = Player(hp=50, max_hp=100)
        player.heal(200)
        assert player.hp <= 100, f"Heal exceeded max silently: {player.hp}"

    def test_no_silent_overflow(self):
        """Переполнение НЕ должно происходить молча."""
        result = calculate_damage(1e308, 0, 1.0)
        assert math.isfinite(result), f"Overflow silent: {result}"

    @pytest.mark.xfail(reason="Слепая зона: SUT не обрабатывает NaN — ожидаемо для L9")
    def test_no_silent_nan_propagation(self):
        """NaN НЕ должен распространяться молча.

        ⚠️ Этот тест упадёт — SUT не обрабатывает NaN.
        Это ожидаемо: тест нашёл слепую зону.
        """
        result = calculate_damage(float('nan'), 20, 1.0)
        assert math.isfinite(result), f"NaN propagated silently: {result}"

    @pytest.mark.xfail(reason="Слепая зона: SUT не обрабатывает NaN/Inf — ожидаемо для L9",
                       strict=False)
    @given(st.floats(allow_nan=True, allow_infinity=True),
           st.floats(allow_nan=True, allow_infinity=True),
           st.floats(allow_nan=True, allow_infinity=True))
    def test_result_always_finite(self, base, armor, crit):
        """Результат ВСЕГДА конечный — даже для NaN/Inf входов.

        ⚠️ Этот тест упадёт — потому что calculate_damage не обрабатывает
        NaN/Inf. Это ожидаемо: тест нашёл слепую зону.
        Действие: добавить проверку NaN/Inf в код, либо документировать
        «функция принимает только конечные числа».
        """
        result = calculate_damage(base, armor, crit)
        assert math.isfinite(result), \
            f"Non-finite for base={base}, armor={armor}, crit={crit}"