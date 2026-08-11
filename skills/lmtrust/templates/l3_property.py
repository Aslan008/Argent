# L3: Property — «Что ВСЕГДА должно быть истинным?»
# Слепая зона: код работает на конкретных входах, но нарушает инварианты на
# случайных. Используй Hypothesis (Python) или QuickCheck-style property testing.
#
# Запуск: pytest templates/l3_property.py -v
# Зависимости: pip install hypothesis

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


# ── Тесты ────────────────────────────────────────────────────────────────────
class TestProperties:
    """L3: Property tests — инварианты на случайных входах."""

    @given(st.floats(min_value=0, max_value=1000),
           st.floats(min_value=0, max_value=500),
           st.floats(min_value=0, max_value=10))
    def test_damage_nonnegative(self, base, armor, crit):
        """Инвариант: результат всегда >= 0."""
        result = calculate_damage(base, armor, crit)
        assert result >= 0

    @given(st.floats(min_value=0, max_value=1000),
           st.floats(min_value=0, max_value=500),
           st.floats(min_value=0, max_value=500))
    def test_monotonic_armor(self, base, armor1, armor2):
        """Инвариант: больше брони → меньше или равно урон (при crit=1)."""
        d1 = calculate_damage(base, armor1, 1.0)
        d2 = calculate_damage(base, armor2, 1.0)
        if armor1 > armor2:
            assert d1 <= d2

    @given(st.floats(min_value=0, max_value=1000),
           st.floats(min_value=0, max_value=500))
    def test_crit_never_reduces_damage(self, base, armor):
        """Инвариант: crit_multiplier >= 1 не уменьшает урон."""
        no_crit = calculate_damage(base, armor, 1.0)
        with_crit = calculate_damage(base, armor, 2.0)
        assert with_crit >= no_crit