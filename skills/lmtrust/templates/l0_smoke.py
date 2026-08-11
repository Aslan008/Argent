# L0: Smoke — «Оно вообще живое?»
# Слепая зона: код не запускается, не импортируется, падает на пустом вызове.
#
# Запуск: pytest templates/l0_smoke.py -v

import pytest


# ── SUT (System Under Test) ──────────────────────────────────────────────────
def calculate_damage(base, armor, crit_multiplier):
    damage = base - armor * 0.5
    if damage < 0:
        damage = 0
    if crit_multiplier > 1:
        damage *= crit_multiplier
    return damage


# ── Тесты ────────────────────────────────────────────────────────────────────
class TestSmoke:
    """L0: Smoke tests — код существует и базово работает."""

    def test_function_callable(self):
        """Функция вызывается и возвращает что-то."""
        result = calculate_damage(100, 20, 1.0)
        assert result is not None

    def test_returns_number(self):
        """Возвращается число (не None, не строка, не список)."""
        result = calculate_damage(100, 20, 1.0)
        assert isinstance(result, (int, float))

    def test_no_crash_on_minimal_input(self):
        """Не падает на минимальном валидном входе."""
        result = calculate_damage(0, 0, 0)
        assert result is not None