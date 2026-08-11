# L2: Boundary — «Что на краях?»
# Слепая зона: код работает в середине диапазона, но ломается на границах.
# 8 обязательных граничных условий для каждого параметра.
#
# Запуск: pytest templates/l2_boundary.py -v

import math
import pytest


# ── SUT ──────────────────────────────────────────────────────────────────────
def calculate_damage(base, armor, crit_multiplier):
    damage = base - armor * 0.5
    if damage < 0:
        damage = 0
    if crit_multiplier > 1:
        damage *= crit_multiplier
    return damage


# ── Тесты: 8 граничных условий ───────────────────────────────────────────────
class TestEdgeCases:
    """L2: Boundary tests — 8 граничных условий."""

    def test_1_empty_zero(self):
        """Пустой ввод: 0, 0.0."""
        assert calculate_damage(0, 0, 0) == 0

    def test_2_single_unit(self):
        """Один элемент: минимальное положительное значение."""
        result = calculate_damage(1, 0, 1.0)
        assert result == 1

    def test_3_min_value(self):
        """Минимальное значение: base=0."""
        assert calculate_damage(0, 20, 1.0) == 0

    def test_4_max_value(self):
        """Максимальное значение: очень большое base."""
        result = calculate_damage(1e6, 0, 1.0)
        assert result == 1e6

    def test_5_out_of_bounds(self):
        """За пределами: отрицательный base (предположение: base >= 0)."""
        result = calculate_damage(-10, 20, 1.0)
        # -10 - 10 = -20 → clamped to 0. Но ожидаем ли мы этого?
        assert result >= 0, f"Negative base produced negative damage: {result}"

    def test_6_inclusion_exclusion(self):
        """Граница включения/исключения: crit=1.0 не умножает, crit>1 — умножает."""
        no_crit = calculate_damage(100, 20, 1.0)
        with_crit = calculate_damage(100, 20, 1.0000001)
        assert no_crit == 90.0, "crit=1.0 should NOT multiply"
        assert with_crit != 90.0, "crit>1 should multiply"

    @pytest.mark.xfail(reason="Слепая зона: SUT не обрабатывает NaN — ожидаемо для L2")
    def test_7_special_values(self):
        """Специальные символы: NaN, Inf.

        ⚠️ Этот тест упадёт — SUT не обрабатывает NaN.
        Это ожидаемо: тест нашёл слепую зону.
        Действие: добавить проверку NaN в код, либо документировать ограничение.
        """
        result = calculate_damage(float('nan'), 20, 1.0)
        assert math.isfinite(result), f"NaN base produced non-finite: {result}"

    def test_8_mixed_types(self):
        """Mixed: int и float в одном вызове."""
        result = calculate_damage(100, 20, 1)  # int crit, not float
        assert result == 90.0