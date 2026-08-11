# L4: Adversarial — «Что если предположения — ложь?»
# КЛЮЧЕВОЙ СЛОЙ LMTrust. Систематически нарушает каждое неявное предположение.
#
# Метод:
# 1. Для каждого предположения — сгенерировать вход, нарушающий его
# 2. Проверить: функция либо обрабатывает gracefully, либо падает явно
# 3. Функция НЕ должна молчаливо возвращать неверный результат
#
# Запуск: pytest templates/l4_adversarial.py -v

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


# ── Тесты ────────────────────────────────────────────────────────────────────
class TestAdversarial:
    """L4: Adversarial tests — нарушение каждого неявного предположения."""

    @pytest.mark.xfail(reason="Слепая зона: SUT не обрабатывает negative armor — ожидаемо для L4")
    def test_negative_armor_increases_damage(self):
        """Предположение: armor >= 0. Что если armor < 0?

        -20 * 0.5 = -10; 100 - (-10) = 110 — броня УВЕЛИЧИВАЕТ урон!
        ⚠️ Этот тест упадёт — SUT молчаливо возвращает неверный результат.
        Это ожидаемо: тест нашёл слепую зону.
        """
        result = calculate_damage(100, -20, 1.0)
        assert result <= 100, f"Negative armor increased damage: {result}"

    def test_nan_crit_produces_nan(self):
        """Предположение: crit is finite. Что если NaN?

        NaN > 1 → False, поэтому умножение не происходит.
        Но base - armor*0.5 с конечными base/armor → конечный результат.
        """
        result = calculate_damage(100, 20, float('nan'))
        assert math.isfinite(result), f"NaN crit produced non-finite: {result}"

    @pytest.mark.xfail(reason="Слепая зона: SUT не обрабатывает Inf — ожидаемо для L4")
    def test_inf_base_produces_inf(self):
        """Предположение: base is finite. Что если Inf?

        Inf - 10 = Inf, Inf > 0 → True, Inf * 2 = Inf.
        ⚠️ Этот тест упадёт — SUT молчаливо возвращает Inf.
        Это ожидаемо: тест нашёл слепую зону.
        """
        result = calculate_damage(float('inf'), 20, 1.0)
        assert math.isfinite(result), f"Inf base propagated to Inf: {result}"

    def test_negative_crit_produces_negative(self):
        """Предположение: crit >= 0. Что если crit < 0?

        -2 > 1 → False, умножения нет. Результат = base - armor*0.5.
        Но если crit = 1.5 (положительный), умножение есть.
        Что если crit = -0.5? -0.5 > 1 → False, нет умножения. OK.
        Но crit = 1.5 → 90 * 1.5 = 135. А crit = -1.5 → 90 (без умножения).
        Это не баг — но проверим.
        """
        result = calculate_damage(100, 20, -2.0)
        assert result >= 0, f"Negative crit produced negative: {result}"