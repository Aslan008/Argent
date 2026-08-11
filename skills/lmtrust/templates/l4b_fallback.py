# L4b: Fallback Path — «Что когда основной путь недоступен?»
# Слепая зона: основной путь работает, но fallback — ломается или не активируется.
#
# Запуск: pytest templates/l4b_fallback.py -v

import pytest


# ── SUT: Player с shield (primary) и HP (fallback) ──────────────────────────
class Player:
    def __init__(self, hp, shield=0):
        self.hp = hp
        self.shield = shield
        self.max_hp = hp

    def take_damage(self, damage):
        if self.shield > 0:
            # Primary path: shield поглощает урон
            absorbed = min(self.shield, damage)
            self.shield -= absorbed
            remaining = damage - absorbed
            if remaining > 0:
                self.hp -= remaining
        else:
            # Fallback path: урон идёт в HP напрямую
            self.hp -= damage
        if self.hp < 0:
            self.hp = 0


# ── Тесты ────────────────────────────────────────────────────────────────────
class TestFallback:
    """L4b: Fallback tests — запасные пути."""

    def test_fallback_activates_when_shield_down(self):
        """Fallback активируется, когда shield = 0 (основной путь недоступен)."""
        player = Player(hp=100, shield=0)
        player.take_damage(30)
        assert player.hp == 70, f"Fallback failed: HP should be 70, got {player.hp}"

    def test_primary_used_when_shield_available(self):
        """Primary путь используется, когда shield > 0."""
        player = Player(hp=100, shield=50)
        player.take_damage(30)
        assert player.shield == 20, "Primary path should be used"
        assert player.hp == 100, "HP should not be damaged when shield active"

    def test_transition_primary_to_fallback(self):
        """Переход: shield исчерпан → урон идёт в HP."""
        player = Player(hp=100, shield=20)
        player.take_damage(50)  # shield: 20→0, HP: 100→70
        assert player.shield == 0
        assert player.hp == 70, f"Transition failed: HP should be 70, got {player.hp}"

    def test_fallback_after_shield_depleted(self):
        """После исчерпания shield — следующий удар идёт в HP (fallback)."""
        player = Player(hp=100, shield=20)
        player.take_damage(20)  # shield: 20→0
        assert player.hp == 100
        player.take_damage(30)  # fallback: HP 100→70
        assert player.hp == 70, f"Fallback after depletion failed: {player.hp}"