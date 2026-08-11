# Слои тестов (L0–L9 + L4b)

11 слоёв, расположенных по возрастанию глубины: от «вообще работает?» до
«что НЕ должно происходить?».

---

## L0: Smoke — «Оно вообще живое?»

**Слепая зона:** Код не запускается, не импортируется, падает на пустом вызове.

```python
class TestSmoke:
    def test_import(self):
        import my_module  # ImportError = катастрофа
    def test_call(self):
        result = calculate_damage(100, 20, 1.0)
        assert result is not None
```

---

## L1: Contract — «Оно делает то, что говорит?»

**Слепая зона:** Код работает на типичных входах, но нарушает свой контракт.

```python
class TestContract:
    def test_documented_examples(self):
        assert calculate_damage(100, 20, 1.0) == 90.0
        assert calculate_damage(100, 20, 2.0) == 180.0
    def test_return_type(self):
        assert isinstance(calculate_damage(100, 20, 1.0), float)
```

### Named Pattern — Known Coordinates (D11)

Проверяй не ЧТО произошло, а ГДЕ. Критично для файловых операций, буферов,
массивов, игровых координат.

```python
class TestCoordinates:
    def test_damage_applies_to_correct_enemy(self):
        enemies = [Enemy(hp=100), Enemy(hp=200), Enemy(hp=300)]
        apply_damage(enemies, 1, 50)  # index 1 = второй враг
        assert enemies[0].hp == 100, "Wrong enemy damaged!"
        assert enemies[1].hp == 150, "Target enemy not damaged!"
        assert enemies[2].hp == 300, "Wrong enemy damaged!"
```

---

## L2: Boundary — «Что на краях?»

**Слепая зона:** Код работает в середине диапазона, но ломается на границах.

**8 обязательных граничных условий для каждого параметра:**

| # | Условие | Пример |
|---|---------|--------|
| 1 | Пустой ввод | `""`, `[]`, `None` |
| 2 | Один элемент | файл из 1 строки, список из 1 элемента |
| 3 | Минимальное значение | `line=0`, `index=0`, `hp=0` |
| 4 | Максимальное значение | последняя строка, последний индекс |
| 5 | За пределами | `line=N+1`, `index=-1`, `hp=-10` |
| 6 | Граница включения/исключения | `end_line=N` — включает N или нет? |
| 7 | Специальные символы | BOM, CRLF, null bytes, NaN, Inf |
| 8 | Mixed | mixed encoding, mixed types, mixed line endings |

```python
class TestEdgeCases:
    def test_zero(self):
        assert calculate_damage(0, 0, 0) == 0.0
    def test_nan(self):
        result = calculate_damage(float('nan'), 20, 1.0)
        assert math.isfinite(result)
    def test_inclusion_exclusion(self):
        # crit_multiplier > 1 — строго больше, не >=
        assert calculate_damage(100, 20, 1.0) == 90.0       # crit=1 → NO mult
        assert calculate_damage(100, 20, 1.0000001) != 90.0  # crit>1 → mult
    def test_mixed_types(self):
        result = calculate_damage(100, 20, 1)  # int crit, not float
        assert result == 90.0
```

---

## L3: Property — «Что ВСЕГДА должно быть истинным?»

**Слепая зона:** Код работает на конкретных входах, но нарушает инварианты на
случайных.

> **⚠️ Property-тесты с `allow_nan=True, allow_infinity=True` могут падать —
> это ожидаемо.** Падение = найденная слепая зона. Не ослабляй assertion —
> предлагай исправление кода (добавить проверку NaN/Inf) или документируй
> ограничение.

```python
from hypothesis import given, strategies as st

class TestProperties:
    @given(st.floats(min_value=0, max_value=1000),
           st.floats(min_value=0, max_value=500),
           st.floats(min_value=0, max_value=10))
    def test_damage_nonnegative(self, base, armor, crit):
        result = calculate_damage(base, armor, crit)
        assert result >= 0

    @given(st.floats(min_value=0, max_value=1000),
           st.floats(min_value=0, max_value=500),
           st.floats(min_value=0, max_value=500))
    def test_monotonic_armor(self, base, armor1, armor2):
        d1 = calculate_damage(base, armor1, 1.0)
        d2 = calculate_damage(base, armor2, 1.0)
        if armor1 > armor2:
            assert d1 <= d2  # больше armor → меньше или равно damage
```

---

## L4: Adversarial — «Что если предположения — ложь?»

**Ключевой слой LMTrust.** Систематически нарушяет каждое неявное предположение.

**Метод:**
1. Для каждого предположения — сгенерировать вход, нарушающий его
2. Проверить: функция либо обрабатывает gracefully, либо падает явно
3. Функция НЕ должна молчаливо возвращать неверный результат

```python
class TestAdversarial:
    def test_negative_armor(self):
        # Предположение: armor >= 0. Что если armor < 0?
        result = calculate_damage(100, -20, 1.0)
        # -20 * 0.5 = -10; 100 - (-10) = 110 — броня УВЕЛИЧИВАЕТ урон!
        assert result <= 100, "Negative armor should not increase damage"

    def test_nan_crit(self):
        result = calculate_damage(100, 20, float('nan'))
        assert math.isfinite(result), "NaN crit must not produce NaN"

    def test_inf_base(self):
        result = calculate_damage(float('inf'), 20, 1.0)
        assert math.isfinite(result), "Inf base must not propagate to Inf"
```

---

## L4b: Fallback — «Что когда основной путь недоступен?»

**Слепая зона:** Основной путь работает, но fallback — ломается или не
активируется.

```python
class TestFallback:
    def test_fallback_when_shield_down(self):
        player = Player(hp=100, shield=0)  # shield system не активна
        player.take_damage(30)
        assert player.hp == 70, f"Fallback failed: HP should be 70, got {player.hp}"

    def test_fallback_not_activated_when_primary_available(self):
        player = Player(hp=100, shield=50)  # shield активна
        player.take_damage(30)
        assert player.shield == 20, "Primary path should be used"
        assert player.hp == 100, "HP should not be damaged when shield active"
```

---

## L5: State Machine — «Все ли состояния достижимы?»

**Слепая зона:** Код работает в нормальной последовательности, но ломается при
необычных переходах.

```python
class TestState:
    def test_dead_after_kill(self):
        enemy = Enemy(hp=100)
        enemy.take_damage(100)
        assert enemy.state == "dead"
        enemy.heal(50)  # Можно ли лечить мёртвого?
        assert enemy.state == "dead", "Dead enemies must not be healable"

    def test_double_death(self):
        enemy = Enemy(hp=100)
        enemy.take_damage(100)
        enemy.take_damage(50)  # Уже мёртв, ещё удар
        assert enemy.hp == 0, "HP must not go below 0 for dead enemy"
```

---

## L6: Cross-System — «A работает, B работает, но A+B ломается»

**Слепая зона:** Каждый компонент проходит свои тесты, но взаимодействие
проваливается.

**Принцип No-Mock E2E:** Хотя бы один тест для каждой внутренней подсистемы —
без единого мока. Если хотя бы один слой замокан, тест не считается
end-to-end. Исключение: внешние API (мокаем транспорт, тестируем логику).

```python
class TestIntegration:
    def test_shield_and_heal(self):
        player = Player(hp=50, shield=30)
        player.take_damage(40)  # Shield: 30→0, HP: 50 (shield gating)
        player.heal(20)          # Heal: HP 50→70
        assert player.hp == 70
        assert player.shield == 0
```

### Named Pattern — Cross-Layer Consistency (D12)

Проверяй преобразования данных между слоями. Каждый слой имеет свои конвенции
(1-indexed vs 0-indexed, тип данных).

```python
class TestLayerConsistency:
    def test_indexing_consistency(self):
        ui_line = 1  # 1-indexed (UI слой)
        internal_index = ui_line - 1  # преобразование: 1→0
        file_line = internal_index  # 0-indexed (файловый слой)
        assert file_line == 0, f"Double conversion: expected 0, got {file_line}"
        result_ui_line = file_line + 1  # обратное преобразование
        assert result_ui_line == ui_line, "Round-trip conversion failed"
```

---

## L7: Compositional Failure — «Что если две вещи ломаются одновременно?»

**Слепая зона:** Код обрабатывает одиночные сбои корректно, но ломается при
комбинации.

```python
class TestCompositional:
    def test_damage_during_heal_failure(self):
        player = Player(hp=50, shield=0)
        try:
            player.heal(float('nan'))  # NaN heal — сбой
        except:
            pass
        result = player.take_damage(30)
        assert math.isfinite(result), "State corrupted by failed heal"
```

---

## L8: Temporal — «Зависит ли порядок от чего-то?»

**Слепая зона:** Код работает в одном порядке вызовов, но ломается в другом.

```python
class TestTemporal:
    def test_heal_then_damage_vs_damage_then_heal(self):
        p1 = Player(hp=50, shield=30)
        p1.heal(20); p1.take_damage(40)

        p2 = Player(hp=50, shield=30)
        p2.take_damage(40); p2.heal(20)

        assert p1.hp == p2.hp, "Order independence violated"

    def test_repeated_zero_damage(self):
        enemy = Enemy(hp=100)
        for _ in range(1000):
            enemy.take_damage(0)
        assert enemy.hp == 100, "Zero damage must not accumulate"
```

---

## L9: Negative Space — «Что НЕ должно происходить?»

**Слепая зона:** Код делает что-то, чего не должен, но никто не проверяет, что
этого НЕ происходит.

```python
class TestNegativeSpace:
    def test_no_silent_data_loss(self):
        player = Player(hp=100)
        player.take_damage(200)  # Больше чем HP
        assert player.hp >= 0, "HP must not go negative silently"

    def test_no_silent_overflow(self):
        result = calculate_damage(1e308, 0, 1.0)
        assert math.isfinite(result), "Overflow must not be silent"

    def test_heal_does_not_exceed_max(self):
        player = Player(hp=50, max_hp=100)
        player.heal(200)
        assert player.hp <= 100, "Heal must not exceed max HP silently"
```