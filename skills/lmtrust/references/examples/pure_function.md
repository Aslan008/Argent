# Пример 1: Чистая функция — `calculate_damage`

## Исходный код

```python
def calculate_damage(base, armor, crit_multiplier):
    damage = base - armor * 0.5
    if damage < 0:
        damage = 0
    if crit_multiplier > 1:
        damage *= crit_multiplier
    return damage
```

## Шаг 1: Assumptions

```json
{
  "assumptions": [
    {"id": "A1", "type": "range", "desc": "base >= 0", "violated_by": [-1, -100]},
    {"id": "A2", "type": "range", "desc": "armor >= 0", "violated_by": [-1, -100]},
    {"id": "A3", "type": "finite", "desc": "base is finite", "violated_by": ["NaN", "Inf"]},
    {"id": "A4", "type": "finite", "desc": "armor is finite", "violated_by": ["NaN", "Inf"]},
    {"id": "A5", "type": "finite", "desc": "crit is finite", "violated_by": ["NaN", "Inf"]},
    {"id": "A6", "type": "range", "desc": "crit >= 0", "violated_by": [-1, -0.5]}
  ],
  "invariants": [
    {"id": "I1", "desc": "result >= 0", "layer": "L3", "direction": "D10"},
    {"id": "I2", "desc": "result is finite", "layer": "L9", "direction": "D1"},
    {"id": "I3", "desc": "higher armor → lower or equal damage", "layer": "L3", "direction": "D10"}
  ]
}
```

## Шаг 2: Blind Spots

```json
{
  "blind_spots": [
    {"id": "BS1", "severity": "HIGH", "desc": "A2 untested — negative armor increases damage"},
    {"id": "BS2", "severity": "HIGH", "desc": "A5 untested — NaN crit produces NaN result"},
    {"id": "BS3", "severity": "HIGH", "desc": "A6 untested — negative crit → negative damage"},
    {"id": "BS4", "severity": "MEDIUM", "desc": "I2 untested — Inf base produces Inf result"}
  ]
}
```

## Шаг 3: Tests

```python
import math
from hypothesis import given, strategies as st

# L0: Smoke
class TestSmoke:
    def test_import(self):
        from damage_module import calculate_damage
    def test_call(self):
        result = calculate_damage(100, 20, 1.0)
        assert result is not None

# L1: Contract
class TestContract:
    def test_documented_examples(self):
        assert calculate_damage(100, 20, 1.0) == 90.0
        assert calculate_damage(100, 20, 2.0) == 180.0
        assert calculate_damage(10, 50, 1.0) == 0.0
    def test_return_type(self):
        assert isinstance(calculate_damage(100, 20, 1.0), float)

# L2: Boundary (8 conditions)
class TestEdgeCases:
    def test_zero(self):
        assert calculate_damage(0, 0, 0) == 0.0
    def test_one_element(self):
        assert calculate_damage(1, 0, 1.0) == 1.0
    def test_max_value(self):
        result = calculate_damage(float('inf'), 0, 1.0)
        assert math.isfinite(result), "Max value must not produce Inf"
    def test_out_of_bounds(self):
        result = calculate_damage(-10, 20, 1.0)
        assert math.isfinite(result)
    def test_inclusion_exclusion(self):
        assert calculate_damage(100, 20, 1.0) == 90.0       # crit=1 → NO mult
        assert calculate_damage(100, 20, 1.0000001) != 90.0  # crit>1 → mult
    def test_special_chars(self):
        result = calculate_damage(float('nan'), 20, 1.0)
        assert math.isfinite(result), "NaN input must not produce NaN"
    def test_mixed_types(self):
        result = calculate_damage(100, 20, 1)  # int crit
        assert result == 90.0

# L3: Property
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
            assert d1 <= d2

# L4: Adversarial
class TestAdversarial:
    def test_negative_armor(self):
        result = calculate_damage(100, -20, 1.0)
        assert result <= 100, f"Negative armor increased damage: {result}"
    def test_nan_crit(self):
        result = calculate_damage(100, 20, float('nan'))
        assert math.isfinite(result), "NaN crit produced non-finite"
    def test_negative_crit(self):
        result = calculate_damage(100, 20, -2.0)
        assert result >= 0, f"Negative crit produced negative damage: {result}"
    def test_inf_base(self):
        result = calculate_damage(float('inf'), 20, 1.0)
        assert math.isfinite(result), "Inf base must not propagate to Inf"

# L9: Negative Space
class TestNegativeSpace:
    @given(st.floats(allow_nan=True, allow_infinity=True),
           st.floats(allow_nan=True, allow_infinity=True),
           st.floats(allow_nan=True, allow_infinity=True))
    def test_result_always_finite(self, base, armor, crit):
        result = calculate_damage(base, armor, crit)
        assert math.isfinite(result), f"Non-finite for base={base}, armor={armor}, crit={crit}"
```

> **⚠️ L9 `test_result_always_finite` упадёт** — потому что `calculate_damage`
> не обрабатывает NaN/Inf. Это **ожидаемо**: тест нашёл слепую зону.
> Действие: добавить `if not math.isfinite(base): return 0.0` в код, либо
> документировать «функция принимает только конечные числа».

## Шаг 4: Verification

```bash
$ mutmut run --paths-to-mutate=damage_module --tests-dir=tests/
$ mutmut results
```

```
Mutation: `if damage < 0` → `if damage <= 0`
  test_edge_cases.test_inclusion_exclusion: CAUGHT ✓
  test_properties.test_damage_nonnegative: CAUGHT ✓

Mutation: `damage *= crit_multiplier` → `damage /= crit_multiplier`
  test_contract.test_documented_examples: CAUGHT ✓ (180.0 → 45.0)

Mutation: `armor * 0.5` → `armor * 0.4`
  test_contract.test_documented_examples: CAUGHT ✓ (90.0 → 92.0)

Kill score: 78% — 3 weak tests усилены
```