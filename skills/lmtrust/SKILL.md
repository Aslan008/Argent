---
name: lmtrust
description: Deep blind spot testing — систематический поиск и тестирование слепых зон в коде через извлечение предположений, нарушение инвариантов и мутационную валидацию.
---

# LMTrust — Deep Blind Spot Testing

> Обычные тесты спрашивают «код делает то, что должен?» — blind spot тесты
> спрашивают «что код делает, чего НЕ должен? Какие предположения он делает?
> Что происходит, когда они ломаются?»

## Когда запускать

- **Новый код** — написана новая функция, класс, модуль
- **Модификация критической логики** — combat math, XP/leveling, economy, damage mitigation
- **Перед мержем** — pre-merge checklist (см. ниже)
- **Баг-репорт** — регрессионный тест для найденного бага

## Pipeline (5 шагов)

```
Код → 1. Analyze → 2. Map → 3. Generate → 4. Verify → 5. Report
```

| Шаг | Что делает | Выход |
|------|-----------|-------|
| **1. Analyze** | Извлекает предположения и инварианты из кода | Список предположений |
| **2. Map** | Картирует слепые зоны — что НЕ протестировано | Coverage matrix |
| **3. Generate** | Генерирует тесты-нарушители для каждой слепой зоны | Файл тестов |
| **4. Verify** | Запускает mutmut/Stryker, усиливает слабые тесты | Kill score |
| **5. Report** | Отчёт: что найдено, что непокрыто, приоритеты | Отчёт |

> **⚠️ Важно: blind spot тесты ДОЛЖНЫ падать.** Падение теста = найденная
> слепая зона. Не ослабляй assertion — предлагай исправление кода. Workflow:
> 1. Тест падает → это ожидаемо (нашли слепую зону)
> 2. Разработчик решает: добавить защиту в код ИЛИ документировать ограничение
> 3. Тест обновляется под решение (либо код исправлен, либо тест отмечен как
>    документирующий ограничение)

---

## Шаг 1: Analyze — извлечение предположений

Каждый кусок кода делает неявные предположения. Обычные тесты работают ВНУТРИ
этих предположений. Blind spot тесты их извлекают и нарушают.

| Тип | Паттерн в коде | Что предполагается | Как нарушить |
|-----|---------------|-------------------|--------------|
| **Range** | `x / y` | y ≠ 0 | `y = 0` |
| **Non-negative** | `math.sqrt(x)` | x ≥ 0 | `x = -1` |
| **Finite** | `x * y` | x, y конечные | `NaN`, `Inf` |
| **Non-null** | `obj.method()` | obj ≠ None | `obj = None` |
| **Non-empty** | `list[0]` | list непустой | `list = []` |
| **Within bounds** | `arr[i]` | 0 ≤ i < len(arr) | `i = len(arr)` |
| **Initialized** | `self.x` | self.x установлен | вызов до init |
| **Ordering** | `init(); use()` | init до use | `use()` без `init()` |
| **State valid** | `self.hp -= dmg` | self.hp ≥ dmg | `dmg > self.hp` |
| **Resource available** | `file.read()` | file открыт | file закрыт |

**Инварианты** — что ВСЕГДА должно быть истинно: damage ≥ 0, hp в [0, max_hp],
xp только растёт, f(f(x)) == f(x).

---

## Шаг 2: Map — картирование слепых зон

Для каждого предположения из шага 1 — отметить: протестировано или нет.
Непротестированные предположения = слепые зоны. Каждая слепая зона получает
severity (HIGH/MEDIUM/LOW) и suggested_test (Layer × Direction).

---

## Шаг 3: Generate — генерация тестов

### Дерево решений: какие слои применять

Не все 11 слоёв применимы к каждому коду. Выбери 3–5 с максимальным ROI:

```
Чистая функция (числа)?
  → L0, L1, L2, L4, L3         D1, D10

Stateful объект (HP, shield, состояние)?
  → L0, L1, L5, L4, L9         D2, D3, D4

Файловый I/O (save/load, чтение/запись)?
  → L0, L1, L2, L6, L4b        D3, D5, Known Coordinates

Внешний API (HTTP, БД, сервис)?
  → L0, L1, L4b, L7, L8        D4, D5, D6

Кросс-слойная система (UI → логика → файл)?
  → L0, L1, L6                 D3, D7, Cross-Layer

Конкурентный код (async/await, threads)?
  → L0, L1, L8, L7             D6, D4

Другое / не уверен?
  → L0, L1, L2, L4, L9         D1, D10
```

### Слои (кратко)

| Слой | Вопрос | Слепая зона |
|------|--------|-------------|
| **L0** Smoke | Оно вообще живое? | Код не запускается |
| **L1** Contract | Делает то, что говорит? | Нарушает контракт |
| **L2** Boundary | Что на краях? | Ломается на границах (8 условий) |
| **L3** Property | Что ВСЕГДА истинно? | Нарушает инварианты на случайных |
| **L4** Adversarial | Что если предположения — ложь? | Ломается при нарушении |
| **L4b** Fallback | Что когда основной путь недоступен? | Fallback ломается |
| **L5** State | Все ли состояния достижимы? | Ломается при переходах |
| **L6** Cross-System | A+B ломается? | Взаимодействие проваливается |
| **L7** Compositional | Две вещи ломаются одновременно? | Комбинация сбоев |
| **L8** Temporal | Зависит ли порядок? | Ломается в другом порядке |
| **L9** Negative Space | Что НЕ должно происходить? | Делает лишнее |

> **Подробные описания, шаблоны кода и примеры для каждого слоя —
> см. `references/layers.md`**

### Направления (D1–D10)

| Направление | Что проверяет |
|-------------|--------------|
| D1: Math/Numeric | Float precision, overflow, NaN/Inf, division by zero |
| D2: State/Transition | State machines, lifecycle, unreachable states |
| D3: Data Integrity | Corruption, loss, consistency, serialization |
| D4: Error Handling | Propagation, recovery, cascading, silent swallowing |
| D5: Resource | Leaks, exhaustion, cleanup on error paths |
| D6: Concurrency | Race conditions, deadlocks, atomicity |
| D7: API Contract | Interface violations, protocol, backwards compat |
| D8: Security | Injection, overflow, privilege escalation |
| D9: Performance | Degradation, memory growth, timeout, scaling |
| D10: Invariant | Conservation, monotonicity, idempotency, bounds |

**Именованные паттерны** (внутри слоёв, не отдельные направления):
- **Known Coordinates** (в L1/L2) — проверяй не ЧТО, а ГДЕ
- **Cross-Layer Consistency** (в L6) — проверяй преобразования между слоями

> **Подробности — см. `references/directions.md`**

### Шаблон файла тестов

```python
# test_<feature>.py
class TestSmoke:              # L0
class TestContract:           # L1 × D7
class TestCoordinates:        # L1 × Known Coordinates
class TestEdgeCases:          # L2 × D1 (8 граничных условий)
class TestProperties:         # L3 × D10 (Hypothesis)
class TestAdversarial:        # L4 × D1-D10 (нарушение предположений)
class TestFallback:           # L4b × D4
class TestState:              # L5 × D2
class TestIntegration:        # L6 × No-Mock E2E
class TestLayerConsistency:   # L6 × Cross-Layer
class TestCompositional:      # L7 × D4
class TestTemporal:           # L8 × D6
class TestNegativeSpace:      # L9 × D3
```

Не все классы нужны — используй дерево решений. Шаблоны: `templates/l*.py`.

**Итого: 11 слоёв (L0–L9 + L4b) × 10 направлений = 110 категорий** (справочник).
Цель — не 110 тестов, а 3–5 целенаправленных на функцию.

---

## Шаг 4: Verify — мутационная валидация

Тест, который не ловит ни одну мутацию — тавтология (AP4).

**Python:**
```bash
pip install mutmut
mutmut run --paths-to-mutate=<module> --tests-dir=tests/
mutmut results          # непойманные мутации
mutmut show <id>        # конкретная мутация
```

**C# / .NET / Unity:**
```bash
dotnet tool install -g dotnet-stryker
dotnet stryker --project <project>.csproj
```

| Kill score | Оценка | Действие |
|-----------|--------|----------|
| ≥ 80% | Отлично | Тесты сильные |
| 50–80% | Приемлемо | Проверить непойманные мутации |
| < 50% | Слабые | Усилить assertions, перезапустить |

---

## Шаг 5: Report

```
LMTrust Report for <function>
=============================
Assumptions: 6 | Invariants: 4 | Blind spots: 6 (3 HIGH, 2 MEDIUM, 1 LOW)
Tests generated: 14 | Kill score: 78%
```

---

## Антипаттерны (AP1–AP8)

### AP1: Мокать то, что тестируешь
```python
# ПЛОХО: мок не проверяет логику
with patch('combat.calculate_damage', return_value=50):
    assert combat.attack(player, enemy) == 50  # Бесполезно

# ХОРОШО: реальный вызов
assert combat.attack(player, enemy) == 45
```

> **Исключение:** Для внешних API (HTTP, БД) моки транспортного слоя —
> необходимость. Мокай транспорт (requests.get), тестируй логику (retry,
> fallback). L6 No-Mock E2E неприменим к внешним сервисам — только к
> внутренним подсистемам проекта.

### AP2: Тестировать только happy path
```python
# ПЛОХО: только существующий текст
assert replace_in_file("old", "new")  # Что если "old" нет?
# ХОРОШО: и happy, и failure
assert replace_in_file("old", "new") == True
assert replace_in_file("nonexistent", "new") == False
```

### AP3: Проверять «что» вместо «где»
```python
# ПЛОХО: "REPLACED" где-то — но где?
assert "REPLACED" in result
# ХОРОШО: конкретная позиция
assert result.lines[5] == "REPLACED"
assert result.lines[4] == "original_line_4"  # соседние не задеты
```

### AP4: Тест, который не может провалиться
```python
# ПЛОХО: проходит всегда
assert len(result) >= 0  # Всегда True
# ХОРОШО: конкретное значение
assert result == 90.0
assert len(result) == 3
```

### AP5: Мокать побочные эффекты, влияющие на результат
```python
# ПЛОХО: мок памяти скрывает баг
with patch('memory.save'): checkpoint.create()
# ХОРОШО: мокаем только то, что не влияет
with patch('logging.write'): checkpoint.create(); assert checkpoint.data == expected
```

### AP6: Предполагать Unix line endings
```python
# ПЛОХО: split("\n") оставляет \r на Windows
lines = content.split("\n")
# ХОРОШО: обрабатываем CRLF
lines = content.splitlines()
```

### AP7: Не проверять тип возвращаемого значения
```python
# ПЛОХО: assert result проходит для [], "", None, 0, False
assert result
# ХОРОШО: конкретный тип и значение
assert isinstance(result, float); assert result == 90.0
```

### AP8: Тестировать реализацию, а не контракт
```python
# ПЛОХО: проверяем внутреннее состояние
assert manager._line == 9  # Приватный атрибут
# ХОРОШО: проверяем публичный контракт
assert result.line == 10
```

---

## Pre-Merge Checklist

### Quick (рутинный мерж — рефакторинг, добавление поля)
- [ ] L0: код импортируется и вызывается
- [ ] L1: документированные примеры проверены
- [ ] L2: граничные условия (0, пустой, NaN/Inf если применимо)
- [ ] L4: хотя бы одно предположение нарушено
- [ ] AP-чек: нет AP4 (тест может провалиться)

### Deep (критическая логика — damage, economy, XP, новая подсистема)
- [ ] **Все Quick пункты**
- [ ] Каждый параметр проверен контрактом (L1 × D7)
- [ ] Каждая операция с координатами проверена (Known Coordinates)
- [ ] Каждое преобразование между слоями проверено (L6 × Cross-Layer)
- [ ] 8 граничных условий для каждого параметра (L2)
- [ ] Хотя бы один тест без моков (L6 × No-Mock E2E)
- [ ] Каждый fallback-путь проверен (L4b × D4)
- [ ] Каждый инвариант проверен на случайных входах (L3 × D10)
- [ ] Хотя бы один композиционный сбой (L7 × D4)
- [ ] Хотя бы один негативный тест — что НЕ должно происходить (L9 × D3)
- [ ] Мутационная проверка — kill score ≥50% (mutmut/Stryker)
- [ ] Каждый найденный баг имеет регрессионный тест

---

## Справочные материалы

| Файл | Содержание |
|------|-----------|
| `references/layers.md` | Полные описания 11 слоёв с примерами кода |
| `references/directions.md` | D1–D10 + Named Patterns |
| `references/examples/` | 4 полных примера прогона |
| `references/kern_integration.md` | Интеграция с Kern Gate |
| `templates/l*.py` | Python-шаблоны для каждого слоя |