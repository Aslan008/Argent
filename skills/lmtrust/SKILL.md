---
name: lmtrust
description: Deep blind spot testing, logic audit, contract polish, and neuro-symbolic verification (Python, C#, Rust, C++) — систематический поиск скрытых дефектов, доказательство инвариантов через Physics-CDCL SMT solver, мутационная валидация (mutmut, Stryker, cargo-mutants, Mull) и runtime healing.
---

# LMTrust — Deep Blind Spot Testing & Quality Intelligence (Python, C#, Rust, C++)

> Обычные тесты спрашивают: «Код делает то, что должен?»
> **LMTrust спрашивает:** «Что код делает, чего НЕ должен? Какие неявные предположения он делает? Что происходит, когда они ломаются? Можно ли математически доказать отсутствие багов?»

---

## 🎯 Когда запускать

- **Новый код** — написана новая функция, структура, класс или модуль.
- **Критическая логика и математика** — combat math, XP/leveling, economy, damage mitigation, физика, сериализация, парсеры.
- **Поиск скрытых багов и несостыковок** — аудит перед релизом, поиск делений на 0, переполнений и уязвимостей с генерацией Proof-of-Concept.
- **Рефакторинг и укрепление контрактов** — перевод неявных валидаций в типы («Parse, don't validate»).
- **Перед мержем** — pre-merge checklist и diff-aware проверка PR.
- **Баг-репорт** — генерация регрессионного теста-нарушителя и автоматический self-healing.

---

## ⚙️ 5 Режимов работы (Modes)

| Режим | Назначение | CLI Команда | Выход |
|---|---|---|---|
| **Mode A: Deep Testing** | Генерация blind spot тестов по матрице 11 слоев × 10 направлений | `lmtrust run / generate / scaffold` | Файлы тестов + Coverage Matrix |
| **Mode B: Logic Audit** | Поиск багов, логических несостыковок и уязвимостей с PoC | `lmtrust audit / review` | Markdown, JSON, SARIF v2.1.0 |
| **Mode C: Code Polish** | Укрепление контрактов, идиоматичный рефакторинг | `lmtrust polish [--diff]` | Unified Diff / Markdown отчет |
| **Mode D: Self-Healing** | Замкнутый цикл устранения дефектов с песочницей | `lmtrust heal [--apply]` | Проверенный патч кода |
| **Mode E: Formal Prove** | Доказательство инвариантов и Translation Validation через CDCL | `lmtrust prove` | 100% доказательство (UNSAT) или контрпример (SAT) |

---

## 🌐 Мультиязычный стек (Python, C#, Rust, C++)

LMTrust автоматически определяет язык по расширению файла или опции `--language`:

| Язык | Расширения | Тестовый фреймворк | Мутационный движок | AST-анализатор | SMT / Формальная верификация |
|---|---|---|---|---|---|
| **Python** | `.py` | `pytest` + `hypothesis` | `mutmut` | Python `ast` | `Physics-CDCL SMT` (QF_BV / QF_ABV) |
| **C#** | `.cs` | `dotnet test` (xUnit/NUnit) | `dotnet-stryker` | `lmtrust-csharp` (Roslyn) | `Physics-CDCL SMT` |
| **Rust** | `.rs` | `cargo test` + `proptest` | `cargo-mutants` | `lmtrust-rust` (Syn) | `Physics-CDCL SMT` |
| **C++** | `.cpp`, `.hpp`, `.cc`, `.h` | `GoogleTest` / `ctest` | `Mull` | `lmtrust-cpp` (Clang) | `Physics-CDCL SMT` |

---

## 📋 CLI команды (`lmtrust`)

```bash
# ── Mode B: Поиск логических багов и несостыковок с Proof-of-Concept ─────────
lmtrust audit <file_or_dir> [--severity high] [--format text|markdown|json|sarif] [-o report.sarif]

# ── Mode C: Полировка кода и укрепление контрактов (Parse, don't validate) ───
lmtrust polish <file> [--focus contracts|idiomatic|performance] [--diff]

# ── Mode D: Замкнутый цикл автоисправления дефектов ──────────────────────────
lmtrust heal <file> [--apply] [--max-retries 3]

# ── Mode E: Формальная верификация через Physics-CDCL SMT Решатель ───────────
lmtrust prove <file> [--threads 8] [--bits 32]

# ── Diff-aware аудит изменений в PR / ветке ──────────────────────────────────
lmtrust review [--against main] [--severity high] [--format markdown]

# ── Mode A: Полный конвейер генерации Blind Spot тестов ──────────────────────
lmtrust run <file> [--test-file <tests>] [--runtime-heal] [--verify]

# ── Каркас тестов на базе AST сигнатур ───────────────────────────────────────
lmtrust scaffold <file> [-o <output_dir>] [--layers L0,L1,L2,L4,L9]

# ── Анализ матрицы покрытия (11 слоев × 10 направлений = 110 ячеек) ─────────
lmtrust coverage <test_file> [--format markdown|json|html]

# ── Мутационная валидация качества тестов ────────────────────────────────────
lmtrust verify <file> [--kill-threshold 0.5]
```

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

## CLI-автоматизация (lmtrust)

Пайплайн автоматизирован в Python-пакете `lmtrust`. Механика (coverage matrix,
mutmut, отчёты, scaffolding) — в фреймворке; семантика (извлечение
предположений, генерация тестов) — в LLM.

### Установка

```bash
pip install lmtrust                    # базовый пакет
pip install lmtrust[analyze]           # + openai для LLM-шагов
pip install lmtrust[test]              # + pytest, hypothesis, mutmut
```

### Команды

| Команда | Что делает |
|---------|-----------|
| `lmtrust run <file.py|file.cs>` | Полный пайплайн: Analyze → Map → Generate → [Verify] → Report |
| `lmtrust analyze <file.py|file.cs>` | Шаг 1: извлечь предположения через LLM |
| `lmtrust map <file.py|file.cs> -t <test.py|test.cs>` | Шаг 2: cross-reference с coverage matrix |
| `lmtrust generate <file.py|file.cs>` | Шаг 3: сгенерировать blind-spot тесты |
| `lmtrust coverage <test.py|test.cs>` | Показать Layer×Direction matrix для файла |
| `lmtrust verify <file.py|file.cs>` | Шаг 4: запустить mutmut/stryker, показать kill score |
| `lmtrust report <module>` | Шаг 5: сгенерировать отчёт |
| `lmtrust scaffold <file.py|file.cs>` | Шаблоны тестов по сигнатурам функций |

> **💡 Поддержка языков:** Инструмент автоматически определяет язык по расширению файла (`.py` или `.cs`). Для C# исходников он прозрачно запускает парсер на базе Roslyn (`lmtrust-csharp`) и использует `.NET SDK` (`dotnet test`, `dotnet stryker`) на этапах генерации и верификации.

### Пример: полный прогон через Ollama

```bash
# Локальный LLM через Ollama (OpenAI-compatible API)
lmtrust run src/lmtrust/coverage.py \
  --test-file tests/dogfood/test_coverage.py \
  --base-url http://localhost:11434/v1 \
  --model kimi-k2.7-code:cloud \
  --output reports/generated/ \
  --report reports/pipeline_demo.md
```

### Dogfood-результат: найден реальный баг

Прогон `lmtrust run` на `coverage.py` сгенерировал 5 blind-spot тестов.
Один нашёл **реальный баг**: `analyze_coverage()` падал с `UnicodeDecodeError`
на non-UTF-8 файлах — не было обработки ошибок кодировки.

**Исправление**: добавлен `try/except UnicodeDecodeError` с поднятием `ValueError`
с понятным сообщением. Регрессионный тест добавлен в dogfood suite.

Остальные 4 теста подтвердили **намеренные design-решения** (пропуск
неизвестных классов, D10 fallback, пропуск standalone функций без маркеров,
MemoryError не ловится) — помечены `@pytest.mark.xfail` с объяснением.

### Dogfood-результат: полный прогон на 6 модулях

Прогон `lmtrust run` на всех 6 исходных модулях (map, generate, analyze,
verify, scaffold, report) через Ollama LLM (`kimi-k2.7-code:cloud`).

**46 LLM-сгенерированных тестов нашли 30 реальных багов** в 4 модулях:

| Модуль | Багов найдено | Багов исправлено | xfail (by-design) |
|--------|--------------|-----------------|-------------------|
| `generate.py` | 9 | 4 | 5 |
| `report.py` | 8 | 2 | 6 |
| `verify.py` | 6 | 6 | 0 |
| `scaffold.py` | 7 | 6 | 1 |
| **Всего** | **30** | **18** | **12** |

**Ключевые исправления**:

- `generate.py`: None-входы для `_parse_generated_code`, `_count_test_functions`,
  `provider`, `GeneratedTests.code` — теперь raise правильные исключения
- `report.py`: `generate_report(fmt="html")` — теперь raise `ValueError`;
  JSON-сериализация — `default=str` для не-serializable объектов
- `verify.py`: `parse_mutmut_results(None)` → `TypeError`; bytes stderr → decode;
  runner returning None → `RuntimeError`; regex false positives на датах →
  требует whitespace после разделителя; timeout → `RuntimeError`
- `scaffold.py`: missing file → `FileNotFoundError`; invalid Python → `ValueError`;
  output_dir как файл → `ValueError`; missing template → `FileNotFoundError`;
  AnnAssign/AugAssign detection; `os.open`/`Path.open` detection

**Итог тест-сьюта**: 695 passed, 1 skipped (mutmut/WSL), 18 xfailed

### Покрытие dogfood-тестов

```
Coverage: 71/110 (64.5%) — 459 marker, 0 inferred
```

Все 11 слоёв представлены. Тесты с маркерами `@pytest.mark.lmtrust(layer=...,
direction=...)` — парсятся из AST, надёжно и однозначно.

### LLM-кеширование

`CacheProvider` оборачивает любой LLM-провайдер (`OpenAIProvider`,
`CallableProvider`) кешем на диске (`.lmtrust_cache/`). Повторные запросы
с теми же промптами не вызывают LLM — экономия времени и токенов.

```python
from lmtrust import CacheProvider, OpenAIProvider

inner = OpenAIProvider(model="gpt-4o-mini", api_key="...")
cached = CacheProvider(inner, cache_dir=".lmtrust_cache")
result = cached.complete(system_prompt, user_prompt)  # cache hit/miss
print(cached.stats)  # {"hits": N, "misses": M, "total": N+M}
```

### CI (GitHub Actions)

`.github/workflows/ci.yml` — автоматический прогон тестов на push/PR.
Python 3.11/3.12/3.13, unit + dogfood + generated tests.

### Runtime Healing & Closed-Loop Engine

Флаг `--runtime-heal` запускает сгенерированные тесты в песочнице pytest
прямо внутри конвейера. Если тесты падают с `ERROR` (неверные импорты,
отсутствующие фикстуры, неверные моки), ошибки отправляются обратно в LLM
для исправления.

---

## Расширенные Режимы LMTrust: Audit, Polish, Heal, Review

LMTrust расширен из генератора тестов в комплексную систему обеспечения качества,
логического аудита и автоматического исправления:

### 1. Режим `Audit` — Поиск багов и логических несостыковок (`lmtrust audit`)
Статически и семантически находит баги (`LOGIC_BUG`), противоречия контрактов (`INCONSISTENCY`),
уязвимости (`VULNERABILITY`) и негативное пространство (`NEGATIVE_SPACE`).
Каждая находка сопровождается **Proof-of-Concept (PoC)** сценарием воспроизведения.

```bash
# Аудит файла или директории
lmtrust audit src/payment_service.py --severity high
lmtrust audit src/ --format sarif -o security_audit.sarif
```

### 2. Режим `Polish` — Полировка, рефакторинг и Contract Hardening (`lmtrust polish`)
Анализирует неявные предположения и предлагает переход на строгие типы
(*"Parse, don't validate"* / `NewType` / `Pydantic` / `records` / `NonZero`), устранение
антипаттернов AP1–AP8 и генерирует unified diff.

```bash
# Рекомендации по полировке и hardening
lmtrust polish src/models.py --focus contracts --diff
```

### 3. Режим `Heal` — Замкнутый цикл самоисправления (`lmtrust heal`)
Автоматизирует цикл: `Поиск бага → PoC тест (падает) → Генерация патча → Верификация в песочнице (проходит) → Применение патча`.

```bash
# Исправление с применением патча к коду
lmtrust heal src/auth.py --apply
```

### 4. Режим `Review` — Diff-Aware проверка изменений (`lmtrust review`)
Проверяет git diff коммита или PR на появление новых слепых зон и логических регрессий.

```bash
lmtrust review --against main --format markdown
```

### 5. Режим `Prove` — Формальная верификация через Physics-CDCL SMT (`lmtrust prove`)
Доказывает инварианты и эквивалентность оптимизаций (Translation Validation) для всех $2^{32}$ или $2^{64}$ состояний через 8-поточный SMT-решатель.

```bash
# Формальное доказательство
lmtrust prove src/combat_math.py --threads 8 --bits 32
```

---

## 🤖 Протокол диалога агента (Agent Interaction Protocol)

При получении запроса `/lmtrust` или задачи на аудит/тестирование, агент следует трёхшаговому протоколу:

1. **Детект стека и фокуса:**
   * Определить язык проекта (Python, C#, Rust, C++).
   * Выявить файлы с критической математикой, состоянием (FSM), сетевыми протоколами или контрактами.
2. **Маршрутизация режима (ROI Routing):**
   * Если задача — найти скрытые ошибки $\rightarrow$ запустить **Mode B (Audit)** с PoC.
   * Если код оптимизируется или содержит битовые формулы $\rightarrow$ запустить **Mode E (Prove)** через CDCL SMT.
   * Если нужно глубокое тестирование $\rightarrow$ выбрать 3–5 слоев по Дереву решений (**Mode A**).
   * Если код полон неявных проверок $\rightarrow$ предложить **Mode C (Polish)**.
3. **Формат выдачи:**
   * Компактный отчет с иконками (`🚨 Bug`, `⚠️ Inconsistency`, `✨ Polish`, `🛡️ Proven`).
   * Для каждого бага — минимальный воспроизводящий сценарий (PoC).
   * Для изменений — готовый unified diff.

---

## 📚 Справочные материалы и шаблоны

| Файл / Каталог | Содержание |
|---|---|
| `references/layers.md` | Полные описания 11 слоёв с примерами кода |
| `references/directions.md` | D1–D10 + Named Patterns (Known Coordinates, Cross-Layer) |
| `references/examples/` | 4 полных сквозных примера прогона |
| `references/ci_cd.md` | Готовый GitHub Actions workflow для автоматического аудита и SARIF |
| `references/kern_integration.md` | Интеграция с Kern Gate и игровой математикой |
| `templates/` (Python) | Python-шаблоны `l0_smoke.py` ... `l9_negative_space.py` |
| `templates/rust/` (Rust) | Rust-шаблоны `l0_smoke.rs` ... `l9_negative_space.rs` (`cargo test`, `proptest`) |
| `templates/csharp/` (C#) | C#-шаблоны `L0SmokeTests.cs` ... `L9NegativeSpaceTests.cs` (`xUnit`, `FsCheck`) |
| `templates/cpp/` (C++) | C++-шаблоны `l0_smoke_test.cpp` ... `l4_adversarial_test.cpp` (`GoogleTest`) |
| `src/lmtrust/` | Python-пакет: analyze, audit, polish, heal, smt_solver, map, generate, verify, report, cli |