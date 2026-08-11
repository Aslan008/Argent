# Направления (D1–D10) + Named Patterns

Направления ортогональны слоям. Каждый тест находится на пересечении Layer ×
Direction.

---

## D1: Mathematical / Numeric
- Float precision, rounding, accumulation errors
- Integer overflow, underflow
- NaN, Inf, -Inf propagation
- Division by zero (и почти-ноль)
- Mixed-type arithmetic (int + float)
- Random number determinism
- Statistical properties (distribution, variance)

## D2: State / Transition
- State machines, lifecycle, ordering
- Unreachable/dead states
- State corruption and recovery
- Transition guards and preconditions
- Concurrent state access

## D3: Data Integrity
- Data corruption (partial writes, truncation)
- Data loss (silent, undetected)
- Consistency (invariants across data structures)
- Persistence (save/load round-trip)
- Serialization/deserialization fidelity
- Reference vs value semantics

## D4: Error Handling
- Error propagation (does the right error reach the right handler?)
- Error recovery (can the system continue after an error?)
- Error during error handling (cascading failures)
- Silent error swallowing
- Error state consistency (system in valid state after error)
- Error type correctness (right exception type)

## D5: Resource
- Memory leaks (references held after use)
- File handle leaks
- Connection leaks (database, network)
- Resource exhaustion (out of memory, disk full)
- Resource cleanup on error paths
- Resource contention (two users, one resource)

## D6: Concurrency
- Race conditions
- Deadlocks
- Atomicity violations
- Visibility (changes visible to other threads?)
- Ordering guarantees (happens-before)
- Lock-free correctness

## D7: API Contract
- Interface violations (wrong types, wrong order)
- Protocol violations (calling before init, after close)
- Backwards compatibility (old callers still work?)
- Versioning (schema changes, migration)
- Nullability contracts
- Exception contracts (what can be thrown?)

## D8: Security
- Injection (SQL, command, path traversal)
- Buffer overflow / underflow
- Privilege escalation
- Authentication bypass
- Authorization checks
- Input validation (malicious input)

## D9: Performance
- Degradation (O(n²) where O(n) expected)
- Memory growth (unbounded collections)
- Timeout behaviour
- Cold start vs warm start
- Scaling (does it work at 10x scale?)

## D10: Invariant
- Conservation laws (total = const)
- Monotonicity (value only increases/decreases)
- Idempotency (f(f(x)) == f(x))
- Symmetry (f(a,b) == f(b,a))
- Bounds (value in [min, max])
- Type invariants (always non-null, always positive)

---

## Named Patterns

### Known Coordinates (внутри L1/L2)

**Вопрос:** Указывают ли операции именно туда, куда должны?

**Концепция:** Проверяет не ЧТО произошло, а ГДЕ. Критично для файловых
операций, буферов/массивов, LSP/AST, игровых координат.

**Метод:**
1. Создать структуру с уникальным маркером на каждой позиции
2. Запросить операцию для позиции N
3. Проверить, что результат содержит маркер N (а не N-1 или N+1)
4. Для операций записи: проверить, что изменён именно маркер N

```python
def test_precision_damage_applies_to_correct_enemy():
    enemies = [Enemy(hp=100), Enemy(hp=200), Enemy(hp=300)]
    apply_damage(enemies, 1, 50)  # index 1 = второй враг
    assert enemies[0].hp == 100, "Wrong enemy damaged!"
    assert enemies[1].hp == 150, "Target enemy not damaged!"
    assert enemies[2].hp == 300, "Wrong enemy damaged!"
```

### Cross-Layer Consistency (внутри L6)

**Вопрос:** Корректно ли преобразуются данные между слоями?

**Концепция:** Каждый слой имеет свои конвенции (1-indexed vs 0-indexed, тип
данных). Ошибки преобразования — классический источник багов.

**Метод:**
1. Идентифицировать каждый слой и его конвенцию
2. Для каждого перехода — проверить, что преобразование корректно
3. Проверить, что преобразование происходит ровно один раз (не дважды, не ноль)
4. Проверить обратное преобразование при возврате

```python
def test_cross_layer_indexing_consistency():
    ui_line = 1  # 1-indexed (UI слой)
    internal_index = ui_line - 1  # преобразование: 1→0
    file_line = internal_index  # 0-indexed (файловый слой)
    assert file_line == 0, f"Double conversion: expected 0, got {file_line}"
    result_ui_line = file_line + 1  # обратное преобразование
    assert result_ui_line == ui_line, "Round-trip conversion failed"
```