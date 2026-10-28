# План реализации: Replay + Agent Bench (и примыкающие направления)

**Дата:** 2026-10-23  
**Статус:** план; код ещё не начат.  
**Главный принцип:** всё новое выключено по умолчанию, включается явно, не меняет текущий поток при выключенных флагах.

## 1. Границы

### Входит в этот план (committed)

- Trace-журнал прогонов.
- Replay: просмотр прогона по шагам.
- Bench: воспроизводимые кейсы с проверками в изолированном worktree.
- Bench из сессий: превращение реальной задачи в черновик кейса.
- Сканер неудачных сессий → черновики кейсов (без автодобавления).

### Ждёт отдельного решения (design-ready, не начинать)

- Типизированные навыки и оркестрация (profile/policy/strategy/procedure/tool/knowledge).
- Уроки с доказательствами и извлечением при ошибке.
- Консолидация опыта в навыки/адаптеры.

### Отложено / исследовательское

- «Агент, доказывающий результат» — parked, см. `reports/idea_verified_agent_2026-10-23.md`.
- Мульти-фьючерсы (branch tournament) — кандидат, зависит от bench.
- Диффузионные модели, canvas denoising, FIM-редактор — исследовательский трек, не в этом плане.
- Автоматическое обновление весов — нет.

## 2. Feature flags

Новый модуль `argent/features.py`:

```python
FEATURES = ("replay", "bench", "skills_v2", "lessons")
def enabled(name: str) -> bool: ...  # env ARGENT_FEATURE_<NAME> или config.features.<name>, default False
```

Правила:

- все флаги по умолчанию **False**;
- при выключенном флаге не создаются файлы, не подключаются sink'и, не меняется промпт;
- включение — только явно: env или `argent config set features.replay true`;
- тест `test_features_default_off` проверяет, что без флагов поведение и файлы не меняются;
- kill switch: снять env/флаг — система возвращается к текущему поведению без миграций.

## 3. Архитектура и файлы

```
argent/
  features.py                 # флаги
  replay/
    __init__.py
    schema.py                 # события trace, версия
    recorder.py               # запись JSONL, редакция, клип
    reader.py                 # чтение/фильтрация
    render.py                 # CLI-таймлайн
    __main__.py               # python -m argent.replay
  bench/
    __init__.py
    case.py                   # схема кейса, загрузка
    runner.py                 # worktree, headless-прогон, проверки
    report.py                 # markdown/json отчёт
    promote.py                # сессия → черновик кейса
    __main__.py               # python -m argent.bench
  cli/                        # команды argent replay / argent bench (если CLI уже есть)
bench/
  cases/                      # кейсы проекта (tracked)
  runs/                       # результаты (gitignored)
tests/
  replay/
  bench/
```

Trace-файл: `~/.argent/sessions/<id>.trace.jsonl` рядом с сессией.  
Bench-кейсы: `bench/cases/<id>.yaml` в проекте; запуски — `bench/runs/<timestamp>/`.

## 4. Фазы

### Фаза 0. Каркас и страховка (1–2 дня)

- `features.py` + тесты.
- Команды-заглушки `argent replay` / `argent bench`, которые при выключенном флаге печатают «disabled» и ничего не делают.
- Пустые пакеты `argent/replay`, `argent/bench`, тесты-заготовки.
- CI: обычный прогон без флагов; отдельный job с флагами (позже).
- **Критерий:** все текущие тесты зелёные; без флагов не появляется ни одного файла.

### Фаза 1. Trace (3–5 дней)

- Схема v0: `meta`, `turn_start`, `model_request` (хэши/параметры), `model_response` (usage/latency), `tool_start`, `tool_end` (exit/duration/result_ref), `approval`, `checkpoint`, `diff`, `error`, `turn_end`.
- Recorder подключается к существующему event sink (`AgentSession`/`ReplBridge`) и к нескольким хукам agent; любые ошибки записи глотаются и логируются.
- Редакция секретов через существующий `secret_redactor`; результаты клипуются; полные тексты не дублируются — ссылки на сообщения сессии.
- Ротация/лимит размера; атомарная запись.
- **Критерий:** overhead < 2% по времени хода или < 50 мс; trace читается; без флага файла нет.

### Фаза 2. Replay (2–3 дня)

- `argent replay <session>`: таймлайн, шаги, инструменты, аргументы/результаты, время/токены, чекпоинты, diff.
- Фильтры: `--turn`, `--tools`, `--errors`, `--json`.
- Fallback для старых сессий без trace: собрать из сообщений (использовать `history_replay.build_items` как основу).
- Replay **ничего не исполняет**; это чтение.
- **Критерий:** одна реальная сессия воспроизводится по шагам; fallback работает; никаких side effects.

### Фаза 3. Bench core (5–7 дней)

- Схема кейса (YAML): `id`, `task`, `start` (checkpoint sha/commit + patch), `setup`, `checks`, `limits`, `allowed_tools`, `tags`.
- Runner:
  - `git worktree add --detach <tmp> <sha>` (fallback: копия дерева);
  - прогон headless через `AgentSession.handle` в subprocess с `cwd=<worktree>`;
  - approval-политика для bench: по умолчанию запрещать destructive/сетевые действия, разрешать safe;
  - проверки: `cmd`, `expect_exit`, `stdout_contains`, `file_exists`, timeout;
  - сбор diff, trace (если включён), usage, времени;
  - очистка worktree в finally.
- Отчёт: JSON + markdown; сравнение двух прогонов.
- **Критерий:** 3 кейса проходят/падают воспроизводимо; worktree всегда чистится; нет интерактива.

### Фаза 4. Bench из сессий (3–4 дня)

- `argent bench promote <session> --turn N`: извлекает запрос, cwd, чекпоинт (если есть в trace), создаёт `bench/cases/<id>.yaml` с `checks: []`.
- Пользователь заполняет проверки; без проверок кейс не запускается.
- Сканер: ошибки инструментов, правки пользователя, откаты, повторные команды → `bench/cases/_drafts/`.
- **Критерий:** одна реальная ошибка становится кейсом; черновик не попадает в прогон без подтверждения.

### Фаза 5 (опционально, после решения). Типизированные навыки

- `skill card` с `kind/scope/activation/authority/affects/conflicts/risk`.
- Реестр и роутер (правила + retrieval; Leeth позже).
- Разделение profile/policy/strategy/procedure/tool/knowledge/orchestration.
- Всё под `features.skills_v2=False`; текущий `skill_manager` не меняется.
- **Критерий:** при выключенном флаге поведение идентично; при включённом — router выбирает top-k и не ломает существующие вызовы.

### Фаза 6 (опционально). Уроки и консолидация

- Уроки с `trigger → wrong → right + evidence`, TTL, scope.
- Извлечение при ошибке; не в системный промпт «на всякий случай».
- Фоновый job: эпизоды → уроки → навыки; экспорт пар для адаптеров.
- **Критерий:** повторяемость ошибок падает на bench; уроки не попадают в глобальный контекст без проверки.

### Фаза 7 (исследование). Fork/counterfactual replay и мульти-фьючерсы

- `git worktree` от чекпоинта, продолжение с другим моделью/промптом.
- Сравнение веток; выбор по проверкам.
- Не включать по умолчанию; только после стабильного bench.

## 5. Что не делаем

- Не трогаем основной цикл `agent.py` без флага.
- Не включаем trace/bench по умолчанию.
- Не проигрываем write/run вслепую.
- Не храним полные промпты/секреты в trace.
- Не мержим worktree автоматически.
- Не запускаем destructive-команды без approval.
- Не добавляем «магическое» обучение весов на каждом шаге.

## 6. Тестирование

- Обычный `pytest` — без флагов, поведение не меняется.
- Новые тесты:
  - `tests/replay/test_schema.py` — сериализация/чтение, неизвестные события.
  - `tests/replay/test_recorder.py` — редакция, клип, ошибка записи не ломает ход, нет файла без флага.
  - `tests/replay/test_render.py` — таймлайн, фильтры, fallback.
  - `tests/bench/test_case.py` — схема, невалидный кейс.
  - `tests/bench/test_runner.py` — worktree, таймаут, cleanup, approval-политика (на фейковом агенте).
  - `tests/bench/test_promote.py` — сессия → черновик.
- Интеграционный: один trace + один replay + один bench-кейс.
- Windows: пути, `git worktree`, удаление, блокировки.

## 7. Метрики и гейты

- Trace: overhead < 2% / < 50 мс; размер < 1 МБ на 100 шагов (после клипа).
- Replay: 95% событий отображаются; fallback на 100% старых сессий.
- Bench: 3 кейса, 3 повтора; pass/fail стабилен; cleanup 100%.
- Bench promote: < 5 минут на кейс.
- Ни одного изменения в существующих тестах без флага.
- Перед включением по умолчанию — неделя dogfood и сравнение с baseline.

## 8. Риски и меры

| Риск | Мера |
|---|---|
| Запись тормозит ход | async writer, лимит, выключено по умолчанию |
| Секреты в trace | redactor, клип, локальное хранение, opt-in |
| Worktree ломает git | отдельный temp, `--detach`, cleanup, не трогать ветки |
| Approval блокирует bench | явная политика, запрет destructive, таймаут |
| Тесты флейкуют | скрытые проверки, повтор 3 раза, фиксированный seed |
| Кейс без проверок | не запускать, статус invalid |
| Разрастание scope | фазы и флаги; skills/lessons только после отдельного решения |

## 9. Порядок и оценки

| Фаза | Оценка |
|---|---|
| 0. Каркас | 1–2 дня |
| 1. Trace | 3–5 дней |
| 2. Replay | 2–3 дня |
| 3. Bench core | 5–7 дней |
| 4. Bench из сессий | 3–4 дня |
| 5. Skills (опц.) | 5–8 дней |
| 6. Lessons (опц.) | 5–7 дней |

Итого до работающего core (0–4): **2–3 недели** спокойного темпа, без изменения текущего поведения.

## 10. Первый шаг

Начать с Фазы 0 + Фазы 1: флаги, пакеты, trace-схема и recorder. Это не меняет UX, не требует TUI и даёт первый реальный артефакт — журнал, по которому можно сделать replay.

---

## 11. Прогресс

### Сделано (2026-10-23)

- **Фаза 0:** `argent/features.py`, флаги `replay/bench/skills_v2/lessons` — по умолчанию выключены; env `ARGENT_FEATURE_*` и `features` в конфиге; неизвестное имя — ошибка, опечатка не включает.
- **Фаза 1 (MVP):** `argent/replay/schema.py`, `recorder.py`; запись нормализованных событий (`turn_start`, `user_message`, `tool_start/end`, `checkpoint`, `diff`, `usage`, `error`, `notice`, `done`, `session_link`), клип, редакция секретов, хэши/размеры результатов.
- **Интеграция:** `AgentSession` пишет trace только при `ARGENT_FEATURE_REPLAY=1`; работает и для `handle()`, и для `_run_turn()` (TUI); `ReplBridge` связывает trace с `session_id` и закрывает его в cleanup.
- **Фаза 2 (MVP):** `reader.py`, `render.py`, `python -m argent.replay` — список trace'ов, таймлайн, `--json`, `--content`, fallback на старые `.json.gz` сессии.
- **Тесты:** `tests/test_features.py` + `tests/replay/*` — 18 passed; bridge regression — 103 passed. Ruff clean.
- **Не сделано в trace:** отдельные `model_request/model_response` хуки (сейчас есть `usage` и события инструментов; запросы модели будут добавлены в Phase 1.1, если нужны для отладки).
- **Не начато:** Фаза 3 (bench runner), Фаза 4 (promote/miner), skills/lessons.

### Как включить

```
ARGENT_FEATURE_REPLAY=1
python -m argent.replay --latest
python -m argent.replay <trace.jsonl> --content
```

Trace по умолчанию: `~/.argent/replay/trace_<timestamp>_<pid>.jsonl` (override `ARGENT_REPLAY_DIR`).

### Сделано: Фаза 3 (MVP)

- `argent/bench/case.py` — YAML/JSON-кейсы, проверки, лимиты, валидация «без checks нельзя».
- `argent/bench/runner.py` — изолированный `git worktree` (fallback: копия дерева), setup, headless-прогон через `AgentSession`, safe-approval (destructive deny), проверки, diff, cleanup.
- `argent/bench/report.py` — markdown/JSON отчёт.
- `python -m argent.bench list|run` под `ARGENT_FEATURE_BENCH=1`; по умолчанию disabled.
- `bench/README.md` — формат кейса и ограничения.
- Тесты: `tests/bench/*` (10) — кейсы, pass/fail, setup, git-worktree изоляция, CLI.
- `.gitignore`: `bench/runs/`, `bench/worktrees/`.

Ограничения MVP: прогон агента в том же процессе (таймаут best-effort), `allowed_tools`/сеть пока не ограничиваются, отчёт только в stdout.

### Сделано: Phase 3.1 (укрепление) и Phase 4

- Агент для bench запускается в **отдельном процессе** (`argent/bench/worker.py`), таймаут убивает процесс; события и результат пишутся в файлы.
- `Case.allowed_tools` ограничивает набор инструментов (если задан); destructive-действия по-прежнему запрещены.
- `--report <path>` пишет markdown-отчёт в файл.
- `promote`: сохранённая сессия → `bench/cases/<id>.yaml` с задачей и `checks: []`; выбор хода, `--force`.
- `mine`: trace/сессии → `bench/cases/_drafts/` по ошибкам (`error`, failed, permission denied, traceback…), без автозапуска.
- CLI: `promote`, `mine`; тесты `tests/bench/test_promote.py`, `test_miner.py`, обновлённый runner.
- Тесты: **137 passed** (новые + bridge); ruff clean.

Ограничения: сеть не ограничена; `mine` — эвристика, черновики требуют ручных checks; отчёт не прикрепляется к trace автоматически.



