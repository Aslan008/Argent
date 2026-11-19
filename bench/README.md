# Agent bench

Кейс — это задача, стартовое состояние и проверки, которые решают pass/fail.
Bench выключен по умолчанию.

## Включение

```bash
export ARGENT_FEATURE_BENCH=1        # PowerShell: $env:ARGENT_FEATURE_BENCH='1'
python -m argent.bench list
python -m argent.bench run bench/cases/*.yaml --report bench/runs/report.md
python -m argent.bench promote --session <id-or-.json.gz> --turn 2
python -m argent.bench mine --limit 10
```

`promote` делает черновик из сохранённой сессии; `mine` ищет ошибки в trace/сессиях и кладёт черновики в `bench/cases/_drafts/`. Черновик без `checks` не запускается — проверки добавляет человек.

## Формат кейса

```yaml
id: fix-import-cycle
task: "Починить циклический импорт в src/foo.py"
start: HEAD                         # или refs/argent/checkpoints/<sha>
setup: ["python -m pip install -e ."]   # опционально
limits:
  timeout_s: 300
checks:
  - cmd: "pytest tests/test_foo.py -q"
    expect_exit: 0
    timeout_s: 120
  - cmd: "python -c 'import src.foo'"
    stdout_contains: ""
tags: [imports]
```

Правила:

- кейс без `checks` не запускается;
- `start` по умолчанию `HEAD`; для воспроизведения используйте чекпоинт;
- destructive-действия агента в bench запрещены, остальные разрешены;
- worktree создаётся во временной папке и удаляется в `finally` (если не `--keep`);
- `bench/runs/` и `bench/worktrees/` не попадают в git.

## Ограничения текущего среза

- агент запускается в отдельном процессе (`argent.bench.worker`) с таймаутом; при зависании процесс убивается;
- `allowed_tools` ограничивает набор инструментов, если указан в кейсе; сеть пока не ограничивается;
- отчёт пишется в `--report` или печатается в stdout;
- `promote`/`mine` создают только черновики без проверок.
