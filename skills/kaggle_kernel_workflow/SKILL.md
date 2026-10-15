# Kaggle Kernel Workflow (Leeth)

Проверенный цикл обучения/экспериментов на Kaggle для Leeth-моделей (паттерн kernel_du + kernel_dopt).

## Цикл
1. **Код кернела**: `.argent/kaggle/kernel_<name>/<name>_kernel.py` + `kernel-metadata.json` (gpu: true, dataset_sources).
2. **Датасеты-зависимости**: заливать через `kaggle datasets create -p .` ИЗНУТРИ папки датасета (CLI ломается на абсолютных путях в temp). Обновление: `kaggle datasets version -p . -m "..."`. Готовность: `kaggle datasets status <id>` → `ready` (exit 1 при успехе — норма CLI).
3. **Передача весов между кернелами** (дообучение поверх чекпоинта): чекпоинт → отдельный датасет (напр. `hunzlee/leeth-dopt-ckpt`), кернел-наследник подключает его в dataset_sources и грузит дифф слоёв в базу до инференса.
4. **Push**: `kaggle kernels push -p .argent/kaggle/kernel_<name>` — exit 1 при успехе НЕ фатален.
5. **Поллер** (start_background_command, проверка раз в 600s):
   ```powershell
   $s = (kaggle kernels status hunzlee/<kernel> 2>$null | Select-String 'status') -join ' '
   ```
   ⚠️ Обязательно `2>$null | Select-String 'status'`: PowerShell-warning ("NativeCommandError") матчится `-match 'ERROR'` case-insensitively → ложный break после первой проверки.
6. **Ожидание**: `wait_heartbeat(until='file_contains("<status_file>", "COMPLETE") or ... "ERROR" ...')`.
7. **Output**: перед скачиванием `Remove-Item <output_dir> -Recurse -Force` (CLI не перезаписывает). Лог — JSON-массив (не JSONL): `json.load`, не построчно.

## Протокол перед пушем (обязательный)
- **AST**: `python -X utf8 -c "import ast; ast.parse(open(...).read())"`.
- **Локальный smoke SMOKE=1**: кернел обязан поддерживать локальный режим (fallback-пути `.argent/models/...` вместо /kaggle/input, мини-размеры данных). Прогон на CPU ловит NameError/shape-баги ДО 2-часового прогона (v7 smoke: 22s, поймал раскачку LR).
- **LR доводки**: если модель уже сошлась (loss ~0.07), полный LR обучения раскачивает её — снижать в 5-10 раз (v7: 1e-4/1e-5 вместо 1e-3/5e-5).
- **Best-трекинг с PROTECT**: при дообучении трекать лучший раунд по целевой метрике, но кандидат обязан не деградировать по остальным (PROTECT-пороги); snapshot/restore весов на CPU; чекпоинт сохранять ДО raise гейтов.

## Ловушки кернелов (реальные баги этого проекта)
- **dataclass + `from __future__ import annotations`**: при importlib-загрузке модуль обязан быть в `sys.modules` до `exec_module`, иначе `KeyError` при резолве аннотаций.
- **`global` для модульных констант, переприсваиваемых в main()** (BASE_DIR, DOMAINS_DIR): иначе UnboundLocalError.
- **label_smoothing + маскированные логиты (−1e9)**: smoothing размазывает массу по замаскированным классам → градиент через LSE давит валидные логиты вниз → loss ~14M, обучение ломается. CE без smoothing.
- **Метрика = оптимальность хода** (argmax ∈ optimal_set эксперта), не совпадение с единственным таргетом: BFS/minimax имеют несколько оптимальных ходов.
- **Mean-pool замороженной базы — плато ~85%** на локально-геометрических задачах; план Б — unfreeze последних 2 слоёв (LR 5e-5) end-to-end: скачок +15-50 п.п. (v5: snake 82→97.7, snake_ood 44→93).
- **Калибровать гейты baseline-экспертом**: winrate гейта без измерения winrate эксперта — гадание (эксперт BFS: 93%).

## Инфраструктура
- База mmBERT: датасет `hunzlee/leeth-mmbert-du` (manifest.json + part_XXX, склейка md5-проверкой).
- Генераторы задач: датасет `hunzlee/leeth-task-domains` (task_domains.py, чистый Python).
- Чекпоинт v6: датасет `hunzlee/leeth-dopt-ckpt` (mmbert_dynamic_opt_v1.pt).
- Чекпоинты качать в `.argent/models/` (kernel_dopt_output перезаписывается).
- Полный лог: `scripts/_dopt_read_log.py` (stdout + ошибки с таймстампами).
- DAgger-паттерн (v7): rollout ТЕКУЩЕЙ политики → состояния + метки эксперта → дообучение на смеси с replay всех доменов (против забывания); лечит imitation-зазор (step-opt 98% но winrate 87% — накопление ошибок за роллаут). Результат v7: winrate 87→90%, snake 98.1→99.3%, snake_ood 93.3→94.9%, ttt 99.9% — все гейты OK.
- Чекпоинт v7: `.argent/models/mmbert_dynamic_opt_v2.pt` (локально) + датасет `hunzlee/leeth-dopt-ckpt` (v1 для наследников).