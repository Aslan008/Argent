---
name: tfshaderc-verify
description: Проверить шейдеры Unity-проекта без запуска редактора — быстрый локальный компилятор tfshaderc (Rust, workspace из двух крейтов) в пакете com.terrafor.terrain. Использовать при правках .shader/.hlsl/.compute, при вопросах «собираются ли шейдеры», для замеров времени и для приёмки изменений инструмента.
---

# tfshaderc: проверка шейдеров без Unity

Инструмент: `Packages/com.terrafor.terrain/Tools~/tfshaderc/`.
Структура: `crates/tfshaderc-core` (ядро: разбор, варианты, граф инклюдов, кеш, бэкенды)
и `crates/tfshaderc-cli` (бинарник). Внешних крейтов нет — только std.

## Запуск

```powershell
cd "C:\Users\mshat\UnityProjects\First Proj\Packages\com.terrafor.terrain\Tools~\tfshaderc"
cargo build --release
.\target\release\tfshaderc.exe --project "C:/Users/mshat/UnityProjects/First Proj" --shader "<путь к .shader>" --no-cache
```

Из корня проекта — `check.cmd` (двойной клик; собирает и проверяет всё в `Assets`).

## Самопроверка инструмента

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File verify_sprint1.ps1
```
Семь проверок (сборка, тесты, 216/216, тёплый кеш, `.compute`, инкремент по инклюду, отсутствие
внешних зависимостей). Код возврата 0 = приёмка пройдена. ВАЖНО: `.ps1` должен быть сохранён
в UTF-8 **с BOM**, иначе PowerShell 5.1 читает кириллицу как ANSI и парсер падает.

## Что важно знать

- Бэкенд HLSL по умолчанию `dxc`, профиль `d3d11`. `--backend fxc` = `dxc -P` + `fxc` (путь DX11 как у Unity), в ~9 раз медленнее.
- Пассы `CGPROGRAM` идут своим препроцессором (`preprocessor.rs`) и `fxc /Gec`. Без `/Gec` fxc падает с X3650 на `half` у глобальных переменных — это 74 ошибки во всём проекте.
- Пропускаются по умолчанию: `EDITOR_VISUALIZATION`, `INSTANCING_ON`. Включаются: `--editor-variants`, `--instancing`. Флаг `--legacy` больше не нужен (принимается для совместимости).
- `.compute` компилируется: каждое `#pragma kernel` — отдельная цель, профиль `cs_5_0`.
- Кеш: `.tfshaderc/cache` (только успехи), ключ = граф всех инклюдов + defines + профиль.
  Правки заголовков пакетов: `.tfshaderc/patch` (удалить = сбросить).
- Код возврата: 0 — всё собралось, 1 — ошибки, 2 — ошибка запуска.

## Замеры (10 ядер, шейдер террейна, 216 вариантов)

| Прогон | Время |
|---|---|
| холодный | 13.3 с |
| тёплый кеш | 3.0 с |
| инкремент после правки инклюда (мини-проект, 4 варианта) | 0.1 с |

## Устройство (при отладке)

| Файл | Назначение |
|---|---|
| `crates/tfshaderc-core/src/shaderlab.rs` | парсер ShaderLab: пассы, прагмы, встроенные наборы |
| `.../builtins.rs` | наборы ключевых слов Unity (из строк `UnityShaderCompiler.exe`) |
| `.../variants.rs` | декартово произведение групп, фильтры, прелюдия |
| `.../include_graph.rs` | транзитивные `#include` и хеш графа |
| `.../compute.rs` | `.compute` и `#pragma kernel` |
| `.../patch.rs` | эмуляция склейки `##` как в Unity + индекс макросов |
| `.../dxc.rs` | вызовы `dxc`/`fxc`, уникальные временные файлы |
| `.../cache.rs` | кеш по ключу варианта |
| `crates/tfshaderc-cli/src/main.rs` | флаги, отчёт, авторемонт, параллельный пул |

## Проверенный факт про URP

URP использует нестандартную склейку: `GBUFFER_IDX_AFTER(x) … ## x` и `DECL_SV_TARGET(idx) SV_Target##idx`.
Unity раскрывает аргумент до `##`, `dxc` и `fxc` — нет, поэтому оба падают одинаково.
Лечится уровнем косвенности в копии заголовка (этим занимается `patch.rs`).
