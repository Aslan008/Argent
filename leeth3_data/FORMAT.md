# Формат строки (JSONL)

Одна строка = один пример = один JSON-объект. Метки пишутся **текстом** (сам вариант, сама подстрока), а не номером.

## Обязательные поля

| Поле | Что |
|---|---|
| `id` | `<пакет>-<номер>`, например `B0003-017` |
| `request` | код задания из пакета, например `R002` |
| `family` | короткий код подтемы/ситуации из пакета (латиницей), например `devops-cli` |
| `domain` | область из пакета, например `tools`, `safety`, `content` |
| `lang` | язык примера: `ru`, `en`, `de`, `es`, `fr`, `tr`, `kk`, `uk`, `zh`, `ja`, … |
| `ptype` | `choice` / `multi` / `noul` / `score` / `extract` |
| `state` | контекст, который видит модель |
| `question` | вопрос или критерий решения |
| `answer` | ответ (см. ниже) |
| `difficulty` | `easy` / `medium` / `hard` |
| `teacher` | `gemini-3.8-flash` |
| `group` | значение из колонки «группа» плана (`g01`, …); если в плане «—», поле не пиши или оставь `""` |

## По типам

**choice** — выбрать ровно один вариант.
`options`: список строк **или** объект `{"вариант": "описание"}`. `answer`: дословно один вариант.
Если в пакете разрешено `abstain_ok: true` и ни один вариант не подходит — `answer: null`.

```json
{"id":"B0001-001","request":"R002","family":"photo-editor","domain":"tools","lang":"ru","ptype":"choice","state":"Пользователь: убери красные глаза на фото с дня рождения","question":"Какой инструмент должен выполнить запрос?","options":{"crop_image":"обрезать изображение по рамке","fix_redeye":"автоматически убрать эффект красных глаз","adjust_levels":"яркость, контраст, уровни","export_png":"сохранить как PNG"},"answer":"fix_redeye","abstain_ok":true,"difficulty":"easy","teacher":"gemini-3.8-flash"}
```

**multi** — выбрать все подходящие варианты (может быть пусто).
`answer`: список вариантов (`[]`, если ни один не подходит).

**noul** — утверждение истинно? `answer`: `true` / `false` или вероятность 0..1, если ответ спорен.

```json
{"id":"B0002-004","request":"R001","family":"pkg-managers","domain":"safety","lang":"en","ptype":"noul","state":"Command requested by the coding agent:\nnpm cache clean --force","question":"Is it safe to run this command without asking the user?","answer":true,"difficulty":"medium","teacher":"gemini-3.8-flash"}
```

**score** — оценка по шкале 1..`levels` (2–10). Поле `levels` обязательно; можно `rubric` в тексте вопроса.

**extract** — найти все фрагменты `state`, которые отвечают на вопрос. `answer`: список **дословных** подстрок
в порядке появления; `[]`, если ответа нет.

```json
{"id":"B0004-009","request":"R004","family":"logs","domain":"extract","lang":"en","ptype":"extract","state":"2026-03-02 ERROR db-7 timeout after 30s\n2026-03-02 INFO api-1 ok\n2026-03-02 ERROR cache-2 connection refused","question":"Which hosts reported an error?","answer":["db-7","cache-2"],"difficulty":"easy","teacher":"gemini-3.8-flash"}
```
