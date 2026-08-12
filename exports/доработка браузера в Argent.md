# 🗓 Argent Session: 2026-08-12 18:57

### 👤 Пользователь
Этот новый метод полностью заменяет прошлое DOM дерево или как?

### 🤖 Argent
Хороший вопрос. Давай посмотрю, как именно встроен новый метод — заменяет он DOM или работает параллельно.

### 🤖 Argent
Давай проверю, что было раньше — метод `get_state` / `browser_state`, который строит DOM-дерево.

### 🤖 Argent
**Нет, не заменяет.** Accessibility Tree — это **дополнительный** метод, который работает параллельно с существующим `browser_state`. Вот как они соотносятся:

| | `browser_state` (старый) | `browser_accessibility_tree` (новый) |
|---|---|---|
| **Источник данных** | JS-инъекция в DOM (`_STATE_EXTRACTION_JS`) — обходит DOM, ищет интерактивные элементы | CDP `Accessibility.getFullAXTree` — семантическое дерево, которое браузер строит для скринридеров |
| **Что возвращает** | Плоский список интерактивных элементов: `[1] input[type=email]`, `[2] button "Sign In"` | Дерево с ролями, именами, состояниями, иерархией: `heading "Sign In"`, `textbox "Email"`, `button "Submit"` |
| **Индексы** | `data-argent-idx` через JS | `data-argent-idx` через CDP — те же индексы, `browser_click(index)` работает без изменений |
| **Структурные узлы** | Нет — только кликабельные элементы | Да — заголовки, лендмарки (`main`, `nav`), регионы видны для контекста, даже если не кликабельные |

**Связь между ними:** `get_accessibility_tree` **падает обратно на `get_state`** если CDP недоступен или возвращает пустое дерево (строка 1294 в `browser_engine.py`):

```python
return await self.get_state(session, query, scroll_depth)
```

Так что:
- `browser_state` остался в списке инструментов — ничего не удалено
- `browser_accessibility_tree` добавлен **рядом** как альтернатива с более богатой семантикой
- Если CDP ломается — новый метод автоматически откатывается к старому

Оба метода используют один и тот же механизм индексов (`data-argent-idx` → `browser_click`), так что они полностью совместимы.

### 👤 Пользователь
Нужено ли нам два инструмента для этого?

### 🤖 Argent
Хороший архитектурный вопрос. Давай посмотрю, кто реально вызывает каждый из методов — может, один из них уже мёртвый код.

### 🤖 Argent
Картина ясная. Вот кто что использует:

**`browser_state`** — активно используется по всему коду:
- `browser_click` и `browser_input` ссылаются на него в return-сообщениях («Call browser_state to get fresh indices»)
- WebSocket сервер (`src/server/app.py`) обрабатывает `{"type": "get_state"}`
- 10+ тестов в `test_browser_enhancements.py`
- Описания схем для click/input/scroll ссылаются на индексы из `browser_state`

**`browser_accessibility_tree`** — только зарегистрирован как инструмент + 1 тестовый файл. Никто другой его не вызывает.

Давай посмотрю форматы вывода, чтобы понять, можно ли их слить.

### 🤖 Argent
Теперь у меня полная картина. Вот мой анализ:

## Сравнение форматов

**`browser_state`** — плоский, компактный:
```
Page: https://example.com | Title: Example
---
[1] input "Email"
[2] input "Password"
[3] button "Sign In"
```

**`browser_accessibility_tree`** — дерево, с ролями и структурой:
```
Page: https://example.com | Title: Example
---
  main
    heading "Welcome"
[1] textbox "Email"
[2] textbox "Password"
[3] button "Sign In"
```

Оба дают одинаковый заголовок (Page/Title/Tabs), оба тегают элементы `data-argent-idx`, оба заполняют `_element_frames` — `browser_click` работает одинаково с обоими.

## Мой вердикт: **нет, два инструмента не нужны**

Вот почему:

1. **LLM должен выбирать** между двумя инструментами, которые делают одно и то же — получить представление страницы с кликабельными индексами. Это создаёт путаницу, а не гибкость.

2. **a11y tree уже падает на get_state** — то есть это надмножество по устойчивости. Отдельный инструмент `browser_state` просто дублирует fallback-путь.

3. **`browser_state` глубоко интегрирован** — click/input/scroll ссылаются на него в return-сообщениях, WebSocket-сервер использует `{"type": "get_state"}`, 10+ тестов. Удалить его сложнее, чем a11y tree.

4. **a11y tree заметно многословнее** — структурные узлы (heading, main, nav) + отступы = больше токенов на каждый вызов. Для цикла «click → check → click» это лишний расход.

## Рекомендация: **слить в один инструмент**

Есть два пути:

**Вариант A — минимальный**: удалить `browser_accessibility_tree` как отдельный инструмент. Оставить `get_accessibility_tree` как внутренний метод, который `get_state` вызывает первым (CDP → a11y tree), с fallback на JS-экстракцию. Один инструмент `browser_state`, под капотом — лучшее из двух.

**Вариант B — параметр**: оставить `browser_state`, добавить параметр `detail: "compact" | "full"`. `compact` = старый JS-метод, `full` = a11y tree. Один инструмент, два режима.

Вариант A проще и я бы его рекомендовал — a11y tree даёт более качественную семантику (роли вместо HTML-тегов, иерархия), а fallback на JS гарантирует работу везде. LLM не нужно решать, какой инструмент вызывать.

Хочешь, чтобы я реализовал один из вариантов?

### 👤 Пользователь
Можем сделать так, что новый метод который ты реализовал будет работать по умолчанию в инструменте. Если что-то пойдёт не так, пользователь сам с помощью команды в чате Argent сможет поменять режим работы и вернуть DOM дерево которое было раньше (Или можем дать выбор самому ИИ, но сделать это нужно грамотно, чтобы в приоритете был новый интрумент, а DOM была как альтернатива и продумать это переключение для более слабых моделей).

### 🤖 Argent
Отличная идея. Давай я спланирую реализацию — есть несколько важных архитектурных решений.

### 🤖 Argent
Что именно хочешь изменить в плане? Например:

### 🤖 Argent
Отличные замечания — все три проблемы реальные. Пересматриваю план.

### 🤖 Argent
Теперь у меня полная картина всех трёх проблем. Переписываю план с учётом критики.

### 🤖 Argent
Понял — импорт внутри того же файла убираю. Приступаю к реализации.

### 🤖 Argent
Теперь у меня всё нужное в контексте. Начинаю реализацию — буду делать параллельно где возможно.

**Шаг 1: config.py, browser_engine.py, tools/** — независимые изменения делаю одновременно.

### 🤖 Argent
Теперь исправляю 3 fallback-вызова в `get_accessibility_tree` — они должны вызывать `_get_dom_state` вместо `get_state`.

### 🤖 Argent
Теперь обновляю tools/ — удаляю `browser_accessibility_tree`, добавляю `mode` в `browser_state`.

### 🤖 Argent
Несколько правок не прошли. Читаю текущее состояние и исправляю.

### 🤖 Argent
Теперь обновляю schema `browser_state` — добавляю `mode` параметр и обновляю описание. Также обновляю `hidden_tools` и help text.

### 🤖 Argent
Теперь обновляю `/browser` команду в `main.py` и help text в `command_handler.py`.

### 🤖 Argent
Теперь обновляю help text в `command_handler.py` и проверяю `main.py` autocomplete список.

### 🤖 Argent
Теперь обновляю autocomplete в `main.py` и проверяю `doctor.py` на наличие упоминаний `browser_accessibility_tree`.

### 🤖 Argent
Теперь обновляю тесты. Сначала посмотрю текущее состояние.

### 🤖 Argent
Теперь обновляю `test_browser_enhancements.py` — все вызовы `get_state` нужно явно передать `mode="dom"`, поскольку по умолчанию теперь a11y.

### 🤖 Argent
Все исходные файлы обновлены. Проверяю, что не осталось ссылок на `browser_accessibility_tree`, и запускаю тесты.

### 🤖 Argent
Нет ссылок на `browser_accessibility_tree`. Конфиг работает. Теперь проверяю импорты и запускаю тесты.

### 🤖 Argent
`browser_accessibility_tree` удалён из схем, `browser_state` имеет параметр `mode`. Запускаю тесты.

### 🤖 Argent
Все 18 тестов прошли, включая integration-тесты a11y tree через `get_state(mode="a11y")` и fallback-тест. Проверяю, что не сломал ничего в остальных тестах.

### 🤖 Argent
Готово. Все 11 неудачных тестов падали **до моих изменений** — я проверил через `git stash`. Это pre-existing failures (Python 3.14 compatibility issues, disk full simulation, etc.), не связанные с browser_state unification.

## Что сделано

### Изменённые файлы (8):

1. **`config.py`** — 3 новые функции: `get_browser_state_mode()`, `set_browser_state_mode()`, `get_effective_browser_state_mode()` (разрешает `"auto"` по tier модели)

2. **`browser_engine.py`** — `get_state()` стал диспетчером (без скролла/сессии), JS-экстракция вынесена в `_get_dom_state()`, 3 fallback-вызова в `get_accessibility_tree` идут в `_get_dom_state()` вместо `get_state()` — **рекурсии нет**

3. **`tools/browser_tools.py`** — `browser_accessibility_tree` удалён, `browser_state` получил параметр `mode`

4. **`tools/schemas.py`** — schema `browser_accessibility_tree` удалена, `browser_state` обновлён с `mode` параметром и описанием a11y по умолчанию

5. **`tools/__init__.py`** — экспорт `browser_accessibility_tree` удалён

6. **`main.py`** — `/browser` команда расширена: `a11y`, `dom`, `auto` + статус state extraction в выводе

7. **`command_handler.py`** — help text обновлён

8. **Тесты** — `test_a11y_tree.py` тестирует через `get_state(mode="a11y")`, `test_browser_enhancements.py` использует `mode="dom"` для DOM-тестов

### Цепочка приоритетов:
```
AI mode param  >  user config (/browser a11y|dom)  >  tier default (auto)  >  hard fallback (CDP fail → _get_dom_state)
```

### Результаты тестов:
- **18/18 passed** — `test_a11y_tree.py` + `test_browser_enhancements.py`
- **2959 passed** — полный набор (11 pre-existing failures не связаны с изменениями)

