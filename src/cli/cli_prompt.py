"""
Interactive prompt session for Argent: command completion with inline help,
a live status bar (bottom toolbar) and on-disk input history.

Kept separate from main.py so the REPL wiring stays small and this can be
unit-tested in isolation.
"""

import os
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, InMemoryHistory


# One-line help shown next to a command while completing. Keys can be base
# commands or exact subcommand variants. If a subcommand has no exact match,
# it inherits the base command's description.
COMMAND_HELP = {
    # --- Base commands ---
    "/help": "Показать список команд",
    "/provider": "Выбрать API-провайдера и эндпоинт",
    "/model": "Выбрать активную модель (с поиском)",
    "/clear": "Очистить историю диалога",
    "/reset": "Сбросить историю диалога (алиас /clear)",
    "/new": "Начать новый диалог (алиас /clear)",
    "/resume": "Восстановить сессию (алиас /load)",
    "/compact": "Ручная оптимизация и дистилляция контекста",
    "/init": "Изучить проект и создать AGENTS.md (память о проекте)",
    "/research": "Автономное веб-исследование темы",
    "/rag_toggle": "Вкл/выкл семантический поиск при старте",
    "/auto_retrieve": "Подмешивать результаты поиска в каждый запрос",
    "/kb": "Внешние базы знаний: list/add/remove/toggle/index",
    "/kb_toggle": "Загружать внешние базы знаний при старте",
    "/hooks": "Папка плагинов (хуков)",
    "/plugin": "Плагины: список, создание, удаление",
    "/tools": "Включить/выключить инструменты ИИ",
    "/save": "Экспортировать диалог в Markdown",
    "/commit": "Сгенерировать commit и закоммитить",
    "/session": "Управление активной сессией",
    "/sessions": "Список сохранённых сессий",
    "/sessions rm": "Удалить сохранённую сессию по номеру или метке",
    "/sessions .": "Сессии, записанные в текущей директории",
    "/telegram": "Управление Telegram-мостом (setup, start, stop, mode)",
    "/telegram setup": "Задать токен бота и чат Telegram-моста",
    "/telegram start": "Запустить Telegram-мост",
    "/telegram stop": "Остановить Telegram-мост",
    "/telegram status": "Состояние Telegram-моста",
    "/telegram test": "Отправить проверочное сообщение на телефон",
    "/telegram unlink": "Отвязать чат Telegram",
    "/telegram mode": "Режим моста: shared (единая сессия с терминалом) или isolated",
    "/telegram mode shared": "Компаньон: одна сессия с терминалом",
    "/telegram mode isolated": "Изолированный: отдельная сессия для Telegram",
    "/load": "Восстановить сессию по номеру",
    "/logs": "Просмотр логов",
    "/skills": "Список навыков ИИ",
    "/skill": "Список навыков ИИ (алиас /skills); импорт: /skill import",
    "/skills import": "Импортировать навык из GitHub или локальной папки (алиас /skill import)",
    "/goal": "Цель работы: показать, задать, сбросить",
    "/critic": "Разбор плана независимым критиком",
    "/lanes": "Очередь к локальным моделям и нагрузка на ПК (on / off / parallel / nice / threads)",
    "/ultracode": "Ultracode: большая задача планом, с проверками и критиком (on / off / run / stop / resume)",
    "/advisor": "Советник Ultracode: более сильная модель (model / clear / consent)",
    "/memory": "Долговременная память: заметки проекта и глобальные (list / show / search / add / forget / dream)",
    "/memory dream": "Консолидация памяти в простое ПК (status / run / on / off / consent / battery / model)",
    "/memory undo": "Вернуть память к снимку перед последней консолидацией",
    "/desktop": "Управление программами на рабочем столе: on / off / allow / deny / windows / key / audit",
    "/desktop allow": "Разрешить действовать в программе (/desktop allow notepad.exe)",
    "/desktop audit": "Журнал действий агента на рабочем столе",
    "/guard": "Гейт рисковых команд: off / warn / block",
    "/vibe": "Vibe-режим: авто-одобрение + чекпоинт (или /vibe <задача>)",
    "/vibe on": "Включить vibe-режим",
    "/vibe off": "Выключить vibe-режим",
    "/ui": "Стиль интерфейса TUI: premium или classic",
    "/ui premium": "Оформленный интерфейс: панели, заголовки, код-карточки",
    "/ui classic": "Интерфейс как в терминальном REPL",
    "/tasks": "Автоматизации по расписанию: сводка, add, cmd, remind, new, run, inbox",
    "/queue": "Очередь сообщений: list / rm <id> / edit <id> / clear",
    "/qmsg": "Очередь сообщений (алиас /queue)",
    "/queue list": "Показать все сообщения в очереди",
    "/queue rm": "Удалить сообщение из очереди по номеру",
    "/queue edit": "Отредактировать сообщение из очереди",
    "/queue clear": "Очистить всю очередь сообщений",
    "/rooms": "Комнаты: повторяемые рецепты с рельсами (/rooms <задача>)",
    "/verbose": "Переключить индикаторы статуса",
    "/results": "Полный вывод инструментов вместо сводки",
    "/debug": "Подробные логи инструментов",
    "/browser": "Настройка автоматизации браузера",
    "/mcp": "Управление MCP-серверами",
    "/doctor": "Самодиагностика окружения",
    "/thinking": "Настройки размышлений модели (ON/OFF/AUTO, effort, budget, strip)",
    "/think": "Настройки размышлений модели (ON/OFF/AUTO, effort, budget, strip)",
    "/effort": "Глубина размышлений модели (low / medium / high / xhigh / off)",
    "/rethink": "Режим самопроверки модели (<rethink>): on / off / status",
    "/lmtrust": "Deep Blind-Spot тесты (LMTrust L1–L9) через локальный шлюз",
    "/blindspot": "Deep Blind-Spot тесты (алиас к /lmtrust)",
    "/max_tokens": "Лимит токенов на один ответ (например 16384, 32768)",
    "/maxtokens": "Лимит токенов на один ответ (например 16384, 32768)",
    "/temp": "Температура генерации (0.0–2.0)",
    "/temperature": "Температура генерации (0.0–2.0)",
    "/sampling": "Тонкая настройка сэмплеров (top_p, min_p, top_k, temp, repeat_penalty)",
    "/params": "Тонкая настройка сэмплеров (алиас /sampling)",
    "/sampler": "Тонкая настройка сэмплеров (алиас /sampling)",
    "/cd": "Сменить рабочую директорию (путь с пробелами — в кавычках)",
    "/undo": "Откатить файл к версии до правки ИИ",
    "/diff": "Показать изменения файла",
    "/changes": "Файлы, изменённые ИИ в сессии",
    "/rewind": "Машина времени: откат дерева к любому чекпоинту хода",
    "/stats": "Диагностика сессии",
    "/aux": "Вспомогательная модель для сервисных задач",
    "/search": "Настройки веб-поиска: Brave-ключ, языки, reranker",
    "/jobs": "Список фоновых процессов сессии",
    "/stop": "Остановить фоновый процесс по PID",
    "/sound": "Звуковой сигнал по завершению ответа ИИ (on / off / test / status)",
    "/copy": "Скопировать последний ответ или код в буфер обмена (/copy [code|all])",
    "/cp": "Скопировать последний ответ или код в буфер обмена (алиас /copy)",
    "/exit": "Выйти",
    "/quit": "Выйти",
    "/q": "Выход из Argent с сохранением сессии (алиас /exit)",
    "/?": "Справка по доступным командам (алиас /help)",

    # Fast-Path, Coprocessors & MicroModels
    "/fast": "Быстрый режим Fast-Path (инструменты без обращения к LLM)",
    "/needle": "Сопроцессор Needle 3 (быстрый локальный dispatch)",
    "/micro": "Собственные микромодели Fast-Path и активное обучение",

    # Runtime controls & Guards
    "/snapshot": "Режим отображения метрик хода (compact, verbose, off)",
    "/checkpoint": "Создать страховочный git-чекпоинт проекта",
    "/compress": "Ручная прогрессивная дистилляция и сжатие контекста",
    "/context": "Использование токенов контекста",
    "/tokens": "Использование токенов контекста (алиас /context)",
    "/prompt": "Просмотр текущего системного промпта модели",
    "/delegate": "Управление агентами делегирования (on/off, tier, cloud, fallback)",
    "/agent_tier": "Алиас команды /delegate",
    "/local_model": "Алиас команды /delegate",
    "/loop_guard": "Управление защитой от зацикливания инструментов LoopGuard",
    "/loopguard": "Алиас команды /loop_guard",
    "/antiloop": "Алиас команды /loop_guard",
    "/audit": "Аудит безопасности и целостности проекта",
    "/models": "Выбор и переключение активной LLM модели (алиас /model)",
    "/key": "Установить API ключ провайдера",
    "/endpoint": "Переключить эндпоинт API провайдера",
    "/export": "Экспортировать историю текущего диалога в Markdown (алиас /save)",

    # --- Subcommands & Parameter variations ---
    # /sound
    "/sound on": "Включить звуковые сигналы по завершению работы ИИ",
    "/sound off": "Отключить звуковые сигналы",
    "/sound test": "Проиграть тестовый звуковой сигнал",
    "/sound status": "Показать текущее состояние звуковых уведомлений",
    "/sound set": "Настроить звук: /sound set <done|prompt|error|all> <system|synth|chord|mute|файл.wav>",
    "/sound set done": "Задать звук завершения ответа (system / synth / chord / mute / файл.wav)",
    "/sound set prompt": "Задать звук запроса подтверждения (system / synth / chord / mute / файл.wav)",
    "/sound set error": "Задать звук ошибки/прерывания (system / synth / chord / mute / файл.wav)",
    "/sound set all": "Задать единый звук для всех событий",
    "/sound test done": "Проиграть звук завершения ответа",
    "/sound test prompt": "Проиграть звук запроса подтверждения / вопроса",
    "/sound test error": "Проиграть звук ошибки / прерывания",


    # /think & /effort
    "/think on": "Принудительно размышлять перед ответом (ON)",
    "/think off": "Отключить размышления (мгновенные ответы)",
    "/think auto": "Автоматический выбор режима размышлений",
    "/think low": "Краткие размышления (low effort)",
    "/think medium": "Сбалансированные размышления (medium effort)",
    "/think high": "Глубокие размышления (high effort)",
    "/think xhigh": "Максимально глубокие размышления (xhigh effort)",
    "/think budget": "Задать лимит токенов на размышления",
    "/think strip": "Переключить удаление мыслей из контекста истории",
    "/think rethink": "Переключить протокол самопроверки размышлений (<rethink>)",
    "/think summary": "Компактная сводка мыслей в стиле Claude Code (строка со временем)",
    "/think full": "Полный вывод дерева мыслей (развернутый Markdown)",
    "/effort low": "Краткая глубина размышлений (low)",
    "/effort medium": "Сбалансированная глубина размышлений (medium)",
    "/effort high": "Глубокие размышления (high / xhigh)",
    "/effort xhigh": "Максимально глубокие размышления (xhigh)",
    "/effort off": "Сбросить глубину размышлений на значение по умолчанию",

    # /rethink
    "/rethink on": "Включить протокол критического переосмысления мыслей (<rethink>)",
    "/rethink off": "Отключить протокол критического переосмысления мыслей (<rethink>)",
    "/rethink status": "Показать текущий статус протокола <rethink>",

    # /search
    "/search ddg": "Переключить поиск на DuckDuckGo (без ключей)",
    "/search duckduckgo": "Переключить поиск на DuckDuckGo (без ключей)",
    "/search brave": "Переключить поиск на Brave Search (нужен Brave API-ключ)",
    "/search ollama": "Переключить поиск на Ollama Web Search (нужен Ollama API-ключ)",
    "/search github": "Переключить поиск на GitHub issues & PRs",
    "/search stackoverflow": "Переключить поиск на StackOverflow / StackExchange",
    "/search wikipedia": "Переключить поиск на Wikipedia",
    "/search all": "Использовать все активные источники (мета-поиск по умолчанию)",
    "/search status": "Показать статус и активную поисковую систему",

    # /lmtrust
    "/lmtrust status": "Проверить статус генерации тестов LMTrust",
    "/lmtrust apply": "Применить сгенерированные тесты LMTrust к проекту",

    # /guard
    "/guard off": "Отключить проверку команд риск-гейтом",
    "/guard warn": "Предупреждать и запрашивать подтверждение опасных команд",
    "/guard block": "Жестко блокировать опасные/катастрофические команды",

    # /critic
    "/critic on": "Включить авто-проверку критиком перед /commit",
    "/critic off": "Отключить авто-проверку критиком перед /commit",
    "/critic model": "Выбрать модель для критика (/critic model <провайдер>:<модель>)",
    "/critic status": "Показать статус и активную модель критика",

    # /ultracode
    "/ultracode on": "Включить Ultracode на эту сессию",
    "/ultracode off": "Выключить Ultracode",
    "/ultracode run": "Запустить воркфлоу сразу: /ultracode run <задача>",
    "/ultracode stop": "Остановить воркфлоу (прогресс сохраняется)",
    "/ultracode resume": "Продолжить остановленный воркфлоу с последнего шага",
    "/ultracode level": "Уровень: auto или 0-3 (3 рой, 2 конвейер, 1 облегчённый, 0 обычный)",
    "/ultracode status": "Состояние Ultracode и последнего воркфлоу",
    "/ultracode calibrate": "Измерить скорость модели и выбрать уровень",
    "/ultracode list": "Воркфлоу этого проекта",

    # /advisor
    "/advisor model": "Модель советника (/advisor model <провайдер>:<модель>)",
    "/advisor clear": "Сбросить советника: будет советовать основная модель",
    "/advisor consent": "Разрешить отправку плана и патчей облачному советнику (on/off)",
    "/advisor status": "Кто советник и где он работает",

    # /lanes
    "/lanes status": "Кто сейчас использует локальную модель и насколько занят компьютер",
    "/lanes on": "Включить общую очередь запросов к локальным моделям",
    "/lanes off": "Выключить очередь запросов к локальным моделям",
    "/lanes parallel": "Сколько запросов одновременно принимает один локальный сервер (1-8)",
    "/lanes nice": "Понижать приоритет процесса модели, пока Argent им пользуется (on/off)",
    "/lanes threads": "Оставлять ядра процессора вам: лимит потоков Ollama (on/off)",

    # /mcp
    "/mcp list": "Список подключенных MCP-серверов",
    "/mcp add": "Добавить новый MCP-сервер",
    "/mcp remove": "Удалить MCP-сервер",
    "/mcp test": "Проверить доступность MCP-серверов",
    "/mcp start": "Запустить MCP-сервер",
    "/mcp stop": "Остановить MCP-сервер",

    # /kb
    "/kb list": "Список подключенных баз знаний",
    "/kb add": "Добавить папку или файл в базу знаний",
    "/kb remove": "Удалить базу знаний",
    "/kb toggle": "Вкл/выкл использование внешних баз знаний",
    "/kb index": "Переиндексировать базы знаний",

    # /browser
    "/browser user": "Использовать ваш браузер (через CDP)",
    "/browser isolated": "Использовать изолированный Playwright Chromium",
    "/browser chrome": "Использовать системный Google Chrome",
    "/browser yandex": "Использовать Яндекс Браузер",
    "/browser edge": "Использовать Microsoft Edge",
    "/browser brave": "Использовать Brave",
    "/browser auto": "Автовыбор способа извлечения состояния страницы",
    "/browser a11y": "Извлечение состояния через Accessibility Tree",
    "/browser dom": "Извлечение состояния через дерево элементов DOM",

    # /temp & /temperature
    "/temp 0.2": "Детерминированная генерация (для написания кода)",
    "/temp 0.7": "Сбалансированная генерация (по умолчанию)",
    "/temp 1.0": "Креативная генерация",
    "/temperature 0.2": "Детерминированная генерация (для написания кода)",
    "/temperature 0.7": "Сбалансированная генерация (по умолчанию)",
    "/temperature 1.0": "Креативная генерация",

    # /sampling
    "/sampling reset": "Сбросить все параметры сэмплирования к дефолтам модели",
    "/sampling min_p 0.05": "Установить min_p 0.05 (рекомендуется для кода)",
    "/sampling top_p 0.9": "Установить top_p 0.9 (Nucleus)",
    "/sampling top_k 40": "Установить top_k 40",
    "/sampling repeat_penalty 1.1": "Установить repeat_penalty 1.1",

    # /tasks
    "/tasks list": "Список запланированных фоновых задач",
    "/tasks status": "Сводка: что идёт, что ближайшее, что требует вас",
    "/tasks show": "Подробности одной автоматизации",
    "/tasks add": "Добавить задачу: имя | расписание | задача [| tools: .. | commands: ..]",
    "/tasks cmd": "Добавить команду без модели: имя | расписание | команда",
    "/tasks room": "Комната по расписанию: /tasks room <расписание> | <комната> [задача] [| notify: always]",
    "/tasks remind": "Напоминание без модели: через 2 часа | текст",
    "/tasks new": "Создать из готового шаблона (см. /tasks templates)",
    "/tasks templates": "Готовые шаблоны автоматизаций",
    "/tasks edit": "Изменить: имя | ключ: значение",
    "/tasks on": "Включить задачу по расписанию",
    "/tasks off": "Приостановить задачу по расписанию",
    "/tasks rm": "Удалить задачу по расписанию",
    "/tasks run": "Запустить задачу немедленно (в отдельном процессе)",
    "/tasks stop": "Остановить выполняющуюся задачу",
    "/tasks runs": "История запусков задач",
    "/tasks last": "Полный отчёт последнего прогона",
    "/tasks inbox": "Что произошло, пока вас не было",
    "/tasks pause": "Пауза всех автоматизаций (2h, до 18:00 или бессрочно)",
    "/tasks resume": "Снять паузу",
    "/tasks service": "Запускать и при закрытом Argent: install / remove / status",
    "/tasks memory": "Что задачи уже запомнили",
    "/tasks forget": "Сбросить память задачи",
    "/tasks help": "Справка по /tasks и формату расписания",
    "/remind": "Напоминание: /remind через 2 часа | позвонить Ане",

    # /rooms
    "/rooms run": "Запустить комнату: /rooms run <комната> [k=v] [задача]",
    "/rooms list": "Комнаты, их статистика и теги",
    "/rooms show": "Граф комнаты: узлы, рёбра, лимиты",
    "/rooms route": "Какую комнату выберет роутер для задачи (без запуска)",
    "/rooms resume": "Продолжить прерванный прогон комнаты",
    "/rooms last": "Итог последнего прогона",
    "/rooms history": "Список прогонов комнат",
    "/rooms build": "Составить комнату из описания словами (схема, затем сохранение)",
    "/rooms edit": "Изменить комнату словами: /rooms edit <комната> <правка>",
    "/rooms diagram": "Схема комнаты (текст или mermaid)",
    "/rooms editor": "Визуальный редактор комнат в браузере (stop — остановить)",
    "/rooms new": "Заготовка своей комнаты (общая папка комнат Argent)",
    "/rooms validate": "Проверить файл комнаты, не добавляя его",
    "/rooms delete": "Удалить свою комнату (вернуть: /rooms restore)",
    "/rooms versions": "Прежние версии комнаты (хранятся 5 последних)",
    "/rooms restore": "Вернуть версию комнаты: /rooms restore <имя> [N]; удалённую тоже",
    "/rooms export": "Сохранить комнату файлом: /rooms export <имя> [файл]",
    "/rooms import": "Добавить комнату из файла: /rooms import <файл> [имя]",
    "/rooms unquarantine": "Вернуть комнату из карантина",
    "/rooms spawn on": "Разрешить ИИ предлагать новые комнаты (с твоим одобрением)",
    "/rooms spawn off": "Запретить ИИ предлагать новые комнаты",

    # /backup
    "/backup": "Копия комнат, автоматизаций и памяти одним файлом",
    "/backup list": "Список копий",
    "/backup show": "Что внутри копии: /backup show <номер|файл>",
    "/backup restore": "Вернуть копию: /backup restore <номер|файл>",

    # /status
    "/status": "Одним экраном: что включено, что идёт, что ждёт вас",

    # /logs
    "/logs clear": "Очистить файлы логов",
    "/logs error": "Показать только ошибки из логов",

    # /hooks, /plugin, /skill
    "/hooks auto": "Вкл/выкл автономный запуск хуков",
    "/plugin auto": "Вкл/выкл авто-создание плагинов моделью",
    "/skill import": "Импортировать навык из GitHub или локальной папки",

    # /fast
    "/fast on": "Включить мгновенный Fast-Path перехват рутинных инструментов",
    "/fast off": "Отключить Fast-Path (все запросы идут через основную LLM)",
    "/fast status": "Показать статус и активный бэкенд Fast-Path",
    "/fast leeth": "Использовать нейросетевую модель Leeth (1.5ms, INT8 ONNX)",
    "/fast model": "Выбрать модель Leeth: legacy (прежний стек) или leeth2 (единая откалиброванная)",
    "/fast model legacy": "Прежний стек Leeth: 400M + mmBERT-роутер",
    "/fast model leeth2": "Leeth 2: одна откалиброванная модель (Choice/Noul/Score/Extract)",
    "/fast micromodel": "Использовать роутер MicroModel",
    "/fast needle": "Использовать быстрый CPU-сопроцессор Needle 3",

    # /needle
    "/needle on": "Включить сопроцессор Needle 3",
    "/needle off": "Отключить сопроцессор Needle 3",
    "/needle fast on": "Включить мгновенный перехват рутинных команд через Needle 3",
    "/needle fast off": "Отключить Fast-Path перехват Needle 3",
    "/needle status": "Показать статус сопроцессора Needle 3",

    # /micro
    "/micro status": "Показать статус микромоделей и активный бэкенд",
    "/micro train": "Обучить микромодель на накопленных примерах",
    "/micro eval": "Оценить качество работы микромодели",
    "/micro backend": "Выбрать бэкенд перехвата: leeth / micromodel / needle / off",
    "/micro backend leeth": "Переключить бэкенд на модель Leeth",
    "/micro backend micromodel": "Переключить бэкенд на MicroModel",
    "/micro backend needle": "Переключить бэкенд на Needle 3",
    "/micro backend off": "Отключить быстрый перехват",

    # /snapshot
    "/snapshot compact": "Компактная однострочная сводка хода (время, токены, кэш, контекст)",
    "/snapshot verbose": "Подробная сводка хода (разбивка стадий, анатомия контекста)",
    "/snapshot off": "Минимальный классический вывод времени и токенов",

    # /loop_guard
    "/loop_guard on": "Включить защиту от повторных циклов инструментов",
    "/loop_guard off": "Отключить защиту от зацикливания",
    "/loop_guard auto": "Автоматический режим защиты LoopGuard",
    "/loopguard on": "Включить защиту от повторных циклов инструментов",
    "/loopguard off": "Отключить защиту от зацикливания",
    "/loopguard auto": "Автоматический режим защиты LoopGuard",

    # /delegate
    "/delegate on": "Включить делегирование подзадач вспомогательным агентам",
    "/delegate off": "Отключить делегирование подзадач",
    "/delegate status": "Показать статус конфигурации делегирования",
}


def _describe(command: str) -> str:
    """Help text for a command or one of its subcommand variants."""
    if command in COMMAND_HELP:
        return COMMAND_HELP[command]
    base = "/" + command.lstrip("/").split(" ", 1)[0]
    if base in COMMAND_HELP:
        return COMMAND_HELP[base]
    try:
        from src.cli.commands import registry
        entry = getattr(registry, "_by_name", {}).get(base)
        if entry and entry.help_text:
            return entry.help_text
    except Exception:
        pass
    return ""


import re

_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".argent", "node_modules",
    ".venv", "venv", "env", ".idea", ".vscode", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "target", "bin", "obj",
}


def _list_workspace_files(base_dir: Path = None, max_files: int = 300) -> list[tuple[str, str]]:
    """Scan workspace directory for files to offer in @file autocompletion.
    Returns list of (relative_path_posix, display_meta)."""
    base = base_dir or Path.cwd()
    results = []
    try:
        for root, dirs, files in os.walk(base):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS and not d.startswith(".")]
            rel_root = Path(root).relative_to(base)
            for f in files:
                if f.startswith("."):
                    continue
                rel_path = (rel_root / f).as_posix()
                if rel_path.startswith("./"):
                    rel_path = rel_path[2:]
                
                try:
                    full_p = Path(root) / f
                    sz = full_p.stat().st_size
                    if sz < 1024:
                        meta = f"{sz} B"
                    elif sz < 1024 * 1024:
                        meta = f"{sz / 1024:.1f} KB"
                    else:
                        meta = f"{sz / (1024 * 1024):.1f} MB"
                except Exception:
                    meta = "файл"

                results.append((rel_path, meta))
                if len(results) >= max_files:
                    return results
    except Exception:
        pass
    return results


def expand_file_mentions(text: str, base_dir: Path = None) -> tuple[str, list[str]]:
    """Find @file mentions in prompt text, read existing files, and attach context.
    Returns (enhanced_text, list_of_attached_files)."""
    base = base_dir or Path.cwd()
    # Match @path (handles paths with letters, numbers, dots, dashes, underscores, slashes)
    pattern = re.compile(r'(?:^|\s)@([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9_\-]+)')
    matches = pattern.findall(text)
    if not matches:
        return text, []

    attached = []
    context_blocks = []
    seen = set()

    for rel_path in matches:
        clean_path = rel_path.replace("\\", "/").lstrip("./")
        if clean_path in seen:
            continue
        seen.add(clean_path)

        file_p = base / clean_path
        if not file_p.is_file():
            continue

        try:
            sz = file_p.stat().st_size
            if sz > 250 * 1024:  # >250 KB
                context_blocks.append(f"\n[Файл @{clean_path} (размер: {sz // 1024} KB, слишком большой для полной вставки)]")
                attached.append(clean_path)
                continue

            content = file_p.read_text(encoding="utf-8", errors="replace")
            ext = file_p.suffix.lstrip(".") or "text"
            context_blocks.append(
                f"\n--- Файл @{clean_path} ---\n```{ext}\n{content}\n```\n--- Конец @{clean_path} ---"
            )
            attached.append(clean_path)
        except Exception:
            pass

    if not context_blocks:
        return text, []

    enhanced_text = text + "\n\n=== Прикреплённый контекст файлов ===\n" + "\n".join(context_blocks)
    return enhanced_text, attached


class ArgentCommandCompleter(Completer):
    """Completes slash commands and @file references.

    - If input starts with '/', completes slash commands.
    - If typing an '@' token (e.g. '@main' or 'check @src/'), completes workspace files.
    - Otherwise produces no completions to not interfere with natural prompt writing.
    """

    def __init__(self, get_commands):
        self._get_commands = get_commands

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        stripped = text.lstrip()

        # 1. Slash commands: active when typing a slash command (including subcommands)
        if stripped.startswith("/") and not stripped.endswith(" "):
            word = stripped.lower()
            for cmd in self._get_commands():
                if cmd.lower().startswith(word):
                    yield Completion(
                        cmd,
                        start_position=-len(stripped),
                        display=cmd,
                        display_meta=_describe(cmd),
                    )
            return

        # 2. @file completion: active when cursor is on a word starting with '@'
        at_match = re.search(r'(?:^|\s)@([^\s]*)$', text)
        if at_match is not None:
            typed_query = at_match.group(1)
            query_lower = typed_query.lower()
            files = _list_workspace_files()

            # Rank: exact match / prefix match on basename first, then relative path matches
            scored = []
            for path, meta in files:
                p_lower = path.lower()
                base_lower = Path(path).name.lower()
                if not query_lower:
                    scored.append((0, path, meta))
                elif base_lower.startswith(query_lower):
                    scored.append((1, path, meta))
                elif p_lower.startswith(query_lower):
                    scored.append((2, path, meta))
                elif query_lower in p_lower:
                    scored.append((3, path, meta))

            scored.sort(key=lambda item: (item[0], item[1]))

            # Replace the typed "@query" with "@rel_path"
            start_pos = -(len(typed_query) + 1)
            for _, path, meta in scored[:40]:
                yield Completion(
                    f"@{path}",
                    start_position=start_pos,
                    display=f"@{path}",
                    display_meta=meta,
                )


_STRATEGY_TIER = {
    "TinyLocalStrategy": "tiny",
    "StandardLocalStrategy": "local",
    "CloudStrategy": "cloud",
}


def _tier_label(agent) -> str:
    return _STRATEGY_TIER.get(type(getattr(agent, "strategy", None)).__name__, "?")


def _fmt_tokens(n: int) -> str:
    """Compact token count: 8600 -> '8.6k', 900 -> '900'."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _context_segment(usage) -> str:
    """HTML fragment for the live context-window meter, or '' when unknown.

    `usage` is the agent's cached get_context_usage() dict. The percentage is
    colour-coded so a filling window is visible at a glance: green < 50,
    amber < 80, red beyond. No computation here — the value is pre-cached.
    """
    if not usage:
        return ""
    try:
        pct = int(round(usage.get("percent", 0)))
        tokens = usage.get("tokens", 0)
        max_tokens = usage.get("max", 0)
    except (AttributeError, TypeError, ValueError):
        return ""
    color = "#4caf50" if pct < 50 else ("#ffb300" if pct < 80 else "#e53935")
    used = f" {_fmt_tokens(tokens)}/{_fmt_tokens(max_tokens)}" if max_tokens else ""
    return f"ctx:<style fg='{color}'>{pct}%</style>{used}"


_SESSION_START_TIME = time.time()


def _fmt_session_uptime() -> str:
    elapsed = time.time() - _SESSION_START_TIME
    mins = int(elapsed // 60)
    hours = mins // 60
    if hours > 0:
        return f"{hours}h {mins % 60}m"
    return f"{mins}m"


def _thinking_segment() -> str:
    """Compact thinking label for the status bar (e.g. 'think:ON/high')."""
    try:
        from config import get_thinking_display_label
        lbl = get_thinking_display_label()
        return f"think:{lbl}" if lbl else ""
    except Exception:
        return ""


def build_bottom_toolbar(agent, ui_state):
    """Return a callable rendering the live status bar.

    Reads current state on every keystroke. All fields are cheap: the context
    meter reads the agent's pre-cached usage dict (refreshed once per turn), so
    no O(history) token counting happens on the keystroke path.
    """
    def _toolbar():
        try:
            provider = getattr(agent, "provider", "?")
            model = getattr(agent, "model_name", "?")
            try:
                from claude_catalog import label_for
                model = label_for(provider, model)
            except Exception:
                pass
            tier = _tier_label(agent)
            mode = ui_state.get("mode", "CHAT")
            try:
                from src.agent.plan_mode import PLAN
                if PLAN.active:
                    mode = "PLAN"
            except Exception:
                pass
            cwd = os.path.basename(os.getcwd()) or os.getcwd()
            think = _thinking_segment()
            think_str = f"| {think} " if think else ""
            # Live context-window meter (cached; see _context_segment).
            ctx = _context_segment(getattr(agent, "_last_context_usage", None))
            ctx_str = f"| {ctx} " if ctx else ""
            # Running session token/cost total, once anything has been spent.
            usage_str = ""
            try:
                from usage_tracker import usage as session_usage
                if session_usage.requests:
                    usage_str = f"| {session_usage.format_session()} "
            except Exception:
                usage_str = ""
            uptime_str = f"| uptime:{_fmt_session_uptime()} "
            # ASCII separators only: legacy Windows consoles (cp1251) choke on
            # box-drawing chars and emoji.
            return HTML(
                f" <b>{mode}</b> | {provider}:{model} | tier:{tier} {think_str}| cwd:{cwd} "
                f"{ctx_str}{usage_str}{uptime_str}| <style fg='#888888'>/help</style> "
            )
        except Exception:
            return ""
    return _toolbar


class _SafeFileHistory(FileHistory):
    """FileHistory that can't take down the REPL.

    Two problems it guards against:
    - The path is absolute (resolved by the caller), so input history keeps
      working after the user changes the working directory with /cd. A relative
      path would be re-resolved against the new cwd on every append and crash
      the prompt_toolkit event loop the moment that cwd has no .argent/ dir.
    - Any disk error while loading or appending is swallowed instead of
      bubbling up out of the key-press handler and killing the session.
    """

    def store_string(self, string: str) -> None:
        try:
            Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
            super().store_string(string)
        except Exception:
            pass

    def load_history_strings(self):
        try:
            yield from super().load_history_strings()
        except Exception:
            return


def _to_pt_style(spec: str) -> str:
    if not spec or spec == "default":
        return ""
    # Map Rich color names that prompt_toolkit does not natively recognize
    rich_to_pt = {
        "bright_white": "#ffffff",
        "bright_black": "#808080",
        "bright_red": "#ff5555",
        "bright_green": "#50fa7b",
        "bright_yellow": "#f1fa8c",
        "bright_blue": "#bd93f9",
        "bright_magenta": "#ff79c6",
        "bright_cyan": "#8be9fd",
    }
    parts = spec.split()
    mapped = [rich_to_pt.get(p.strip(), p.strip()) for p in parts if p.strip()]
    return " ".join(mapped)


def build_prompt_session(get_commands, agent, ui_state, history_path=None) -> PromptSession:
    """Create the PromptSession used by the main REPL loop."""
    if history_path:
        try:
            # Anchor to an absolute path now, so /cd later can't strand it.
            hp = Path(history_path).expanduser().resolve()
            hp.parent.mkdir(parents=True, exist_ok=True)
            history = _SafeFileHistory(str(hp))
        except Exception:
            history = InMemoryHistory()
    else:
        history = InMemoryHistory()

    from prompt_toolkit.styles import Style
    from ui import c

    try:
        sym_color = _to_pt_style(c.get("user_prompt_symbol", "#58a6ff"))
        prompt_attr = f"{sym_color} bold".strip() if sym_color else "bold"
        user_color = _to_pt_style(c.get("user_prompt", "#ffffff"))

        style_dict = {'prompt': prompt_attr}
        if user_color:
            style_dict[''] = user_color
        prompt_style = Style.from_dict(style_dict)
    except Exception:
        prompt_style = Style.from_dict({'prompt': 'bold #58a6ff', '': '#ffffff'})

    from prompt_toolkit.key_binding import KeyBindings
    kb = KeyBindings()

    @kb.add('enter')
    def _handle_enter(event):
        # Shift+Enter on Windows: insert newline instead of submitting
        if os.name == 'nt':
            try:
                import ctypes
                VK_SHIFT = 0x10
                if ctypes.windll.user32.GetKeyState(VK_SHIFT) & 0x8000:
                    event.current_buffer.insert_text('\n')
                    return
            except Exception:
                pass
        event.current_buffer.validate_and_handle()

    @kb.add('escape', 'enter')
    def _handle_alt_enter(event):
        event.current_buffer.insert_text('\n')

    @kb.add('c-j')
    def _handle_ctrl_j(event):
        event.current_buffer.insert_text('\n')

    return PromptSession(
        completer=ArgentCommandCompleter(get_commands),
        history=history,
        bottom_toolbar=build_bottom_toolbar(agent, ui_state),
        complete_while_typing=True,
        style=prompt_style,
        key_bindings=kb,
    )
