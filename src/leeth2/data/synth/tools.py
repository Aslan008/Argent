"""Arbitrary tool sets: pick the right tool from a described list (or none), and pull its argument out of the request.

The tool list is part of the question and changes every time (3..25 tools out of a library), so the model has to read
descriptions instead of remembering names. A slice of tools is reserved for evaluation to measure unseen-tool transfer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...schema import Sample

CITIES = ["Paris", "Berlin", "Tokyo", "Moscow", "Лондон", "Казань", "Madrid", "Oslo", "Нью-Йорк", "Санкт-Петербург", "Cairo", "Lima"]
NAMES = ["Anna", "John", "Maria", "Dmitry", "Olga", "Peter", "Ivan", "Sophie", "Алексей", "Наталья", "Mike", "Лена"]
FILES = ["/home/user/report.pdf", "C:\\Users\\Ann\\notes.txt", "~/projects/app/main.py", "./config.yaml", "/var/log/syslog", "D:\\photos\\trip.jpg", "src/utils.ts", "README.md"]
DIRS = ["/home/user/docs", "C:\\Projects", "~/Downloads", "./src", "/etc/nginx", "D:\\Music"]
URLS = ["https://example.com/pricing", "https://news.site/world", "https://docs.python.org/3/", "https://github.com/torvalds/linux", "https://wikipedia.org/wiki/Moon"]
EMAILS = ["anna@example.com", "boss@company.org", "support@shop.io", "ivan.petrov@mail.ru", "team@startup.dev"]
QUERIES = ["best pizza in town", "how to change a tire", "python asyncio tutorial", "курс доллара сегодня", "рецепт борща", "cheap flights to Rome", "weather radar", "новости технологий"]
TEXTS = ["the meeting is moved to 5pm", "thanks for the quick reply", "встреча перенесена на пятницу", "please review the draft", "I will be late", "до завтра"]
SONGS = ["Bohemian Rhapsody", "Imagine", "Кино — Группа крови", "Shape of You", "Smells Like Teen Spirit", "Мой друг"]
CMDS = ["ls -la", "git status", "npm test", "docker ps", "pip list", "df -h", "pytest -q", "ping 8.8.8.8"]
TIMES = ["9:00", "14:30", "18:00", "7 am", "8 pm", "в 10 утра", "в 21:00"]
DATES = ["Friday", "tomorrow", "March 3", "next Monday", "в пятницу", "завтра", "3 марта", "on 2025-06-01"]
ROOMS = ["kitchen", "bedroom", "living room", "garage", "кухне", "спальне", "гостиной"]
TASKS = ["buy milk", "call the dentist", "pay the rent", "submit the report", "купить хлеб", "позвонить маме", "оплатить интернет"]
LANGS = ["French", "German", "Spanish", "Japanese", "английский", "французский", "немецкий"]
TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "BTC"]
DISHES = ["margherita pizza", "sushi set", "burger", "пицца пепперони", "ролл Калифорния", "борщ"]
NUMS = ["12", "45", "7", "130", "2.5", "999"]


@dataclass
class Tool:
    name: str
    desc_en: str
    desc_ru: str
    tpl_en: List[str]
    tpl_ru: List[str]
    slots: Dict[str, Any]  # slot -> list of values or callable(rng)->str
    extract_slot: Optional[str] = None
    extract_en: str = ""
    extract_ru: str = ""


def S(**kw: Any) -> Dict[str, Any]:
    return kw


TOOLS: List[Tool] = [
    Tool("read_file", "Read the contents of a file", "Прочитать содержимое файла", ["Open {path} and show me what is inside.", "What does {path} contain?", "Show the file {path}."], ["Открой {path} и покажи, что внутри.", "Что находится в файле {path}?", "Покажи файл {path}."], S(path=FILES), "path", "the file path", "путь к файлу"),
    Tool("write_file", "Create or overwrite a file with given text", "Создать или перезаписать файл с заданным текстом", ["Create {path} with the text “{text}”.", "Write “{text}” into {path}."], ["Создай файл {path} с текстом «{text}».", "Запиши «{text}» в {path}."], S(path=FILES, text=TEXTS), "path", "the file path", "путь к файлу"),
    Tool("delete_file", "Delete a file from disk", "Удалить файл с диска", ["Delete {path}.", "Remove the file {path} please."], ["Удали файл {path}.", "Сотри {path}, пожалуйста."], S(path=FILES), "path", "the file path", "путь к файлу"),
    Tool("list_directory", "List files in a directory", "Показать список файлов в каталоге", ["What files are in {dir}?", "List the contents of {dir}."], ["Какие файлы лежат в {dir}?", "Покажи содержимое каталога {dir}."], S(dir=DIRS), "dir", "the directory", "каталог"),
    Tool("run_command", "Run a shell command", "Выполнить команду в терминале", ["Run `{cmd}`.", "Execute {cmd} in the terminal."], ["Выполни `{cmd}`.", "Запусти в терминале {cmd}."], S(cmd=CMDS), "cmd", "the command", "команда"),
    Tool("web_search", "Search the web for information", "Искать информацию в интернете", ["Search the web for {query}.", "Look up {query} online.", "Google {query}."], ["Найди в интернете {query}.", "Поищи {query} в сети.", "Загугли {query}."], S(query=QUERIES), "query", "the search query", "поисковый запрос"),
    Tool("open_url", "Open a web page by URL", "Открыть веб-страницу по адресу", ["Open {url}.", "Go to {url} in the browser."], ["Открой {url}.", "Перейди по адресу {url} в браузере."], S(url=URLS), "url", "the URL", "адрес страницы"),
    Tool("send_email", "Send an email message", "Отправить электронное письмо", ["Email {email} that {text}.", "Send a mail to {email}: {text}."], ["Напиши на {email}, что {text}.", "Отправь письмо {email}: {text}."], S(email=EMAILS, text=TEXTS), "email", "the recipient address", "адрес получателя"),
    Tool("calendar_create_event", "Create a calendar event", "Создать событие в календаре", ["Schedule a meeting {date} at {time}.", "Put a call on my calendar for {date}, {time}."], ["Назначь встречу на {date} {time}.", "Добавь в календарь звонок: {date}, {time}."], S(date=DATES, time=TIMES), "date", "the date", "дата"),
    Tool("set_reminder", "Set a reminder", "Поставить напоминание", ["Remind me to {task} at {time}.", "Set a reminder: {task}, {time}."], ["Напомни {task} {time}.", "Поставь напоминание: {task}, {time}."], S(task=TASKS, time=TIMES), "task", "the task to remember", "о чём напомнить"),
    Tool("get_weather", "Get the weather forecast for a city", "Получить прогноз погоды для города", ["What is the weather in {city}?", "Will it rain in {city} tomorrow?"], ["Какая погода в городе {city}?", "Будет ли завтра дождь: {city}?"], S(city=CITIES), "city", "the city", "город"),
    Tool("calculator", "Evaluate an arithmetic expression", "Вычислить арифметическое выражение", ["What is {a} times {b}?", "Calculate {a} + {b}."], ["Сколько будет {a} умножить на {b}?", "Посчитай {a} + {b}."], S(a=NUMS, b=NUMS)),
    Tool("translate_text", "Translate text into another language", "Перевести текст на другой язык", ["Translate “{text}” into {lang}.", "How do you say “{text}” in {lang}?"], ["Переведи «{text}» на {lang}.", "Как будет «{text}» на языке: {lang}?"], S(text=TEXTS, lang=LANGS), "text", "the text to translate", "текст для перевода"),
    Tool("play_music", "Play a song or playlist", "Включить песню или плейлист", ["Play {song}.", "Put on {song} please."], ["Включи {song}.", "Поставь, пожалуйста, {song}."], S(song=SONGS), "song", "the song", "песня"),
    Tool("set_timer", "Start a countdown timer", "Запустить таймер обратного отсчёта", ["Set a timer for {n} minutes.", "Timer {n} min."], ["Поставь таймер на {n} минут.", "Таймер: {n} мин."], S(n=NUMS), "n", "the number of minutes", "число минут"),
    Tool("take_screenshot", "Capture the screen", "Сделать снимок экрана", ["Take a screenshot.", "Capture my screen."], ["Сделай скриншот.", "Сфотографируй экран."], S()),
    Tool("git_commit", "Commit staged changes to git", "Зафиксировать изменения в git", ["Commit my changes with message “{text}”.", "git commit: {text}"], ["Закоммить изменения с сообщением «{text}».", "Сделай коммит: {text}"], S(text=TEXTS), "text", "the commit message", "сообщение коммита"),
    Tool("create_task", "Add an item to the to-do list", "Добавить задачу в список дел", ["Add “{task}” to my to-do list.", "Put {task} on my task list."], ["Добавь «{task}» в список дел.", "Занеси в задачи: {task}."], S(task=TASKS), "task", "the task", "задача"),
    Tool("get_stock_price", "Get the current price of a stock or crypto", "Узнать текущую цену акции или криптовалюты", ["What is the price of {ticker} right now?", "How is {ticker} trading?"], ["Сколько сейчас стоит {ticker}?", "Как торгуется {ticker}?"], S(ticker=TICKERS), "ticker", "the ticker", "тикер"),
    Tool("send_message", "Send a chat message to a contact", "Отправить сообщение контакту в мессенджере", ["Message {name}: {text}.", "Tell {name} on chat that {text}."], ["Напиши {name} в чат: {text}.", "Передай {name} в мессенджере, что {text}."], S(name=NAMES, text=TEXTS), "name", "the contact", "контакт"),
    Tool("call_contact", "Place a phone call", "Позвонить по телефону", ["Call {name}.", "Ring {name} for me."], ["Позвони {name}.", "Набери {name}."], S(name=NAMES), "name", "the contact", "контакт"),
    Tool("navigate_map", "Get directions to a place", "Построить маршрут до места", ["Directions to {city}.", "How do I get to {city}?"], ["Построй маршрут до {city}.", "Как доехать до {city}?"], S(city=CITIES), "city", "the destination", "пункт назначения"),
    Tool("convert_currency", "Convert an amount between currencies", "Конвертировать сумму между валютами", ["Convert {n} dollars to euros.", "How much is {n} USD in EUR?"], ["Переведи {n} долларов в евро.", "Сколько будет {n} USD в EUR?"], S(n=NUMS)),
    Tool("search_files", "Find files by name", "Найти файлы по имени", ["Find files named {name}.", "Where is the file {name}?"], ["Найди файлы с именем {name}.", "Где лежит файл {name}?"], S(name=["report.pdf", "main.py", "notes.txt", "photo.jpg", "invoice.xlsx"]), "name", "the file name", "имя файла"),
    Tool("book_flight", "Book a flight ticket", "Забронировать авиабилет", ["Book a flight to {city} for {date}.", "I need a plane ticket to {city}, {date}."], ["Забронируй билет в {city} на {date}.", "Нужен авиабилет до {city}, {date}."], S(city=CITIES, date=DATES), "city", "the destination", "пункт назначения"),
    Tool("order_food", "Order food delivery", "Заказать доставку еды", ["Order {dish} for delivery.", "I want {dish} delivered."], ["Закажи доставку: {dish}.", "Хочу, чтобы привезли {dish}."], S(dish=DISHES), "dish", "the dish", "блюдо"),
    Tool("toggle_lights", "Switch smart lights on or off", "Включить или выключить умный свет", ["Turn off the lights in the {room}.", "Switch on the {room} light."], ["Выключи свет на {room}.", "Включи свет в {room}."], S(room=ROOMS), "room", "the room", "комната"),
    Tool("adjust_thermostat", "Change the thermostat temperature", "Изменить температуру термостата", ["Set the temperature to {n} degrees.", "Make it {n}° in here."], ["Установи температуру {n} градусов.", "Сделай тут {n}°."], S(n=["18", "21", "23", "25"]), "n", "the temperature", "температура"),
    Tool("clipboard_copy", "Copy text to the clipboard", "Скопировать текст в буфер обмена", ["Copy “{text}” to the clipboard.", "Put {text} in my clipboard."], ["Скопируй «{text}» в буфер обмена.", "Положи {text} в буфер."], S(text=TEXTS), "text", "the text to copy", "текст для копирования"),
    Tool("create_folder", "Create a new directory", "Создать новую папку", ["Create a folder {dir}.", "Make a directory called {dir}."], ["Создай папку {dir}.", "Сделай каталог {dir}."], S(dir=DIRS), "dir", "the directory", "каталог"),
    Tool("summarize_page", "Summarize a web page", "Кратко пересказать веб-страницу", ["Summarize {url}.", "Give me the gist of {url}."], ["Перескажи кратко {url}.", "О чём страница {url}?"], S(url=URLS), "url", "the URL", "адрес страницы"),
    Tool("check_email", "Read new email messages", "Прочитать новые письма", ["Do I have new mail?", "Check my inbox."], ["Есть новые письма?", "Проверь мою почту."], S()),
    Tool("lock_screen", "Lock the computer screen", "Заблокировать экран компьютера", ["Lock my screen.", "Lock the computer."], ["Заблокируй экран.", "Заблокируй компьютер."], S()),
]
NO_TOOL_EN = ["Tell me a joke.", "What is the capital of France?", "Why is the sky blue?", "Thanks, that was helpful!", "Explain recursion in simple words.", "What do you think about teamwork?", "Who wrote War and Peace?"]
NO_TOOL_RU = ["Расскажи анекдот.", "Какая столица Франции?", "Почему небо голубое?", "Спасибо, было полезно!", "Объясни рекурсию простыми словами.", "Что ты думаешь о командной работе?", "Кто написал «Войну и мир»?"]
NONE_OPT = {"en": ["no tool needed — answer directly", "none of the tools — just reply", "No suitable tool"], "ru": ["инструмент не нужен — ответить напрямую", "ни один — просто ответить", "Подходящего инструмента нет"]}

HOLDOUT_EVERY = 7  # tools with index % 7 == 3 appear only in evaluation pools


def is_holdout(i: int) -> bool:
    return i % HOLDOUT_EVERY == 3


def _fill(rng: random.Random, tpl: str, slots: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    vals = {k: (v(rng) if callable(v) else rng.choice(v)) for k, v in slots.items()}
    return tpl.format(**vals), vals


def _tool_option(rng: random.Random, t: Tool, lang: str, style: str) -> str:
    d = t.desc_ru if lang == "ru" else t.desc_en
    return {"name": t.name, "desc": d, "both": f"{t.name} — {d}"}[style]


def gen_toolpick(rng: random.Random, holdout: bool = False) -> Optional[List[Sample]]:
    """``holdout=True`` builds evaluation samples whose target is a tool never seen in training."""
    lang = rng.choice(["en", "ru"])
    all_ids = list(range(len(TOOLS)))
    train_ids = [i for i in all_ids if not is_holdout(i)]
    ti = rng.choice([i for i in all_ids if is_holdout(i)] if holdout else train_ids)
    tool = TOOLS[ti]
    no_tool = rng.random() < 0.12
    missing = (not no_tool) and rng.random() < 0.10  # the right tool is absent from the list -> "none"
    others = [i for i in (all_ids if holdout else train_ids) if i != ti]
    k = min(rng.randint(3, 25), len(others) + 1)
    chosen_ids = rng.sample(others, k - 1)
    vals: Dict[str, str] = {}
    if no_tool:
        request = rng.choice(NO_TOOL_RU if lang == "ru" else NO_TOOL_EN)
        chosen_ids = chosen_ids + [ti]
    else:
        request, vals = _fill(rng, rng.choice(tool.tpl_ru if lang == "ru" else tool.tpl_en), tool.slots)
        if not missing:
            chosen_ids = chosen_ids + [ti]
    rng.shuffle(chosen_ids)
    style = rng.choice(["name", "desc", "both", "both"])
    opts = [_tool_option(rng, TOOLS[i], lang, style) for i in chosen_ids]
    opts.append(rng.choice(NONE_OPT[lang]))
    if len(set(opts)) != len(opts):
        return None
    target = len(opts) - 1 if (no_tool or missing) else chosen_ids.index(ti)
    if lang == "ru":
        state = f"Запрос пользователя: «{request}»"
        q = rng.choice(["Какой инструмент нужно вызвать?", "Выберите подходящий инструмент или ответьте без инструмента."])
    else:
        state = f"User request: “{request}”"
        q = rng.choice(["Which tool should be called?", "Pick the right tool, or answer without one."])
    out = [Sample("tool_pick", "tools", lang, "choice", state, q, opts, target, meta={"holdout_tool": is_holdout(ti)})]
    lst = "; ".join(opts[:-1])[:400]
    qn = ("Можно ли выполнить этот запрос одним из перечисленных инструментов? " if lang == "ru" else "Can this request be handled by one of these tools? ") + lst
    out.append(Sample("tool_available", "tools", lang, "noul", state, qn, None, 0.0 if (no_tool or missing) else 1.0))
    if not (no_tool or missing) and tool.extract_slot and tool.extract_slot in vals:
        v = vals[tool.extract_slot]
        i = state.find(v)
        if i >= 0:
            qe = f"Извлеките аргумент: {tool.extract_ru}." if lang == "ru" else f"Extract the argument: {tool.extract_en}."
            out.append(Sample("tool_arg", "tools", lang, "extract", state, qe, None, [i, i + len(v)]))
    return out


GENERATORS: Dict[str, Tuple[Callable[[random.Random], Optional[List[Sample]]], float]] = {"toolpick": (gen_toolpick, 6.0)}
