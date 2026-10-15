"""What Leeth 2 v1/v2 never saw: the states Argent really sends, and the requests people really type.

Audit of the live path (src/leeth/dispatcher.py::_build_state) against the v2 training data found three gaps that the
hand-written exams could not show, because they used the bare state and 15 options:

1. When the session has context, Argent sends it:  "Active background processes: ...", "Recent session context:
   - Last target / Last tool / Previous turn".  No training row had these lines.
2. Argent offers 16 options: `read_background_command` sits between ask_user_questions and NO_TOOL.  No row had it.
3. Real traffic (the user's own input history) is mostly NOT a one-tool request: pasted errors, "use skill X and ...",
   weather questions, long multi-step tasks, "ок, продолжай", corrections.  Those must be handed to the LLM
   (UNFAMILIAR / NO_TOOL); the template-built UNFAMILIAR set was too uniform to teach that.

Every label here is decided by a rule from the pieces the sample is built from (context present or not, which kind of
sentence), never guessed.  The hand-written and real-usage exams are never read here; finetune.py also drops any sample
that repeats one of their requests.
"""

from __future__ import annotations

import random
import re
from typing import List, Optional, Sequence, Tuple

from ...schema import Sample
from ..argent import TOOL_OPTIONS, TOOL_QUESTION
from . import argent_router as AR
from .argent_router import ARG_QUESTION, NEED_TOOL_QUESTION, _pick, dress

#: the option list Argent really sends (dispatcher._build_state_and_questions)
PROD_OPTIONS: List[str] = list(TOOL_OPTIONS)
PROD_OPTIONS.insert(PROD_OPTIONS.index("NO_TOOL"), "read_background_command")

SERVER_COMMANDS = ["python -m http.server 8000", "npm run dev", "npm start", "uvicorn app:app --reload", "python manage.py runserver", "node server.js",
                   "flask run", "cargo watch -x run", "docker compose up", "pytest --watch", "npm run build", "python train.py", "dotnet watch run",
                   "vite", "next dev", "ng serve", "go run main.go", "tail -f app.log", "python bot.py", "make build", "cargo build --release"]
PREV_TURNS = ["Готово, файл создан.", "Запустил сервер на порту 8000.", "Нашёл 3 совпадения в src/app.py.", "Исправил ошибку в обработчике.", "Тесты прошли: 42 passed.",
              "Вот что я нашёл в проекте.", "Сборка завершена без ошибок.", "Прочитал файл, вот краткое содержание.", "Не нашёл такой функции в проекте.",
              "Что именно вы хотите изменить?", "Готов приступить к работе.", "Сделал, проверьте результат.", "I created the file and ran the tests.",
              "Here is what I found in the repository.", "Запустил команду в фоне.", "Проект проанализирован, вот план."]
LAST_TOOLS = ["read_file", "write_file", "replace_in_file", "run_command", "grep_search", "search_files", "list_directory", "browser_open", "calculate",
              "read_background_command", "search_web", "semantic_search"]
SKILLS = ["ccc", "unity-dev", "blind-spot-pass", "unfairgaps", "code-review", "frontend-design", "pdf", "docx", "skill-creator", "security-review", "simplify", "deploy"]
CITIES = ["Караганде", "Москве", "Алматы", "Астане", "Берлине", "Токио", "Лондоне", "Пионерском", "Санкт-Петербурге", "Нью-Йорке", "Минске", "Стамбуле"]
ERRORS = [
    "TypeError: 'NoneType' object is not subscriptable", "AttributeError: 'sqlite3.Connection' object has no attribute 'lastrowid'", "Uncaught SyntaxError: Unexpected token '}'",
    "ModuleNotFoundError: No module named '{m}'", "KeyError: '{w}'", "ImportError: cannot import name '{w}' from '{m}'", "Uncaught ReferenceError: {w} is not defined",
    "error[E0382]: borrow of moved value: `{w}`", "NullReferenceException: Object reference not set to an instance of an object", "CS0103: The name '{w}' does not exist in the current context",
    "FileNotFoundError: [Errno 2] No such file or directory: '{f}'", "ValueError: invalid literal for int() with base 10: '{w}'", "IndexError: list index out of range",
    "Traceback (most recent call last): File \"{f}\", line {n}, in <module>", "npm ERR! code ERESOLVE", "TypeError: Cannot read properties of undefined (reading '{w}')",
    "SyntaxError: invalid syntax ({f}, line {n})", "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0", "PermissionError: [Errno 13] Permission denied: '{f}'",
    "{f}:{n} Uncaught SyntaxError: Unexpected end of input", "ConnectionRefusedError: [WinError 10061]", "Failed to load resource: the server responded with a status of 404 (Not Found)",
]
ERROR_LEADS = ["", "", "", "Теперь появляется ошибка: ", "Вот ошибка: ", "Ошибка при запуске: ", "получаю такое: ", "не работает, пишет ", "исправь ошибку: ", "fix this: ", "тест упал: ", "в консоли: "]
BIG_TASKS = [
    "изучи весь проект и составь по нему анализ в {f}", "проведи полноценное тестирование этого сайта, понажимай все кнопки", "создай любое полезное приложение и приведи пример использования",
    "создай на рабочем столе папку и реализуй в ней проект по ТЗ из файла {p}", "изучи текущий проект и доведи его до ума", "проверь проект, почему не работает {x}",
    "найди и исправь все ошибки в проекте, затем запусти тесты", "сделай ревью всей кодовой базы и предложи улучшения", "подготовь релиз: обнови версию, собери, прогони тесты и напиши changelog",
    "проанализируй архитектуру и составь план рефакторинга", "перепиши интерфейс в тёмной теме и проверь в браузере", "разберись, как устроена авторизация, и добавь вход через Google",
    "изучи репозиторий {u} и расскажи, что он делает и как его использовать", "реализуй плагин {w}", "добавь в игру систему {x} и протестируй", "сделай так, чтобы игра запускалась на весь экран",
    "set up CI for this project and make it green", "investigate why the build is slow and fix it", "migrate the project to the new framework version step by step",
    "найди в документации {w} все настройки {x} и сделай краткий конспект", "составь отчёт по {x} и сохрани его в {f}",
]
BIG_FILL = dict(
    f=["AGENTS.md", "REPORT.md", "docs/analysis.md", "NOTES.md"], p=["C:\\Docs\\ТЗ.md", "C:\\Obsidian Folder\\Промпт.md", "D:\\projects\\spec.md"],
    x=["освещение", "магазин", "авторизация", "мутация артефактов", "запуск игры в браузере", "сохранения", "звук", "инвентарь"], w=["Unity", "prompt_enhancer", "SaaS", "Godot", "URP", "Redis"],
    u=["https://github.com/AyanbekDos/unfairgaps-os", "https://github.com/org/repo", "https://github.com/user/project"],
)
WEB_QUESTIONS = ["какая погода сейчас в {c}?", "какая погода в {c}", "погода в {c} на завтра", "какой сейчас курс доллара?", "что нового в мире ИИ?", "последние новости про {w}",
                 "сколько стоит {w} сейчас?", "what's the weather in {e}?", "latest news about {w}", "найди в интернете отзывы о {w}", "загугли, как настроить {w}",
                 "найди в документации {w} про {x}", "что написано в документации {w} про {x}?", "что говорится в документации {w} про {x}", "кто выиграл вчера матч?"]
FEEDBACK = [
    "Отлично, ты всё исправил. Можешь заканчивать", "отлично, ошибок нет", "спасибо, всё работает", "да, так и надо", "нет, мы работаем через мою собственную платформу",
    "не используй {s} на весь проект, ищи в папке scripts", "продолжи без использования {s}", "не переходи по пути cd, иди прямо в текущей директории", "просто скажи мне что нужно изменить, я сделаю сам",
    "каков результат?", "каков результат? ничего не трогай, просто ответь", "откуда ты это знаешь?", "то есть, это нужно только для помещений?", "что думаешь насчёт существующего кода?",
    "переведи на русский то, что ты написал", "объясни простыми словами и приведи пример", "а если по-другому?", "ты закончил работу?", "что ты умеешь?", "почему ты так решил?",
    "хорошо, но коротко", "ладно, понял", "ты уверен?", "повтори, пожалуйста", "сделай короче", "хм, а почему так", "now explain that in simple words", "you did great, thanks",
    "файл записан верно, ты просто решил прочитать его через консоль", "нет, не то", "не надо, оставь как есть", "стоп, не делай этого", "подожди, я передумал",
]
CONTINUE = ["продолжай", "продолжай если это необходимо", "приступай", "отлично, приступай", "приступай к задаче", "да, делай", "давай, начинай", "ок, продолжай", "делай", "дальше",
            "go ahead", "continue", "ok, proceed", "да, запускай", "хорошо, приступай к работе", "поехали"]

BG_BARE = ["покажи вывод фоновой команды {pid}", "прочитай вывод фонового процесса {pid}", "read the output of background command {pid}", "что вывела фоновая команда {pid}",
           "покажи логи фоновой задачи {pid}", "статус фонового процесса {pid}", "check background command {pid}", "посмотри вывод процесса {pid}", "как там фоновая задача {pid}?"]
BG_CTX = ["что там с сервером?", "что там с процессом?", "как там сборка?", "он уже закончил?", "есть ошибки в выводе?", "покажи вывод", "покажи, что он выводит", "как дела у фонового процесса?",
          "check the output", "is it done yet?", "what is the server printing?", "посмотри логи сервера", "сервер что-нибудь пишет?", "закончилось?", "что с тестами, которые запущены?",
          "проверь, работает ли процесс", "покажи вывод сборки", "есть ли ошибки в логах?", "ну что там", "как успехи у запущенной команды?", "прочитай вывод фоновой команды"]
KILL_CTX = ["останови сервер", "останови работу сервера", "убей процесс", "заверши фоновую задачу", "kill the server", "stop the dev server", "останови фоновую команду", "выключи сервер",
            "останови его", "прекрати выполнение", "stop it", "заверши процесс {pid}", "убей процесс {pid}", "kill process {pid}", "останови процесс с pid {pid}", "заверши сервер"]
FILE_ANAPHORA = ["открой его", "покажи его ещё раз", "прочитай его снова", "что в нём?", "покажи этот файл", "show it again", "open it", "what's in it?", "выведи его содержимое",
                 "покажи содержимое этого файла", "прочитай этот файл", "перечитай файл", "глянь ещё раз что в файле"]
FILE_ANAPHORA_DELEGATE = ["исправь в нём ошибку", "перепиши его", "добавь туда обработку ошибок", "отрефактори этот файл", "fix the bug in it", "объясни, что делает этот файл", "улучши его",
                          "добавь комментарии в этот файл", "найди проблемы в этом файле и исправь"]


_SLOT = re.compile(r"\{([a-z])\}")


def lower_first(text: str) -> str:
    """The dispatcher lowercases the first letter before it builds the state; training must look the same."""
    return text[:1].lower() + text[1:] if text[:1].isalpha() else text


def make_context(rng: random.Random, *, tasks: Optional[bool] = None, target: Optional[bool] = None, prev: Optional[bool] = None) -> dict:
    """A plausible session context.  Each part is decided (True/False) or drawn at random when left None."""
    ctx = {"active_tasks": [], "last_target": None, "last_tool": None, "prev_turn": None}
    if tasks if tasks is not None else rng.random() < 0.25:
        for _ in range(rng.choice([1, 1, 1, 2])):
            ctx["active_tasks"].append({"pid": rng.randint(1000, 99999) if rng.random() < 0.7 else rng.randint(1, 12), "command": _pick(rng, SERVER_COMMANDS),
                                        "status": "running", "elapsed": f"{rng.choice([1, 4, 12, 35, 95, 240, 1800])}s"})
    if target if target is not None else rng.random() < 0.5:
        ctx["last_target"] = rng.choice([AR.filepath(rng), AR.filepath(rng), AR.url(rng), AR.directory(rng)])
    if prev if prev is not None else rng.random() < 0.6:
        ctx["prev_turn"] = _pick(rng, PREV_TURNS)
        if rng.random() < 0.8:
            ctx["last_tool"] = _pick(rng, LAST_TOOLS)
    elif rng.random() < 0.2:
        ctx["last_tool"] = _pick(rng, LAST_TOOLS)
    return ctx


def build_state(query: str, ctx: Optional[dict], cwd: str) -> str:
    """Byte-for-byte what dispatcher._build_state produces."""
    q = lower_first(query)
    has = bool(ctx and (ctx["active_tasks"] or ctx["last_target"] or ctx["last_tool"] or ctx["prev_turn"]))
    if not has:
        return f'Incoming user request in Argent session:\n"{q}"\nWorking directory: {cwd}'
    tasks = ctx["active_tasks"]
    active = "; ".join(f"PID {t['pid']} ({t['command']}, status={t['status']}, elapsed={t['elapsed']})" for t in tasks) if tasks else "None"
    return (f'Incoming user request in Argent session:\n"{q}"\nActive background processes: {active}\nRecent session context:\n'
            f"- Last target: {ctx['last_target'] or 'None'}\n- Last tool: {ctx['last_tool'] or 'None'}\n- Previous turn: {ctx['prev_turn'] or 'None'}\nWorking directory: {cwd}")


def _route(task: str, state: str, tool: str) -> Sample:
    return Sample(task, "argent", "ru", "choice", state, TOOL_QUESTION, list(PROD_OPTIONS), PROD_OPTIONS.index(tool), meta={"tool": tool})


def _bundle(rng: random.Random, task: str, state: str, tool: str, key: Optional[str] = None) -> List[Sample]:
    out = [_route(task, state, tool)]
    if rng.random() < 0.4:
        out.append(Sample("argent_need_tool_v3", "argent", "ru", "noul", state, NEED_TOOL_QUESTION, None, 0.0 if tool in ("NO_TOOL", "UNFAMILIAR") else 1.0))
    if key and rng.random() < 0.5:
        i = state.find(key)
        if i >= 0:
            out.append(Sample("argent_arg_v3", "argent", "ru", "extract", state, ARG_QUESTION, None, [i, i + len(key)]))
    return out


def _clean(request: str) -> str:
    return request.replace('"', "'")


# ------------------------------------------------------------------------------------------------ generators
def gen_route16(rng: random.Random, owner: bool = False) -> List[Sample]:
    """The ordinary routing grammar, re-asked the way production asks: 16 options, lower-cased first letter and, half the
    time, an unrelated session context (which must not change the answer).  ``owner``: two-step requests are labelled the way
    the owner's batteries label them (argent_router.compound)."""
    names = list(AR.MAKERS)
    tool = rng.choices(names, [AR.MAKERS[n][1] for n in names])[0]
    request, key = AR.MAKERS[tool][0](rng)
    request, tool = AR.compound(rng, tool, request, owner)
    request = _clean(dress(rng, request))
    ctx = make_context(rng) if rng.random() < 0.5 else None
    state = build_state(request, ctx, _pick(rng, AR.WORKDIRS))
    return _bundle(rng, "argent_route_v3", state, tool, key if key and key in request else None)


def gen_background(rng: random.Random) -> List[Sample]:
    """Background processes: read their output, stop them — and what to do when there are none."""
    cwd = _pick(rng, AR.WORKDIRS)
    r = rng.random()
    pid = str(rng.randint(1000, 99999))
    if r < 0.18:   # explicit pid, no context needed
        request = dress(rng, _pick(rng, BG_BARE).format(pid=pid))
        ctx = make_context(rng, tasks=rng.random() < 0.4)
        return _bundle(rng, "argent_bg_v3", build_state(_clean(request), ctx, cwd), "read_background_command", pid)
    if r < 0.50:   # anaphora + a running process -> read its output
        ctx = make_context(rng, tasks=True)
        request = dress(rng, _pick(rng, BG_CTX))
        return _bundle(rng, "argent_bg_v3", build_state(_clean(request), ctx, cwd), "read_background_command")
    if r < 0.72:   # stop it -> run_command (kill)
        ctx = make_context(rng, tasks=True)
        used = ctx["active_tasks"][0]["pid"]
        request = dress(rng, _pick(rng, KILL_CTX).format(pid=used))
        return _bundle(rng, "argent_bg_v3", build_state(_clean(request), ctx, cwd), "run_command", str(used) if str(used) in request else None)
    if r < 0.84:   # same words, nothing running: the LLM has to work out what is meant
        ctx = make_context(rng, tasks=False)
        request = dress(rng, _pick(rng, BG_CTX + KILL_CTX).format(pid=pid))
        return _bundle(rng, "argent_bg_v3", build_state(_clean(request), ctx, cwd), "NO_TOOL" if request.lower().startswith(("что ", "как ", "ну ")) else "UNFAMILIAR")
    if r < 0.93:   # file anaphora with a file in context
        ctx = make_context(rng, target=True, tasks=rng.random() < 0.15)
        ctx["last_target"] = AR.filepath(rng)
        request = dress(rng, _pick(rng, FILE_ANAPHORA))
        return _bundle(rng, "argent_anaphora_v3", build_state(_clean(request), ctx, cwd), "read_file")
    ctx = make_context(rng, target=True)
    request = dress(rng, _pick(rng, FILE_ANAPHORA_DELEGATE))
    return _bundle(rng, "argent_anaphora_v3", build_state(_clean(request), ctx, cwd), "UNFAMILIAR")


def _fill(rng: random.Random, text: str) -> str:
    vals = dict(f=AR.filepath(rng), n=rng.randint(3, 400), w=_pick(rng, AR.STEMS), m=_pick(rng, AR.MODULES), s=_pick(rng, SKILLS), c=_pick(rng, CITIES),
                e=rng.choice(["London", "Berlin", "Tokyo", "Almaty", "Paris"]))
    for k, v in BIG_FILL.items():          # request-specific pools win over the generic ones, except the file name of an error
        if k != "f" or "{f}" in text and ("анализ в" in text or "сохрани его в" in text):
            vals[k] = _pick(rng, v)
    return _SLOT.sub(lambda m: str(vals.get(m.group(1), m.group(0))), text)   # only {x}-style slots: error texts hold real braces


def gen_delegate(rng: random.Random) -> List[Sample]:
    """What people really type that is NOT one tool call: hand it to the LLM (UNFAMILIAR) or just answer (NO_TOOL)."""
    r = rng.random()
    cwd = _pick(rng, AR.WORKDIRS)
    ctx = make_context(rng) if rng.random() < 0.55 else None
    if r < 0.20:   # use a skill
        s = _pick(rng, SKILLS)
        task = _pick(rng, ["найди, где реализовано {x}", "найди в проекте, что отвечает за {x}", "выяви системные дыры", "проверь проект и исправь найденное", "сделай {x}", "проанализируй {x}",
                           "find where {x} is implemented", "review the changes"])
        lead = _pick(rng, ["используй скилл {s} и ", "с помощью скилла {s} ", "используй {s}: ", "use the {s} skill to ", "примени скилл {s}: "])
        request, tool = _fill(rng, (lead + task).replace("{s}", s)), "UNFAMILIAR"
    elif r < 0.38:   # a pasted error
        err = _fill(rng, _pick(rng, ERRORS))
        request, tool = _pick(rng, ERROR_LEADS) + err, "UNFAMILIAR"
    elif r < 0.52:   # live information
        request, tool = _fill(rng, _pick(rng, WEB_QUESTIONS)), "UNFAMILIAR"
    elif r < 0.72:   # big agentic task
        request, tool = _fill(rng, _pick(rng, BIG_TASKS)), "UNFAMILIAR"
    elif r < 0.88:   # reactions, corrections, questions about the last answer
        request, tool = _fill(rng, _pick(rng, FEEDBACK)), "NO_TOOL"
    else:            # "go on"
        request, tool = _pick(rng, CONTINUE), "NO_TOOL"
        ctx = ctx or make_context(rng, tasks=False, prev=True)
    request = _clean(dress(rng, request, polite=0.1))
    if len(request) > 260 and rng.random() < 0.5:
        request = request[:260]
    return _bundle(rng, "argent_delegate_v3", build_state(request, ctx, cwd), tool)


# ---------------------------------------------------------------------------------------- more harmless commands
SAFE_MORE = [
    "python -m http.server 8000", "python3 -m http.server", "npm run dev", "npm start", "npm run serve", "npm i", "npm ci", "npm install", "pnpm dev", "yarn dev", "yarn install", "bun install", "bun run dev",
    "pip install -e .", "pip install -U pip", "pip install --upgrade requests", "pip install -r requirements-dev.txt", "uv pip install ruff", "uv run pytest", "poetry install", "poetry run pytest",
    "uvicorn main:app --reload", "uvicorn app:app --port 8001", "flask run", "python manage.py runserver", "python manage.py makemigrations", "python manage.py test", "gunicorn app:app",
    "docker compose up", "docker compose up -d --build", "docker compose logs -f", "docker compose ps", "docker build .", "docker pull nginx", "docker stats --no-stream", "docker network ls",
    "git clone https://github.com/user/repo.git", "git checkout main", "git switch -c fix/login", "git restore --staged src/app.py", "git stash pop", "git cherry-pick abc1234", "git log --stat -3",
    "git config user.name", "git remote add origin https://github.com/user/repo.git", "git commit -am 'wip'", "git push", "git push -u origin feature/x", "git rebase --continue",
    "cargo run", "cargo run --release", "cargo clippy", "cargo fmt", "cargo update", "cargo doc --open", "rustc --version", "rustup update", "go run .", "go mod tidy", "go build -o app", "dotnet run",
    "dotnet restore", "dotnet publish -c Release", "mvn clean install", "gradle build", "./gradlew test", "make clean", "make install", "cmake -B build", "cmake --build build", "ninja -C build",
    "node --version", "npx tsc", "npx prettier --write .", "npx eslint src", "npx vite build", "npx create-react-app demo", "npx skills add user/skill", "tsc -p .", "jest --coverage",
    "ipconfig /flushdns", "ipconfig", "ipconfig /all", "netstat -ano", "netstat -ano | findstr :8000", "tasklist | findstr python", "Get-NetTCPConnection -LocalPort 8000", "Get-Process python", "Get-Process | Sort-Object CPU -Descending | Select-Object -First 10",
    "systeminfo | findstr /B /C:\"OS\"", "wmic cpu get name", "Get-WmiObject Win32_OperatingSystem", "Get-CimInstance Win32_Processor", "hostname", "echo %PATH%", "echo $env:PATH", "set", "where git", "Get-Command python",
    "python -m venv .venv", ".venv\\Scripts\\activate", "source .venv/bin/activate", "python -m pip --version", "py -3.12 -V", "pytest -x --lf", "pytest --cov=src", "python -m unittest discover", "black --check .", "isort .",
    "mkdir logs", "New-Item -ItemType Directory logs", "Set-Location src", "cd src", "cd ..", "Get-Location", "start .", "explorer .", "code .", "notepad notes.txt", "start chrome https://example.com",
    "curl -s https://httpbin.org/get", "curl -X POST -H 'Content-Type: application/json' -d '{\"a\":1}' http://localhost:8000/api", "curl http://localhost:3000/health", "Invoke-RestMethod http://localhost:8000/health",
    "Invoke-WebRequest http://localhost:8000 -UseBasicParsing", "ping localhost", "nslookup example.com", "tar -xzf archive.tar.gz", "unzip build.zip -d build", "Expand-Archive build.zip -DestinationPath build",
    "ffmpeg -i in.mp4 out.gif", "magick convert a.png b.jpg", "sqlite3 app.db .tables", "sqlite3 app.db 'SELECT count(*) FROM users;'", "psql -c 'SELECT 1'", "redis-cli ping", "mongosh --eval 'db.stats()'",
    "kubectl get nodes", "kubectl config current-context", "helm list", "terraform plan", "terraform validate", "aws s3 ls", "gcloud config list", "az account show", "heroku logs --tail", "vercel --prod",
]
SAFE_MORE += [   # read-only git / lint variants: the v3 exam still hesitated on neighbours of these
    "git log", "git log --oneline", "git log --oneline -20", "git log --oneline --graph", "git log -5", "git log -p -1", "git log --since=yesterday", "git branch", "git branch -r", "git branch --list",
    "git branch -vv", "git branch --show-current", "git rev-parse HEAD", "git describe --tags", "git status -s", "git status --short", "git diff --cached", "git diff HEAD~1", "git shortlog -sn",
    "git ls-files", "git tag", "git tag -l", "git remote", "git stash list", "git reflog", "black --check src", "black --diff .", "ruff format --check .", "isort --check .", "flake8 src", "mypy .",
    "npm ci --silent", "npm ci --prefer-offline", "npm audit", "npm ls --depth=0", "yarn install --frozen-lockfile", "pnpm install --frozen-lockfile", "pip check", "pip install -q pytest",
    "pip install --user requests", "python -m pip install requests", "python -m pytest --collect-only", "pytest --collect-only -q", "pytest --co", "pytest -v", "pytest --version",
]
REVIEW_MORE = [
    "git push --force-with-lease origin feature/x", "git reset --hard origin/main", "git clean -fdx", "git checkout .", "rm -rf node_modules", "rm -rf build dist", "Remove-Item -Recurse -Force .\\build",
    "rd /s /q node_modules", "del *.log", "rm *.tmp", "pip uninstall -y numpy", "npm uninstall lodash", "docker rm -f web", "docker compose down -v", "docker system prune -f", "kill 4242", "taskkill /f /pid 4242",
    "Stop-Process -Name python -Force", "sudo apt-get remove nginx", "sudo systemctl stop postgresql", "chmod -R 755 .", "mv build build_old", "sed -i 's/foo/bar/g' src/*.py", "truncate -s 0 app.log", "alembic downgrade base",
    "terraform apply -auto-approve", "kubectl delete deployment web", "npm publish --access public", "cargo publish", "git tag -d v1.0", "git branch -D old-feature",
]


def gen_safety_more(rng: random.Random) -> List[Sample]:
    """More of what an agent really runs on a developer machine, with the same classes and labels as the v2 safety grammar."""
    r = rng.random()
    if r < 0.72:
        cmd = _pick(rng, SAFE_MORE)
        if rng.random() < 0.3:
            cmd = _pick(rng, ["cd {d} && ", "cd {d}; ", ""]).format(d=AR.directory(rng)) + cmd
        return AR._command_sample(rng, cmd, rng.choice([1.0, 1.0, 0.97, 0.95]), rng.uniform(1, 14), "execute_immediately")
    if r < 0.88:
        return AR._command_sample(rng, AR._fill_cmd(rng, _pick(rng, AR.LOOKALIKE_TEMPLATES)), rng.choice([1.0, 0.97, 0.95]), rng.uniform(3, 16), "execute_immediately")
    cmd = _pick(rng, REVIEW_MORE)
    return AR._command_sample(rng, cmd, rng.choice([0.35, 0.4, 0.45]), rng.uniform(30, 58), "request_user_approval")


GENERATORS = [
    ("route16", gen_route16, 60000),
    ("background", gen_background, 26000),
    ("delegate", gen_delegate, 40000),
    ("safety_more", gen_safety_more, 22000),
]


# ------------------------------------------------------------------------- gaps the real-traffic exam found in v3
# (generic patterns, not the exam's sentences: finetune.py also drops anything that repeats an exam request)
REVIEW_NOUNS = ["план", "отчёт", "анализ", "ревью", "результаты", "предложение", "резюме", "диагностику", "аудит", "отчёт аудита", "plan", "report", "review", "analysis"]
REVIEW_VERBS = ["оцени", "проанализируй", "посмотри и оцени", "изучи и оцени", "проверь и оцени", "посмотри ревью:", "посмотри отчёт", "прочитай и оцени", "оцени теперь", "evaluate the", "review the", "assess the", "реализуй всё, что описано в"]
PROSE_NUMBERS = [
    "на сколько ты оценишь это из 100?", "есть что-то, что делается за 1-2 часа?", "оцени от 1 до 10", "это займёт часов 3-4?", "через 5 минут будет готово?", "сколько это займёт, примерно 2-3 дня?",
    "дай оценку из 10", "около 3-4 вариантов есть?", "как ты оценишь из 5?", "rate it from 1 to 10", "is there anything that takes 1-2 hours?", "how long, 2-3 days?",
    "поставь оценку от 0 до 100", "в 2-3 предложениях объясни", "в двух-трёх словах", "что лучше из 3 вариантов?", "топ-5 причин?", "дай 3-5 идей",
]
COMPARE = ["сравни {a} и {b}", "сравни ии модели {a} и {b}", "в чём разница между {a} и {b}", "что лучше: {a} или {b}", "compare {a} and {b}", "{a} против {b}: что выбрать", "сравни {a} с {b} по скорости"]
COMPARE_THINGS = ["nanbeige4.1", "qwen3 и llama3", "react", "vue", "postgres", "mysql", "rust", "go", "gpt-4o", "claude sonnet", "mistral-7b", "gemma2", "redis", "kafka", "tailwind", "bootstrap", "numpy", "polars"]
BROWSER_WINDOW = ["открой окно браузера", "открой видимое окно браузера", "открой браузер", "open a browser window", "открой окно браузера, я сам зайду", "запусти браузер чтобы я мог войти", "покажи окно браузера"]


BS = chr(92)
WIN_ROOTS = [BS.join(["C:", "Users", "Anna", ""]), BS.join(["D:", "projects", ""]), BS.join(["C:", "work", ""]), BS.join(["E:", "repo", ""])]
WIN_LEAVES = ["app", "game", "My Proj" + BS + "src", "tools"]


def _win_doc(rng: random.Random, ext: str) -> str:
    return BS.join(["C:", "Users", "Anna", "Docs", _pick(rng, AR.STEMS) + ext])


def gen_real_gaps(rng: random.Random) -> List[Sample]:
    cwd = _pick(rng, AR.WORKDIRS)
    ctx = make_context(rng) if rng.random() < 0.4 else None
    r = rng.random()
    key = None
    if r < 0.20:   # a shell `cd`: a command, not a directory listing
        d = AR.directory(rng) if rng.random() < 0.6 else AR.filepath(rng).rsplit("/", 1)[0] or "src"
        if rng.random() < 0.5:
            d = _pick(rng, WIN_ROOTS) + _pick(rng, WIN_LEAVES)
        request = _pick(rng, ["cd {d}", "cd {d}", "cd \"{d}\"", "cd /d {d}", "Set-Location {d}", "pushd {d}"]).format(d=d)
        tool, key = "run_command", request   # the extracted argument is the whole command (v3.1 returned just "cd")
        request = request.replace('"', "'") if request.startswith("cd '") else request
    elif r < 0.45:  # asked to judge a document: the LLM reads and thinks
        path = rng.choice([AR.filepath(rng), _win_doc(rng, ".md"), _win_doc(rng, ".txt"), AR.filepath(rng)])
        verb = _pick(rng, REVIEW_VERBS)
        noun = "" if verb.endswith(("в", ":")) or "ревью" in verb or "отчёт" in verb else " " + _pick(rng, REVIEW_NOUNS)
        request = f"{verb}{noun}: {path}" if rng.random() < 0.7 else f"{verb}{noun} {path}"
        tool = "UNFAMILIAR"
    elif r < 0.60:
        request, tool = _pick(rng, PROSE_NUMBERS), "NO_TOOL"
    elif r < 0.78:
        request, tool = _pick(rng, COMPARE).format(a=_pick(rng, COMPARE_THINGS), b=_pick(rng, COMPARE_THINGS)), "UNFAMILIAR"
    elif r < 0.90:   # the dispatcher only follows browser_open with a URL, so a bare "open a window" is the LLM's job
        request, tool = _pick(rng, BROWSER_WINDOW), "UNFAMILIAR"
    else:            # a path-only message ("C:\...\report.md") says nothing about what to do with it
        request, tool = rng.choice([_win_doc(rng, ".md"), "D:/projects/" + _pick(rng, AR.STEMS) + "/README.md", "D:/notes/" + _pick(rng, AR.STEMS) + ".txt"]), "UNFAMILIAR"
    request = _clean(dress(rng, request, polite=0.12))
    return _bundle(rng, "argent_gaps_v3", build_state(request, ctx, cwd), tool, key)


GENERATORS.append(("real_gaps", gen_real_gaps, 24000))
