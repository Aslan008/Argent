"""Where Leeth 2 still grabs a request that needed the LLM -- taught as clusters, with their close neighbours that DO deserve a tool.

The real-traffic exam (scripts/leeth2_real_exam.py) counts a FALSE DISPATCH when the fast path takes a request that was not one clean
tool call.  v3.1 and the merged models share the same blind spots, each a *kind* of request rather than a sentence:

  * "open <site name>" with no address           the LLM has to find the URL; a browser_open without one goes nowhere
  * "launch <app / game / demo / server>"        no shell command is given; which one is the LLM's job
  * a vague "check <something>"                  check what, how?
  * "search the web" with nothing to search for
  * an anaphora with nothing to refer to         "open it" when no file was mentioned
  * the user REPORTING what they did             "I restarted it" is not a request
  * talk ABOUT a tool                            "and if it is the plain read_file command?"
  * arithmetic buried in chat or in LaTeX
  * "don't run anything, just tell me"

Each cluster below has a delegate side (UNFAMILIAR / NO_TOOL) and, where the boundary matters, a tool side with the same words but
a concrete target (a real URL, an explicit command, a file the session just touched), so the model learns the *difference* rather
than "never fast-path anything with these verbs".  Slots are filled from pools of their own; finetune.py also drops any sample
that repeats one of the exam's requests.
"""

from __future__ import annotations

import random
import re
from typing import List, Optional

from ...schema import Sample
from . import argent_router as AR
from .argent_context import _bundle, _clean, build_state, make_context
from .argent_router import _pick, dress

SITES = ["epic games", "steam", "ozon", "avito", "wildberries", "kaspi", "hh", "github", "яндекс карты", "мой банк", "почту", "ютуб", "википедию", "reddit", "пинтерест", "гугл диск",
         "twitch", "linkedin", "habr", "aliexpress", "booking", "tripadvisor", "notion", "figma", "telegram web", "хабр", "кинопоиск", "2гис", "сбербанк онлайн", "госуслуги"]
DOMAINS = ["example.com", "github.com/torvalds/linux", "docs.python.org", "news.ycombinator.com", "kaspi.kz/shop", "habr.com", "en.wikipedia.org/wiki/Rust", "pypi.org/project/requests",
           "stackoverflow.com", "arxiv.org/abs/1706.03762", "localhost:3000", "127.0.0.1:8000/docs"]
APPS = ["epic games", "steam", "discord", "блокнот", "калькулятор", "unity hub", "obs", "spotify", "telegram", "chrome", "godot", "blender", "photoshop", "проводник", "диспетчер задач",
        "visual studio", "docker desktop", "postman", "vlc", "ableton"]
DEMOS = ["демо платформера", "тестовую версию игры", "прототип", "пример из документации", "мой проект", "последнюю сборку", "лаунчер", "эмулятор", "сервер", "бота", "игру", "приложение", "тесты в фоне"]
VAGUE_TARGETS = ["сеть", "результат", "реализацию", "работоспособность", "новую версию", "V7", "V3", "обновление", "интернет", "систему",
                 "последние изменения", "состояние проекта", "сборку", "корректность", "правильность", "готовность"]
VAGUE_TARGETS_EN = ["the network", "the result", "the implementation", "whether it works", "the new version", "the update", "the build", "the system", "the latest changes"]
STATEMENTS = ["перезапустил ещё раз", "я перезапустил {a}", "я всё починил", "обновил драйвера", "готово, я закончил", "я запустил сервер", "только что сохранил файл", "сделал, как ты сказал",
              "я нажал на кнопку, ничего не произошло", "я переустановил зависимости", "я закрыл все окна", "сеть вернулась", "я проверил сам, всё работает", "вроде заработало",
              "перезагрузил компьютер", "поставил обновление", "я вышел из аккаунта", "включил vpn"]
TOOL_NAMES = ["read_file", "run_command", "grep_search", "list_directory", "browser_open", "calculate", "write_file", "search_files", "replace_in_file"]
TOOL_TALK = ["а если обычная команда {t}", "что делает {t}?", "зачем тебе {t}", "почему ты вызвал {t}", "не используй {t}", "для чего нужен {t}", "в чём разница между {t} и {u}",
             "объясни, как работает {t}", "ты можешь вызывать {t}?", "what does {t} do?", "don't call {t} here", "why did you use {t}?"]
FILE_STATEMENTS = ["так я положил подсказку в {f}", "я сохранил заметки в {f}", "{f} уже обновлён", "в {f} лежит инструкция", "я переписал {f} вручную", "кстати, {f} я не трогал",
                   "{f} теперь в другой папке", "обрати внимание: в {f} всё по-новому"]
AREA_READS = ["прочитай {d} в папке проекта", "посмотри {d} в проекте", "найди {d} где-нибудь в репозитории", "глянь {d} в текущей папке", "открой {d} из проекта", "покажи {d} из корня"]
AREA_WHAT = ["ТЗ", "конфиг", "настройки", "инструкцию", "описание задачи", "заметки", "лог ошибок", "договор", "список задач", "схему базы"]
NO_RUN = ["{r}. ничего не запускай, просто {s}", "{r}, но команды не выполняй — только объясни", "не трогай файлы и ничего не запускай: {r}", "{r} (без запуска инструментов, только ответ)",
          "только отвечай словами, без инструментов: {r}", "don't run anything, just answer: {r}", "{r} — но ничего не меняй, просто расскажи"]
NO_RUN_BASE = ["что ты думаешь про этот план", "в чём может быть причина", "как бы ты это сделал", "что лучше выбрать", "какие тут риски", "как это проверить", "объясни ход мысли"]
NO_RUN_END = ["укажи, что нужно проверить", "скажи своё мнение", "опиши шаги", "перечисли варианты", "объясни почему"]
MATH_CHAT = ["pytest крутит другой агент, не обращай на него внимания. А теперь давай проверим. Сколько будет {e}?", "пока я пью чай, посчитай в голове: {e}", "кстати, раз уж ты здесь — {e}? и как ты это посчитал?",
             "не запускай ничего, я просто хочу понять, как ты считаешь: {e}", "в прошлый раз ты ошибся в арифметике. проверим? {e}", "мне нужно знать {e}, но объясни решение по шагам, а не просто число",
             "смотри, у меня тут спор с другом: {e}. рассуди нас и объясни"]
LATEX = ["\\frac{{{a}}}{{{b}}} + \\sqrt{{{c}}} - {d}", "(((({a} \\cdot {b} - {c}) : {d} + \\sqrt{{{e}}}) \\cdot 3 - 20) : 2 + 55 \\cdot (10 - 2^3))", "\\int_{{0}}^{{{a}}} x^{{{b}}} dx + {c}",
         "\\sum_{{i=1}}^{{{a}}} i^{{{b}}} - {c}", "\\lim_{{x \\to {a}}} \\frac{{x^{{{b}}} - 1}}{{x - 1}}"]
GIT_STATUS_PHRASES = ["покажи статус git", "покажи статус гита", "что в git status", "git status", "какой статус у репозитория (git status)", "show the git status", "покажи git diff --stat"]
SEARCH_BARE = ["найди в интернете", "загугли", "поищи в сети", "search the web", "найди информацию", "поищи в интернете, пожалуйста", "поищи", "найди в гугле", "look it up online"]
SEARCH_WITH = ["найди в интернете {q}", "загугли {q}", "поищи в сети {q}", "search the web for {q}", "найди информацию про {q}"]
SEARCH_Q = ["лучшие шрифты для кода", "как настроить nginx", "курс евро", "рецепт борща", "rust vs go", "python 3.13 новинки", "отзывы о ноутбуке lenovo", "погода в Алматы", "docker compose volumes"]
EXPRS = ["12*7", "1234+4321-1+9*8", "(3+4)*5", "100-123", "2+2*2", "15% от 240", "2**10", "45/9+7", "999-111", "7*8-6"]
SKILL_LEADS = ["используй", "примени", "запусти скилл"]


OPEN_FIRST = ["Открой страницу {d} в моём браузере", "Открой {d} в открытом браузере", "зайди на {d} в видимом окне", "перейди на {d}", "открой https://{d}", "открой {d}"]
OPEN_FIRST_EN = ["open {d} in my browser", "go to https://{d}", "open https://{d}"]
OPEN_THEN_EN = ["{r} and then run {act} via {fw}", "{r} and take a screenshot of it", "{r} and fill in the form there", "{r} and log in", "{r} and click the subscribe button",
                "{r} and extract every heading with {fw}"]
MODELS = ["Llama 3 8B", "модель Mistral 7B в формате gguf", "Stable Diffusion XL", "Cyberpunk 2077", "эту игру", "Blender 4.2", "квантованную модель на 27B", "Whisper large", "Unreal Engine 5"]
RUNNERS = ["koboldcpp", "ollama", "LM Studio", "Steam", "wine", "docker", "llama.cpp", "на встроенной графике"]
TRACKED = ["страницу отслеживания запуска", "панель мониторинга", "документацию по API", "отчёт о сборке", "страницу с логами", "дашборд с метриками", "страницу статуса сервиса"]
FEASIBILITY = ["смогу ли я запустить {w} в {u}?", "потянет ли мой пк {w}?", "какая ссылка на {t}?", "где посмотреть {t}?", "как мне запустить {w} в {u}?", "можно ли запустить {w} без видеокарты?",
               "подойдёт ли {u} для {w}?", "can I run {w} in {u}?", "what is the link to {t}?", "where can I find {t}?", "will my laptop handle {w}?"]
DID_NOT = [("запускал", "запустить", "тесты"), ("проверял", "проверить", "сборку"), ("обновлял", "обновить", "зависимости"), ("отправлял", "отправить", "отчёт"),
           ("сохранял", "сохранить", "изменения"), ("перезапускал", "перезапустить", "сервер"), ("удалял", "удалить", "старые логи"), ("читал", "прочитать", "инструкцию"),
           ("запускал", "запустить", "обучение на Kaggle"), ("делал", "сделать", "резервную копию")]
REPROACH = ["ты разве не {past} {obj}? Я же просил", "почему ты не {past} {obj}?", "я же просил {inf} {obj}", "ты забыл {inf} {obj}", "а {obj} ты {past}?", "ты так и не {past} {obj}",
            "didn't I ask you to {inf_en} the {obj_en}?", "you never {past_en} the {obj_en}"]
REPROACH_EN = {"тесты": ("run", "ran", "tests"), "сборку": ("check", "checked", "build"), "зависимости": ("update", "updated", "dependencies"), "отчёт": ("send", "sent", "report"),
               "изменения": ("save", "saved", "changes"), "сервер": ("restart", "restarted", "server"), "старые логи": ("delete", "deleted", "old logs"),
               "инструкцию": ("read", "read", "instructions"), "обучение на Kaggle": ("start", "started", "training run"), "резервную копию": ("make", "made", "backup")}


# The read-only / single-call routes whose request makers exist; the keys are the router's own labels (AR.MAKERS), so a renamed label cannot drift here.
MULTI_TOOLS = [t for t in AR.MAKERS if t not in ("write_file", "replace_in_file", "browser_click", "browser_type", "browser_upload_file", "ask_user_questions", "NO_TOOL", "UNFAMILIAR")]
MULTI_JOIN = [". ", ". Потом ", " и ", "; ", ", а ещё ", ". А ещё ", ", затем ", ", и заодно "]
MULTI_JOIN_EN = [". Then ", " and also ", ". Also ", ", and then "]


def _chat(rng: random.Random, request: str, tool: str, ctx: Optional[dict] = None, key: Optional[str] = None, polite: float = 0.1) -> List[Sample]:
    request = _clean(dress(rng, request, polite=polite))
    return _bundle(rng, "argent_hard_v4", build_state(request, ctx, _pick(rng, AR.WORKDIRS)), tool, key)


def _extra(rng: random.Random, ctx: Optional[dict]) -> List[Sample]:
    """Clusters the first real-traffic run left over: "open a page AND do the rest", questions ABOUT running / finding something, reproaches."""
    k = rng.random()
    d = _pick(rng, DOMAINS)
    fill = dict(act=_pick(rng, AR.BROWSER_ACTS), fw=_pick(rng, AR.BROWSER_TOOLS), x=_pick(rng, AR.BROWSER_THINGS))
    if k > 0.78:                                                # two requests in one message: each is one clean call, together they are not
        a, b = rng.sample(MULTI_TOOLS, 2)
        first, second = AR.MAKERS[a][0](rng)[0].rstrip(".?!"), AR.MAKERS[b][0](rng)[0].rstrip(".?!")
        join = _pick(rng, MULTI_JOIN if re.search("[а-яё]", second, re.I) else MULTI_JOIN_EN)
        return _chat(rng, first + join + second[:1].lower() + second[1:] + ("?" if rng.random() < 0.3 and b in ("calculate", "read_file") else ""), "UNFAMILIAR", ctx)
    k = k / 0.78 if k <= 0.78 else k
    if k < 0.45:                                                # the second step is a job of its own -> the LLM (the open itself is one call, the rest is not)
        if rng.random() < 0.2:
            return _chat(rng, _pick(rng, OPEN_THEN_EN).format(r=_pick(rng, OPEN_FIRST_EN).format(d=d), **fill), "UNFAMILIAR", ctx)
        return _chat(rng, _pick(rng, AR.COMPOUND_JOB["browser_open"]).format(r=_pick(rng, OPEN_FIRST).format(d=d), **fill), "UNFAMILIAR", ctx)
    if k < 0.58:                                                # the second step only reports -> still one browser_open
        return _chat(rng, _pick(rng, AR.COMPOUND_REPORT["browser_open"]).format(r=_pick(rng, OPEN_FIRST).format(d=d)), "browser_open", ctx, d)
    if k < 0.80:                                                # a question about doing / finding it: an answer, not a command
        return _chat(rng, _pick(rng, FEASIBILITY).format(w=_pick(rng, MODELS), u=_pick(rng, RUNNERS), t=_pick(rng, TRACKED)), "NO_TOOL", ctx)
    past, inf, obj = _pick(rng, DID_NOT)                        # the user reproaches the agent for something it did not do: it has to look at what happened
    en = REPROACH_EN[obj]
    t = _pick(rng, REPROACH).format(past=past, inf=inf, obj=obj, inf_en=en[0], past_en=en[1], obj_en=en[2])
    return _chat(rng, t, "UNFAMILIAR", ctx)


def gen_hardening(rng: random.Random) -> List[Sample]:
    r = rng.random()
    ctx = make_context(rng) if rng.random() < 0.45 else None
    if r < 0.14:
        return _extra(rng, ctx)
    r = (r - 0.14) / 0.86                                       # the older clusters below keep their relative weights
    # ---- open <site> : no address -> LLM ; with an address -> browser_open
    if r < 0.13:
        if rng.random() < 0.15:
            site = _pick(rng, ["epic games", "steam", "github", "twitch", "reddit", "linkedin", "notion", "figma", "booking"])
            t = _pick(rng, ["go to {s}", "open the {s} website", "take me to {s}", "can you open {s}?", "check what is new on {s}"])
        else:
            site = _pick(rng, SITES)
            t = _pick(rng, ["открой сайт {s}", "зайди на {s}", "перейди на сайт {s}", "открой {s} в браузере", "можешь открыть {s}?", "посмотри, что сейчас на {s}",
                            "зайди на вакансии на сайте {s}", "найди на {s} акции и открой их", "проверь результаты на {s}", "открой раздел помощи на {s}"])
        return _chat(rng, t.format(s=site), "UNFAMILIAR", ctx)
    if r < 0.18:
        d = _pick(rng, DOMAINS)
        t = _pick(rng, ["открой {d}", "открой https://{d}", "зайди на https://{d}", "open https://{d}", "перейди по ссылке https://{d}", "открой страницу {d} в браузере"])
        return _chat(rng, t.format(d=d), "browser_open", ctx, d)
    # ---- launch an app / game / demo without a command
    if r < 0.30:
        what = _pick(rng, APPS + DEMOS)
        t = _pick(rng, ["запусти {w} на моём пк", "запусти {w}", "включи {w}", "запусти {w} в фоне", "открой {w}", "start the {w}", "можешь запустить {w}?", "запусти демо {w} с python 3.11",
                        "запусти {w}, я хочу посмотреть", "поднимай {w}"])
        return _chat(rng, t.format(w=what), "UNFAMILIAR", ctx)
    if r < 0.34:
        cmd = _pick(rng, ["pytest -x", "npm run dev", "python main.py", "cargo test", "docker compose up -d", "uvicorn app:app --reload", "make build", "go test ./...", "python -m http.server 8000"])
        t = _pick(rng, ["запусти {c}", "выполни {c}", "run {c}", "запусти команду {c}", "прогони {c}"])
        return _chat(rng, t.format(c=cmd), "run_command", ctx, cmd)
    # ---- vague checks
    if r < 0.44:
        if rng.random() < 0.15:
            t = _pick(rng, ["check {w}", "have a look at {w}", "make sure {w} is fine", "verify {w}"]).format(w=_pick(rng, VAGUE_TARGETS_EN))
        else:
            t = _pick(rng, ["проверь {w}", "проверь, пожалуйста, {w}", "посмотри {w}", "глянь {w}", "оцени {w}", "убедись, что {w} в порядке", "проверь {w} и расскажи"]).format(w=_pick(rng, VAGUE_TARGETS))
        return _chat(rng, t, "UNFAMILIAR", ctx)
    # ---- search with nothing to search for / with a query
    if r < 0.49:
        return _chat(rng, _pick(rng, SEARCH_BARE), "UNFAMILIAR", ctx)
    if r < 0.53:
        q = _pick(rng, SEARCH_Q)
        return _chat(rng, _pick(rng, SEARCH_WITH).format(q=q), "UNFAMILIAR", ctx)           # live information -> the LLM (as in gen_delegate's WEB_QUESTIONS)
    # ---- anaphora with nothing to refer to
    if r < 0.58:
        c = make_context(rng, target=False, tasks=False, prev=rng.random() < 0.5)
        return _chat(rng, _pick(rng, ["открой его", "покажи его", "прочитай его", "что в нём?", "open it", "show it", "покажи этот файл", "прочитай этот файл", "а теперь открой тот файл"]), "UNFAMILIAR", c)
    # ---- the user reports what they did
    if r < 0.66:
        t = _pick(rng, STATEMENTS).replace("{a}", _pick(rng, ["Argent", "сервер", "компьютер", "роутер", "проект"]))
        tool = "UNFAMILIAR" if rng.random() < 0.3 and "перезапустил" in t else "NO_TOOL"
        if tool == "UNFAMILIAR":
            t += _pick(rng, [". можешь проверить работоспособность?", ", проверь пожалуйста", ", посмотри что получилось"])
        return _chat(rng, t, tool, ctx)
    # ---- talk about a tool
    if r < 0.71:
        t = _pick(rng, TOOL_TALK).format(t=_pick(rng, TOOL_NAMES), u=_pick(rng, TOOL_NAMES))
        return _chat(rng, t, "NO_TOOL", ctx)
    # ---- statements about files
    if r < 0.76:
        t = _pick(rng, FILE_STATEMENTS).format(f=rng.choice(["AGENTS.md", "README.md", "notes.txt", "config.yaml", "TODO.md", "plan.md", "settings.json"]))
        return _chat(rng, t, "NO_TOOL", ctx)
    # ---- "read the X in the project": no concrete path
    if r < 0.82:
        t = _pick(rng, AREA_READS).format(d=_pick(rng, AREA_WHAT))
        return _chat(rng, t, "UNFAMILIAR", ctx)
    # ---- arithmetic in chat / LaTeX -> the LLM; clean arithmetic -> calculate
    if r < 0.89:
        if rng.random() < 0.55:
            e = _pick(rng, EXPRS)
            return _chat(rng, _pick(rng, MATH_CHAT).format(e=e), "UNFAMILIAR", ctx)
        tex = _pick(rng, LATEX).format(a=rng.randint(2, 30), b=rng.randint(2, 9), c=rng.randint(2, 60), d=rng.randint(2, 9), e=rng.choice([16, 25, 36, 49, 64]))
        return _chat(rng, rng.choice(["{x}", "реши: {x}", "посчитай {x}", "вычисли \\({x}\\)", "чему равно {x}?"]).format(x=tex), "UNFAMILIAR", ctx)
    if r < 0.93:
        e = _pick(rng, EXPRS)
        t = _pick(rng, ["сколько будет {e}?", "посчитай {e}", "вычисли {e}", "{e} = ?", "calculate {e}", "what is {e}"])
        return _chat(rng, t.format(e=e), "calculate", ctx, e)
    # ---- "don't run anything, just ..."
    if r < 0.97:
        t = _pick(rng, NO_RUN).format(r=_pick(rng, NO_RUN_BASE), s=_pick(rng, NO_RUN_END))
        return _chat(rng, t, "NO_TOOL", ctx)
    # ---- git status in words
    return _chat(rng, _pick(rng, GIT_STATUS_PHRASES), "run_command", ctx)


GENERATORS = [("hardening", gen_hardening, 48000)]
