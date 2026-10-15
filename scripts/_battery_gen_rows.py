# -*- coding: utf-8 -*-
"""scripts/_battery_gen_rows.py — таргетированные строки по категориям промахов батареи.

Категории из _battery_miss_diff (прод, 36 промахов):
  1. git/python run_command -> UNFAMILIAR
  2. grep_search <-> find_by_name <-> list_directory путаница
  3. find_by_name -> UNFAMILIAR
  4. list_directory -> UNFAMILIAR
  5. calculate -> UNFAMILIAR
  6. browser_open -> browser_click/UNFAMILIAR
  7. NO_TOOL (эссе/рассуждения) -> run_command
Формат: state/question/target_choice/confuser — как docker-строки.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(".argent/kaggle/dataset_v2/battery_rows.jsonl")
TQ = "Какой инструмент Argent должен быть вызван в первую очередь?"


def state(q):
    return (f'Incoming user request in Argent session:\n"{q}"\n'
            f"Working directory: C:/Users/user/Desktop/Argent")


rows = []  # (query, target, confuser)


def add(q, target, confuser=None):
    rows.append((q, target, confuser))


# ── 1. git/python run_command (прод промахивается в UNFAMILIAR/ask/find) ────
git_cmds = [
    ("выполни git log --oneline -5", "git log --oneline -5"),
    ("покажи последние 5 коммитов", "git log --oneline -5"),
    ("проверь статус репозитория", "git status"),
    ("какой текущий статус git", "git status"),
    ("сделай коммит с сообщением fix", 'git commit -m "fix"'),
    ("закамить изменения с текстом fix", 'git commit -m "fix"'),
    ("запусти python scripts/mine_shadow.py", "python scripts/mine_shadow.py"),
    ("запусти python scripts/train.py", "python scripts/train.py"),
    ("исполни python main.py", "python main.py"),
    ("проверь установленные пакеты pip", "pip list"),
    ("покажи список pip пакетов", "pip list"),
    ("выполни git diff", "git diff"),
    ("покажи git diff", "git diff"),
    ("git branch", "git branch"),
    ("покажи ветки git", "git branch"),
    ("запусти pytest -k test_router", "pytest -k test_router"),
    ("прогони тесты pytest", "pytest"),
    ("запусти cargo check --release", "cargo check --release"),
    ("запусти npm run build", "npm run build"),
    ("выполни docker ps -a", "docker ps -a"),
    ("покажи все контейнеры docker", "docker ps -a"),
    ("запусти docker images", "docker images"),
    ("покажи образы docker", "docker images"),
    ("docker logs app", "docker logs app"),
    ("покажи логи контейнера app", "docker logs app"),
]
for q, cmd in git_cmds:
    add(q, "run_command")

# ── 2. grep_search (путаница с find_by_name/read_file/NO_TOOL) ──────────────
grep_rows = [
    ("найди где используется replace_in_file", "replace_in_file"),
    ("найди где используется dispatcher", "dispatcher"),
    ("где в коде вызывается decide_batch", "decide_batch"),
    ("где в коде вызывается log_decision", "log_decision"),
    ("поиск по коду: def finetune_from_feedback", "def finetune_from_feedback"),
    ("поиск по коду: class LeethDispatcher", "class LeethDispatcher"),
    ("найди импорты torch в src", "import torch"),
    ("найди упоминания UNFAMILIAR в проекте", "UNFAMILIAR"),
    ("найди функцию arbitrate в проекте", "arbitrate"),
    ("поищи строку DENY_PATTERNS", "DENY_PATTERNS"),
    ("найди все использования try_fast_dispatch", "try_fast_dispatch"),
    ("где определяется ALLOWED_FAST_TOOLS", "ALLOWED_FAST_TOOLS"),
    ("найди вхождения feedback в src", "feedback"),
    ("поищи mark_skipped в коде", "mark_skipped"),
    ("найди где определяется ROUTER_MODE", "ROUTER_MODE"),
    ("найди упоминания onnx в проекте", "onnx"),
]
for q, pat in grep_rows:
    add(q, "grep_search", confuser="find_by_name")

# ── 3. find_by_name (прод: UNFAMILIAR/list_directory/search_files) ──────────
find_rows = [
    ("найди все файлы с расширением .cs", "*.cs"),
    ("найди все файлы с расширением .py", "*.py"),
    ("найди все тестовые файлы test_*.py", "test_*.py"),
    ("найди файл leeth_vocab.json", "leeth_vocab.json"),
    ("найди файл settings.json в проекте", "settings.json"),
    ("найди файл config.py", "config.py"),
    ("найди файл runtime.py", "runtime.py"),
    ("найди файл prod_battery.json", "prod_battery.json"),
    ("найди все readme файлы", "readme"),
    ("найди файл dispatcher.py", "dispatcher.py"),
]
for q, name in find_rows:
    add(q, "find_by_name", confuser="grep_search")

# ── 4. list_directory ────────────────────────────────────────────────────────
ls_rows = [
    ("покажи файлы в .argent/artifacts", ".argent/artifacts"),
    ("покажи содержимое папки src", "src"),
    ("покажи файлы в scripts", "scripts"),
    ("что в папке .argent/models", ".argent/models"),
    ("покажи содержимое директории src/leeth", "src/leeth"),
    ("список файлов в корне проекта", "."),
]
for q, d in ls_rows:
    add(q, "list_directory", confuser="find_by_name")

# ── 5. calculate ─────────────────────────────────────────────────────────────
calc_rows = [
    ("переведи 1024 МБ в гигабайты", "1024 / 1024"),
    ("сколько гигабайт в 512 МБ", "512 / 1024"),
    ("посчитай 15 процентов от 800", "800 * 0.15"),
    ("вычисли корень из 144", "sqrt(144)"),
    ("сколько будет 2 в степени 10", "2 ** 10"),
    ("посчитай 128 * 64", "128 * 64"),
    ("сколько будет 10-5+2?", "10-5+2"),
    ("вычисли 3600 секунд в часах", "3600 / 3600"),
]
for q, expr in calc_rows:
    add(q, "calculate", confuser="run_command")

# ── 6. browser_open ──────────────────────────────────────────────────────────
web_rows = [
    ("загрузи страницу github.com/QwenLM", "https://github.com/QwenLM"),
    ("открой сайт docs.python.org", "https://docs.python.org"),
    ("открой https://example.com", "https://example.com"),
    ("загрузи страницу pytorch.org", "https://pytorch.org"),
]
for q, url in web_rows:
    add(q, "browser_open", confuser="browser_click")

# ── 7. NO_TOOL (эссе/рассуждения — прод диспатчит run_command) ───────────────
notool_rows = [
    "напиши эссе о программировании",
    "расскажи о преимуществах Rust",
    "объясни разницу между TCP и UDP",
    "напиши стихотворение про код",
    "сравни подходы к тестированию",
]
for q in notool_rows:
    add(q, "NO_TOOL", confuser="run_command")

# ── 8. итерация 2: конфьюзер=UNFAMILIAR для сваливаний + защита read ────────
fixup = [
    # read-запросы, перетянутые run_command-блоком итерации 1
    ("прочитай AGENTS.md", "read_file", "run_command"),
    ("прочитай README.md", "read_file", "run_command"),
    ("прочитай main.py", "read_file", "run_command"),
    # сваливания в UNFAMILIAR — конфьюзер именно UNFAMILIAR
    ("проверь статус репозитория", "run_command", "UNFAMILIAR"),
    ("сделай коммит с сообщением fix", "run_command", "UNFAMILIAR"),
    ("найди все файлы с расширением .cs", "find_by_name", "UNFAMILIAR"),
    ("покажи файлы в .argent/artifacts", "list_directory", "UNFAMILIAR"),
    ("загрузи скриншот в форму", "browser_upload_file", "UNFAMILIAR"),
    ("задай вопрос о выборе фреймворка", "ask_user_questions", "UNFAMILIAR"),
    ("напиши эссе о программировании", "NO_TOOL", "UNFAMILIAR"),
    # прочие точечные
    ("уточни у пользователя, какой вариант нужен", "ask_user_questions", "run_command"),
    ("продолжай", "NO_TOOL", "run_command"),
]
for q, t, cf in fixup:
    add(q, t, confuser=cf)

# ── 9. итерация 3: 5 регрессий cand9 (конфьюзер = наблюдаемый промах) ───────
# из _battery_miss_diff (cand, 5 регрессий): все — перетяг в run_command/NO_TOOL
iter3 = [
    # write_file: cand сваливает в NO_TOOL
    ("сохрани отчёт в отчёт.md", "write_file", "NO_TOOL"),
    ("сохрани список файлов в files.txt", "write_file", "NO_TOOL"),
    ("запиши итоги в summary.md", "write_file", "NO_TOOL"),
    ("сохрани заметку в notes.md", "write_file", "NO_TOOL"),
    ("запиши результат в output.txt", "write_file", "NO_TOOL"),
    # ask_user_questions: cand диспатчит run_command
    ("уточни у пользователя, какой вариант нужен", "ask_user_questions", "run_command"),
    ("задай вопрос о выборе фреймворка", "ask_user_questions", "run_command"),
    ("спроси пользователя, какую БД использовать", "ask_user_questions", "run_command"),
    ("уточни требования у пользователя", "ask_user_questions", "run_command"),
    ("задай пользователю вопрос о приоритетах", "ask_user_questions", "run_command"),
    # NO_TOOL (анекдоты/стихи): cand диспатчит run_command
    ("расскажи анекдот", "NO_TOOL", "run_command"),
    ("расскажи шутку про программистов", "NO_TOOL", "run_command"),
    ("напиши стихотворение про осень", "NO_TOOL", "run_command"),
    ("расскажи что-нибудь весёлое", "NO_TOOL", "run_command"),
    # UNFAMILIAR (рассуждения о причине): cand диспатчит run_command
    ("почему тесты падают?", "UNFAMILIAR", "run_command"),
    ("почему сборка ломается?", "UNFAMILIAR", "run_command"),
    ("объясни причину падения тестов", "UNFAMILIAR", "run_command"),
    ("как исправить эту ошибку компиляции?", "UNFAMILIAR", "run_command"),
]
for q, t, cf in iter3:
    add(q, t, confuser=cf)

# ── запись ───────────────────────────────────────────────────────────────────
class_to_idx = json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8"))
with open(OUT, "w", encoding="utf-8") as f:
    for q, target, confuser in rows:
        rec = {
            "state": state(q), "question": TQ, "question_type": "choice",
            "target_choice": target,
            "confuser_choice": confuser,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"записано {len(rows)} строк -> {OUT}")
from collections import Counter
print(Counter(r[1] for r in rows))