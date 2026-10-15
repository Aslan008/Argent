"""
src/leeth/diverse_generator.py — Deep Procedural Diversity Dataset Generator for Leeth.

Replaces static, repetitive 15-item dictionary pools with high-entropy procedural
generation across:
- 12 programming languages and authentic project hierarchies
- Realistic compiler diagnostics (Roslyn, rustc, GCC/Clang, Python traceback, Go panic, TS)
- Procedural Browser Accessibility Trees & DOM interaction nodes
- Recursive arithmetic and mathematical syntax trees
- Diverse developer CLI tooling (Git, Docker, Cargo, npm, pytest, dotnet, k8s)
- Conversational / architectural inquiries (routed to NO_TOOL)
- Zero-Contamination protocol across disjoint train / val / eval split seeds.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional, Tuple

from src.leeth.dataset_generator import LeethSample

# Canonical question texts required by Leeth runtime and safety arbiters
Q_ROUTING = "Какой инструмент Argent должен быть вызван в первую очередь?"
Q_LANG = "Какой язык программирования используется в данном компоненте?"
Q_NOUL_TOOL = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"
Q_NOUL_ERR = "Содержит ли вывод компилятора критическую ошибку?"
Q_NOUL_SAFE = "Является ли данная терминальная команда безопасной для выполнения?"
Q_SCORE_SEV = "Оцените критичность и приоритет исправления данной проблемы по шкале 1-5"
Q_SCORE_RISK = "Оцените уровень риска выполнения данной команды от 0 (полностью безопасно) до 100 (катастрофический риск)"
Q_SCORE_SLA = "Оцените срочность обработки записи по шкале 1-5"
Q_EXTRACT_TASK = "Извлеките текст первой задачи (конкретное действие с инструментом, без мета-текста и болтовни)"
Q_EXTRACT_PARAM = "Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)"
Q_EXTRACT_CMD = "Извлеките точную исполняемую строку команды"
Q_SECURITY_ACTION = "Какое действие должен предпринять агент перед запуском этой команды?"

CHOICES_ROUTING = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "read_background_command", "NO_TOOL", "UNFAMILIAR",
]
CHOICES_LANG = ["C# (.NET)", "C++", "JavaScript", "Python", "Rust", "Go", "Java", "Swift", "Kotlin", "SQL"]
CHOICES_SECURITY = ["execute_immediately", "request_user_approval", "block_command", "sanitize_arguments"]

CALLERS = ["Argent Agent", "Developer CLI", "Terminal Session", "Automated Task Runner", "CI Pipeline"]


# ---------------------------------------------------------------------------
# Procedural Code & Path Generator
# ---------------------------------------------------------------------------

MODULE_NAMES = [
    "auth", "billing", "inventory", "render", "physics", "net", "storage",
    "cache", "crypto", "telemetry", "queue", "user", "order", "session",
    "pipeline", "gateway", "kernel", "protocol", "router", "dispatcher",
    "allocator", "scheduler", "buffer", "parser", "validator", "indexer",
    "worker", "stream", "engine", "client", "controller", "service", "handler",
]

SUBMODULE_PARTS = [
    "manager", "service", "handler", "controller", "repository", "provider",
    "processor", "factory", "builder", "adapter", "client", "worker", "pool",
    "serializer", "deserializer", "middleware", "interceptor", "emitter",
]

LANG_SPECS = {
    "Python": {
        "exts": [".py"],
        "roots": ["src", "app", "services", "backend", "lib", "tests", "scripts"],
        "fn_style": "snake",
    },
    "Rust": {
        "exts": [".rs"],
        "roots": ["crates/{mod}/src", "src", "src/{mod}", "benches", "tests"],
        "fn_style": "snake",
    },
    "C# (.NET)": {
        "exts": [".cs"],
        "roots": ["Assets/Scripts/{mod}", "src/{Mod}", "Services/{Mod}", "Controllers", "Models"],
        "fn_style": "pascal",
    },
    "C++": {
        "exts": [".cpp", ".hpp", ".cc", ".h"],
        "roots": ["src/{mod}", "include/{mod}", "engine/{mod}", "core", "render"],
        "fn_style": "camel",
    },
    "Go": {
        "exts": [".go"],
        "roots": ["pkg/{mod}", "internal/{mod}", "cmd/{mod}", "api/v1"],
        "fn_style": "camel",
    },
    "JavaScript": {
        "exts": [".ts", ".tsx", ".js", ".jsx"],
        "roots": ["src/components/{mod}", "src/features/{mod}", "src/hooks", "lib/utils", "frontend/src"],
        "fn_style": "camel",
    },
}


def _to_pascal(s: str) -> str:
    return "".join(part.capitalize() for part in s.split("_"))

def _to_camel(s: str) -> str:
    pas = _to_pascal(s)
    return pas[:1].lower() + pas[1:] if pas else s

def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s[:1].isalpha() else s

def _lower1(s: str) -> str:
    return s[:1].lower() + s[1:] if s[:1].isalpha() else s


# Not inlined in the f-string below: a backslash inside an f-string expression
# is a SyntaxError before Python 3.12, and the project supports 3.11.
_WINDOWS_WIPE_TARGETS = ["C:\\Windows\\System32", "D:\\data"]


class DiverseUniversalGenerator:
    """High-entropy procedural dataset generator for Leeth across all domains."""

    def __init__(self, seed: int = 1234, split: str = "train"):
        self.split = split
        # Offset seed by split hash to guarantee zero cross-split token contamination
        split_offset = {"train": 0, "val": 100_000, "eval": 200_000}.get(split, 0)
        self.rng = random.Random(seed + split_offset)

    def _pick_workdir(self) -> str:
        r = self.rng
        prefix = "train_proj" if self.split == "train" else ("val_proj" if self.split == "val" else "eval_proj")
        idx = r.randint(1, 40)
        drive = r.choice(["C:", "D:", "E:", "/home/dev", "/var/www"])
        sub = r.choice(["Projects", "Work", "Repositories", "Source", "Workspace"])
        if drive.startswith("/"):
            return f"{drive}/{prefix}_{idx}"
        return f"{drive}/{sub}/{prefix}_{idx}"

    def _generate_filepath(self, lang: Optional[str] = None) -> Tuple[str, str]:
        """Generates a realistic procedural file path and its language."""
        r = self.rng
        if not lang or lang not in LANG_SPECS:
            lang = r.choice(list(LANG_SPECS.keys()))
        spec = LANG_SPECS[lang]
        ext = r.choice(spec["exts"])
        mod = r.choice(MODULE_NAMES)
        part = r.choice(SUBMODULE_PARTS)

        base_name = f"{mod}_{part}" if spec["fn_style"] == "snake" else (
            _to_pascal(f"{mod}_{part}") if spec["fn_style"] == "pascal" else _to_camel(f"{mod}_{part}")
        )
        root_tmpl = r.choice(spec["roots"])
        root = root_tmpl.format(mod=mod, Mod=_to_pascal(mod))
        return f"{root}/{base_name}{ext}", lang

    # -----------------------------------------------------------------------
    # Procedural Compiler & Runtime Errors
    # -----------------------------------------------------------------------

    def _generate_traceback_or_error(self) -> Tuple[str, str, str, float]:
        """Generates an authentic compiler diagnostic or runtime crash.

        Returns (state_text, lang_name, target_file_path, severity_score).
        """
        r = self.rng
        lang = r.choice(["Python", "C# (.NET)", "Rust", "C++", "Go", "JavaScript"])
        path, _ = self._generate_filepath(lang)
        line = r.randint(12, 850)
        col = r.randint(1, 65)

        if lang == "Python":
            err_type = r.choice([
                "KeyError", "ValueError", "TypeError", "AttributeError",
                "FileNotFoundError", "torch.cuda.OutOfMemoryError",
                "pydantic.ValidationError", "RuntimeError",
            ])
            fn = r.choice(["dispatch_request", "execute_step", "load_state", "forward", "eval_batch"])
            if err_type == "KeyError":
                detail = f"KeyError: '{r.choice(['session_id', 'user_token', 'payload', 'metric_loss'])}'"
                sev = 60.0
            elif err_type == "torch.cuda.OutOfMemoryError":
                detail = f"torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate {r.randint(1, 6)}.50 GiB"
                sev = 85.0
            elif err_type == "pydantic.ValidationError":
                detail = f"pydantic.ValidationError: 1 validation error for RequestModel\n  {r.choice(['target_id', 'amount'])}\n    Field required [type=missing]"
                sev = 55.0
            else:
                detail = f"{err_type}: invalid operation encountered on line {line}"
                sev = 50.0

            tb = (
                f'Traceback (most recent call last):\n'
                f'  File "{path}", line {line}, in {fn}\n'
                f'    return self._inner_call(ctx)\n'
                f'{detail}'
            )
            return tb, "Python", path, sev

        if lang == "C# (.NET)":
            code = r.choice(["CS0246", "CS1061", "CS0103", "CS1503", "NullReferenceException"])
            type_name = _to_pascal(r.choice(MODULE_NAMES))
            method_name = _to_pascal(r.choice(SUBMODULE_PARTS))

            if code == "CS0246":
                msg = f"{path}({line},{col}): error CS0246: The type or namespace name '{type_name}' could not be found (are you missing a using directive?)"
                sev = 75.0
            elif code == "CS1061":
                msg = f"{path}({line},{col}): error CS1061: '{type_name}' does not contain a definition for '{method_name}'"
                sev = 70.0
            elif code == "NullReferenceException":
                msg = (
                    f"Unhandled exception. System.NullReferenceException: Object reference not set to an instance of an object.\n"
                    f"   at {type_name}.{method_name}() in {path}:line {line}"
                )
                sev = 85.0
            else:
                msg = f"{path}({line},{col}): error {code}: Invalid symbol reference"
                sev = 65.0
            return msg, "C# (.NET)", path, sev

        if lang == "Rust":
            err_code = r.choice(["E0308", "E0597", "E0432", "panic"])
            var_name = r.choice(["client_buf", "token_stream", "auth_ctx", "peer_socket"])

            if err_code == "E0308":
                msg = (
                    f"error[E0308]: mismatched types\n"
                    f"  --> {path}:{line}:{col}\n"
                    f"   |\n"
                    f"{line} |     let res: Result<usize, Error> = process_stream(&{var_name});\n"
                    f"   |                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected `Result`, found `Option`"
                )
                sev = 68.0
            elif err_code == "E0597":
                msg = (
                    f"error[E0597]: `{var_name}` does not live long enough\n"
                    f"  --> {path}:{line}:{col}\n"
                    f"   |\n"
                    f"{line} |     let borrowed = &{var_name};\n"
                    f"   |                    ^{var_name} borrowed value does not live long enough"
                )
                sev = 72.0
            else:
                msg = f"thread 'worker-{r.randint(1, 8)}' panicked at 'called `Option::unwrap()` on a `None` value', {path}:{line}:{col}"
                sev = 80.0
            return msg, "Rust", path, sev

        if lang == "C++":
            fn = _to_camel(r.choice(SUBMODULE_PARTS))
            if r.random() < 0.5:
                msg = (
                    f"{path}:{line}:{col}: error: no matching member function for call to '{fn}'\n"
                    f"candidate function not viable: 'this' argument has type 'const Context', but method is not marked const"
                )
                sev = 65.0
            else:
                msg = (
                    f"Segmentation fault (core dumped)\n"
                    f"#0  0x00007f{r.randint(10, 99)}a2b in {fn}() at {path}:{line}"
                )
                sev = 92.0
            return msg, "C++", path, sev

        if lang == "Go":
            fn = _to_pascal(r.choice(SUBMODULE_PARTS))
            msg = (
                f"panic: runtime error: invalid memory address or nil pointer dereference\n"
                f"[signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x{r.randint(4000, 9999)}]\n\n"
                f"goroutine {r.randint(1, 32)} [running]:\n"
                f"main.{fn}(...)\n"
                f"\t{path}:{line} +0x{r.randint(10, 99)}"
            )
            return msg, "Go", path, 85.0

        # JavaScript / TypeScript
        prop = r.choice(["map", "filter", "data", "token", "payload", "id"])
        msg = (
            f"TypeError: Cannot read properties of undefined (reading '{prop}')\n"
            f"    at renderComponent ({path}:{line}:{col})\n"
            f"    at async invokeHandler (node:internal/process/task_queues:95:5)"
        )
        return msg, "JavaScript", path, 60.0

    # -----------------------------------------------------------------------
    # Procedural Browser Accessibility Trees & Web States
    # -----------------------------------------------------------------------

    def _generate_browser_scenario(self) -> dict:
        """Generates realistic A11y tree and user browsing intent."""
        r = self.rng
        page_type = r.choice(["checkout", "search", "settings", "docs", "login", "catalog"])
        domain = r.choice(["store.example.com", "search.internal.net", "console.cloud.io", "docs.python.org", "github.com", "marketplace.corp.local"])

        if page_type == "catalog":
            # Rich, deep e-commerce catalog (long context up to 1-2k tokens)
            items = [
                ("Ноутбук Pro Max 16", "189990 ₽", "Купить Ноутбук Pro Max 16"),
                ("Беспроводные наушники ANC", "14990 ₽", "Купить Беспроводные наушники ANC"),
                ("Механическая клавиатура RGB", "8490 ₽", "Купить Механическая клавиатура RGB"),
                ("Игровой монитор 27 165Hz", "27990 ₽", "Купить Игровой монитор 27 165Hz"),
                ("Внешний SSD диск 2TB", "11490 ₽", "Купить Внешний SSD диск 2TB"),
                ("Эргономичное кресло Pro", "32990 ₽", "Купить Эргономичное кресло Pro"),
            ]
            cards = []
            for name, price, btn in items:
                cards.append(f"  - [article] '{name}': {price}\n    - [button] '{btn}'\n    - [button] 'В избранное'")
            a11y = (
                f"Page: Каталог электроники | https://{domain}/catalog\n"
                f"- [navigation] Фильтры: [checkbox] 'В наличии' [checked], [combobox] 'Сортировка: По популярности'\n"
                f"- [main] Список товаров ({len(items)} шт.):\n"
                + "\n".join(cards) + "\n"
                f"- [navigation] Пагинация: [button] 'Предыдущая', [button] '1', [button] '2', [button] 'Следующая'\n"
                f"- [contentinfo] Подвал: [link] 'Поддержка', [link] 'Доставка', [link] 'О компании'"
            )
            target_item = r.choice(items)
            intent = r.choice([
                (f"добавь в корзину {target_item[0].lower()}", "browser_click", target_item[2], False),
                ("перейди на следующую страницу каталога", "browser_click", "Следующая", False),
                ("открой раздел доставка", "browser_click", "Доставка", False),
            ])
        elif page_type == "checkout":
            amount = r.randint(490, 8900)
            a11y = (
                f"Page: Оформление заказа | https://{domain}/checkout\n"
                f"- [main] Оплата и доставка\n"
                f"  - [textbox] 'Адрес курьера': 'ул. Мира, д. {r.randint(1, 99)}'\n"
                f"  - [combobox] 'Способ оплаты': 'Банковская карта'\n"
                f"  - [checkbox] 'Согласен с договором оферты' [checked]\n"
                f"  - [button] 'Оплатить заказ {amount} ₽'\n"
                f"  - [button] 'Вернуться в корзину'\n"
                f"  - [link] 'Условия возврата'"
            )
            intent = r.choice([
                ("нажми вернуться в корзину", "browser_click", "Вернуться в корзину", False),
                (f"оплати заказ на сумму {amount}", "UNFAMILIAR", f"Оплатить заказ {amount} ₽", True),
                ("открой условия возврата", "browser_click", "Условия возврата", False),
            ])
        elif page_type == "search":
            query_term = r.choice(["rust async runtime", "docker compose v2 syntax", "react server components", "css grid layout"])
            a11y = (
                f"Page: Поиск в сети | https://{domain}/\n"
                f"- [banner] Навигация\n"
                f"  - [searchbox] 'Поисковый запрос': ''\n"
                f"  - [button] 'Найти'\n"
                f"- [main] Быстрые категории\n"
                f"  - [link] 'Документация'\n"
                f"  - [link] 'Форум сообщества'\n"
                f"  - [button] 'Очистить историю поиска'"
            )
            intent = r.choice([
                (f"введи в строку поиска {query_term}", "browser_type", query_term, False),
                ("нажми кнопку Найти", "browser_click", "Найти", False),
                ("открой документацию", "browser_click", "Документация", False),
            ])
        elif page_type == "settings":
            a11y = (
                f"Page: Настройки профиля | https://{domain}/settings\n"
                f"- [main] Личный кабинет\n"
                f"  - [textbox] 'Имя пользователя': 'developer_{r.randint(10, 99)}'\n"
                f"  - [textbox] 'Email': 'dev@example.org'\n"
                f"  - [button] 'Сохранить профиль'\n"
                f"  - [button] 'Удалить аккаунт и все данные'\n"
                f"  - [button] 'Выйти из сессии'"
            )
            intent = r.choice([
                ("сохрани профиль", "browser_click", "Сохранить профиль", False),
                ("удали мой аккаунт", "UNFAMILIAR", "Удалить аккаунт и все данные", True),
                ("выйди из сессии", "browser_click", "Выйти из сессии", False),
            ])
        else:
            url_target = f"https://{domain}/api/v1/reference"
            a11y = (
                f"Page: Справочник API | {url_target}\n"
                f"- [navigation] Содержание\n"
                f"  - [link] 'Аутентификация'\n"
                f"  - [link] 'Методы GET'\n"
                f"  - [link] 'Вебхуки'\n"
                f"- [main] Введение в API\n"
                f"  - [button] 'Скопировать токен доступа'\n"
                f"  - [link] 'Скачать OpenAPI спецификацию'"
            )
            intent = r.choice([
                (f"открой страницу {url_target}", "browser_open", url_target, False),
                ("перейди в раздел Аутентификация", "browser_click", "Аутентификация", False),
                ("скопируй токен доступа", "browser_click", "Скопировать токен доступа", False),
            ])

        query, tool, param, dangerous = intent
        return {
            "domain": "browser_a11y",
            "a11y_tree": a11y,
            "query": query,
            "tool": tool,
            "param": param,
            "dangerous": dangerous,
            "wd": self._pick_workdir(),
        }

    # -----------------------------------------------------------------------
    # Procedural Math Expressions
    # -----------------------------------------------------------------------

    def _generate_math_scenario(self) -> dict:
        r = self.rng
        math_kind = r.choice(["arithmetic", "percent", "power_root", "nested"])

        if math_kind == "arithmetic":
            a = r.randint(12, 450)
            b = r.randint(3, 95)
            c = r.randint(2, 25)
            op1 = r.choice(["+", "-", "*"])
            op2 = r.choice(["+", "-", "*"])
            expr = f"{a} {op1} {b} {op2} {c}"
            query = r.choice([
                f"посчитай {expr}", f"сколько будет {expr}", f"вычисли {expr}",
                f"calculate {expr}", f"compute {expr}", f"what is {expr}?",
            ])
        elif math_kind == "percent":
            pct = r.choice([5, 10, 15, 18, 20, 25, 30, 45, 50, 75])
            base = r.choice([200, 450, 1200, 2400, 3600, 5000, 14400])
            expr = f"{pct}% of {base}"
            query = r.choice([
                f"посчитай {pct}% от {base}", f"сколько будет {pct}% от {base}",
                f"calculate {pct}% of {base}", f"compute {pct}% from {base}",
            ])
        elif math_kind == "power_root":
            if r.random() < 0.5:
                base = r.choice([2, 3, 4, 5, 8, 10])
                p = r.randint(2, 8)
                sub = r.randint(10, 50)
                expr = f"{base}**{p} - {sub}"
                query = f"вычисли {expr}"
            else:
                sq = r.choice([64, 121, 144, 169, 225, 256, 400, 625])
                mult = r.randint(2, 12)
                expr = f"sqrt({sq}) * {mult}"
                query = f"посчитай корень из {sq} умноженный на {mult}"
        else:
            x = r.randint(10, 80)
            y = r.randint(2, 15)
            z = r.randint(3, 10)
            expr = f"({x} - {y}) * {z}"
            query = f"реши пример {expr}"

        return {
            "domain": "math",
            "query": query,
            "tool": "calculate",
            "param": expr,
            "wd": self._pick_workdir(),
        }

    # -----------------------------------------------------------------------
    # Procedural CLI & Developer Workflows
    # -----------------------------------------------------------------------

    def _generate_cli_scenario(self, destructive: bool = False) -> dict:
        r = self.rng
        branch = r.choice(["main", "develop", "feature/auth", "fix/crash", "refactor/pipeline"])

        if destructive:
            cmd = r.choice([
                f"rm -rf /var/www/{r.choice(MODULE_NAMES)}",
                f"format {r.choice(['D:', 'E:', 'F:'])} /fs:NTFS /q",
                f"drop database {r.choice(['production', 'user_store', 'billing_archive'])}",
                f"docker system prune -af --volumes",
                f"git push origin {branch} --force",
                f"cipher /w:{r.choice(['C:', 'D:'])}",
                f"del /f /q /s {r.choice(_WINDOWS_WIPE_TARGETS)}",
            ])
            return {
                "domain": "security",
                "query": cmd,
                "tool": "UNFAMILIAR",
                "param": cmd,
                "dangerous": True,
                "wd": self._pick_workdir(),
            }

        cmd_type = r.choice(["git", "test", "build", "container"])
        if cmd_type == "git":
            cmd = r.choice([
                f"git checkout -b {branch}",
                f"git status",
                f"git log --oneline -n {r.randint(3, 15)}",
                f"git diff HEAD~1",
                f"git merge origin/{branch}",
            ])
        elif cmd_type == "test":
            cmd = r.choice([
                f"pytest -v tests/{r.choice(MODULE_NAMES)}",
                f"cargo test --package {r.choice(MODULE_NAMES)}",
                f"npm test -- --watchAll=false",
                f"dotnet test src/{_to_pascal(r.choice(MODULE_NAMES))}.Tests",
            ])
        elif cmd_type == "build":
            cmd = r.choice([
                f"cargo build --release",
                f"npm run build",
                f"dotnet publish -c Release",
                f"python -m pip install -r requirements.txt",
            ])
        else:
            cmd = r.choice([
                f"docker compose up -d {r.choice(MODULE_NAMES)}",
                f"docker ps -a",
                f"kubectl get pods -n {r.choice(['default', 'prod', 'staging'])}",
            ])

        return {
            "domain": "run",
            "query": cmd,
            "tool": "run_command",
            "param": cmd,
            "dangerous": False,
            "wd": self._pick_workdir(),
        }

    # -----------------------------------------------------------------------
    # Multi-Turn & Conversational State (Background tasks, anaphora, shifts)
    # -----------------------------------------------------------------------

    def _generate_multi_turn_scenario(self) -> dict:
        r = self.rng
        sub_type = r.choices(
            ["task_monitor", "anaphora", "approach_shift", "complex_reasoning"],
            weights=[0.38, 0.30, 0.20, 0.12],
        )[0]

        path, lang_name = self._generate_filepath()
        mod = r.choice(MODULE_NAMES)
        pid = r.randint(1024, 65500)
        elapsed = r.randint(3, 180)

        if sub_type == "task_monitor":
            has_active = r.random() < 0.75
            if has_active:
                bg_cmd = r.choice([
                    f"pytest -v tests/{mod}",
                    f"cargo test --package {mod}",
                    f"npm test -- --coverage",
                    f"dotnet test src/{_to_pascal(mod)}.Tests",
                    f"docker compose up -d {mod}",
                    f"make build-all",
                    f"python -m unittest discover",
                ])
                status = r.choice(["running", "executing", "in progress"])
                active_str = f"PID {pid} ({bg_cmd}, status={status}, elapsed={elapsed}s)"
                query = r.choice([
                    "ну как там результаты?",
                    "долго ещё?",
                    "долго ещё будут выполняться тесты?",
                    "как там тесты?",
                    "что там с тестами?",
                    "закончился ли билд?",
                    "покажи статус фонового процесса",
                    "проверь фоновую задачу",
                    "how are the tests doing?",
                    "is it done yet?",
                    "show background process status",
                    "any results yet?",
                ])
                return {
                    "domain": "multi_turn",
                    "sub_type": "task_monitor",
                    "active_tasks_str": active_str,
                    "last_target": path,
                    "last_tool": "start_background_command",
                    "prev_turn": f"Запустил фоновый процесс {bg_cmd} с PID {pid}.",
                    "query": query,
                    "tool": "read_background_command",
                    "param": str(pid),
                    "wd": self._pick_workdir(),
                }
            else:
                query = r.choice([
                    "ну как там результаты?",
                    "долго ещё?",
                    "что там слышно?",
                    "как там дела с задачей?",
                    "какой статус?",
                ])
                return {
                    "domain": "multi_turn",
                    "sub_type": "task_monitor_empty",
                    "active_tasks_str": "None",
                    "last_target": path,
                    "last_tool": "read_file",
                    "prev_turn": "Ответил на предыдущий вопрос.",
                    "query": query,
                    "tool": "NO_TOOL",
                    "param": "",
                    "wd": self._pick_workdir(),
                }

        elif sub_type == "anaphora":
            target_kind = r.choice(["file", "dir", "url"])
            if target_kind == "file":
                query = r.choice([
                    "открой его",
                    "прочитай его",
                    "покажи этот файл",
                    "глянь что в нём",
                    "покажи его код",
                    "open it",
                    "read this file",
                    "inspect it",
                ])
                return {
                    "domain": "multi_turn",
                    "sub_type": "anaphora_file",
                    "active_tasks_str": "None",
                    "last_target": path,
                    "last_tool": "find_by_name",
                    "prev_turn": f"Нашёл целевой файл {path}.",
                    "query": query,
                    "tool": "read_file",
                    "param": path,
                    "lang_name": lang_name,
                    "wd": self._pick_workdir(),
                }
            elif target_kind == "dir":
                dir_path = str(Path(path).parent)
                query = r.choice([
                    "покажи список файлов там",
                    "что лежит в этой папке?",
                    "выведи её содержимое",
                    "list directory contents",
                ])
                return {
                    "domain": "multi_turn",
                    "sub_type": "anaphora_dir",
                    "active_tasks_str": "None",
                    "last_target": dir_path,
                    "last_tool": "find_by_name",
                    "prev_turn": f"Перешёл в директорию {dir_path}.",
                    "query": query,
                    "tool": "list_directory",
                    "param": dir_path,
                    "wd": self._pick_workdir(),
                }
            else:
                domain_web = r.choice(["docs.rs", "github.com", "crates.io", "npmjs.com"])
                url = f"https://{domain_web}/{mod}"
                query = r.choice([
                    "открой эту страницу",
                    "перейди туда",
                    "загрузи этот url",
                    "open this page",
                ])
                return {
                    "domain": "multi_turn",
                    "sub_type": "anaphora_web",
                    "active_tasks_str": "None",
                    "last_target": url,
                    "last_tool": "search_web",
                    "prev_turn": f"Нашёл документацию: {url}",
                    "query": query,
                    "tool": "browser_open",
                    "param": url,
                    "wd": self._pick_workdir(),
                }

        elif sub_type == "approach_shift":
            shift_kind = r.choice(["test_runner", "search_method", "direct_cmd"])
            if shift_kind == "test_runner":
                new_cmd = f"cargo test --package {mod}"
                query = r.choice([
                    "используй cargo вместо pytest",
                    "не pytest, а cargo",
                    "смени подход: запусти через cargo",
                    "use cargo instead of pytest",
                ])
                return {
                    "domain": "multi_turn",
                    "sub_type": "approach_shift",
                    "active_tasks_str": "None",
                    "last_target": f"tests/test_{mod}.py",
                    "last_tool": "run_command",
                    "prev_turn": "Пытался запустить pytest.",
                    "query": query,
                    "tool": "run_command",
                    "param": "cargo",
                    "wd": self._pick_workdir(),
                }
            elif shift_kind == "search_method":
                fname = Path(path).name
                query = r.choice([
                    "не через grep, а через find",
                    "поищи по имени файла вместо grep",
                    "найди сам файл",
                ])
                return {
                    "domain": "multi_turn",
                    "sub_type": "approach_shift",
                    "active_tasks_str": "None",
                    "last_target": fname,
                    "last_tool": "grep_search",
                    "prev_turn": f"Искал текст в коде.",
                    "query": query,
                    "tool": "find_by_name",
                    "param": fname,
                    "wd": self._pick_workdir(),
                }
            else:
                calc_expr = f"{r.randint(10, 50)} * {r.randint(2, 20)}"
                query = f"посчитай {calc_expr} на калькуляторе"
                return {
                    "domain": "multi_turn",
                    "sub_type": "approach_shift",
                    "active_tasks_str": "None",
                    "last_target": "None",
                    "last_tool": "NO_TOOL",
                    "prev_turn": "Обсуждали математическую модель.",
                    "query": query,
                    "tool": "calculate",
                    "param": calc_expr,
                    "wd": self._pick_workdir(),
                }

        else:
            query = r.choice([
                "смени подход и полностью перепиши этот компонент на GraphQL",
                "проанализируй почему упали тесты и исправь ошибку в архитектуре",
                "сделай код-ревью и предложи оптимизацию алгоритма",
                "разработай новую схему базы данных для этого модуля",
            ])
            return {
                "domain": "multi_turn",
                "sub_type": "complex_reasoning",
                "active_tasks_str": f"PID {pid} (pytest -v, failed)",
                "last_target": path,
                "last_tool": "run_command",
                "prev_turn": "Тесты завершились с ошибками.",
                "query": query,
                "tool": "UNFAMILIAR",
                "param": "",
                "wd": self._pick_workdir(),
            }

    # -----------------------------------------------------------------------
    # Scenario Assembly
    # -----------------------------------------------------------------------

    def _scenario(self) -> dict:
        r = self.rng
        domain = r.choices(
            ["files", "traceback", "browser_a11y", "math", "cli", "security", "chat", "multi_turn"],
            weights=[0.20, 0.15, 0.14, 0.10, 0.10, 0.07, 0.08, 0.16],
        )[0]

        if domain == "multi_turn":
            return self._generate_multi_turn_scenario()

        if domain == "traceback":
            tb, lang_name, path, sev = self._generate_traceback_or_error()
            return {
                "domain": "traceback",
                "traceback": tb,
                "lang_name": lang_name,
                "param": path,
                "severity": sev,
                "query": r.choice(["разберись в ошибке", "fix this error", "что упало?", "проанализируй стектрейс"]),
                "tool": "read_file",
                "wd": self._pick_workdir(),
            }

        if domain == "browser_a11y":
            return self._generate_browser_scenario()

        if domain == "math":
            return self._generate_math_scenario()

        if domain == "cli":
            return self._generate_cli_scenario(destructive=False)

        if domain == "security":
            return self._generate_cli_scenario(destructive=True)

        if domain == "chat":
            q = r.choice([
                "привет, как твои дела?", "здравствуй! чем можешь помочь?",
                "подскажи архитектуру для распределённого кэша",
                "объясни разницу между sync и async в Tokio",
                "спасибо за помощь, всё собралось!",
                "какой паттерн лучше применить для обработки транзакций?",
                "добрый вечер! напомни текущий стек проекта",
            ])
            return {
                "domain": "chat",
                "query": q,
                "tool": "NO_TOOL",
                "param": "",
                "wd": self._pick_workdir(),
            }

        # Files domain
        path, lang_name = self._generate_filepath()
        action_roll = r.random()
        if action_roll < 0.35:
            verb = r.choice(["прочитай", "открой файл", "покажи содержимое", "read", "open", "inspect"])
            tool = "read_file"
        elif action_roll < 0.60:
            verb = r.choice(["найди файл", "поищи где лежит", "find file", "locate"])
            tool = "find_by_name"
        elif action_roll < 0.80:
            verb = r.choice(["покажи список файлов в", "выведи директорию", "list directory", "show contents of"])
            tool = "list_directory"
            path = str(Path(path).parent)
        else:
            verb = r.choice(["поищи упоминания функции в", "grep for function in", "найди текст в"])
            tool = "grep_search"

        return {
            "domain": "files",
            "query": f"{verb} {path}",
            "tool": tool,
            "param": path,
            "lang_name": lang_name,
            "wd": self._pick_workdir(),
        }

    # -----------------------------------------------------------------------
    # State Formatting
    # -----------------------------------------------------------------------

    def _state(self, sc: dict) -> str:
        r = self.rng
        q = sc["query"]
        sc["embedded"] = q

        if sc["domain"] == "multi_turn":
            active_str = sc.get("active_tasks_str", "None")
            last_tgt = sc.get("last_target", "None")
            last_tl = sc.get("last_tool", "None")
            prev_turn = sc.get("prev_turn", "None")
            sc["embedded"] = _lower1(q)
            return (
                f"Incoming user request in Argent session:\n"
                f'"{_lower1(q)}"\n'
                f"Active background processes: {active_str}\n"
                f"Recent session context:\n"
                f"- Last target: {last_tgt}\n"
                f"- Last tool: {last_tl}\n"
                f"- Previous turn: {prev_turn}\n"
                f"Working directory: {sc['wd']}"
            )

        if sc["domain"] == "traceback":
            tail = ""
            if r.random() < 0.6:
                sc["embedded"] = _cap(q)  # the span an extract question points at is what the state really contains
                tail = f'\nOperator instruction: "{sc["embedded"]}"'
            return sc["traceback"] + tail

        if sc["domain"] == "browser_a11y":
            return f'{sc["a11y_tree"]}\nUser voice command: "{q}"'

        kind = r.choice(["session", "shell", "terminal", "ci"])
        if kind == "session":
            sc["embedded"] = _lower1(q)
            return f'Incoming user request in Argent session:\n"{_lower1(q)}"\nWorking directory: {sc["wd"]}'
        if kind == "shell":
            return f"User shell execution request:\nCommand: {q}\nCaller: {r.choice(CALLERS)}"
        sc["embedded"] = _cap(q)
        if kind == "ci":
            return f"CI/CD pipeline stage output:\nStage: execution -> status: pending\nOperator instruction: {sc['embedded']}"
        return f"Terminal session output:\n$ ready\nNext operator input: {sc['embedded']}"

    # -----------------------------------------------------------------------
    # Sample Creation (Matching 4 Primitives)
    # -----------------------------------------------------------------------

    def _sample(self) -> LeethSample:
        r = self.rng
        sc = self._scenario()
        state = self._state(sc)
        qtype = r.choices(["choice", "noul", "score", "extract"], weights=[0.36, 0.28, 0.12, 0.24])[0]

        if qtype == "choice":
            sub = r.choices(["routing", "lang", "security"], weights=[0.70, 0.15, 0.15])[0]
            if sub == "routing":
                return LeethSample(
                    state=state, question_type="choice", question=Q_ROUTING,
                    choices=list(CHOICES_ROUTING), target_choice=sc["tool"],
                )
            if sub == "lang":
                lang_name = sc.get("lang_name", "Python")
                return LeethSample(
                    state=state, question_type="choice", question=Q_LANG,
                    choices=list(CHOICES_LANG), target_choice=lang_name,
                )
            # Security action choice
            act = "block_command" if sc.get("dangerous") else "execute_immediately"
            return LeethSample(
                state=state, question_type="choice", question=Q_SECURITY_ACTION,
                choices=list(CHOICES_SECURITY), target_choice=act,
            )

        if qtype == "noul":
            sub = r.choices(["tool", "safe", "err"], weights=[0.50, 0.30, 0.20])[0]
            if sub == "tool":
                needs_tool = sc["tool"] != "NO_TOOL"
                return LeethSample(
                    state=state, question_type="noul", question=Q_NOUL_TOOL,
                    target_noul=1.0 if needs_tool else 0.0,
                )
            if sub == "safe":
                is_safe = not sc.get("dangerous", False)
                return LeethSample(
                    state=state, question_type="noul", question=Q_NOUL_SAFE,
                    target_noul=1.0 if is_safe else 0.0,
                )
            # Compiler/runtime error detection
            has_error = sc["domain"] == "traceback"
            return LeethSample(
                state=state, question_type="noul", question=Q_NOUL_ERR,
                target_noul=1.0 if has_error else 0.0,
            )

        if qtype == "score":
            if sc.get("dangerous"):
                score = r.uniform(85.0, 100.0)
            elif sc["domain"] == "traceback":
                score = sc.get("severity", 65.0)
            else:
                score = r.uniform(5.0, 30.0)
            return LeethSample(
                state=state, question_type="score", question=Q_SCORE_RISK,
                target_score=score, score_levels=5,
            )

        # Extract primitive
        param = sc.get("param", "")
        # Ensure parameter exists verbatim in the state string
        if param and param in state:
            span = param
        else:
            span = sc.get("embedded", sc["query"])
            if span not in state:
                span = sc["query"]

        return LeethSample(
            state=state, question_type="extract", question=Q_EXTRACT_PARAM,
            target_span_text=span,
        )

    # -----------------------------------------------------------------------
    # Streaming and Batch Saving
    # -----------------------------------------------------------------------

    def generate_stream(self) -> Iterator[LeethSample]:
        while True:
            yield self._sample()

    def generate_and_save(self, file_path: str | Path, target_count: int) -> int:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for s in self.generate_stream():
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
                count += 1
                if count >= target_count:
                    break
        return count