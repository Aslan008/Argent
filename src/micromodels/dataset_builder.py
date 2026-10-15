"""
Dataset builder for Argent MicroModels.

Extracts real training pairs from ~/.argent/sessions/*.json.gz and ~/.argent/logs/tools.log,
and combines them with rich synthetic bilingual (Russian + English) intent and extraction examples.
Outputs a structured dataset ready for tokenizer training, intent router, and specialist models.
"""

import gzip
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from logger import get_logger

logger = get_logger("dataset_builder")

# Intent classes
INTENTS = [
    "NO_TOOL",
    "read_file",
    "grep_search",
    "search_files",
    "run_command",
    "list_directory",
    "delete_file",
    "create_directory",
    "calculate",
    "search_web",
    "read_webpage",
    "read_git_diff",
]

INTENT2ID = {intent: idx for idx, intent in enumerate(INTENTS)}
ID2INTENT = {idx: intent for idx, intent in enumerate(INTENTS)}


class DatasetBuilder:
    """Builds and augments training datasets for Argent MicroModels."""

    def __init__(
        self,
        sessions_dir: Optional[str] = None,
        logs_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        home = os.path.expanduser("~")
        self.sessions_dir = Path(sessions_dir or os.path.join(home, ".argent", "sessions"))
        self.logs_dir = Path(logs_dir or os.path.join(home, ".argent", "logs"))
        self.output_dir = Path(
            output_dir
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_sessions(self) -> List[Dict[str, Any]]:
        """Extracts user -> tool call pairs from recorded Argent sessions."""
        pairs: List[Dict[str, Any]] = []
        if not self.sessions_dir.exists():
            logger.warning(f"Sessions dir not found: {self.sessions_dir}")
            return pairs

        session_files = list(self.sessions_dir.glob("*.json.gz"))
        logger.info(f"Scanning {len(session_files)} session files...")

        for sfile in session_files:
            try:
                with gzip.open(sfile, "rt", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                    messages = data.get("messages", [])
                    for i in range(len(messages) - 1):
                        curr = messages[i]
                        nxt = messages[i + 1]

                        if curr.get("role") == "user" and nxt.get("role") == "assistant":
                            # Extract user query
                            u_content = curr.get("content", "")
                            if isinstance(u_content, list):
                                u_content = " ".join(
                                    part.get("text", "")
                                    for part in u_content
                                    if isinstance(part, dict)
                                )
                            if not isinstance(u_content, str):
                                continue

                            u_text = u_content.strip()
                            if not u_text or len(u_text) > 400 or u_text.startswith("/"):
                                continue

                            tool_calls = nxt.get("tool_calls", [])
                            if tool_calls:
                                first_call = tool_calls[0]
                                fn = first_call.get("function", {})
                                name = fn.get("name")
                                args = fn.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except Exception:
                                        args = {"raw": args}

                                mapped_intent = self._map_tool_to_intent(name)
                                if mapped_intent:
                                    pairs.append({
                                        "text": u_text,
                                        "intent": mapped_intent,
                                        "arguments": args,
                                        "source": "session",
                                    })
                            else:
                                # Normal conversational message (no tool call)
                                if random.random() < 0.25 and len(u_text) < 250:
                                    pairs.append({
                                        "text": u_text,
                                        "intent": "NO_TOOL",
                                        "arguments": {},
                                        "source": "session",
                                    })
            except Exception as e:
                logger.debug(f"Error parsing session {sfile.name}: {e}")

        logger.info(f"Extracted {len(pairs)} pairs from sessions.")
        return pairs

    def extract_from_logs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extracts tool arguments from tools.log to use for realistic templates."""
        tool_args: Dict[str, List[Dict[str, Any]]] = {
            "run_command": [],
            "read_file": [],
            "grep_search": [],
            "search_files": [],
            "list_directory": [],
        }
        tools_log = self.logs_dir / "tools.log"
        if not tools_log.exists():
            return tool_args

        pat = re.compile(r"\[INFO\]\s+([a-zA-Z0-9_]+):\s+(.*)$")
        with open(tools_log, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if not m:
                    continue
                tool_name, rest = m.group(1), m.group(2).strip()
                if tool_name == "run_command" and len(rest) < 150:
                    tool_args["run_command"].append({"command": rest})
                elif tool_name in ("read_file", "write_file", "delete_file"):
                    # file path is typically before any trailing info
                    path = rest.split(" (")[0].strip()
                    if path and not path.startswith("{") and len(path) < 120:
                        tool_args["read_file"].append({"file_path": path})

        return tool_args

    def _map_tool_to_intent(self, tool_name: str) -> Optional[str]:
        """Maps an Argent tool name to an intent category."""
        if not tool_name:
            return None
        if tool_name in ("read_file", "read_file_outline", "view_image"):
            return "read_file"
        if tool_name in ("grep_search", "find_references", "find_definition"):
            return "grep_search"
        if tool_name in ("search_files", "filter_new_items"):
            return "search_files"
        if tool_name in ("run_command", "run_admin_command", "read_background_command"):
            return "run_command"
        if tool_name in ("list_directory", "list_desktop_windows"):
            return "list_directory"
        if tool_name == "delete_file":
            return "delete_file"
        if tool_name == "create_directory":
            return "create_directory"
        if tool_name == "calculate":
            return "calculate"
        if tool_name == "search_web":
            return "search_web"
        if tool_name == "read_webpage":
            return "read_webpage"
        if tool_name in ("read_git_diff", "git_status", "git_diff"):
            return "read_git_diff"
        return None

    def generate_synthetic_data(
        self, samples_per_intent: int = 150
    ) -> List[Dict[str, Any]]:
        """Generates realistic bilingual synthetic training queries."""
        synthetic: List[Dict[str, Any]] = []

        # Common code paths
        sample_paths = [
            "src/agent.py",
            "src/main.py",
            "config.py",
            "README.md",
            "requirements.txt",
            "package.json",
            "src/agent/needle_coprocessor.py",
            "tests/test_agent.py",
            "tests/test_needle_coprocessor.py",
            "src/cli/commands/config_commands.py",
            "tools/file_ops.py",
            "tools/search_ops.py",
            "src/micromodels/tokenizer.py",
            "src/micromodels/intent_router.py",
            "setup.py",
            "pyproject.toml",
            "index.html",
            "src/index.css",
            ".env",
            "docker-compose.yml",
        ]

        sample_dirs = [
            "src",
            "tests",
            "src/agent",
            "src/cli",
            "tools",
            "docs",
            "config",
            ".argent",
            "src/micromodels",
            ".",
            "src/automation",
            "tests/unit",
        ]

        sample_symbols = [
            "NeedleCoprocessor",
            "TinyIntentEncoder",
            "BilingualIntentNormalizer",
            "FastDispatchResult",
            "get_needle_enabled",
            "try_fast_dispatch",
            "MicroModelRuntime",
            "AVAILABLE_TOOLS",
            "read_file",
            "grep_search",
            "run_command",
            "def initialize",
            "class Agent",
            "TOKEN_LIMIT",
            "api_key",
            "temperature",
            "logger",
        ]

        sample_commands = [
            "pytest tests/",
            "pytest tests/test_agent.py -v",
            "git status",
            "git diff",
            "git log -n 5",
            "git branch",
            "python -m unittest",
            "npm run test",
            "npm run build",
            "python --version",
            "pip list",
            "ruff check .",
            "mypy src/",
            "cargo test",
            "dir src",
            "ls -la",
        ]

        sample_math = [
            "2 + 2",
            "125 * 38",
            "1024 / 16",
            "2**10",
            "500 * 1.2",
            "sqrt(256) + 14",
            "100 - 37.5",
            "(45 + 55) * 3",
            "3.14159 * 25",
            "2048 / 8",
        ]

        sample_web_queries = [
            "onnxruntime python tutorial",
            "pytorch transformer encoder example",
            "bpe tokenizer implementation python",
            "fastapi async background tasks",
            "rust token killer rtk",
            "latest python 3.14 release notes",
            "github actions ci cd best practices",
            "как обучить transformer с нуля",
            "квантование моделей int8 onnx",
        ]

        sample_urls = [
            "https://pytorch.org/docs/stable/index.html",
            "https://onnxruntime.ai/docs/",
            "https://docs.python.org/3/library/ast.html",
            "https://github.com/huggingface/transformers",
            "https://habr.com/ru/post/123456/",
            "https://en.wikipedia.org/wiki/Byte_pair_encoding",
        ]

        # 1. NO_TOOL (conversations, reasoning, questions, code writing)
        no_tool_templates = [
            "Привет, как дела?", "Привет!", "Здравствуй", "Спасибо за помощь", "Отлично, спасибо",
            "Понял тебя", "Ясно", "Давай продолжим", "Почему падает этот тест?",
            "Объясни мне, как работает архитектура агента", "Напиши функцию для быстрой сортировки на Python",
            "Зарефактори класс Agent чтобы он был более модульным", "Как мне лучше организовать базу данных для проекта?",
            "Придумай архитектуру микросервисов", "Помоги составить план разработки",
            "Исправь ошибку переполнения стека", "Что думаешь по поводу использования Rust вместо Go?",
            "Напиши тесты для модуля аутентификации", "Сделай красивый UI для главной страницы",
            "Hello, how are you?", "Thanks a lot!", "Can you explain how this works?",
            "Why did the build fail?", "Write a Python function to parse JSON files",
            "Refactor this method to use async/await", "How should I structure my database schema?",
            "Help me write a technical specification", "What is the best way to handle concurrency here?",
            "Fix the bug in the login route", "Draft a design doc for the new subsystem",
            "Посоветуй библиотеку для работы с графами", "Как ускорить загрузку приложения?",
            "В чем разница между процессами и потоками?", "Объясни паттерн проектирования Фабрика",
            "Напиши dockerfile для nodejs приложения", "Что такое инверсия зависимостей?",
            "Как настроить nginx в качестве reverse proxy?", "Какая сложность у алгоритма Дейкстры?",
            "Привет! Чем можешь помочь?", "Подскажи хорошую статью по трансформерам",
            "Хорошо, давай так", "Ладно, посмотрим позже", "Отмени последнее действие",
            "Расскажи анекдот", "Какая сейчас погода?", "Кто создал Python?",
            "What is the difference between TCP and UDP?", "How does garbage collection work in V8?",
            "Write a prompt for an LLM that generates unit tests", "Good morning!",
            "Can we discuss the project roadmap?", "Explain quantum computing in simple terms",
            "How do I optimize SQL queries for postgres?", "What are the SOLID principles?",
            "Summarize the conversation so far", "Thank you, that helped a lot",
            "I disagree, let's try a different approach", "Let's take a break",
            "Привет! Как устроен этот проект?", "Объясни как устроен проект",
            "Что делает этот репозиторий?", "Расскажи про архитектуру",
            "Как запустить проект локально?", "Какие технологии здесь используются?",
            "Помоги разобраться в коде", "Что ты умеешь делать?",
            "Расскажи о себе", "Дай совет по архитектуре",
        ]
        for text in no_tool_templates:
            synthetic.append({
                "text": text,
                "intent": "NO_TOOL",
                "arguments": {},
                "source": "synthetic",
            })
            # Add conversational lowercase variants
            synthetic.append({
                "text": text.lower().rstrip("?!.,"),
                "intent": "NO_TOOL",
                "arguments": {},
                "source": "synthetic",
            })

        # 2. read_file
        read_prefixes_ru = [
            "прочитай", "открой", "покажи", "покажи содержимое", "выведи", "глянь в",
            "прочти", "посмотри файл", "прочитай файл", "открой файл", "покажи код",
            "выведи на экран файл", "дай глянуть", "отобрази файл", "считай файл",
        ]
        read_prefixes_en = [
            "read", "open", "show", "show content of", "cat", "display", "view",
            "inspect file", "read file", "open file", "view source of", "print file",
            "examine file", "get file contents",
        ]

        extra_paths = [
            "docker/Dockerfile", "scripts/build.sh", "src/core/engine.cpp", "include/header.h",
            "internal/server/handler.go", "Cargo.toml", "src/lib.rs", "assets/style.scss",
            "config/settings.yaml", "manifest.json", "public/favicon.ico", "tests/e2e/test_ui.py",
            "src/utils/helpers.ts", "components/Button.vue", "templates/index.jinja2",
            ".gitignore", ".github/workflows/ci.yml", "alembic.ini", "prisma/schema.prisma"
        ]
        all_paths = sample_paths + extra_paths

        for path in all_paths:
            for pfx in random.sample(read_prefixes_ru, 3):
                synthetic.append({
                    "text": f"{pfx} {path}",
                    "intent": "read_file",
                    "arguments": {"file_path": path},
                    "source": "synthetic",
                })
            for pfx in random.sample(read_prefixes_en, 3):
                synthetic.append({
                    "text": f"{pfx} {path}",
                    "intent": "read_file",
                    "arguments": {"file_path": path},
                    "source": "synthetic",
                })
            # Direct path query
            synthetic.append({
                "text": path,
                "intent": "read_file",
                "arguments": {"file_path": path},
                "source": "synthetic",
            })
            synthetic.append({
                "text": f"./{path}",
                "intent": "read_file",
                "arguments": {"file_path": f"./{path}"},
                "source": "synthetic",
            })

        # 3. grep_search
        grep_prefixes_ru = [
            "найди где используется", "поищи", "где объявлен", "поиск по коду",
            "найди вхождения", "где вызывается", "найди", "поищи в коде",
            "прогрепай", "поиск подстроки", "где упоминается", "найди использование",
        ]
        grep_prefixes_en = [
            "find where", "search for", "grep for", "where is", "find references to",
            "search in code for", "locate", "grep", "search codebase for", "find all occurrences of",
            "where does", "look for symbol",
        ]

        extra_symbols = [
            "handle_request", "execute_query", "AuthMiddleware", "parse_arguments",
            "CONNECTION_POOL", "MAX_RETRIES", "OnDestroy", "EventEmitter", "subagent_id",
            "async def step", "render_template", "validate_token", "DispatchError"
        ]
        all_symbols = sample_symbols + extra_symbols

        for sym in all_symbols:
            for pfx in random.sample(grep_prefixes_ru, 3):
                synthetic.append({
                    "text": f"{pfx} {sym}",
                    "intent": "grep_search",
                    "arguments": {"query": sym},
                    "source": "synthetic",
                })
                # In-directory query
                d_sample = random.choice(sample_dirs)
                synthetic.append({
                    "text": f"{pfx} {sym} в {d_sample}",
                    "intent": "grep_search",
                    "arguments": {"query": sym, "path": d_sample},
                    "source": "synthetic",
                })
            for pfx in random.sample(grep_prefixes_en, 3):
                synthetic.append({
                    "text": f"{pfx} {sym}",
                    "intent": "grep_search",
                    "arguments": {"query": sym},
                    "source": "synthetic",
                })
                d_sample = random.choice(sample_dirs)
                synthetic.append({
                    "text": f"{pfx} {sym} in {d_sample}",
                    "intent": "grep_search",
                    "arguments": {"query": sym, "path": d_sample},
                    "source": "synthetic",
                })

        # 4. search_files
        file_patterns = [
            "*.py", "*.json", "test_*.py", "*.md", "*.css", "*.ts", "Dockerfile*",
            "*.yaml", "*.rs", "*.go", "*.cpp", "*config*", "*.txt", "*.sh", "*.sql"
        ]
        search_file_ru = [
            "найди файлы", "поищи файлы с именем", "список файлов по маске",
            "где лежат файлы", "поиск файлов", "найди все файлы", "поищи файл по шаблону",
        ]
        search_file_en = [
            "find files matching", "search for files named", "locate files with pattern",
            "find all files", "search files", "find file by name", "glob for",
        ]
        for pat_str in file_patterns:
            for pfx in random.sample(search_file_ru, 3):
                synthetic.append({
                    "text": f"{pfx} {pat_str}",
                    "intent": "search_files",
                    "arguments": {"pattern": pat_str},
                    "source": "synthetic",
                })
            for pfx in random.sample(search_file_en, 3):
                synthetic.append({
                    "text": f"{pfx} {pat_str}",
                    "intent": "search_files",
                    "arguments": {"pattern": pat_str},
                    "source": "synthetic",
                })

        # 5. run_command
        cmd_prefixes_ru = [
            "запусти", "выполни команду", "запусти команду", "выполни",
            "запусти в терминале", "прогони", "выполни в шелле", "прогони команду",
        ]
        cmd_prefixes_en = [
            "run", "execute", "run command", "execute command",
            "exec", "run in terminal", "run shell command", "execute in bash",
        ]
        extra_commands = [
            "git pull", "git checkout main", "git push origin dev", "npm install",
            "pip install -r requirements.txt", "docker ps", "docker compose up -d",
            "pytest -k test_router", "python -m pytest", "cargo check", "go build ./...",
            "dotnet build", "python setup.py develop", "flake8 src/", "black --check ."
        ]
        all_commands = sample_commands + extra_commands

        for cmd in all_commands:
            for pfx in random.sample(cmd_prefixes_ru, 2):
                synthetic.append({
                    "text": f"{pfx} {cmd}",
                    "intent": "run_command",
                    "arguments": {"command": cmd},
                    "source": "synthetic",
                })
            for pfx in random.sample(cmd_prefixes_en, 2):
                synthetic.append({
                    "text": f"{pfx} {cmd}",
                    "intent": "run_command",
                    "arguments": {"command": cmd},
                    "source": "synthetic",
                })
            # Bare command
            synthetic.append({
                "text": cmd,
                "intent": "run_command",
                "arguments": {"command": cmd},
                "source": "synthetic",
            })

        # 6. list_directory
        list_prefixes_ru = [
            "покажи файлы в", "что лежит в папке", "список файлов в директории",
            "покажи содержимое папки", "покажи директорию", "содержимое папки",
            "выведи структуру папки", "глянь что в папке", "покажи каталог",
        ]
        list_prefixes_en = [
            "list files in", "show contents of folder", "ls", "dir",
            "list directory", "show files in", "inspect folder", "list contents of",
        ]
        extra_dirs = ["build", "dist", "node_modules", "vendor", "scripts", "public", "assets", "bin"]
        all_dirs = sample_dirs + extra_dirs

        for d in all_dirs:
            for pfx in random.sample(list_prefixes_ru, 3):
                synthetic.append({
                    "text": f"{pfx} {d}",
                    "intent": "list_directory",
                    "arguments": {"path": d},
                    "source": "synthetic",
                })
            for pfx in random.sample(list_prefixes_en, 3):
                synthetic.append({
                    "text": f"{pfx} {d}",
                    "intent": "list_directory",
                    "arguments": {"path": d},
                    "source": "synthetic",
                })

        # 7. delete_file
        del_paths = [
            "temp.txt", "cache.tmp", "scratch.py", "out.log", "old_backup.json", "debug.log",
            "dummy.onnx", "test.tmp", "temp_output.json", "dump.sql", "core.dump", "junk.dat"
        ]
        del_prefixes_ru = [
            "удали", "удали файл", "сотри файл", "удали временный файл", "убери", "снеси файл", "уничтожь файл"
        ]
        del_prefixes_en = [
            "delete", "remove file", "delete file", "rm", "unlink", "erase file", "drop file"
        ]
        for p in del_paths:
            for pfx in random.sample(del_prefixes_ru, 3):
                synthetic.append({
                    "text": f"{pfx} {p}",
                    "intent": "delete_file",
                    "arguments": {"file_path": p},
                    "source": "synthetic",
                })
            for pfx in random.sample(del_prefixes_en, 3):
                synthetic.append({
                    "text": f"{pfx} {p}",
                    "intent": "delete_file",
                    "arguments": {"file_path": p},
                    "source": "synthetic",
                })

        # 8. create_directory
        dirs_to_create = [
            "dist", "build", "logs", "temp", "src/models", "assets/icons", "coverage",
            "reports", "data/raw", "data/processed", "migrations", "fixtures", "bin"
        ]
        mkdir_prefixes_ru = [
            "создай папку", "создай директорию", "сделай директорию", "mkdir", "создай каталог", "сделай папку"
        ]
        mkdir_prefixes_en = [
            "create directory", "create folder", "make directory", "mkdir", "make folder", "create dir"
        ]
        for d in dirs_to_create:
            for pfx in random.sample(mkdir_prefixes_ru, 3):
                synthetic.append({
                    "text": f"{pfx} {d}",
                    "intent": "create_directory",
                    "arguments": {"path": d},
                    "source": "synthetic",
                })
            for pfx in random.sample(mkdir_prefixes_en, 3):
                synthetic.append({
                    "text": f"{pfx} {d}",
                    "intent": "create_directory",
                    "arguments": {"path": d},
                    "source": "synthetic",
                })

        # 9. calculate
        calc_prefixes_ru = ["посчитай", "сколько будет", "вычисли", "расчитай", "каков результат", "реши пример"]
        calc_prefixes_en = ["calculate", "calc", "compute", "evaluate", "what is", "solve"]
        extra_math = [
            "365 * 24", "1000 / 60", "2**16 - 1", "sqrt(100) * 5", "15 * 15",
            "4096 * 4", "100 * 0.15", "50 / 2.5", "180 / 3.1415", "(12 + 18) * (5 - 2)"
        ]
        all_math = sample_math + extra_math
        for expr in all_math:
            for pfx in random.sample(calc_prefixes_ru, 2):
                synthetic.append({
                    "text": f"{pfx} {expr}",
                    "intent": "calculate",
                    "arguments": {"expression": expr},
                    "source": "synthetic",
                })
            for pfx in random.sample(calc_prefixes_en, 2):
                synthetic.append({
                    "text": f"{pfx} {expr}",
                    "intent": "calculate",
                    "arguments": {"expression": expr},
                    "source": "synthetic",
                })
            synthetic.append({
                "text": expr,
                "intent": "calculate",
                "arguments": {"expression": expr},
                "source": "synthetic",
            })

        # 10. search_web
        web_prefixes_ru = [
            "найди в интернете", "погугли", "поищи в сети", "найди инфу про", "поищи в гугле",
            "поиск в яндексе", "пробей в интернете", "найди информацию о"
        ]
        web_prefixes_en = [
            "search the web for", "google", "look up online", "search internet for", "find info on",
            "web search for", "search duckduckgo for", "find online"
        ]
        extra_web = [
            "pep 703 nogil python", "vLLM architecture diagram", "transformer attention formula",
            "best practice for onnx quantization", "how to build agentic workflows",
            "apple m4 max benchmarks", "fastest json parser python", "clickhouse vs duckdb"
        ]
        all_web = sample_web_queries + extra_web
        for q in all_web:
            for pfx in random.sample(web_prefixes_ru, 2):
                synthetic.append({
                    "text": f"{pfx} {q}",
                    "intent": "search_web",
                    "arguments": {"query": q},
                    "source": "synthetic",
                })
            for pfx in random.sample(web_prefixes_en, 2):
                synthetic.append({
                    "text": f"{pfx} {q}",
                    "intent": "search_web",
                    "arguments": {"query": q},
                    "source": "synthetic",
                })

        # 11. read_webpage
        url_prefixes_ru = [
            "прочитай страницу", "открой ссылку", "скачай сайт", "загрузи страницу", "прочти по ссылке",
            "скачай содержимое url", "покажи веб-страницу"
        ]
        url_prefixes_en = [
            "read webpage", "fetch url", "open link", "read url", "download webpage",
            "fetch web content", "scrape webpage"
        ]
        extra_urls = [
            "https://developer.mozilla.org/en-US/", "https://news.ycombinator.com",
            "https://pypi.org/project/torch/", "https://platform.openai.com/docs/",
            "https://en.cppreference.com/w/", "https://docs.docker.com/get-started/"
        ]
        all_urls = sample_urls + extra_urls
        for url in all_urls:
            for pfx in random.sample(url_prefixes_ru, 2):
                synthetic.append({
                    "text": f"{pfx} {url}",
                    "intent": "read_webpage",
                    "arguments": {"url": url},
                    "source": "synthetic",
                })
            for pfx in random.sample(url_prefixes_en, 2):
                synthetic.append({
                    "text": f"{pfx} {url}",
                    "intent": "read_webpage",
                    "arguments": {"url": url},
                    "source": "synthetic",
                })
            synthetic.append({
                "text": url,
                "intent": "read_webpage",
                "arguments": {"url": url},
                "source": "synthetic",
            })

        # 12. read_git_diff
        diff_queries = [
            "покажи дифф", "покажи diff", "покажи изменения в гите", "что изменилось в коде",
            "какие изменения внесены", "покажи git diff", "гит дифф", "покажи разницу файлов",
            "show git diff", "git diff", "show changes", "what changed in git",
            "view git diff", "display unstaged changes", "show diff", "глянь дифф",
            "покажи последние изменения", "покажи unstaged diff", "git diff HEAD",
            "что я поменял в коде", "покажи гит изменения", "выведи diff", "git status diff",
            "show working tree diff", "show uncommitted changes", "view git status and diff",
            "покажи статус гита", "git status", "какие файлы изменены в репозитории",
            "покажи измененные файлы", "список модифицированных файлов", "что я изменил",
            "покажи правки", "какие файлы modified", "проверь статус git", "статус git",
            "show modified files", "git diff --stat", "git diff staged", "view changes",
            "git status -s", "покажи diff файлов", "какая разница с последним коммитом",
            "diff HEAD", "git diff HEAD~1", "покажи изменения в git репозитории",
            "проверь изменения в проекте", "глянь что изменилось", "открой diff",
            "покажи правки в коде", "какие файлы поправлены", "what files are modified",
            "show git status and modified files", "git diff summary", "покажи дельту",
            "дельту изменений покажи", "что поменялось", "какие диффы", "диффы",
        ]
        for q in diff_queries:
            synthetic.append({
                "text": q,
                "intent": "read_git_diff",
                "arguments": {},
                "source": "synthetic",
            })

        logger.info(f"Generated {len(synthetic)} synthetic training queries.")
        return synthetic

    def build(self) -> Path:
        """Assembles all data sources and saves training_pairs.jsonl."""
        all_samples: List[Dict[str, Any]] = []

        # 1. Real session data
        session_data = self.extract_from_sessions()
        all_samples.extend(session_data)

        # 2. Synthetic augmented data
        synthetic_data = self.generate_synthetic_data()
        all_samples.extend(synthetic_data)

        random.seed(42)
        random.shuffle(all_samples)

        out_path = self.output_dir / "training_pairs.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Stats
        intent_counts: Dict[str, int] = {}
        for s in all_samples:
            it = s["intent"]
            intent_counts[it] = intent_counts.get(it, 0) + 1

        print(f"Dataset assembled: {len(all_samples)} items saved to {out_path}")
        print("Intent distribution:")
        for k, v in sorted(intent_counts.items(), key=lambda x: -x[1]):
            print(f"  {k:20s}: {v}")

        return out_path


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()
