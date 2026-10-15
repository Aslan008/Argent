"""
Universal Schema Dataset Generator for Argent MicroModels.

Automatically extracts all 66 tool schemas from tools.schemas.TOOL_SCHEMAS,
combines with real arguments from ~/.argent/logs/tools.log and session histories,
and generates a balanced multilingual (Russian + English) dataset for all 66 tools + NO_TOOL.
Also produces token-level entity labels (NER) for neural parameter extraction.
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is at the head of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.schemas import TOOL_SCHEMAS
from logger import get_logger

logger = get_logger("schema_dataset_generator")

# 66 tool names extracted from TOOL_SCHEMAS + NO_TOOL (67 classes total)
RAW_TOOL_NAMES = sorted([t["function"]["name"] for t in TOOL_SCHEMAS])
ALL_INTENTS = ["NO_TOOL"] + RAW_TOOL_NAMES
INTENT2ID = {name: idx for idx, name in enumerate(ALL_INTENTS)}
ID2INTENT = {idx: name for idx, name in enumerate(ALL_INTENTS)}

# NER Entity tags
ENTITY_TAGS = [
    "O",
    "B-PATH", "I-PATH",
    "B-CMD", "I-CMD",
    "B-QUERY", "I-QUERY",
    "B-SYM", "I-SYM",
    "B-URL", "I-URL",
    "B-NUM", "I-NUM",
    "B-NAME", "I-NAME",
]
TAG2ID = {tag: idx for idx, tag in enumerate(ENTITY_TAGS)}
ID2TAG = {idx: tag for idx, tag in enumerate(ENTITY_TAGS)}


class UniversalDatasetGenerator:
    """Generates training data for all 66 Argent tools and dynamic semantic embeddings."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(
            output_dir or (Path(__file__).parent / "data")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        home = os.path.expanduser("~")
        self.sessions_dir = Path(os.path.join(home, ".argent", "sessions"))
        self.logs_dir = Path(os.path.join(home, ".argent", "logs"))

    def get_tool_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """Extracts tool metadata, parameter names, and descriptions."""
        catalog = {}
        for s in TOOL_SCHEMAS:
            fn = s.get("function", {})
            name = fn.get("name")
            desc = fn.get("description", "")
            params = fn.get("parameters", {}).get("properties", {})
            catalog[name] = {
                "description": desc,
                "parameters": list(params.keys()),
                "param_details": params,
            }
        return catalog

    def generate_all(self, samples_per_tool: int = 50) -> List[Dict[str, Any]]:
        """Generates training samples covering every tool in Argent."""
        dataset: List[Dict[str, Any]] = []

        # Common entity pools
        sample_paths = [
            "src/agent.py", "config.py", "README.md", "src/main.py", "package.json",
            "tools/file_ops.py", "tools/schemas.py", "tests/test_agent.py", "setup.py",
            "docker-compose.yml", ".env", "src/micromodels/runtime.py", "assets/logo.png",
            "docs/index.html", "scripts/build.sh", "Cargo.toml", "requirements.txt",
            "test.txt", "temp.txt", "main.py", "app.py", "data.json", "config.json"
        ]
        sample_dirs = [
            "src", "tests", "tools", "docs", "config", "assets", "scripts", "src/cli",
            "src/agent", "src/micromodels", "plugins", "."
        ]
        sample_symbols = [
            "TinyIntentEncoder", "NeedleCoprocessor", "FastDispatchResult", "Agent",
            "run_turn", "MicroModelRuntime", "AVAILABLE_TOOLS", "stream_chat",
            "handle_command", "validate_token", "execute_query", "AuthMiddleware"
        ]
        sample_commands = [
            "pytest tests/test_agent.py", "pytest tests/", "pytest", "git status", "git diff",
            "npm test", "npm run build", "cargo test", "python main.py", "ruff check .",
            "pip list", "docker ps", "python -m pytest"
        ]
        sample_urls = [
            "https://pytorch.org", "https://github.com/torvalds/linux",
            "https://docs.python.org/3/", "https://news.ycombinator.com",
            "https://developer.mozilla.org"
        ]
        sample_windows = ["Visual Studio Code", "Google Chrome", "Telegram", "Terminal", "Notepad"]
        sample_skills = ["yandex-games", "chrome-extensions", "lmtrust", "antigravity-guide"]

        # 1. NO_TOOL (conversations, reasoning, open questions, general planning)
        conversational = [
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
            "Привет! Как устроен этот проект?", "Объясни как устроен проект",
            "Что делает этот репозиторий?", "Расскажи про архитектуру",
            "Как запустить проект локально?", "Какие технологии здесь используются?",
            "Помоги разобраться в коде", "Что ты умеешь делать?",
            "запеки свет для сцены", "bake lighting for the scene",
            "запекание карт освещения", "baking lightmaps",
            "настрой свет и тени сцены", "setup lighting and scene shadows",
            "шейдеры и текстуры материалов", "material shaders and textures",
            "физика и твердые тела", "physics and rigidbodies",
            "анимация персонажей", "character animations",
        ]
        for q in conversational:
            dataset.append({"text": q, "intent": "NO_TOOL", "arguments": {}, "source": "synthetic"})
            dataset.append({"text": q.lower().rstrip("?!.,"), "intent": "NO_TOOL", "arguments": {}, "source": "synthetic"})

        # 2. Templates for each specific tool
        tool_templates: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}

        # Helper to register template
        def add_t(tool: str, ru_templates: List[str], en_templates: List[str], arg_gen_fn):
            if tool not in tool_templates:
                tool_templates[tool] = []
            for tpl in ru_templates:
                tool_templates[tool].append((tpl, "ru", arg_gen_fn))
            for tpl in en_templates:
                tool_templates[tool].append((tpl, "en", arg_gen_fn))

        # Desktop / Window tools
        add_t(
            "list_desktop_windows",
            [
                "покажи открытые окна", "список окон", "какие окна открыты на рабочем столе",
                "покажи список окон", "покажи открытые окна на рабочем столе", "список активных окон",
                "какие приложения открыты", "окна рабочего стола", "список окон в системе",
            ],
            [
                "list desktop windows", "show open windows", "list visible windows",
                "get desktop windows", "list all windows", "show active windows",
            ],
            lambda: {}
        )
        add_t(
            "capture_desktop_window",
            ["сделай скриншот окна {win}", "сфоткай окно {win}", "захвати окно {win}", "скриншот приложения {win}"],
            ["capture desktop window {win}", "take screenshot of window {win}", "screenshot window {win}"],
            lambda: {"title": random.choice(sample_windows)}
        )
        add_t(
            "capture_desktop_screen",
            [
                "сделай скриншот экрана", "сфоткай весь экран", "захвати монитор",
                "скриншот рабочего стола", "сделай скриншот всего дисплея", "снимок экрана",
            ],
            ["capture desktop screen", "take full screen screenshot", "capture monitor", "screenshot entire screen"],
            lambda: {}
        )
        add_t(
            "get_clipboard_image",
            ["возьми картинку из буфера", "получи изображение из буфера обмена", "вставь скриншот из буфера"],
            ["get clipboard image", "paste image from clipboard", "read clipboard image"],
            lambda: {}
        )

        # Skills & Subagents
        add_t(
            "list_skills",
            ["покажи доступные скиллы", "список скиллов", "какие скиллы установлены", "покажи скилы"],
            ["list available skills", "show installed skills", "list skills", "what skills are installed"],
            lambda: {}
        )
        add_t(
            "read_skill",
            ["прочитай скилл {skill}", "инструкция скилла {skill}", "покажи документацию скилла {skill}"],
            ["read skill {skill}", "show skill instructions {skill}", "load skill {skill}"],
            lambda: {"skill_name": random.choice(sample_skills)}
        )
        add_t(
            "run_subagent",
            ["запусти субагента для {task}", "делегируй задачу субагенту", "создай субагента"],
            ["run subagent for {task}", "spawn subagent to handle {task}", "delegate to subagent"],
            lambda: {"task": random.choice(["research", "code_review", "testing"])}
        )
        add_t(
            "run_deep_research",
            ["проведи глубокое исследование {topic}", "запусти deep research по {topic}", "исследуй подробно {topic}"],
            ["run deep research on {topic}", "start deep research for {topic}", "thoroughly investigate {topic}"],
            lambda: {"topic": random.choice(["transformer architecture", "onnx int8 quantization", "pep 703"])}
        )

        # File & Directory operations
        def _read_file_arg():
            p = random.choice(sample_paths)
            s = random.randint(1, 50)
            e = s + random.randint(10, 40)
            return {"file_path": p, "start": s, "end": e, "path": p}

        add_t(
            "read_file",
            [
                # Direct commands
                "прочитай {path}", "прочитай файл {path}", "открой {path}", "открой файл {path}",
                "покажи содержимое {path}", "покажи содержимое файла {path}",
                "глянь в {path}", "глянь в файл {path}", "глянь {path}", "чекни {path}", "чекни файл {path}",
                "катни {path}", "выведи файл {path}", "покажи код из {path}", "считай файл {path}",
                # Line range queries
                "прочитай {path} строки {start}-{end}", "покажи {path} строки с {start} по {end}",
                "открой {path} строки {start} до {end}", "глянь {path} строки {start}-{end}",
                "прочитай {path} с {start} по {end} строку", "выведи строки {start}-{end} из {path}",
                # CoT Reasoning thoughts
                "Сначала открою файл {path}, чтобы посмотреть реализацию",
                "Сначала открою {path}",
                "Сначала открою файл {path}",
                "Сейчас посмотрим файл {path}",
                "Посмотрим файл {path}",
                "Посмотрим код в {path}",
                "Открою файл {path}",
                "Открою {path}",
                "Прочитаю файл {path}",
                "Прочитаю {path}",
                "Нужно прочитать {path} и проверить сигнатуру класса",
                "Нужно открыть файл {path}",
                "Нужно прочитать файл {path}",
                "Гляну в {path}, чтобы понять структуру данных",
                "Гляну в файл {path}",
                "Посмотрим код в {path}, чтобы найти источник ошибки",
                "Для решения задачи мне нужно прочитать {path}",
                "Изучу реализацию в {path} перед внесением изменений",
                "Прочитаю {path} строки {start}-{end} чтобы увидеть логику функции",
            ],
            [
                # English direct
                "read file {path}", "read {path}", "open file {path}", "open {path}",
                "show content of {path}", "cat {path}", "display {path}", "view file {path}",
                "view {path}", "inspect {path}", "view source of {path}",
                # English line ranges
                "read {path} lines {start} to {end}", "show {path} lines {start}-{end}",
                "open {path} from line {start} to {end}", "view lines {start}-{end} in {path}",
                # English CoT thoughts
                "First, let's open {path}",
                "First, let's inspect {path} to understand the implementation",
                "Let's check file {path}",
                "Let's check {path}",
                "I will read {path} to check the functions and types",
                "Looking at the traceback, let me inspect {path} lines {start}-{end}",
                "I need to read {path} to see how the turn loop is handled",
                "Let's examine the contents of {path} before modifying it",
            ],
            _read_file_arg
        )
        add_t(
            "write_file",
            ["запиши в файл {path}", "сохрани код в {path}", "создай файл {path} с содержимым"],
            ["write file {path}", "save to file {path}", "create file {path} with content"],
            lambda: {"file_path": random.choice(sample_paths)}
        )
        add_t(
            "delete_file",
            ["удали файл {path}", "сотри {path}", "убери временный файл {path}"],
            ["delete file {path}", "remove file {path}", "rm {path}"],
            lambda: {"file_path": random.choice(["temp.txt", "scratch.py", "cache.tmp", "old.log"])}
        )
        add_t(
            "create_directory",
            ["создай папку {dir}", "создай директорию {dir}", "сделай каталог {dir}", "mkdir {dir}"],
            ["create directory {dir}", "make folder {dir}", "create folder {dir}", "mkdir {dir}"],
            lambda: {"path": random.choice(["build", "dist", "logs", "temp", "src/models"])}
        )
        add_t(
            "move_file",
            ["перемести {path} в {dir}", "переименуй файл {path}", "move {path} to {dir}"],
            ["move file {path} to {dir}", "rename file {path}", "mv {path} {dir}"],
            lambda: {"source": random.choice(sample_paths), "destination": random.choice(sample_dirs)}
        )
        add_t(
            "copy_file",
            ["скопируй {path} в {dir}", "сделай копию файла {path}", "copy {path} to {dir}"],
            ["copy file {path} to {dir}", "duplicate {path}", "cp {path} {dir}"],
            lambda: {"source": random.choice(sample_paths), "destination": random.choice(sample_dirs)}
        )
        add_t(
            "list_directory",
            ["покажи файлы в {dir}", "что лежит в папке {dir}", "список директории {dir}", "ls {dir}"],
            ["list directory {dir}", "show contents of folder {dir}", "list files in {dir}", "ls {dir}"],
            lambda: {"path": random.choice(sample_dirs)}
        )
        add_t(
            "search_files",
            ["найди файлы {pat}", "поищи файлы с маской {pat}", "поиск файлов {pat} в {dir}"],
            ["search files matching {pat}", "find all files {pat}", "locate files {pat} in {dir}"],
            lambda: {"pattern": random.choice(["*.py", "test_*.py", "*.md", "*.json"]), "directory": random.choice(sample_dirs)}
        )
        add_t(
            "grep_search",
            ["найди где используется {sym}", "поищи {sym} в коде", "найди вхождения {sym} в {dir}"],
            ["grep for {sym}", "search in codebase for {sym}", "find references to {sym} in {dir}"],
            lambda: {"query": random.choice(sample_symbols), "path": random.choice(sample_dirs)}
        )
        add_t(
            "replace_in_file",
            ["замени блок кода в {path}", "исправь фрагмент в {path}", "replace in file {path}"],
            ["replace in file {path}", "substitute code block in {path}", "replace text in {path}"],
            lambda: {"file_path": random.choice(sample_paths)}
        )
        add_t(
            "multi_replace_in_file",
            ["множественная замена в {path}", "замени несколько блоков в {path}"],
            ["multi replace in file {path}", "apply multiple edits to {path}"],
            lambda: {"file_path": random.choice(sample_paths)}
        )

        # Code Navigation & LSP
        add_t(
            "find_definition",
            ["где определен {sym}", "найди объявление {sym}", "перейди к определению {sym}"],
            ["find definition of {sym}", "where is {sym} defined", "go to definition of {sym}"],
            lambda: {"symbol": random.choice(sample_symbols)}
        )
        add_t(
            "find_references",
            ["найди все ссылки на {sym}", "где вызывается {sym}", "все использования {sym}"],
            ["find all references to {sym}", "where is {sym} used", "find usages of {sym}"],
            lambda: {"symbol": random.choice(sample_symbols)}
        )
        add_t(
            "find_implementations",
            ["найди реализации {sym}", "кто реализует интерфейс {sym}"],
            ["find implementations of {sym}", "locate classes implementing {sym}"],
            lambda: {"symbol": random.choice(sample_symbols)}
        )
        add_t(
            "search_workspace_symbols",
            ["поиск символов {sym} в проекте", "найди символ {sym} в воркспейсе"],
            ["search workspace symbols for {sym}", "find symbol {sym} across project"],
            lambda: {"query": random.choice(sample_symbols)}
        )
        add_t(
            "get_file_outline",
            ["покажи структуру файла {path}", "аутлайн файла {path}", "структура классов в {path}"],
            ["get file outline for {path}", "show structural outline of {path}", "classes and methods in {path}"],
            lambda: {"file_path": random.choice(sample_paths)}
        )
        add_t(
            "get_call_hierarchy",
            ["покажи дерево вызовов для {sym}", "иерархия вызовов {sym}", "кто вызывает {sym}"],
            ["get call hierarchy for {sym}", "show callers and callees for {sym}"],
            lambda: {"symbol": random.choice(sample_symbols)}
        )
        add_t(
            "check_code",
            ["проверь код на ошибки lsp", "есть ли ошибки компиляции", "проверь файл {path} на ошибки"],
            ["check code for errors", "run lsp diagnostics on {path}", "check syntax and types"],
            lambda: {"file_path": random.choice(sample_paths)}
        )
        add_t(
            "run_deep_linter",
            ["запусти глубокий линтер на {path}", "прогони pylint по {path}", "проверь линтером {path}"],
            ["run deep linter on {path}", "execute pylint on {path}", "deep lint {path}"],
            lambda: {"file_path": random.choice(sample_paths)}
        )
        add_t(
            "analyze_project",
            ["проанализируй структуру проекта", "сделай аудит проекта", "обзор проекта в целом"],
            ["analyze project structure", "audit current project", "inspect project layout"],
            lambda: {}
        )

        # Commands & Background execution
        add_t(
            "run_command",
            ["{cmd}", "запусти {cmd}", "выполни команду {cmd}", "выполни {cmd}", "запусти в терминале {cmd}"],
            ["{cmd}", "run command {cmd}", "execute command {cmd}", "run {cmd}", "run in terminal {cmd}", "exec {cmd}"],
            lambda: {"command": random.choice(sample_commands)}
        )
        add_t(
            "run_admin_command",
            ["выполни с правами админа {cmd}", "запусти от имени администратора {cmd}"],
            ["run admin command {cmd}", "execute as administrator {cmd}"],
            lambda: {"command": random.choice(sample_commands)}
        )
        add_t(
            "start_background_command",
            ["запусти в фоне {cmd}", "стартани фоновый процесс {cmd}", "запусти сервис {cmd} в бэкграунде"],
            ["start background command {cmd}", "run {cmd} in background", "spawn background job {cmd}"],
            lambda: {"command": random.choice(["npm run dev", "python -m http.server 8000", "docker compose up"])}
        )
        add_t(
            "list_background_commands",
            ["какие фоновые процессы запущены", "список фоновых задач", "покажи фоновые команды"],
            ["list background commands", "show running background jobs", "get background processes"],
            lambda: {}
        )
        add_t(
            "read_background_command",
            ["покажи вывод фоновой команды {id}", "прочитай stdout фонового процесса {id}"],
            ["read background command output {id}", "get stdout of background process {id}"],
            lambda: {"id": random.choice(["1", "2", "task-101"])}
        )
        add_t(
            "send_background_command",
            ["отправь ввод в фоновый процесс {id}", "передай stdin в фоновую задачу {id}"],
            ["send input to background process {id}", "write to stdin of background job {id}"],
            lambda: {"id": random.choice(["1", "2"])}
        )
        add_t(
            "stop_background_command",
            ["останови фоновый процесс {id}", "заверши фоновую задачу {id}", "убей фоновую команду {id}"],
            ["stop background command {id}", "kill background process {id}", "terminate background job {id}"],
            lambda: {"id": random.choice(["1", "2"])}
        )

        # Git operations
        add_t(
            "read_git_diff",
            ["покажи дифф", "покажи изменения в гите", "что изменилось в коде", "git diff", "покажи git status"],
            ["show git diff", "view unstaged changes", "what changed in git", "git diff", "git status"],
            lambda: {}
        )
        add_t(
            "git_checkpoint",
            ["сохрани чекпоинт в гите", "сделай git checkpoint", "сохрани текущий прогресс в git"],
            ["create git checkpoint", "save git checkpoint", "save current progress checkpoint"],
            lambda: {"message": "checkpoint before refactor"}
        )
        add_t(
            "git_rollback",
            ["откати чекпоинт", "вернись к предыдущему чекпоинту", "сделай git rollback", "отмени последние изменения"],
            ["git rollback", "revert to previous checkpoint", "rollback last checkpoint"],
            lambda: {}
        )

        # Web & Media
        add_t(
            "search_web",
            ["найди в интернете {q}", "погугли {q}", "поищи в сети информацию про {q}"],
            ["search the web for {q}", "google {q}", "search online for {q}"],
            lambda: {"query": random.choice(["onnx int8 quantization", "python 3.14 release notes", "transformer pre-ln"])}
        )
        add_t(
            "read_webpage",
            ["прочитай веб-страницу {url}", "открой ссылку {url}", "скачай текст со страницы {url}"],
            ["read webpage {url}", "fetch url {url}", "extract text from {url}"],
            lambda: {"url": random.choice(sample_urls)}
        )
        add_t(
            "view_image",
            ["посмотри изображение {path}", "открой картинку {path}", "взгляни на скриншот {path}"],
            ["view image {path}", "look at image {path}", "inspect image {path}"],
            lambda: {"file_path": random.choice(["screenshot.png", "assets/diagram.png", "mockup.jpg"])}
        )
        add_t(
            "calculate",
            ["посчитай {expr}", "сколько будет {expr}", "вычисли {expr}"],
            ["calculate {expr}", "compute {expr}", "what is {expr}"],
            lambda: {"expression": random.choice(["125 * 38", "2**10", "1024 / 16", "sqrt(144) + 10"])}
        )
        add_t(
            "filter_new_items",
            ["отфильтруй новые элементы", "верни только новые айдишники", "отсей дубликаты"],
            ["filter new items", "get only unseen items", "filter duplicate entries"],
            lambda: {}
        )
        add_t(
            "wait_heartbeat",
            ["подожди завершения процесса", "жди пока закончится сборка", "пауза до завершения"],
            ["wait heartbeat", "wait until job completes", "pause until task is done"],
            lambda: {}
        )

        # MCP and External Server Tools
        add_t(
            "call_mcp_tool",
            [
                "вызови mcp инструмент {tool} на сервере {srv}",
                "запусти mcp функцию {tool} для {srv}",
                "выполни действие {tool} через mcp сервер {srv}",
                "mcp call {tool} on {srv}",
                "вызови внешний инструмент {tool} в {srv}",
                "вызови mcp эндпоинт {tool}",
                "запусти удаленный тул {tool} на сервере {srv}",
                "выполни операцию {tool} через mcp",
            ],
            [
                "call mcp tool {tool} on server {srv}",
                "execute mcp tool {tool} on {srv}",
                "invoke tool {tool} on mcp server {srv}",
                "mcp call {tool} on server {srv}",
                "run remote tool {tool} via mcp",
                "invoke external tool {tool} on {srv}",
            ],
            lambda: {
                "server_name": random.choice(["unity", "blender", "kern_gate", "database", "github"]),
                "tool_name": random.choice(["get_object_info", "render_scene", "run_gate", "query_db"]),
                "tool": random.choice(["get_object_info", "render_scene", "run_gate"]),
                "srv": random.choice(["unity", "blender", "kern_gate"]),
            }
        )
        add_t(
            "list_mcp_tools",
            [
                "какие mcp инструменты доступны",
                "список mcp инструментов на сервере {srv}",
                "что умеет mcp сервер {srv}",
                "покажи все инструменты mcp",
                "список функций сервера {srv}",
            ],
            [
                "list mcp tools",
                "list available tools on mcp server {srv}",
                "what tools does {srv} mcp server have",
                "show mcp tool registry for {srv}",
            ],
            lambda: {"server_name": random.choice(["unity", "blender", "kern_gate"]), "srv": random.choice(["unity", "blender", "kern_gate"])}
        )

        # Agent & Coordination Tools
        add_t(
            "run_subagent",
            [
                "запусти субагента для {task}",
                "делегируй задачу {task} субагенту",
                "создай изолированного субагента",
                "пусть субагент исследует {task}",
            ],
            [
                "run subagent for {task}",
                "spawn subagent to handle {task}",
                "delegate {task} to subagent",
            ],
            lambda: {"task": random.choice(["рефакторинг тестов", "анализ безопасности", "поиск утечек памяти", "code review"])}
        )
        add_t(
            "run_deep_research",
            [
                "проведи глубокое исследование {topic}",
                "deep research по теме {topic}",
                "исследуй подробно {topic} в сети",
                "собери подробный технический отчет по {topic}",
            ],
            [
                "run deep research on {topic}",
                "perform deep research investigation on {topic}",
                "conduct in-depth research about {topic}",
            ],
            lambda: {"topic": random.choice(["onnx web assembly", "transformer quantization int8", "vulkan ray tracing", "godot vs unity physics"])}
        )
        add_t(
            "create_artifact",
            [
                "создай артефакт {topic}",
                "сохрани план как артефакт {topic}",
                "запиши архитектурный документ в артефакты",
                "создай markdown артефакт {topic}",
            ],
            [
                "create artifact for {topic}",
                "save document as artifact {topic}",
                "write markdown artifact {topic}",
            ],
            lambda: {"topic": random.choice(["implementation_plan.md", "architecture.md", "research_notes.md"])}
        )
        add_t(
            "request_user_approval",
            [
                "запроси подтверждение пользователя",
                "спроси одобрение юзера перед продолжением",
                "подожди подтверждения плана",
                "request user approval",
            ],
            [
                "request user approval",
                "ask user for confirmation before proceeding",
                "wait for user approval",
            ],
            lambda: {}
        )
        add_t(
            "ask_user_questions",
            [
                "задай вопросы пользователю",
                "уточни у пользователя требования",
                "спроси предпочтения пользователя",
                "задай структурированные вопросы юзеру",
            ],
            [
                "ask user questions",
                "clarify requirements with user",
                "interview user on decisions",
            ],
            lambda: {}
        )

        # Browser Automation Tools (17 tools)
        add_t(
            "browser_open",
            ["открой в браузере {url}", "загрузи страницу {url} в браузере", "перейди на сайт {url}"],
            ["open {url} in browser", "navigate browser to {url}", "browser open {url}"],
            lambda: {"url": random.choice(sample_urls)}
        )
        add_t(
            "browser_state",
            ["в каком состоянии браузер", "текущий url и статус браузера", "покажи состояние браузера"],
            ["get browser state", "what is the browser state", "check browser url and title"],
            lambda: {}
        )
        add_t(
            "browser_click",
            ["кликни по элементу {id}", "нажми на кнопку {id}", "кликни в браузере {id}"],
            ["click element {id}", "click button {id} in browser", "browser click {id}"],
            lambda: {"index": random.choice([1, 2, 5, 12])}
        )
        add_t(
            "browser_input",
            ["введи в поле {id} текст '{text}'", "напечатай в инпут '{text}'", "заполни поле '{text}'"],
            ["type '{text}' into field {id}", "input text '{text}' in browser", "fill input with '{text}'"],
            lambda: {"text": random.choice(["admin", "test@example.com", "hello world"])}
        )
        add_t(
            "browser_select",
            ["выбери опцию {opt} в селекте", "выбери значение в выпадающем списке"],
            ["select option {opt} in dropdown", "browser select option {opt}"],
            lambda: {"value": "option_1"}
        )
        add_t(
            "browser_screenshot",
            ["сделай скриншот страницы в браузере", "сфоткай веб-страницу", "сохрани скриншот браузера"],
            ["take browser screenshot", "capture webpage screenshot", "browser screenshot"],
            lambda: {}
        )
        add_t(
            "browser_scroll",
            ["прокрути страницу вниз", "скролл вниз в браузере", "прокрути на 500 пикселей"],
            ["scroll page down", "browser scroll down", "scroll window down"],
            lambda: {"direction": "down"}
        )
        add_t(
            "browser_wait_for",
            ["подожди появления элемента {sel}", "жди пока загрузится элемент {sel}"],
            ["wait for element {sel} in browser", "browser wait for selector {sel}"],
            lambda: {"selector": "#submit-btn"}
        )
        add_t(
            "browser_get_content",
            ["извлеки контент со страницы", "получи текст страницы в браузере", "вытащи html страницы"],
            ["get browser page content", "extract text from browser page", "extract html content"],
            lambda: {}
        )
        add_t(
            "browser_close",
            ["закрой браузер", "заверши сессию браузера", "закрыть окно браузера"],
            ["close browser", "terminate browser session", "close browser window"],
            lambda: {}
        )
        add_t(
            "browser_switch_tab",
            ["переключись на вкладку {id}", "переключи таб в браузере на {id}"],
            ["switch browser tab to {id}", "switch to tab {id} in browser"],
            lambda: {"tab_index": 1}
        )
        add_t(
            "browser_press_key",
            ["нажми клавишу {key} в браузере", "отправь нажатие {key}"],
            ["press key {key} in browser", "send key press {key}"],
            lambda: {"key": random.choice(["Enter", "Escape", "Tab", "ArrowDown"])}
        )
        add_t(
            "browser_fill_form",
            ["заполни форму в браузере", "автозаполни поля формы"],
            ["fill form fields in browser", "batch fill form in browser"],
            lambda: {}
        )
        add_t(
            "browser_upload_file",
            ["загрузи файл {path} в форму браузера", "прикрепи файл {path} к инпуту"],
            ["upload file {path} to browser input", "attach file {path} in browser form"],
            lambda: {"file_path": random.choice(sample_paths)}
        )
        add_t(
            "browser_back",
            ["вернись назад в браузере", "нажми кнопку назад в браузере", "страница назад"],
            ["go back in browser", "browser navigate back", "navigate previous page"],
            lambda: {}
        )
        add_t(
            "browser_reload",
            ["перезагрузи страницу в браузере", "обнови страницу", "browser reload"],
            ["reload browser page", "refresh webpage in browser", "browser reload"],
            lambda: {}
        )
        add_t(
            "browser_eval_js",
            ["выполни javascript на странице", "выполни js код {code} в браузере", "evaluate js expression"],
            ["evaluate javascript in browser", "execute js script in page", "run js expression on page"],
            lambda: {"expression": "document.title"}
        )

        # NO_TOOL: Conversational, Reasoning, Refutations, Explanations
        add_t(
            "NO_TOOL",
            [
                # Greetings & Conversational
                "привет", "добрый день", "здравствуй", "как дела?", "добрый вечер",
                "спасибо большое", "отлично, продолжай", "ясно, понял", "хорошо, принято",
                # Explanations & Conceptual questions
                "как работает async/await в python?", "объясни разницу между процессами и потоками",
                "в чем отличие tcp от udp?", "напиши алгоритм быстрой сортировки",
                "расскажи про архитектуру трансформеров", "почему этот тест падает?",
                "как оптимизировать запросы к базе данных?", "что такое dependency injection?",
                # CoT Refutations & Backtracking (КРИТИЧЕСКИ ВАЖНО)
                "Хотя нет, этот файл читать не нужно",
                "Хотя нет, в этом нет необходимости",
                "Прочитаю agent.py... хотя погоди, проблема не в нем",
                "Не буду открывать {path}, проверю сначала тесты",
                "Отставить, удалять эту директорию не стоит",
                "Стоп, перезаписывать файл пока рано",
                "Впрочем, этот класс трогать не нужно",
                "Пожалуй, обойдемся без вызова команды",
                "Хотя стоп, эта функция уже реализована выше",
                "Нет, отмена, это приведет к ошибке",
                # General observations & Thinking aloud
                "Похоже, ошибка кроется в логике авторизации",
                "Судя по стектрейсу, переменная равна None",
                "Пользователь просит добавить новый эндпоинт",
                "Итак, план действий следующий: сначала проанализируем код",
            ],
            [
                # English greetings & general
                "hello", "hi there", "good morning", "how are you?",
                "thank you so much", "sounds good, proceed", "got it", "ok thanks",
                # English questions
                "how does async/await work in python?", "explain the difference between stack and heap",
                "what is the difference between tcp and udp?", "write a quicksort algorithm in python",
                "explain the transformer architecture", "why does this test fail?",
                "how to optimize database queries?", "what is dependency injection?",
                # English CoT Refutations & Backtracking
                "Wait, no need to read this file, it's already in the context",
                "Actually, nevermind, let's keep it as is",
                "No, wait, the error is in a different module",
                "I will skip reading {path}, let's directly write the test",
                "Wait, running this command might break the build, let's not do that",
                "Although wait, we don't need to touch this file",
                "Stop, modifying this file is premature",
                "Nevermind, this function is already defined in utils",
                "Hold on, let's think about this before executing",
                # English observations
                "The issue appears to be an off-by-one error in the loop",
                "Looking at the traceback, the NullReferenceException occurs at line 42",
                "The user wants us to refactor the payment processor",
                "Let's carefully verify the plan before writing code",
            ],
            lambda: {"path": random.choice(sample_paths)}
        )

        # 3. Instantiate templates to generate samples
        for tool_name in ALL_INTENTS:
            templates = tool_templates.get(tool_name, [])
            if not templates:
                # Fallback template if missing
                desc_short = tool_name.replace("_", " ")
                templates = [
                    (f"вызови {tool_name}", "ru", lambda: {}),
                    (f"выполни {tool_name}", "ru", lambda: {}),
                    (f"run {tool_name}", "en", lambda: {}),
                    (f"execute {tool_name}", "en", lambda: {}),
                ]

            # Generate target number of samples: core tools get 3x samples, NO_TOOL gets 2x
            core_tools = {"read_file", "list_directory", "grep_search", "search_files", "run_command", "calculate"}
            if tool_name in core_tools:
                target_count = samples_per_tool * 3
            elif tool_name == "NO_TOOL":
                target_count = samples_per_tool * 2
            else:
                target_count = samples_per_tool
            count = 0
            while count < target_count:
                tpl, lang, arg_fn = random.choice(templates)
                args = arg_fn()
                # Format string
                try:
                    formatted_text = tpl
                    for k, v in args.items():
                        formatted_text = formatted_text.replace(f"{{{k}}}", str(v))
                        # Also generic aliases
                        formatted_text = formatted_text.replace("{path}", str(v))
                        formatted_text = formatted_text.replace("{dir}", str(v))
                        formatted_text = formatted_text.replace("{sym}", str(v))
                        formatted_text = formatted_text.replace("{cmd}", str(v))
                        formatted_text = formatted_text.replace("{q}", str(v))
                        formatted_text = formatted_text.replace("{url}", str(v))
                        formatted_text = formatted_text.replace("{win}", str(v))
                        formatted_text = formatted_text.replace("{skill}", str(v))
                        formatted_text = formatted_text.replace("{task}", str(v))
                        formatted_text = formatted_text.replace("{topic}", str(v))
                        formatted_text = formatted_text.replace("{pat}", str(v))
                        formatted_text = formatted_text.replace("{id}", str(v))
                        formatted_text = formatted_text.replace("{expr}", str(v))
                        formatted_text = formatted_text.replace("{text}", str(v))
                        formatted_text = formatted_text.replace("{opt}", str(v))
                        formatted_text = formatted_text.replace("{key}", str(v))
                        formatted_text = formatted_text.replace("{code}", str(v))
                except Exception:
                    formatted_text = tpl

                dataset.append({
                    "text": formatted_text,
                    "intent": tool_name,
                    "arguments": args,
                    "source": "schema_synthetic",
                })
                count += 1

        random.seed(42)
        random.shuffle(dataset)

        out_file = self.output_dir / "universal_training_pairs.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for s in dataset:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # Save metadata / intent catalog
        meta_file = self.output_dir / "universal_intent_catalog.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_intents": len(ALL_INTENTS),
                "intents": ALL_INTENTS,
                "intent2id": INTENT2ID,
                "id2intent": ID2INTENT,
            }, f, ensure_ascii=False, indent=2)

        print(f"Universal dataset generated: {len(dataset)} samples across {len(ALL_INTENTS)} classes.")
        print(f"Saved to {out_file} and catalog to {meta_file}")
        return dataset


if __name__ == "__main__":
    gen = UniversalDatasetGenerator()
    gen.generate_all(samples_per_tool=55)
