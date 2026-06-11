# Argent: The Elite AI Coding Assistant

[![CI](https://github.com/Aslan008/Argent/actions/workflows/ci.yml/badge.svg)](https://github.com/Aslan008/Argent/actions/workflows/ci.yml)

Argent is a high-performance, professional AI pair programmer designed to live in your terminal. It supports local **Ollama** models, **Z.ai**, and **KoboldCPP** API providers, leveraging advanced architectural patterns to provide an autonomous, efficient, and secure development environment.

> [!NOTE]
> Argent is a **personal experiment** in building high-autonomy AI agents for terminal-based development.
> [!CAUTION]
> **Warning**: This project is in active development and is considered experimental. Things may not work as expected, logic might fail, or code could potentially break. Use with caution.

---

## 🚀 Key Features

### 🧠 Project Brain (Autonomous Mode)
Argent doesn't just answer questions; it builds entire projects. Utilizing a multi-state machine, Argent can:
- **Research**: Scour the web for the latest documentation.
- **Architect**: Design high-level system structures.
- **Spec**: Generate detailed per-file specifications.
- **Execute**: Implement code autonomously while managing its own context limits.

### 🧪 TDD Mode (Test-Driven Development)
Enable strict TDD to force the AI into a professional **Red-Green-Refactor** cycle. Argent will write failing tests, implement the code, and verify results before moving forward.

### 🪝 Global Plugins & Hooks
Extend Argent's core logic with Python plugins stored in `~/.argent/hooks/` (configurable via `/hooks`).
- **Autonomous Extension**: Argent can autonomously create new specialized tools when needed (toggle via `/hooks auto on`).
- **Lifecycle Hooks**: `on_startup`, `pre_prompt`, `post_response`, `on_tool_call`.
- **Custom Commands**: Define functions starting with `command_` to create your own slash commands.

### 🔍 Synchronous RAG (Semantic Search)
Never worry about outdated knowledge. Argent automatically indexes your codebase using **ChromaDB**. When files are modified, the RAG index is updated incrementally in real-time.

### 🌐 Intelligent Browser Control (CDP & Playwright)
Argent includes a powerful browser automation engine based on Playwright:
- **CDP Mode (User's Browser)**: Connects directly to your real browser (Chrome, Yandex Browser, Edge, Brave) to leverage your active sessions, cookies, and login states.
- **Auto-Attach**: Automatically attaches to your already open tabs without interrupting your workflow.
- **Visual & Structural Analysis**: Recursively crawls DOM trees across nested `iframe` and Shadow DOM boundaries.
- **Smart Filtering**: Supports multi-keyword search queries (logical OR) to filter complex web pages and reduce prompt size, with automatic feedback warnings for over-filtering.
- **Robust Actions**: Interact with elements using dynamic ID indexing, precise CSS selectors, or visible text, with automatic scrolling into view.

### 🚦 Automatic Model-Tier Adaptation (Zero Configuration)
Argent detects the model size from its name and silently adapts the whole pipeline — switch between a 1B local model and a cloud giant without touching a single setting:
- **Tiny models (Ollama)**: every agent step is **grammar-constrained at the decoder level** to a strict JSON schema — malformed tool calls become physically impossible to generate, not merely repaired afterwards. A compact tool catalog and an objective reminder are pinned near the end of the prompt, where small models actually attend.
- **Standard local models**: native tool calling plus the objective anchor for long histories.
- **Cloud models**: completely unburdened — no schemas, no catalogs, no reminders.
- **All tiers**: a deterministic **loop guard** catches verbatim-repeated tool calls (the classic small-model failure mode) and force-ends runaway turns; `replace_in_file` auto-corrects whitespace-broken edit targets with re-indentation instead of bouncing errors back.

### 🛡 Centralized Approval & Safe Autonomy
Every dangerous action (shell commands, file deletion, git rollback) passes through a single approval gate:
- **Destructive-command detection**: `rm` / `Remove-Item` / `format` / `git reset --hard` / `taskkill` and friends always require explicit confirmation — even in autonomous mode.
- **Session grants**: approve once with *"always allow `git` this session"* and stop clicking through repeated prompts.
- **Working autonomy**: in `/auto` mode safe actions are auto-approved so the agent can actually run unattended, while destructive ones still pause for you.

### 🧮 Exact Arithmetic
The `calculate` tool evaluates math expressions through a whitelisted AST interpreter (no `eval`, no code execution). Small local models no longer guess numbers — they compute them.

### 🤝 Professional Git Integration
- **Smart Commits**: Use `/commit` to let the AI analyze your diffs and generate professional Conventional Commit messages.
- **Diff Awareness**: Argent can read its own changes to ensure context consistency.

### 🔌 MCP Server Support (Model Context Protocol)
Integrate external tools and resources seamlessly. Argent supports **stdio**, **SSE**, and **REST** MCP transports to connect to filesystem, github, database, or other custom APIs.

### 📂 Obsidian Integration
Link Argent to your Obsidian vault to automatically create, search, and manage markdown notes, building an external long-term memory and knowledge base.

### 💻 Direct Terminal Execution & Self-Repair
Run local shell commands directly from the prompt by prefixing them with `!`. If a command fails, Argent can analyze the error and automatically suggest fixes.

---

## 🛠 Commands

- `/project [prompt]` — Start a massive multi-step project from scratch.
- `/work [prompt]` — Modify or fix an existing codebase autonomously.
- `/commit` — Generate AI commit message and commit staged changes.
- `/enable_rag` — Enable Semantic Search for the current project.
- `/disable_rag` — Turn off Semantic Search.
- `/rag_provider` — Switch embedding provider (sentence-transformers / Ollama).
- `/browser [mode/name]` — Configure browser automation (mode: isolated/user, name: auto/yandex/chrome/edge/brave).
- `/hooks [path]` — Manage global plugin (hook) directory.
- `/hooks auto [on/off]` — Toggle autonomous AI plugin creation.
- `/research [topic]` — Deep autonomous web research.
- `/tools` — Interactive menu to enable/disable specific AI capabilities.
- `/setup_terminal` — UI optimization guide (Fonts & Colors).
- `/provider` — Select API Provider (Ollama / Z.ai / KoboldCPP) and endpoint.
- `/model` — Select active LLM model.
- `/obsidian [path]` — Set the path to your Obsidian vault.
- `/mcp [subcommand]` — Manage MCP servers (list / add / remove / start / stop / test).
- `/save [name]` — Export the current conversation to a Markdown file.
- `/sessions` — List saved sessions.
- `/load <n>` — Restore a saved session by number.
- `/diff [file]` — Show changes made to files.
- `/undo [file]` — Restore a file to its previous version.
- `/undo_all` — Restore all modified files.
- `/copy <n>` — Copy code block #n to clipboard.
- `/logs [module] [n]` — View logs (e.g. `/logs tools 20`, `/logs error`).
- `/skills` — List available AI skills.
- `/auto [task]` — Run task in experimental full autonomous mode.
- `/verbose` — Toggle live status indicators (spinners).
- `/thinking` — Toggle forced removal of reasoning blocks from history.
- `/temp [value]` — Set or view the model temperature (range: 0.0 - 2.0).
- `/clear` — Clear conversation history.
- `/help` — Show this help message.
- `/exit` (or `/quit`) — Exit the application.

---

## Argent: Элитный ИИ-Ассистент для Программирования

Argent — это высокопроизводительный профессиональный ИИ-напарник, который живет в вашем терминале. Он поддерживает локальные модели **Ollama**, а также API-провайдеров **Z.ai** и **KoboldCPP**, используя продвинутые архитектурные паттерны для создания автономной и безопасной среды разработки.

> [!NOTE]
> Argent является моим **личным экспериментом** по созданию высокоавтономных ИИ-агентов для терминальной разработки.
> [!CAUTION]
> **Внимание**: Проект находится в стадии активной разработки и является экспериментальным. Всё может работать не так, как задумывалось, логика может давать сбои, а код — ломаться. Используйте на свой страх и риск.

---

## 🚀 Основные Возможности

### 🧠 Project Brain (Автономный режим)
Argent не просто отвечает на вопросы — он строит целые проекты. Используя сложную машину состояний, Argent умеет:
- **Исследовать**: Собирать актуальную документацию из сети.
- **Проектировать**: Создавать архитектуру системы верхнего уровня.
- **Специфицировать**: Генерировать детальные описания для каждого файла.
- **Исполнять**: Писать код автономно, управляя собственными лимитами контекста.

### 🧪 TDD Режим (Разработка через тестирование)
Включите строгий TDD, чтобы заставить ИИ следовать профессиональному циклу **Red-Green-Refactor**. Argent будет писать падающие тесты, реализовывать код и проверять результаты перед тем, как двигаться дальше.

### 🪝 Глобальные Плагины и Хуки
Расширяйте логику Argent с помощью Python-плагинов (путь настраивается через `/hooks`).
- **Автономное расширение**: Argent может сам создавать новые инструменты, если это нужно для задачи (включается через `/hooks auto on`).
- **Lifecycle Хуки**: `on_startup`, `pre_prompt`, `post_response`, `on_tool_call`.
- **Свои Команды**: Создавайте функции, начинающиеся с `command_`, чтобы добавить собственные слэш-команды.

### 🔍 Синхронный RAG (Семантический поиск)
Забудьте об устаревших знаниях. Argent автоматически индексирует вашу кодовую базу через **ChromaDB**. При изменении файлов индекс RAG обновляется инкрементально в реальном времени.

### 🌐 Интеллектуальное управление браузером (CDP & Playwright)
Argent содержит мощный движок автоматизации браузера на базе Playwright:
- **Режим CDP (Реальный браузер)**: Подключается напрямую к вашему установленному браузеру (Яндекс.Браузер, Chrome, Edge, Brave) через Chrome DevTools Protocol, используя ваши сессии, куки и авторизации.
- **Автоподключение**: Автоматически «присоединяется» к вашим уже открытым вкладкам для бесшовной совместной работы.
- **Глубокий анализ DOM**: Рекурсивно сканирует интерактивные элементы через границы `iframe` и Shadow DOM.
- **Умная фильтрация**: Фильтрует сложные страницы по ключевым словам (через запятую, логическое ИЛИ) с выводом предупреждения о скрытом контенте для оптимизации контекста ИИ.
- **Гибкое управление**: Кликает и заполняет поля по автоиндексам, CSS-селекторам или тексту с автопрокруткой элементов в зону видимости.

### 🚦 Автоматическая адаптация под размер модели (нулевая настройка)
Argent определяет размер модели по имени и незаметно перестраивает весь конвейер — переключайтесь между локальной 1B-моделью и облачным гигантом, не трогая ни одной настройки:
- **Tiny-модели (Ollama)**: каждый шаг агента **ограничен грамматикой на уровне декодера** строгой JSON-схемой — некорректный tool-call становится физически невозможным, а не «чинится» постфактум. Компактный каталог инструментов и напоминание о цели закрепляются в конце промпта — там, куда маленькие модели реально смотрят.
- **Стандартные локальные модели**: нативные tool-calls плюс якорь цели для длинных историй.
- **Облачные модели**: полностью разгружены — никаких схем, каталогов и напоминаний.
- **Все ярусы**: детерминированный **детектор циклов** ловит дословно повторяющиеся tool-call'ы (классический режим отказа маленьких моделей) и принудительно завершает зациклившиеся ходы; `replace_in_file` автоматически исправляет цели правок со сломанными пробелами через пере-индентацию вместо возврата ошибок.

### 🛡 Централизованные подтверждения и безопасная автономия
Каждое опасное действие (команды оболочки, удаление файлов, git-откаты) проходит через единый шлюз подтверждений:
- **Детектор деструктивных команд**: `rm` / `Remove-Item` / `format` / `git reset --hard` / `taskkill` и подобные всегда требуют явного подтверждения — даже в автономном режиме.
- **Сессионные разрешения**: одобрите один раз с опцией *«всегда разрешать `git` в этой сессии»* — и повторные запросы исчезнут.
- **Рабочая автономия**: в режиме `/auto` безопасные действия одобряются автоматически, поэтому агент действительно может работать без присмотра, а деструктивные — по-прежнему ставятся на паузу.

### 🧮 Точная арифметика
Инструмент `calculate` вычисляет выражения через AST-интерпретатор с белым списком операций (никакого `eval` и исполнения кода). Маленькие локальные модели больше не угадывают числа — они их считают.

### 🤝 Профессиональная интеграция с Git
- **Умные коммиты**: Используйте `/commit`, чтобы ИИ проанализировал ваши diff'ы и составил профессиональные сообщения в стиле Conventional Commits.
- **Понимание Diff**: Argent видит собственные изменения для обеспечения целостности контекста.

### 🔌 Поддержка MCP-серверов (Model Context Protocol)
Бесшовная интеграция внешних инструментов и ресурсов. Argent поддерживает транспорты **stdio**, **SSE** и **REST** для подключения к файловой системе, GitHub, базам данных и любым другим сторонним API.

### 📂 Интеграция с Obsidian
Подключите Argent к вашему хранилищу (Vault) Obsidian. ИИ сможет автоматически создавать, искать и редактировать заметки, формируя внешнюю базу знаний и долгосрочную память.

### 💻 Прямой запуск команд терминала с автоисправлением
Выполняйте консольные команды прямо из ввода Argent, добавив префикс `!`. В случае ошибки выполнения ИИ проанализирует вывод и предложит варианты исправления.

---

## 🛠 Команды

- `/project [prompt]` — Запустить создание масштабного проекта с нуля.
- `/work [prompt]` — Автономно модифицировать или починить существующий код.
- `/commit` — Сгенерировать AI-сообщение и закоммитить изменения.
- `/enable_rag` — Включить семантический поиск по текущему проекту.
- `/disable_rag` — Выключить семантический поиск.
- `/rag_provider` — Сменить провайдер эмбеддингов (sentence-transformers / Ollama).
- `/browser [mode/name]` — Настройка автоматизации браузера (режим: isolated/user, имя: auto/yandex/chrome/edge/brave).
- `/hooks [path]` — Управление папкой глобальных плагинов.
- `/hooks auto [on/off]` — Переключить режим создания плагинов самим ИИ.
- `/research [topic]` — Глубокое автономное исследование темы в сети.
- `/tools` — Интерактивное меню для настройки инструментов ИИ.
- `/setup_terminal` — Гайд по настройке интерфейса (Шрифты и Цвета).
- `/provider` — Выбрать провайдера API (Ollama / Z.ai / KoboldCPP) и эндпоинт.
- `/model` — Выбрать активную модель ИИ.
- `/obsidian [path]` — Задать путь к хранилищу Obsidian.
- `/mcp [subcommand]` — Управление MCP-серверами (list / add / remove / start / stop / test).
- `/save [name]` — Экспортировать историю текущего диалога в Markdown-файл.
- `/sessions` — Показать сохраненные сессии диалогов.
- `/load <n>` — Восстановить сохраненную сессию по номеру.
- `/diff [file]` — Показать изменения, внесенные в файлы.
- `/undo [file]` — Откатить файл к предыдущей сохраненной версии.
- `/undo_all` — Откатить все измененные файлы к исходному состоянию.
- `/copy <n>` — Скопировать блок кода №n из последнего ответа в буфер обмена.
- `/logs [module] [n]` — Посмотреть логи (например, `/logs tools 20`, `/logs error`).
- `/skills` — Показать список доступных навыков ИИ.
- `/auto [task]` — Запустить выполнение задачи в экспериментальном полностью автономном режиме.
- `/verbose` — Включить/выключить интерактивные спиннеры статуса.
- `/thinking` — Включить/выключить принудительное удаление рассуждений из истории контекста.
- `/temp [value]` — Просмотреть или задать температуру генерации модели (от 0.0 до 2.0).
- `/clear` — Очистить историю текущего диалога.
- `/help` — Показать справку по командам.
- `/exit` (или `/quit`) — Выйти из приложения.

---

## 🧑‍💻 Development

```bash
pip install -r requirements.txt   # full runtime dependencies
python -m pytest                  # unit tests (browser tests excluded)
python -m pytest -m integration   # browser tests (launch a real CDP browser)
```

CI runs the unit suite on `windows-latest` for every push and pull request
using the lightweight `requirements-ci.txt` set.