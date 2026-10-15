# -*- coding: utf-8 -*-
"""scripts/_d4_gen_unfamiliar_negs.py — генерация недостающих UNFAMILIAR-негативов.

Проблема (аудит _d4_corpus_audit.py): в корпусе 1530 routing-UNFAMILIAR строк,
все — деструктивные/OOD («сотри логи», «rm -rf», «water the plants»). Русские
СОСТАВНЫЕ задачи (доминирующие ошибки батареи: expected=UNFAMILIAR ->
best=run_command/read_file ×12) в корпусе отсутствуют полностью.

Генерируем routing-строки (формат dispatcher._build_state) по семьям батареи:
планирование/дизайн, ревью, тесты, исследование/анализ, отладка-рассуждение,
оптимизация, составные двух-шаговые, документация, вопросы о возможностях.
Все комбинации слотов, дедупликация против корпуса.

Выход: src/leeth/data_400m/leeth_400m_train_extra.jsonl
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path("src/leeth/data_400m/leeth_400m_train.jsonl")
VAL = Path("src/leeth/data_400m/leeth_400m_val.jsonl")
OUT = Path("src/leeth/data_400m/leeth_400m_train_extra.jsonl")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR",
]
P = "Incoming user request in Argent session:"
WD = "Working directory: C:/Users/user/Desktop/Argent"

MODULES = ["auth", "payments", "dispatcher", "runtime", "router", "cache", "api",
           "storage", "notifications", "billing", "search", "export", "sync",
           "users", "orders", "reports", "webhooks", "queue", "scheduler", "metrics"]
FILES = ["config.py", "main.rs", "app.ts", "models.py", "server.go", "utils.js",
         "PaymentService.cs", "index.tsx", "db.py", "auth.py", "api.py", "test_user.py"]
FUNCS = ["calculate_total", "process_payment", "validate_user", "parse_config",
         "handle_request", "compress_data", "serialize_order", "authenticate"]
SERVICES = ["PaymentService", "UserService", "OrderService", "AuthService",
            "CartService", "InventoryService", "NotificationService", "SearchService"]
TOPICS = ["кеширования", "авторизации", "очередей сообщений", "репликации базы",
          "контейнеризации", "observability", "rate limiting", "полнотекстового поиска"]


def routing_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'{P}\n"{normalized}"\n{WD}'


def row(query: str, target: str = "UNFAMILIAR") -> dict:
    return {
        "state": routing_state(query),
        "question_type": "choice",
        "question": TEMPLATE_Q,
        "choices": CHOICES,
        "target_choice_idx": None,
        "target_choice": target,
        "target_noul": None,
        "target_score": None,
        "score_levels": None,
        "target_span_text": None,
    }


def main() -> None:
    queries: list[str] = []

    # ── 1. Планирование / дизайн / архитектура ────────────────────────────────
    for m in MODULES:
        queries += [f"составь план рефакторинга модуля {m}",
                    f"спроектируй архитектуру сервиса {m}",
                    f"составь план миграции {m} на PostgreSQL",
                    f"продумай план внедрения кеширования для {m}"]
    for v in ["2.0", "3.1", "1.5"]:
        queries += [f"составь план релиза версии {v}",
                    f"подготовь roadmap на следующий квартал для версии {v}"]
    queries += ["спроектируй схему базы данных для магазина",
                "спроектируй схему базы данных для соцсети",
                "спроектируй микросервисную архитектуру для маркетплейса",
                "придумай архитектуру системы уведомлений",
                "составь план декомпозиции монолита",
                "спроектируй API для мобильного приложения",
                "составь план перехода на микросервисы",
                "продумай стратегию масштабирования базы"]

    # ── 2. Ревью / анализ качества ────────────────────────────────────────────
    for m in MODULES:
        queries += [f"проведи код-ревью модуля {m}",
                    f"сделай ревью архитектуры {m}"]
    for f in FILES:
        queries += [f"проведи код-ревью файла {f}",
                    f"сделай ревью изменений в {f}"]
    queries += ["сделай ревью плана миграции",
                "сделай ревью архитектуры проекта",
                "проведи ревью пулл-реквеста",
                "оцени качество тестового покрытия",
                "проанализируй соответствие код-стайлу",
                "сделай аудит безопасности модуля auth",
                "проверь архитектуру на соответствие SOLID"]

    # ── 3. Тесты (написание — рассуждение, не запуск) ─────────────────────────
    for m in MODULES:
        queries += [f"напиши тесты для модуля {m}",
                    f"добавь тесты для {m}"]
    for s in SERVICES:
        queries += [f"напиши тесты для {s}",
                    f"напиши unit-тесты для {s}",
                    f"добавь интеграционные тесты для {s}"]
    queries += ["напиши тесты для граничных случаев",
                "составь тест-план для новой фичи",
                "придумай кейсы для нагрузочного тестирования"]

    # ── 4. Исследование / сравнение / анализ ──────────────────────────────────
    for t in TOPICS:
        queries += [f"исследуй лучшие практики {t}",
                    f"изучи современные подходы к {t}",
                    f"проанализируй варианты реализации {t}"]
    queries += ["сравни два подхода к кешированию и предложи лучший",
                "сравни PostgreSQL и MongoDB для наших задач",
                "сравни REST и GraphQL и предложи вариант",
                "проанализируй причины падения производительности",
                "проанализируй причины утечки памяти",
                "разберись, почему растёт время ответа API",
                "исследуй, почему деградирует латентность",
                "проанализируй метрики и найди узкие места"]

    # ── 5. Отладка-рассуждение ────────────────────────────────────────────────
    for f in FUNCS:
        queries += [f"исправь баг в функции {f}",
                    f"почини функцию {f}",
                    f"найди и исправь ошибку в {f}"]
    queries += ["как исправить эту ошибку компиляции?",
                "почему тесты падают?",
                "почему сборка падает на CI?",
                "из-за чего падает прод?",
                "почему приложение медленно запускается?",
                "найди причину падения сервиса",
                "разберись в ошибке сериализации",
                "почему линтер выдаёт ошибки?",
                "как убрать циклическую зависимость?",
                "почему дедлок в потоках?"]

    # ── 6. Оптимизация / улучшение ────────────────────────────────────────────
    for m in MODULES[:10]:
        queries += [f"оптимизируй модуль {m}",
                    f"улучши производительность {m}"]
    queries += ["оптимизируй этот SQL-запрос",
                "оптимизируй потребление памяти",
                "уменьши размер бандла",
                "улучши время холодного старта",
                "оптимизируй рендер списка"]

    # ── 7. Составные двух-шаговые (два действия в одном запросе) ──────────────
    for f in FILES:
        queries += [f"прочитай {f} и исправь ошибки",
                    f"открой {f} и найди баг"]
    queries += ["найди и удали дубликаты в коде",
                "найди и удали мёртвый код",
                "проверь содержимое .env и обнови ключи",
                "запусти тесты и исправь падения",
                "запусти линтер и исправь предупреждения",
                "покажи содержимое src и найди все классы",
                "покажи структуру проекта и найди дубликаты",
                "проверь логи и исправь ошибку деплоя",
                "посмотри метрики и почини алерты",
                "прочитай конфиг и обнови токены",
                "проверь зависимости и обнови версии",
                "найди все TODO и закрой их",
                "собери статистику по багам и составь отчёт"]

    # ── 8. Документация ───────────────────────────────────────────────────────
    for m in ["dispatcher", "runtime", "api", "auth", "billing", "webhooks"]:
        queries += [f"задокументируй API модуля {m}",
                    f"напиши документацию для {m}"]
    queries += ["задокументируй публичные endpoints",
                "составь changelog для релиза",
                "опиши архитектуру в ADR",
                "напиши гайд по развертыванию"]

    # ── 9. Вопросы о возможностях / мета ──────────────────────────────────────
    queries += ["какие у тебя есть возможности?", "что ты умеешь?", "расскажи о себе",
                "как ты работаешь?", "что ты можешь?", "какие инструменты у тебя есть?",
                "перечисли свои возможности", "что ты умеешь делать?",
                "какие у тебя есть навыки?", "чем ты можешь помочь?",
                "какие задачи ты решаешь?", "расскажи, что ты умеешь",
                "как устроен твой рабочий процесс?", "что входит в твои функции?",
                "какие у тебя ограничения?", "с чем ты справляешься лучше всего?"]

    # ── 10. КОНТРАСТНЫЕ ПАРЫ: позитивные близнецы негативов ───────────────────
    # Учат границу на уровне префикса: «прочитай X» -> read_file, но
    # «прочитай X и исправь ошибки» -> UNFAMILIAR. Без них голова схлопывает
    # префикс (провал первой попытки: 'прочитай config.py' -> UNFAMILIAR 0.94).
    positives: list[str] = []
    for f in FILES:
        positives += [(f"прочитай {f}", "read_file"),
                      (f"открой файл {f}", "read_file"),
                      (f"покажи содержимое {f}", "read_file")]
    positives += [("прочитай файл config.py", "read_file"),
                  ("открой файл package.json", "read_file"),
                  ("прочитай AGENTS.md", "read_file"),
                  ("проверь содержимое .env", "read_file"),
                  ("прочитай конфиг", "read_file"),
                  ("запусти тесты", "run_command"),
                  ("запусти линтер", "run_command"),
                  ("прогони тесты", "run_command"),
                  ("запусти сборку", "run_command"),
                  ("покажи содержимое src", "list_directory"),
                  ("покажи структуру проекта", "list_directory"),
                  ("проверь логи", "run_command"),
                  ("проверь зависимости", "run_command"),
                  ("найди дубликаты в коде", "grep_search"),
                  ("найди мёртвый код", "grep_search"),
                  ("найди все TODO", "grep_search"),
                  ("найди все классы в src", "grep_search"),
                  ("поищи строку в коде", "grep_search")]
    for f in ["src/leeth/runtime.py", "config.py", "package.json", "README.md",
              "src/leeth/dispatcher.py", "app.ts", "main.rs"]:
        positives += [(f"исправь импорт в {f}", "replace_in_file"),
                      (f"замени строку подключения в {f}", "replace_in_file"),
                      (f"поправь опечатку в {f}", "replace_in_file"),
                      (f"обнови версию в {f}", "replace_in_file")]
    positives += [("в файле main.py замени print на logger", "replace_in_file"),
                  ("запиши результат в output.txt", "write_file"),
                  ("сохрани отчёт в отчёт.md", "write_file"),
                  ("переведи 1024 МБ в гигабайты", "calculate"),
                  ("посчитай 15 процентов от 2400", "calculate"),
                  ("загрузи страницу github.com", "browser_open"),
                  ("кликни по ссылке habr.com", "browser_click"),
                  ("выполни pytest -x", "run_command"),
                  ("запусти python scripts/main.py", "run_command"),
                  ("найди все файлы с расширением .py", "find_by_name"),
                  ("найди все тестовые файлы test_*.py", "find_by_name"),
                  ("найди где используется replace_in_file", "grep_search"),
                  ("где в коде вызывается decide_batch", "grep_search"),
                  ("поищи строку DENY_PATTERNS", "grep_search"),
                  ("найди импорты torch в src", "grep_search")]

    # ── дедупликация против корпуса ───────────────────────────────────────────
    existing = set()
    for path in (TRAIN, VAL):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            m = re.search(r'"(.+)"', r.get("state", ""))
            if m:
                existing.add(m.group(1).lower())
    before = len(queries)
    seen: set = set()
    deduped = []
    for q_ in queries:
        if q_.lower() in existing or q_.lower() in seen:
            continue
        seen.add(q_.lower())
        deduped.append(q_)
    queries = deduped
    print(f"негативов сгенерировано {before}, после дедупликации {len(queries)}")

    # позитивные близнецы: тоже дедуплицируем против корпуса и негативов
    neg_set = {q_.lower() for q_ in queries}
    pos_rows, pos_seen = [], set()
    for q_, target in positives:
        if q_.lower() in existing or q_.lower() in neg_set or q_.lower() in pos_seen:
            continue
        pos_seen.add(q_.lower())
        pos_rows.append(row(q_, target))
    print(f"позитивных близнецов: {len(pos_rows)}")

    rows = [row(q_) for q_ in queries] + pos_rows
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    print(f"записано {len(rows)} строк ({len(queries)} UNFAMILIAR + {len(pos_rows)} позитивов) -> {OUT}")
    for r in rows[:3]:
        print(" пример:", re.search(r'"(.+)"', r["state"]).group(1)[:80], "->", r["target_choice"])


if __name__ == "__main__":
    main()