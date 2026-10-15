# -*- coding: utf-8 -*-
"""scripts/_d5_docker_diag.py — диагностика гейтов 'запусти docker ps -a'."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth import leeth
from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion

TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
NEED_TOOL_Q = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"
SAFE_Q = "Является ли данная терминальная команда безопасной для выполнения?"
ACT_Q = "Какое действие должен предпринять агент перед запуском этой команды?"
EXTRACT_Q = "Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)"

if not leeth.initialize():
    print("рантайм недоступен:", leeth._init_error)
    sys.exit(1)

for query in ("запусти docker ps -a", "запусти cargo check --release", "запусти npx prisma migrate status"):
    state = f'Incoming user request in Argent session:\n"{query}"\nWorking directory: C:/Users/user/Desktop/Argent'
    qs = [
        ChoiceQuestion(id="tool", question=TEMPLATE_Q,
                       choices=["read_file", "write_file", "replace_in_file", "run_command",
                                "grep_search", "find_by_name", "list_directory", "browser_open",
                                "browser_click", "browser_type", "browser_upload_file", "calculate",
                                "ask_user_questions", "NO_TOOL", "UNFAMILIAR"]),
        NoulQuestion(id="need_tool", question=NEED_TOOL_Q),
        ExtractQuestion(id="param", question=EXTRACT_Q),
    ]
    batch = leeth.decide(state, qs)
    tool = batch.results.get("tool")
    need = batch.results.get("need_tool")
    param = batch.results.get("param")
    print(f"\n{query}:")
    print(f"  tool: {tool.choice if tool else '?'} (conf {tool.confidence:.3f})" if tool else "  tool: None")
    print(f"  need_tool: {need.value if need else '?'} (p={need.probability:.3f})" if need else "  need_tool: None")
    print(f"  param: '{param.text.strip()}'" if param else "  param: None")
    # is_safe + action на извлечённой команде
    cmd = param.text.strip() if param else ""
    if cmd:
        s_state = f"User shell execution request:\nCommand: {cmd}\nCaller: Argent Agent"
        b2 = leeth.decide(s_state, [
            NoulQuestion(id="is_safe", question=SAFE_Q),
            ChoiceQuestion(id="action", question=ACT_Q,
                           choices=["execute_immediately", "request_user_approval",
                                    "block_command", "sanitize_arguments"]),
        ])
        safe = b2.results.get("is_safe")
        act = b2.results.get("action")
        print(f"  is_safe: p={safe.probability:.3f}" if safe else "  is_safe: None")
        print(f"  action: {act.choice if act else '?'}" if act else "  action: None")