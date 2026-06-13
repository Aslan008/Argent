"""
Mini model benchmark for Argent.

Runs a small set of agent-shaped tasks against one or more models on the
ACTIVE provider and prints a scorecard, so picking a model becomes a table
instead of a feeling. Focuses on the capabilities Argent actually relies on:
native tool selection, argument validity, preferring the calculator over
mental math, instruction following, code generation and tool restraint.

Usage:
    python scripts/benchmark_models.py                      # current model
    python scripts/benchmark_models.py qwen3.5:9b gemma4:e4b
    python scripts/benchmark_models.py --provider ollama m1 m2

For tiny models that work through constrained decoding rather than native
tool calls, see scripts/check_constrained.py instead.
"""

import ast
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers import create_provider, ProviderError


READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string", "description": "Path to the file"}},
            "required": ["file_path"],
        },
    },
}

CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression exactly. Use this instead of doing math yourself.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


def _called(tool_calls, name):
    return any(tc.get("name") == name for tc in tool_calls)


def _args_of(tool_calls, name):
    for tc in tool_calls:
        if tc.get("name") == name:
            return tc.get("arguments")
    return None


TASKS = [
    {
        "name": "tool_select",
        "prompt": "Прочитай файл с именем app.py.",
        "tools": [READ_FILE_TOOL],
        "check": lambda content, tcs: _called(tcs, "read_file")
                 and "app.py" in json.dumps(_args_of(tcs, "read_file") or {}),
    },
    {
        "name": "args_valid",
        "prompt": "Открой файл src/main.py и покажи его содержимое.",
        "tools": [READ_FILE_TOOL],
        "check": lambda content, tcs: isinstance(_args_of(tcs, "read_file"), dict)
                 and "file_path" in (_args_of(tcs, "read_file") or {}),
    },
    {
        "name": "calc_tool",
        "prompt": "Сколько будет 1847 умножить на 23? Посчитай точно.",
        "tools": [CALCULATE_TOOL],
        "check": lambda content, tcs: _called(tcs, "calculate"),
    },
    {
        "name": "follow_instruction",
        "prompt": "Ответь ТОЛЬКО одним словом: столица Японии.",
        "tools": None,
        "check": lambda content, tcs: ("токио" in content.lower() or "tokyo" in content.lower())
                 and len(content.split()) <= 4,
    },
    {
        "name": "code_compiles",
        "prompt": "Напиши функцию на Python is_even(n), возвращающую True для чётных. Только код.",
        "tools": None,
        "check": lambda content, tcs: _code_ok(content),
    },
    {
        "name": "tool_restraint",
        "prompt": "Поздоровайся со мной в одном коротком предложении.",
        "tools": [READ_FILE_TOOL, CALCULATE_TOOL],
        "check": lambda content, tcs: len(tcs) == 0 and len(content.strip()) > 0,
    },
]


def _code_ok(content: str) -> bool:
    code = content
    if "```" in code:
        # Pull out the first fenced block.
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if code.startswith("python"):
                code = code[len("python"):]
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return "def is_even" in code


def run_task(provider, model, task):
    """Return (passed, elapsed_seconds, error_or_None)."""
    messages = [{"role": "user", "content": task["prompt"]}]
    content = ""
    tool_acc = {}
    start = time.time()
    try:
        for chunk in provider.stream_chat(model=model, messages=messages,
                                          tools=task["tools"], temperature=0.1):
            content += chunk.get("content", "")
            for d in chunk.get("tool_call_deltas", []):
                i = d["index"]
                slot = tool_acc.setdefault(i, {"name": "", "arguments": ""})
                slot["name"] += d.get("function_name_delta", "")
                slot["arguments"] += d.get("function_arguments_delta", "")
    except (ProviderError, Exception) as e:
        return False, time.time() - start, str(e)[:80]

    tool_calls = []
    for slot in tool_acc.values():
        args = slot["arguments"]
        try:
            args = json.loads(args) if isinstance(args, str) and args.strip() else (args or {})
        except json.JSONDecodeError:
            pass
        tool_calls.append({"name": slot["name"], "arguments": args})

    try:
        passed = bool(task["check"](content, tool_calls))
    except Exception:
        passed = False
    return passed, time.time() - start, None


def main():
    args = [a for a in sys.argv[1:]]
    provider_name = None
    if "--provider" in args:
        i = args.index("--provider")
        provider_name = args[i + 1]
        del args[i:i + 2]

    import config
    models = args or [config.get_current_model()]
    provider = create_provider(provider_name)

    err = provider.validate_config()
    if err:
        print(f"Provider not ready: {err}")
        return

    print(f"Provider: {provider.name} | Tasks: {len(TASKS)} | Models: {len(models)}\n")
    task_names = [t["name"] for t in TASKS]
    col_w = max(len(n) for n in task_names + ["MODEL"]) + 1

    header = "MODEL".ljust(28) + "".join(n[:col_w - 1].ljust(col_w) for n in task_names) + "SCORE   TIME"
    print(header)
    print("-" * len(header))

    for model in models:
        cells = []
        passed_count = 0
        total_time = 0.0
        for task in TASKS:
            ok, elapsed, error = run_task(provider, model, task)
            total_time += elapsed
            passed_count += int(ok)
            cells.append(("PASS" if ok else "fail").ljust(col_w))
        row = model[:27].ljust(28) + "".join(cells)
        row += f"{passed_count}/{len(TASKS)}".ljust(8) + f"{total_time:.1f}s"
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
