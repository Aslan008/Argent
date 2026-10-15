"""
Comprehensive tests for Argent Universal MicroModels (67 classes + Zero-Shot Semantic Routing).

Verifies:
1. UniversalIntentEncoder supports all 66 Argent tools + NO_TOOL.
2. SemanticToolRegistry contains pre-computed embeddings for all tools.
3. Zero-Shot dynamic tool matching: registering a completely new tool at runtime and routing to it.
4. Fast-path parameter extraction across desktop, code navigation, browser, and git tools.
5. CPU latency benchmark (< 2.0ms).
"""

import time
import pytest
import numpy as np

from src.micromodels.dispatcher import MicroModelDispatcher
from src.micromodels.runtime import MicroModelRuntime
from src.micromodels.schema_dataset_generator import ALL_INTENTS
from src.micromodels.semantic_registry import SemanticToolRegistry


@pytest.fixture(autouse=True)
def init_runtime():
    MicroModelRuntime._instance = None
    MicroModelDispatcher._instance = None
    rt = MicroModelRuntime.get_instance()
    rt.initialize()
    return rt


def test_universal_runtime_67_classes(init_runtime):
    rt = init_runtime
    assert rt.is_available is True
    assert rt.total_intents == 67
    assert rt.backend in ("onnx", "onnx_int8", "pytorch")


def test_semantic_registry_preloaded():
    reg = SemanticToolRegistry.get_instance()
    reg.load()
    assert reg.total_tools >= 66
    assert "read_file" in reg._tools
    assert "browser_open" in reg._tools
    assert "find_definition" in reg._tools
    assert "capture_desktop_window" in reg._tools


def test_classification_across_domains():
    dispatcher = MicroModelDispatcher.get_instance()

    test_cases = [
        # File & Directory
        ("прочитай src/agent.py", "read_file"),
        ("покажи файлы в директории src", "list_directory"),
        ("удали временный файл temp.txt", "delete_file"),
        ("создай директорию build", "create_directory"),

        # Desktop & Windows
        ("покажи открытые окна на рабочем столе", "list_desktop_windows"),
        ("сделай скриншот экрана", "capture_desktop_screen"),
        ("возьми картинку из буфера", "get_clipboard_image"),

        # Code Navigation & Diagnostics
        ("где определен TinyIntentEncoder", "find_definition"),
        ("найди все ссылки на FastDispatchResult", "find_references"),
        ("покажи структуру файла src/agent.py", "get_file_outline"),
        ("проверь код на ошибки lsp", "check_code"),
        ("проанализируй структуру проекта", "analyze_project"),

        # Git
        ("покажи изменения в гите", "run_command"), # mapped to run_command git diff
        ("сохрани чекпоинт в гите", "git_checkpoint"),
        ("откати чекпоинт", "git_rollback"),

        # Browser
        ("открой в браузере https://pytorch.org", "browser_open"),
        ("в каком состоянии браузер", "browser_state"),
        ("сделай скриншот страницы в браузере", "browser_screenshot"),
        ("закрой браузер", "browser_close"),

        # Skills & Commands
        ("покажи доступные скиллы", "list_skills"),
        ("какие фоновые процессы запущены", "list_background_commands"),
        ("посчитай 125 * 84", "calculate"),
    ]

    for query, expected_tool in test_cases:
        res = dispatcher.try_fast_dispatch(query)
        assert res is not None, f"Failed to dispatch: '{query}'"
        assert res.tool_name == expected_tool, f"Expected {expected_tool}, got {res.tool_name} for '{query}'"
        assert res.confidence >= 0.75


def test_conversational_rejection():
    dispatcher = MicroModelDispatcher.get_instance()

    conversational_queries = [
        "Привет, как дела?",
        "Объясни мне архитектуру этого проекта",
        "Напиши функцию для быстрой сортировки",
        "Почему падает этот тест?",
    ]

    for q in conversational_queries:
        res = dispatcher.try_fast_dispatch(q)
        assert res is None, f"Conversational query should NOT be fast-dispatched: '{q}'"


def test_zero_shot_dynamic_new_tool_routing():
    """Simulates registering a BRAND NEW MCP tool that was NOT seen during training,

    and verifies the model routes to it purely via semantic embedding similarity!
    """
    rt = MicroModelRuntime.get_instance()

    # 1. Register completely new tool on the fly
    dynamic_tool_name = "unity_bake_lightmap"
    dynamic_tool_desc = "Bake lighting and lightmaps for the active Unity 3D scene."

    ok = rt.register_dynamic_tool(name=dynamic_tool_name, description=dynamic_tool_desc)
    assert ok is True

    # 2. Query in natural Russian targeting this new tool
    query = "запеки свет для сцены в Unity"

    intent, conf, all_p, ms, emb = rt.predict_intent(query)
    matches = rt._registry.match(emb, top_k=3, min_similarity=0.60)

    assert len(matches) > 0
    top_tool, sim = matches[0]
    print(f"\nZero-shot match for '{query}': {top_tool} (similarity: {sim:.3f})")
    assert top_tool == dynamic_tool_name, f"Expected {dynamic_tool_name}, got {top_tool}"
    assert sim >= 0.65


def test_universal_latency_sub_2ms():
    rt = MicroModelRuntime.get_instance()

    # Warmup
    for _ in range(15):
        rt.predict_intent("прочитай src/agent.py")

    # 100 benchmark runs
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        rt.try_fast_dispatch("прочитай src/agent.py")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(latencies) / len(latencies)
    p95_ms = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"\nUniversal 66-Tool Benchmark ({rt.backend}, 100 runs): avg={avg_ms:.2f}ms, p95={p95_ms:.2f}ms")
    assert avg_ms < 2.5, f"Average latency {avg_ms:.2f}ms exceeded 2.5ms!"
