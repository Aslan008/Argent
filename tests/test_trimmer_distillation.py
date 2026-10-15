import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from memory_manager import memory
from src.agent.trimmer import (
    _is_substantive,
    _find_active_goal,
    _find_active_window_start,
    _prepare_for_summarizer,
    _extract_deterministic_facts,
    _infer_tool_context,
    _build_aged_stub,
    age_tool_results,
    _rehydrate_files,
    _create_distillate,
    _update_memory_from_messages,
    hard_reset_with_memory,
    soft_trim_with_summarization,
)


def test_is_substantive():
    # Empty / short
    assert not _is_substantive("")
    assert not _is_substantive("   ")
    assert not _is_substantive("short")
    assert not _is_substantive("ок")
    assert not _is_substantive("продолжай")
    assert not _is_substantive("давай дальше!")
    
    # Auto impulse / heartbeat / commands
    assert not _is_substantive("[Режим Автоматизма] Проанализируй состояние...")
    assert not _is_substantive("[Heartbeat] continue")
    assert not _is_substantive("/auto 10")
    
    # Substantive user tasks
    assert _is_substantive("Добавь новые кнопки в интерфейс игры")
    assert _is_substantive("Перепиши функцию calculate_score в Game.ts")
    assert _is_substantive("Why is the screen black when loading the game?")


def test_find_active_goal():
    messages = [
        {"role": "system", "content": "You are Argent."},
        {"role": "user", "content": "Старая задача: улучши UI игры"},
        {"role": "assistant", "content": "Делаю UI..."},
        {"role": "user", "content": "ок"},
        {"role": "user", "content": "Актуальная задача: добавь баннер Яндекс Игр"},
        {"role": "assistant", "content": "Работаю над баннером..."},
        {"role": "user", "content": "продолжай"},
        {"role": "assistant", "content": "Завершаю..."},
    ]
    # If active window starts at index 6:
    goal = _find_active_goal(messages, active_window_start=6)
    assert goal == "Актуальная задача: добавь баннер Яндекс Игр"

    # If active window starts at index 4 (so the actual goal is inside the active window):
    goal_early = _find_active_goal(messages, active_window_start=4)
    assert goal_early == "Старая задача: улучши UI игры"


def test_find_active_window_start():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Goal 1: do something"},
        {"role": "assistant", "content": "resp 1"},
        {"role": "user", "content": "Goal 2: do something else"},
        {"role": "assistant", "content": "resp 2"},
        {"role": "user", "content": "Goal 3: finish up"},
        {"role": "assistant", "content": "resp 3"},
    ]
    # 3 cycles: should start at index 1 (Goal 1)
    start = _find_active_window_start(messages, keep_cycles=3, model_name="mock", token_budget=10000)
    assert start == 1

    # 2 cycles: should start at index 3 (Goal 2)
    start_2 = _find_active_window_start(messages, keep_cycles=2, model_name="mock", token_budget=10000)
    assert start_2 == 3


def test_prepare_for_summarizer():
    long_file_content = "line\n" * 200
    messages = [
        {"role": "user", "content": "Read the main file and fix errors."},
        {
            "role": "assistant",
            "content": "I will read the file and inspect it.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "read_file", "arguments": json.dumps({"file_path": "src/main.ts"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": long_file_content},
    ]

    distilled = _prepare_for_summarizer(messages)
    assert "src/main.ts" in distilled
    assert "truncated" in distilled or "chars" in distilled
    assert len(distilled) < len(long_file_content)  # Massively reduced


def test_extract_deterministic_facts():
    messages = [
        {
            "role": "assistant",
            "content": "Editing game file\nReplacing score logic",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "read_file", "arguments": json.dumps({"file_path": "src/Game.ts"})},
                },
                {
                    "id": "c2",
                    "function": {"name": "replace_in_file", "arguments": json.dumps({"file_path": "src/Game.ts"})},
                },
                {
                    "id": "c3",
                    "function": {"name": "run_command", "arguments": json.dumps({"command": "npm test"})},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c3", "content": "Error: test failed on line 42 in Game.spec.ts"},
        {"role": "user", "content": "Пожалуйста исправь эту ошибку в тестах"},
    ]

    facts = _extract_deterministic_facts(messages)
    assert "src/Game.ts" in facts
    assert "npm test" in facts
    assert "Error: test failed" in facts
    assert "Пожалуйста исправь эту ошибку в тестах" in facts


def test_in_flight_aging():
    messages = [
        {"role": "system", "content": "system"},
        # Turn 1 (old)
        {"role": "user", "content": "Read file A"},
        {
            "role": "assistant",
            "content": "Reading...",
            "tool_calls": [{"id": "call_a", "function": {"name": "read_file", "arguments": json.dumps({"file_path": "A.ts"})}}],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "A" * 500},
        # More turns to exceed age_threshold=2
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "resp 2"},
        {"role": "user", "content": "Turn 3"},
        {"role": "assistant", "content": "resp 3"},
        {"role": "user", "content": "Turn 4 (recent)"},
        {
            "role": "assistant",
            "content": "Reading recent...",
            "tool_calls": [{"id": "call_b", "function": {"name": "read_file", "arguments": json.dumps({"file_path": "B.ts"})}}],
        },
        {"role": "tool", "tool_call_id": "call_b", "content": "B" * 500},
    ]

    aged = age_tool_results(messages, age_threshold=2)
    # Old tool result at index 3 should be aged
    assert aged[3]["content"].startswith("[aged: read_file(A.ts)")
    # Recent tool result at index 10 should NOT be aged
    assert aged[10]["content"] == "B" * 500


def test_rehydrate_files():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as tf:
        tf.write("export const BANNER_ACTIVE = true;\nconsole.log('banner ready');\n")
        temp_path = tf.name

    try:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps({"file_path": temp_path}),
                        },
                    }
                ],
            }
        ]
        rehydrated = _rehydrate_files(messages, max_files=1)
        assert "BANNER_ACTIVE = true" in rehydrated
        assert os.path.basename(temp_path) in rehydrated
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_update_memory_from_messages():
    memory.clear()
    messages = [
        {"role": "user", "content": "Реализуй поддержку баннера Яндекс Игр"},
        {"role": "assistant", "content": "Добавил баннер в UIManager.ts\nНастроил стили"},
    ]
    _update_memory_from_messages(messages)
    assert memory.data["current_task"] == "Реализуй поддержку баннера Яндекс Игр"
    assert any("Добавил баннер" in act for act in memory.data["completed"])


def test_hard_reset_with_memory():
    memory.clear()
    messages = [
        {"role": "system", "content": "System prompt instructions"},
        {"role": "user", "content": "Сделай баннерную рекламу"},
        {
            "role": "assistant",
            "content": "Working on banner...",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "write_file", "arguments": json.dumps({"file_path": "src/Banner.ts"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Successfully wrote 50 lines."},
        {"role": "user", "content": "Как успехи?"},
    ]

    reset_msgs = hard_reset_with_memory(messages, model_name="mock")
    # Must preserve system prompt
    assert reset_msgs[0]["role"] == "system"
    assert reset_msgs[0]["content"] == "System prompt instructions"
    # Second message must be distillate
    assert reset_msgs[1]["role"] == "system"
    assert "CONTEXT DISTILLATE (emergency reset)" in reset_msgs[1]["content"]
    assert "**ACTIVE GOAL:** Сделай баннерную рекламу" in reset_msgs[1]["content"]
    assert "src/Banner.ts" in reset_msgs[1]["content"]
    # Last user message preserved
    assert reset_msgs[-1]["role"] == "user"
    assert reset_msgs[-1]["content"] == "Как успехи?"


def test_soft_trim_with_summarization():
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Улучши производительность игры"},
        {"role": "assistant", "content": "Оптимизирую рендер..."},
        {"role": "user", "content": "ок"},
        {"role": "user", "content": "Теперь добавь баннер Яндекс Игр"},
        {"role": "assistant", "content": "Работаю над баннером..."},
        {"role": "tool", "tool_call_id": "c1", "content": "Updated UIManager.ts"},
        {"role": "assistant", "content": "Баннер почти готов."},
        {"role": "user", "content": "Проверь стили баннера"},
        {"role": "assistant", "content": "Стили проверены."},
    ]

    with patch("src.agent.trimmer._summarize_with_llm", return_value="## Done\n- Оптимизирован рендер\n## Current State\n- Проверка стилей"):
        trimmed = soft_trim_with_summarization(
            messages=messages,
            model_name="mock",
            max_history_messages=4,
            max_context_tokens=1000,
        )

        # Distillate was injected
        has_distillate = any("CONTEXT DISTILLATE" in m.get("content", "") for m in trimmed)
        assert has_distillate

        # The active window is preserved at the tail
        assert trimmed[-1]["role"] == "assistant"
        assert trimmed[-1]["content"] == "Стили проверены."


def test_handle_slash_command_compact():
    from command_handler import handle_slash_command

    mock_agent = MagicMock()
    mock_agent.messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Сделай фичу 1"},
        {"role": "assistant", "content": "Ответ 1"},
        {"role": "user", "content": "Сделай фичу 2"},
        {"role": "assistant", "content": "Ответ 2"},
        {"role": "user", "content": "Сделай фичу 3"},
        {"role": "assistant", "content": "Ответ 3"},
        {"role": "user", "content": "Сделай фичу 4"},
        {"role": "assistant", "content": "Ответ 4"},
    ]
    mock_agent.max_history_messages = 4
    mock_agent.max_context_tokens = 1000
    mock_agent.model_name = "mock"

    with patch("src.agent.trimmer._summarize_with_llm", return_value="## Done\n- Фича 1"):
        res = handle_slash_command("/compact", mock_agent)
        assert res is False  # Does not exit REPL
        assert any("CONTEXT DISTILLATE" in m.get("content", "") for m in mock_agent.messages)


def test_soft_trim_large_history_actually_compresses():
    """Verify that a long 50+ message conversation actually drops old messages
    and inserts a distillate rather than returning unchanged history.
    """
    messages = [{"role": "system", "content": "System prompt"}]
    for i in range(25):
        messages.append({"role": "user", "content": f"User query {i}"})
        messages.append({"role": "assistant", "content": f"Assistant response {i}"})

    assert len(messages) == 51

    with patch("src.agent.trimmer._summarize_with_llm", return_value="## Done\n- Summary of old turns"):
        trimmed = soft_trim_with_summarization(
            messages=messages,
            model_name="mock",
            max_history_messages=20,
            max_context_tokens=30000,
        )
        assert len(trimmed) < len(messages)
        assert len(trimmed) <= 12
        assert any("CONTEXT DISTILLATE" in m.get("content", "") for m in trimmed)
        assert trimmed[-1]["content"] == "Assistant response 24"


