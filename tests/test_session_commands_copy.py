# -*- coding: utf-8 -*-
"""Tests for /copy command and clipboard integration."""

import pytest
from unittest.mock import MagicMock
from src.cli.commands.base import CommandContext
from src.cli.commands.session_commands import handle_copy


def test_handle_copy_last_assistant_message(monkeypatch):
    copied = []
    monkeypatch.setattr("src.cli.input_queue.set_clipboard_text", lambda text: copied.append(text) or True)

    agent = MagicMock()
    agent.messages = [
        {"role": "user", "content": "Вопрос"},
        {"role": "assistant", "content": "Полный ответ модели с деталями."},
    ]

    ctx = CommandContext(agent=agent, user_input="/copy")
    res = handle_copy(ctx)
    assert not res.should_exit
    assert copied == ["Полный ответ модели с деталями."]


def test_handle_copy_code_blocks(monkeypatch):
    copied = []
    monkeypatch.setattr("src.cli.input_queue.set_clipboard_text", lambda text: copied.append(text) or True)

    agent = MagicMock()
    agent.messages = [
        {"role": "user", "content": "Напиши код"},
        {"role": "assistant", "content": "Вот код:\n```python\ndef foo():\n    return 42\n```\nПользуйся!"},
    ]

    ctx = CommandContext(agent=agent, user_input="/copy code")
    res = handle_copy(ctx)
    assert not res.should_exit
    assert len(copied) == 1
    assert "def foo():" in copied[0]
    assert "return 42" in copied[0]
    assert "Вот код:" not in copied[0]


def test_handle_copy_empty_messages(monkeypatch):
    agent = MagicMock()
    agent.messages = []
    ctx = CommandContext(agent=agent, user_input="/copy")
    res = handle_copy(ctx)
    assert not res.should_exit
