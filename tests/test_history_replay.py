"""A restored session is shown again as blocks, not as two clipped lines."""

import json

import pytest

from src.agent import history_replay as hr


def _roles(items):
    return [i["role"] for i in items]


def test_user_assistant_and_tool_blocks_in_order():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "почини меню"},
        {"role": "assistant", "content": "Смотрю файл.", "thinking": "надо прочитать",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": json.dumps({"file_path": "menu.py"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "1\tdef menu(): ..."},
        {"role": "assistant", "content": "Готово."},
    ]
    items = hr.build_items(messages)
    assert _roles(items) == ["user", "assistant", "tool", "assistant"]
    tool = items[2]
    assert tool["name"] == "read_file" and tool["args"] == {"file_path": "menu.py"}
    assert tool["result"].startswith("1\tdef menu")
    assert items[1]["thinking"] == "надо прочитать"


def test_results_find_their_call_by_id_even_when_out_of_order():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "function": {"name": "one", "arguments": "{}"}},
            {"id": "b", "function": {"name": "two", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "result-two"},
        {"role": "tool", "tool_call_id": "a", "content": "result-one"},
    ]
    tools = {i["name"]: i["result"] for i in hr.build_items(messages) if i["role"] == "tool"}
    assert tools == {"one": "result-one", "two": "result-two"}


def test_providers_without_call_ids_answer_in_call_order():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "one", "arguments": {}}}, {"function": {"name": "two", "arguments": {}}}]},
        {"role": "tool", "content": "first"},
        {"role": "tool", "content": "second"},
    ]
    tools = [(i["name"], i["result"]) for i in hr.build_items(messages) if i["role"] == "tool"]
    assert tools == [("one", "first"), ("two", "second")]


def test_a_call_that_never_finished_says_so():
    messages = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "x", "function": {"name": "run_command", "arguments": "{}"}}]}]
    tool = [i for i in hr.build_items(messages) if i["role"] == "tool"][0]
    assert tool["result"] == "не завершено"


def test_long_results_and_arguments_are_clipped():
    messages = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "w", "function": {"name": "write_file",
                                             "arguments": json.dumps({"file_path": "a.py", "content": "x" * 5000})}}]},
                {"role": "tool", "tool_call_id": "w", "content": "y" * 10000}]
    tool = [i for i in hr.build_items(messages) if i["role"] == "tool"][0]
    assert len(tool["args"]["content"]) < 400 and "симв." in tool["args"]["content"]
    assert len(tool["result"]) < hr.MAX_RESULT_CHARS + 60


def test_only_the_last_turns_are_shown_and_the_rest_is_counted():
    messages = [{"role": "system", "content": "s"}]
    for i in range(10):
        messages += [{"role": "user", "content": f"request {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    items = hr.build_items(messages, max_turns=3)
    users = [i["text"] for i in items if i["role"] == "user"]
    assert users == ["request 7", "request 8", "request 9"]
    assert items[0]["role"] == "note" and "7" in items[0]["text"]


def test_a_compaction_summary_becomes_a_note_and_other_system_messages_are_hidden():
    messages = [
        {"role": "system", "content": "You are Argent."},
        {"role": "system", "content": "=== CONTEXT DISTILLATE (compression #2) ===\nbody\n=== END DISTILLATE ==="},
        {"role": "user", "content": "дальше"},
        {"role": "assistant", "content": "ок"},
    ]
    items = hr.build_items(messages)
    assert _roles(items) == ["note", "user", "assistant"]
    assert "CONTEXT DISTILLATE" in items[0]["text"] or "сводк" in items[0]["text"]


def test_list_content_with_images_is_flattened():
    messages = [{"role": "user", "content": [{"type": "text", "text": "что здесь?"},
                                             {"type": "image_url", "image_url": {"url": "data:..."}}]}]
    items = hr.build_items(messages)
    assert items[0]["text"] == "что здесь?\n[изображение]"


def test_show_sends_one_history_event_with_a_frame():
    sent = []
    hr.set_sink(sent.append)
    try:
        ok = hr.show([{"role": "system", "content": "s"}, {"role": "user", "content": "привет"},
                      {"role": "assistant", "content": "здравствуйте"}], replace=True, saved_at="2026-10-11 10:00")
    finally:
        hr.set_sink(None)
    assert ok and len(sent) == 1
    event = sent[0]
    assert event["type"] == "history" and event["replace"] is True
    assert event["items"][0]["role"] == "note" and "2026-10-11 10:00" in event["items"][0]["text"]
    assert event["items"][-1]["role"] == "note"


def test_empty_conversation_shows_nothing():
    sent = []
    hr.set_sink(sent.append)
    try:
        assert hr.show([{"role": "system", "content": "s"}]) is False
    finally:
        hr.set_sink(None)
    assert sent == []


def test_load_replaces_the_screen_before_the_notes(monkeypatch):
    """`/load` in the middle of a chat: the history event comes first, the 'restored' note after it."""
    from unittest.mock import MagicMock
    from src.cli.commands import session_commands as sc

    order = []
    hr.set_sink(lambda ev: order.append(("history", ev["replace"])))
    data = {"id": "s9", "saved_at": "2026-10-11T09:00:00", "model": "", "cwd": "",
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "привет"},
                         {"role": "assistant", "content": "здравствуйте"}]}
    monkeypatch.setattr(sc, "load_session", lambda sid: data)
    monkeypatch.setattr(sc, "print_system", lambda text, *a, **k: order.append(("note", text)))
    try:
        agent = MagicMock()
        agent.repair_history.return_value = False
        assert sc.restore_session(agent, {"id": "s9"}) is True
    finally:
        hr.set_sink(None)
    assert order[0] == ("history", True)
    assert any(kind == "note" and "Восстановлено" in text for kind, text in order[1:])
