"""Unit tests for InputQueue, LiveInputState, TurnInputListener, and /queue command."""

import pytest
from unittest.mock import MagicMock

from src.cli.input_queue import (
    InputQueue,
    LiveInputState,
    TurnInputListener,
    get_input_queue,
    get_live_input_state,
    pause_active_input_listener,
    resume_active_input_listener,
)
from src.cli.commands import dispatch_command, CommandContext


class TestInputQueue:
    def test_push_and_fifo_pop(self):
        q = InputQueue()
        assert not q.has_items()
        assert q.count() == 0

        q.push("first message")
        q.push("second message")
        assert q.count() == 2
        assert q.has_items()
        assert q.items() == ["first message", "second message"]

        assert q.pop() == "first message"
        assert q.pop() == "second message"
        assert q.pop() is None
        assert not q.has_items()

    def test_pop_last_lifo(self):
        q = InputQueue()
        q.push("first")
        q.push("second")
        q.push("third")

        assert q.pop_last() == "third"
        assert q.count() == 2
        assert q.items() == ["first", "second"]

    def test_remove_by_index(self):
        q = InputQueue()
        q.push("item 0")
        q.push("item 1")
        q.push("item 2")

        removed = q.remove(1)
        assert removed == "item 1"
        assert q.items() == ["item 0", "item 2"]
        assert q.remove(99) is None

    def test_edit_by_index(self):
        q = InputQueue()
        q.push("original 0")
        q.push("original 1")

        assert q.edit(0, "edited 0") is True
        assert q.items()[0] == "edited 0"
        assert q.edit(99, "invalid") is False
        assert q.edit(0, "   ") is False

    def test_clear(self):
        q = InputQueue()
        q.push("msg 1")
        q.push("msg 2")

        cleared = q.clear()
        assert cleared == ["msg 1", "msg 2"]
        assert q.count() == 0

    def test_empty_and_whitespace_ignored(self):
        q = InputQueue()
        q.push("")
        q.push("   ")
        assert q.count() == 0


class TestLiveInputState:
    def test_insert_and_cursor(self):
        q = InputQueue()
        state = LiveInputState(q)

        state.insert_char("a")
        state.insert_char("b")
        state.insert_char("c")
        assert state.current_text == "abc"
        assert state.cursor_pos == 3

        state.move_cursor_left()
        assert state.cursor_pos == 2
        state.insert_char("X")
        assert state.current_text == "abXc"
        assert state.cursor_pos == 3

    def test_multiline_insert(self):
        q = InputQueue()
        state = LiveInputState(q)

        state.insert_text("line 1\nline 2")
        assert state.current_text == "line 1\nline 2"
        assert state.cursor_pos == 13

    def test_backspace_and_delete(self):
        q = InputQueue()
        state = LiveInputState(q)

        state.insert_text("hello")
        state.backspace()
        assert state.current_text == "hell"

        state.move_cursor_home()
        state.delete()
        assert state.current_text == "ell"

    def test_cursor_home_end(self):
        q = InputQueue()
        state = LiveInputState(q)

        state.insert_text("testing")
        state.move_cursor_home()
        assert state.cursor_pos == 0
        state.move_cursor_end()
        assert state.cursor_pos == 7

    def test_submit_moves_to_queue(self):
        q = InputQueue()
        state = LiveInputState(q)

        state.insert_text("new prompt")
        res = state.submit()
        assert res == "new prompt"
        assert state.current_text == ""
        assert state.cursor_pos == 0
        assert q.items() == ["new prompt"]

    def test_recall_last_from_queue(self):
        q = InputQueue()
        q.push("first in queue")
        q.push("second in queue")

        state = LiveInputState(q)
        recalled = state.recall_last()

        assert recalled == "second in queue"
        assert state.current_text == "second in queue"
        assert state.cursor_pos == len("second in queue")
        assert q.items() == ["first in queue"]

    def test_clear_buffer(self):
        q = InputQueue()
        state = LiveInputState(q)
        state.insert_text("draft")
        cleared = state.clear()
        assert cleared == "draft"
        assert state.current_text == ""
        assert state.cursor_pos == 0


class TestTurnInputListener:
    def test_listener_context_and_pause(self):
        agent = MagicMock()
        state = LiveInputState(InputQueue())
        listener = TurnInputListener(agent, state=state)

        assert not listener._running
        assert not listener._paused

        with listener:
            assert listener._running
            assert getattr(agent, "_turn_input_listener", None) is listener

            pause_active_input_listener()
            assert listener._paused is True

            resume_active_input_listener()
            assert listener._paused is False

        assert not listener._running

    def test_trigger_stop_calls_agent_stop(self):
        agent = MagicMock()
        state = LiveInputState(InputQueue())
        listener = TurnInputListener(agent, state=state)

        listener._trigger_stop()
        assert agent.request_stop.called or agent._stop_requested is True


class TestQueueSlashCommands:
    def test_queue_list_and_empty(self):
        q = get_input_queue()
        q.clear()

        ctx = CommandContext(
            agent=MagicMock(),
            user_input="/queue list",
            ui_state={},
        )
        res = dispatch_command(ctx.user_input, ctx)
        assert res.handled is True

    def test_queue_push_and_rm(self):
        q = get_input_queue()
        q.clear()

        # Add item via /queue add
        ctx_add = CommandContext(
            agent=MagicMock(),
            user_input="/queue add Fix the bug in auth",
            ui_state={},
        )
        dispatch_command(ctx_add.user_input, ctx_add)
        assert q.items() == ["Fix the bug in auth"]

        # Edit item via /queue edit
        ctx_edit = CommandContext(
            agent=MagicMock(),
            user_input="/queue edit 1 Fix the bug in parser",
            ui_state={},
        )
        dispatch_command(ctx_edit.user_input, ctx_edit)
        assert q.items() == ["Fix the bug in parser"]

        # Remove item via /queue rm
        ctx_rm = CommandContext(
            agent=MagicMock(),
            user_input="/queue rm 1",
            ui_state={},
        )
        dispatch_command(ctx_rm.user_input, ctx_rm)
        assert q.count() == 0

    def test_queue_clear(self):
        q = get_input_queue()
        q.clear()
        q.push("task 1")
        q.push("task 2")

        ctx = CommandContext(
            agent=MagicMock(),
            user_input="/queue clear",
            ui_state={},
        )
        dispatch_command(ctx.user_input, ctx)
        assert q.count() == 0


class TestUIRendering:
    def test_render_queue_input_display_empty(self):
        from src.cli.cli_ui import _render_queue_input_display
        state = get_live_input_state()
        state.clear()
        state.queue.clear()

        assert _render_queue_input_display() is None

    def test_render_queue_input_display_with_items_and_typing(self):
        from src.cli.cli_ui import _render_queue_input_display
        state = get_live_input_state()
        state.clear()
        state.queue.clear()

        state.queue.push("Item 1 in queue")
        state.insert_text("typing message")

        disp = _render_queue_input_display()
        assert disp is not None
        # Clean up after test
        state.clear()
        state.queue.clear()
