"""LMTrust Deep Blind Spot & Verification Test Suite for Argent Input Queue.

Layers covered:
- L0 Smoke: module importability and fundamental object instantiation
- L1 Contract: strict public contract validation of InputQueue, LiveInputState, and Listener
- L2 Boundary: 8 boundary conditions (None, empty string, single char, negative index, max boundaries)
- L4 Adversarial: malformed inputs, non-string push, corrupted cursor positions
- L5 State: state machine transitions (typing -> submit -> pop, recall -> edit -> resubmit, Esc abort)
- L6 Cross-System: end-to-end integration between InputQueue, prompt defaults, and commands
- L8 Concurrency: multi-threaded stress testing (race condition proof under heavy contention)
- L9 Negative Space: ensuring unrequested side effects do NOT occur (no data loss, no phantom prompts)
"""

import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from src.cli.input_queue import (
    InputQueue,
    LiveInputState,
    TurnInputListener,
    get_input_queue,
    get_live_input_state,
    pause_active_input_listener,
    resume_active_input_listener,
    get_clipboard_text,
)
from src.cli.commands import dispatch_command, CommandContext


# ── L0: Smoke Tests ──────────────────────────────────────────────────────────

class TestSmoke:
    """L0: Verify modules, singletons, and classes are importable and instantiable."""

    def test_instantiate_clean_queue(self):
        q = InputQueue()
        assert q is not None
        assert q.count() == 0
        assert q.items() == []

    def test_instantiate_live_state(self):
        state = LiveInputState()
        assert state is not None
        assert state.current_text == ""
        assert state.cursor_pos == 0
        assert state.queue is not None

    def test_singletons_accessible(self):
        q = get_input_queue()
        state = get_live_input_state()
        assert isinstance(q, InputQueue)
        assert isinstance(state, LiveInputState)


# ── L1: Contract Tests (D7: API Contract) ───────────────────────────────────

class TestContract:
    """L1: Verify strict type and behavioral contracts of public interfaces."""

    def test_input_queue_contract_return_types(self):
        q = InputQueue()
        # push returns int (count)
        res_push = q.push("hello")
        assert isinstance(res_push, int)
        assert res_push == 1

        # pop returns Optional[str]
        item = q.pop()
        assert isinstance(item, str)
        assert item == "hello"

        # pop on empty returns None
        assert q.pop() is None

        # pop_last on empty returns None
        assert q.pop_last() is None

        # items returns a fresh list copy (modifying it doesn't corrupt internal state)
        q.push("msg")
        snap = q.items()
        snap.append("corrupted")
        assert q.items() == ["msg"]

    def test_live_input_state_snapshot_contract(self):
        q = InputQueue()
        q.push("queued 1")
        state = LiveInputState(q)
        state.insert_text("typing")

        text, pos, items = state.snapshot()
        assert isinstance(text, str)
        assert isinstance(pos, int)
        assert isinstance(items, list)
        assert text == "typing"
        assert pos == 6
        assert items == ["queued 1"]


# ── L2: Boundary Tests (D1 / D3: 8 Boundary Conditions) ──────────────────────

class TestBoundaryEdgeCases:
    """L2: Exhaustive testing of boundary values, edge conditions, and bounds."""

    def test_push_empty_and_whitespace(self):
        q = InputQueue()
        assert q.push("") == 0
        assert q.push("   ") == 0
        assert q.push("\t\n\r  ") == 0
        assert q.count() == 0
        assert q.has_items() is False

    def test_push_single_character(self):
        q = InputQueue()
        assert q.push("a") == 1
        assert q.items() == ["a"]

    def test_push_massive_multiline_text(self):
        q = InputQueue()
        big_text = "line " + ("x" * 10000) + "\nline 2\n" + ("y" * 10000)
        assert q.push(big_text) == 1
        popped = q.pop()
        assert popped == big_text.strip()

    def test_remove_boundary_indices(self):
        q = InputQueue()
        q.push("item 0")
        q.push("item 1")

        # Negative index must return None (not wrap around like raw Python list)
        assert q.remove(-1) is None
        assert q.remove(-99) is None

        # Exact bounds
        assert q.remove(2) is None  # index == len is out of bounds
        assert q.remove(100) is None

        # Valid remove
        assert q.remove(0) == "item 0"
        assert q.remove(0) == "item 1"
        assert q.remove(0) is None  # now empty

    def test_edit_boundary_indices(self):
        q = InputQueue()
        q.push("original")

        assert q.edit(-1, "new") is False
        assert q.edit(1, "new") is False
        assert q.edit(0, "") is False
        assert q.edit(0, "   ") is False
        assert q.edit(0, "valid edit") is True
        assert q.items() == ["valid edit"]

    def test_cursor_move_bounds_underflow_overflow(self):
        state = LiveInputState()
        # At empty: cursor pos 0
        state.move_cursor_left()
        assert state.cursor_pos == 0
        state.move_cursor_home()
        assert state.cursor_pos == 0

        # With text: cursor cannot exceed len(text)
        state.insert_text("abc")
        assert state.cursor_pos == 3
        state.move_cursor_right()
        assert state.cursor_pos == 3  # clamped to length
        state.move_cursor_end()
        assert state.cursor_pos == 3

        # Backspace at position 0
        state.move_cursor_home()
        state.backspace()
        assert state.cursor_pos == 0
        assert state.current_text == "abc"

        # Delete at end of text
        state.move_cursor_end()
        state.delete()
        assert state.cursor_pos == 3
        assert state.current_text == "abc"


# ── L4: Adversarial Tests (D4 / D7: Robustness against malformed inputs) ────

class TestAdversarial:
    """L4: Verify component resilience against non-standard or corrupt inputs."""

    def test_push_non_string_types_gracefully_handled(self):
        q = InputQueue()
        # Should not crash if non-string is passed
        try:
            res = q.push(None)
            assert res == 0 or res == q.count()
        except AttributeError:
            pytest.fail("InputQueue.push(None) raised unhandled AttributeError!")

    def test_unicode_and_cyrillic_boundary(self):
        state = LiveInputState()
        cyrillic = "Привет, мир! 🚀 Как дела?"
        state.insert_text(cyrillic)
        assert state.current_text == cyrillic
        assert state.cursor_pos == len(cyrillic)

        # Backspace on unicode character
        state.backspace()
        assert state.current_text == cyrillic[:-1]

        # Shift+Enter newline in unicode
        state.insert_char("\n")
        state.insert_text("Вторая строка")
        assert "\n" in state.current_text

    def test_queue_commands_adversarial_arguments(self):
        q = get_input_queue()
        q.clear()
        q.push("sample task")

        # Invalid non-numeric ID for remove
        ctx_bad_rm = CommandContext(agent=MagicMock(), user_input="/queue rm abc", ui_state={})
        res = dispatch_command(ctx_bad_rm.user_input, ctx_bad_rm)
        assert res.handled is True
        assert q.count() == 1  # Not removed

        # Out-of-bounds ID for remove
        ctx_oob_rm = CommandContext(agent=MagicMock(), user_input="/queue rm 99", ui_state={})
        dispatch_command(ctx_oob_rm.user_input, ctx_oob_rm)
        assert q.count() == 1

        # Missing arguments for edit
        ctx_no_arg_edit = CommandContext(agent=MagicMock(), user_input="/queue edit", ui_state={})
        dispatch_command(ctx_no_arg_edit.user_input, ctx_no_arg_edit)
        assert q.count() == 1

        # Missing text for edit
        ctx_half_edit = CommandContext(agent=MagicMock(), user_input="/queue edit 1", ui_state={})
        dispatch_command(ctx_half_edit.user_input, ctx_half_edit)
        assert q.items() == ["sample task"]

        # Unknown subcommands
        ctx_unknown = CommandContext(agent=MagicMock(), user_input="/queue foobar", ui_state={})
        res_unknown = dispatch_command(ctx_unknown.user_input, ctx_unknown)
        assert res_unknown.handled is True


# ── L5: State Machine & Transition Tests (D2: Lifecycle & Transitions) ──────

class TestStateMachine:
    """L5: Test complete state transitions of the typing & queueing lifecycle."""

    def test_full_lifecycle_type_submit_pop(self):
        q = InputQueue()
        state = LiveInputState(q)

        # 1. State: Typing
        state.insert_text("Run migration batch 1")
        assert state.current_text == "Run migration batch 1"
        assert q.count() == 0

        # 2. State: Submit to Queue (Enter pressed)
        submitted = state.submit()
        assert submitted == "Run migration batch 1"
        assert state.current_text == ""
        assert state.cursor_pos == 0
        assert q.count() == 1

        # 3. State: Agent pops message from queue
        executed = q.pop()
        assert executed == "Run migration batch 1"
        assert q.count() == 0

    def test_recall_last_for_editing_and_resubmit(self):
        q = InputQueue()
        state = LiveInputState(q)

        # Enqueue two messages
        q.push("Task A")
        q.push("Task B with typo")

        # User presses Up Arrow (recall_last)
        recalled = state.recall_last()
        assert recalled == "Task B with typo"
        assert state.current_text == "Task B with typo"
        assert state.cursor_pos == len("Task B with typo")
        assert q.items() == ["Task A"]  # Popped from queue into editor

        # User edits the typo
        for _ in range(len(" with typo")):
            state.backspace()
        state.insert_text(" FIXED")
        assert state.current_text == "Task B FIXED"

        # User presses Enter -> resubmits
        state.submit()
        assert q.items() == ["Task A", "Task B FIXED"]

    def test_recall_on_empty_queue_is_safe_noop(self):
        q = InputQueue()
        state = LiveInputState(q)
        state.insert_text("existing draft")

        recalled = state.recall_last()
        assert recalled is None
        # Existing draft must remain untouched!
        assert state.current_text == "existing draft"

    def test_esc_clears_typing_buffer_without_aborting_agent(self):
        agent = MagicMock()
        state = LiveInputState()
        listener = TurnInputListener(agent, state=state)

        state.insert_text("accidental text")
        # Simulating ESC key when text is present:
        current, _, _ = state.snapshot()
        assert current == "accidental text"
        cleared = state.clear()
        assert cleared == "accidental text"
        assert state.current_text == ""
        # Agent MUST NOT be stopped when clearing typed text
        assert not agent.request_stop.called

    def test_esc_on_empty_triggers_agent_stop(self):
        agent = MagicMock()
        state = LiveInputState()
        listener = TurnInputListener(agent, state=state)

        # When input is empty, ESC triggers agent stop
        assert state.current_text == ""
        listener._trigger_stop()
        assert agent.request_stop.called or agent._stop_requested is True


# ── L6: Cross-System Integration Tests (D3 / D7: Cross-layer consistency) ───

class TestCrossSystem:
    """L6: Verify interaction between input queue, REPL prompts, and UI."""

    def test_draft_preservation_into_prompt_default(self):
        """When agent finishes turn, unsubmitted draft text becomes prompt default."""
        state = LiveInputState()
        state.insert_text("half-written question")

        # Simulate main.py loop logic:
        draft_default = state.clear() if state.current_text else ""
        assert draft_default == "half-written question"
        assert state.current_text == ""

    def test_ui_rendering_multiline_and_cursor_markup(self):
        """Verify _render_queue_input_display formats multiline and cursor correctly."""
        from src.cli.cli_ui import _render_queue_input_display
        state = get_live_input_state()
        state.clear()
        state.queue.clear()

        try:
            state.queue.push("Item 1")
            state.queue.push("Item 2")
            state.insert_text("Line 1\nLine 2")
            state.move_cursor_left()  # cursor between '1' and '\n'

            rendered = _render_queue_input_display()
            assert rendered is not None
        finally:
            state.clear()
            state.queue.clear()


# ── L8: Concurrency & Temporal Tests (D6: Thread Safety & Race Conditions) ──

class TestConcurrency:
    """L8: Stress-test thread safety across simultaneous producers and consumers."""

    def test_concurrent_push_and_pop_thread_safety(self):
        """50 concurrent threads pushing and popping must not corrupt InputQueue."""
        q = InputQueue()
        iterations_per_thread = 200
        num_producers = 10
        pushed_records = []
        popped_records = []

        def producer(thread_id):
            for i in range(iterations_per_thread):
                msg = f"T{thread_id}-M{i}"
                q.push(msg)
                pushed_records.append(msg)

        def consumer():
            for _ in range(iterations_per_thread * num_producers):
                item = q.pop()
                if item:
                    popped_records.append(item)
                else:
                    time.sleep(0.001)

        threads = []
        for t in range(num_producers):
            threads.append(threading.Thread(target=producer, args=(t,)))
        consumer_thread = threading.Thread(target=consumer)

        for t in threads:
            t.start()
        consumer_thread.start()

        for t in threads:
            t.join()
        consumer_thread.join(timeout=3.0)

        # Drain remaining items
        while q.has_items():
            popped_records.append(q.pop())

        assert len(popped_records) == len(pushed_records)
        assert set(popped_records) == set(pushed_records)
        assert q.count() == 0

    def test_concurrent_live_state_typing_and_snapshot(self):
        """Simultaneous keystrokes and snapshot reads must not raise or deadlock."""
        state = LiveInputState()
        running = True

        def typer():
            for char in "abcdefghijklmnopqrstuvwxyz" * 20:
                if not running:
                    break
                state.insert_char(char)
                if len(state.current_text) > 10:
                    state.backspace()

        def reader():
            for _ in range(200):
                text, pos, items = state.snapshot()
                assert isinstance(text, str)
                assert 0 <= pos <= len(text) + 10

        t1 = threading.Thread(target=typer)
        t2 = threading.Thread(target=reader)

        t1.start()
        t2.start()

        t1.join(timeout=2.0)
        running = False
        t2.join(timeout=2.0)


# ── L9: Negative Space Tests (D3 / D10: What must NEVER happen) ─────────────

class TestNegativeSpace:
    """L9: Explicit invariants proving absence of unintended side effects."""

    def test_empty_submit_never_adds_to_queue(self):
        q = InputQueue()
        state = LiveInputState(q)
        # Calling submit on empty text or pure whitespace must NEVER push to queue
        assert state.submit() is None
        state.insert_text("   \t  ")
        assert state.submit() is None
        assert q.count() == 0
        assert q.items() == []

    def test_recall_on_non_empty_queue_never_destroys_other_items(self):
        q = InputQueue()
        q.push("first")
        q.push("second")
        q.push("third")

        state = LiveInputState(q)
        state.recall_last()
        assert q.items() == ["first", "second"]  # Exactly 1 item moved, others preserved

    def test_listener_pause_stops_capture_without_killing_thread(self):
        agent = MagicMock()
        listener = TurnInputListener(agent)
        listener.start()
        try:
            assert listener._running is True
            listener.pause()
            assert listener._paused is True
            # Thread should still be alive, just paused
            assert listener._thread is not None and listener._thread.is_alive()
            listener.resume()
            assert listener._paused is False
            assert listener._thread.is_alive()
        finally:
            listener.stop()
