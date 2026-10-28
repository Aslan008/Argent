"""An unattended run must not treat a forced stop as progress.

The loop guard ends the turn when a tool call repeats. The auto-continue decision
only looked at "did this turn call any tool" — and a spinning turn always does. So
the guard stopped the turn, and the next one started a second later, forever: the
"busy dead end" that made leaving the agent alone a gamble.
"""

import pytest

from src.bridge import repl_bridge as rb


class _Session:
    def __init__(self, tools=3, end="final answer"):
        self.last_turn_tools = tools
        self.last_turn_end = end
        self.last_tool_output = ""


class _Noop:
    daemon = False

    def start(self):
        pass

    def cancel(self):
        pass


class _Bridge:
    """Just enough of the bridge to run the auto-continue decision without a UI."""

    _maybe_auto_continue = rb.ReplBridge._maybe_auto_continue
    _cancel_auto = rb.ReplBridge._cancel_auto

    def __init__(self, session, vibe=True):
        self.session = session
        self.vibe_mode = vibe
        self._shutting_down = False
        self._idle_turns = 0
        self._wakes_in_a_row = 0
        self._auto_timer = None
        self.events = []
        self.vibe_off = False
        self.queued = []

    def set_vibe(self, on):
        self.vibe_mode = on
        self.vibe_off = not on

    def _emit(self, event):
        self.events.append(event)

    def _auto_turn(self, prompt):
        self.queued.append(prompt)


@pytest.fixture
def timers(monkeypatch):
    made = []
    monkeypatch.setattr(rb.threading, "Timer",
                        lambda delay, fn, args=(): made.append((delay, fn, args)) or _Noop())
    return made


def test_a_forced_stop_ends_the_run_instead_of_waking_again(timers):
    # Plenty of tools were called — that is exactly why the old check missed it.
    bridge = _Bridge(_Session(tools=7, end="loop guard"))

    bridge._maybe_auto_continue()

    assert not bridge.vibe_off, "vibe — переключатель одобрений, сам он не выключается"
    assert timers == [], "следующий автоход всё равно запланирован"
    assert any("принудительно" in e.get("text", "") for e in bridge.events)


def test_a_textual_call_stop_ends_it_too(timers):
    bridge = _Bridge(_Session(tools=2, end="tool call written as text"))
    bridge._maybe_auto_continue()
    assert not bridge.vibe_off
    assert timers == []


def test_a_normal_turn_still_wakes_again(timers):
    # The healthy path must be untouched: the turn did not conclude, the run continues.
    bridge = _Bridge(_Session(tools=4, end="tool call limit"))
    bridge._maybe_auto_continue()
    assert not bridge.vibe_off
    assert len(timers) == 1


def test_a_finished_answer_ends_the_run(timers):
    # The agent said it was done. A heartbeat one second later, while nothing is
    # running, is exactly what made vibe look broken.
    session = _Session(tools=3, end="final answer")
    bridge = _Bridge(session)

    bridge._maybe_auto_continue()

    assert not bridge.vibe_off, "vibe остаётся включён: его выключает человек, а не конец хода"
    assert timers == [], "автоход всё равно запланирован"
    assert not any("vibe выключен" in e.get("text", "").lower() for e in bridge.events)


def test_a_turn_the_person_stopped_ends_the_run_too(timers):
    session = _Session(tools=5, end="turn stopped by user request during stream")
    bridge = _Bridge(session)

    bridge._maybe_auto_continue()

    assert not bridge.vibe_off, "остановка хода прекращает прогон, но не режим"
    assert timers == []


def test_the_run_ending_never_flips_the_switch(timers, monkeypatch):
    """Only the person turns vibe off. The auto-continue decision may end a run,
    but it must never touch the mode itself — that is what made /vibe last one task."""
    bridge = _Bridge(_Session(tools=3, end="final answer"))
    flips = []
    monkeypatch.setattr(bridge, "set_vibe", lambda on: flips.append(on))

    bridge._maybe_auto_continue()

    assert flips == [], "авто-прогон выключил vibe за спиной у человека"


def test_an_unfinished_turn_continues(timers):
    # A tool-call limit, a provider error, a lost network: the work is not done and the
    # run is allowed to pick itself up.
    session = _Session(tools=9, end="tool call limit")
    bridge = _Bridge(session)

    bridge._maybe_auto_continue()

    assert not bridge.vibe_off
    assert len(timers) == 1, "оборванный по лимиту ход должен продолжаться"


def test_a_wake_that_was_asked_for_still_wakes(timers):
    # wait_heartbeat asked to be woken: that is not an idle turn, whatever the tools.
    session = _Session(tools=0, end="final answer")
    session.last_tool_output = "[HEARTBEAT_REQUEST: 7 | ждём сборку]"
    bridge = _Bridge(session)

    bridge._maybe_auto_continue()

    assert not bridge.vibe_off
    assert len(timers) == 1, "запрошенное пробуждение потерялось"
