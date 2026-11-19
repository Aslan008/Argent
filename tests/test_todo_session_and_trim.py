"""The checklist and the history budget, as they behave across a restart.

Three things are checked here, each of which used to be a real annoyance:

* a 1M-token model distilling 150 messages of 108k tokens on the first message
  after a restart, because a constant meant for small windows said 80;
* the same "Осталось 3 из 5" at the end of every turn, including turns that never
  touched the plan;
* the current step's clock restarting when its wording was corrected.
"""

import json
import time
import types

import argent.agent as agent_module
from argent.agent import ArgentAgent as Agent
from src.agent import todos as T
from src.agent.todos import COMPLETED, IN_PROGRESS, PENDING, Todo, TodoStore


# ── when a summarising trim is allowed to run ────────────────────────────────

class _History:
    """Only the parts of the agent `_trim_history` touches."""

    def __init__(self, messages, limit, budget):
        self.messages = messages
        self.max_history_messages = limit
        self.max_context_tokens = 1_000_000
        self.model_name = "stub"
        self.budget = budget
        self.trimmed = None
        self.strategy = types.SimpleNamespace(trim_history=self._trim)

    def _trim(self, messages, model_name, max_msgs, budget):
        self.trimmed = {"max_msgs": max_msgs, "budget": budget}
        return messages

    def _estimate_message(self, message):
        return 10

    def _history_budget(self):
        return self.budget


def _conversation(turns):
    messages = [{"role": "system", "content": "sys"}]
    for i in range(turns):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    return messages


def test_a_long_history_that_fits_the_window_is_not_distilled(monkeypatch):
    """The case that was reported: 150 messages, 108k tokens, a 1M window.

    The count limit alone fired here on the first message after every restart —
    a distillation paid for, and detail lost, that fitted many times over.
    """
    monkeypatch.setattr(agent_module, "get_context_window", lambda: 1_000_000)
    history = _History(_conversation(75), limit=80, budget=800_000)

    Agent._trim_history(history)

    assert history.trimmed is None, "a history inside its budget must be left alone"


def test_a_history_over_both_limits_is_cut_deep_once(monkeypatch):
    monkeypatch.setattr(agent_module, "get_context_window", lambda: 1_000_000)
    history = _History(_conversation(75), limit=80, budget=500)

    Agent._trim_history(history)

    assert history.trimmed is not None
    assert history.trimmed["max_msgs"] == 48          # 60% of the configured 80


# ── the step clock ────────────────────────────────────────────────────────────

def test_a_reworded_step_keeps_its_clock():
    store = TodoStore()
    store.set([Todo("rewrite the panel", IN_PROGRESS)], scope="t-reword")
    started = store.snapshot("t-reword")[0]["since"]

    time.sleep(0.02)
    store.set([Todo("rewrite the panel (folded header)", IN_PROGRESS)], scope="t-reword")

    assert store.snapshot("t-reword")[0]["since"] == started


def test_a_reordered_step_keeps_its_clock():
    store = TodoStore()
    store.set([Todo("a", IN_PROGRESS), Todo("b")], scope="t-order")
    started = store.snapshot("t-order")[0]["since"]

    time.sleep(0.02)
    store.set([Todo("b"), Todo("a", IN_PROGRESS)], scope="t-order")

    assert store.snapshot("t-order")[1]["since"] == started


def test_a_new_step_starts_its_own_clock():
    store = TodoStore()
    store.set([Todo("a", IN_PROGRESS), Todo("b")], scope="t-new")
    first = store.snapshot("t-new")[0]["since"]

    time.sleep(0.02)
    store.set([Todo("a", COMPLETED), Todo("b", IN_PROGRESS)], scope="t-new")

    assert store.snapshot("t-new")[1]["since"] > first


def test_a_finished_step_has_no_clock():
    store = TodoStore()
    store.set([Todo("a", COMPLETED)], scope="t-done")
    assert "since" not in store.snapshot("t-done")[0]


# ── the end-of-turn warning ───────────────────────────────────────────────────

def test_an_untouched_plan_does_not_warn_every_turn():
    items = [Todo("write it", COMPLETED), Todo("test it", PENDING)]
    assert T.end_of_turn_note(items, changed=False) == ""


def test_a_half_done_step_still_speaks():
    items = [Todo("write it", IN_PROGRESS), Todo("test it", PENDING)]
    note = T.end_of_turn_note(items, changed=False)
    assert "Осталось 2 из 2" in note


def test_a_plan_advanced_this_turn_reports_what_is_left():
    items = [Todo("write it", COMPLETED), Todo("test it", PENDING)]
    note = T.end_of_turn_note(items, changed=True)
    assert "Осталось 1 из 2" in note


def test_a_stopped_turn_says_nothing():
    items = [Todo("write it", IN_PROGRESS)]
    assert T.end_of_turn_note(items, stopped=True, changed=True) == ""


def test_a_finished_plan_never_warns():
    items = [Todo("write it", COMPLETED)]
    assert T.end_of_turn_note(items, changed=True) == ""


# ── sub-agent checklists ──────────────────────────────────────────────────────

def test_a_subagents_list_is_visible_next_to_the_main_one():
    store = TodoStore()
    store.set([Todo("delegate the rewrite")], scope="main")
    store.set([Todo("read ui.rs", COMPLETED), Todo("rewrite it", IN_PROGRESS)], scope="sub-1")

    subs = store.snapshot_subagents()

    assert list(subs) == ["sub-1"]
    assert [t["content"] for t in subs["sub-1"]] == ["read ui.rs", "rewrite it"]
    assert "since" in subs["sub-1"][1]


def test_the_main_list_is_not_reported_as_a_subagent():
    store = TodoStore()
    store.set([Todo("only step")], scope="main")
    assert store.snapshot_subagents() == {}


# ── what a session carries ────────────────────────────────────────────────────

def test_the_checklist_survives_a_restart(tmp_path, monkeypatch):
    from argent import session as session_mod

    monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(session_mod, "INDEX_PATH", tmp_path / "index.json", raising=False)

    store = TodoStore()
    store.set([Todo("read the code", COMPLETED), Todo("fix it", IN_PROGRESS)], scope="main")
    state = {"todos": json.dumps(store.snapshot("main"), ensure_ascii=False)}

    sid = session_mod.save_session(
        [{"role": "user", "content": "go"}, {"role": "assistant", "content": "on it"}],
        {"model": "test", "provider": "test", "cwd": str(tmp_path)},
        state=state,
    )
    assert sid, "the session must be written"

    data = session_mod.load_session(sid)
    T.restore((data.get("state") or {}).get("todos"))

    assert [t.content for t in T.TODOS.get("main")] == ["read the code", "fix it"]
    assert T.TODOS.get("main")[1].status == IN_PROGRESS


def test_a_restored_step_keeps_its_clock():
    """A step that was running before the restart did not start three seconds ago."""
    started = time.time() - 1200
    snapshot = [
        {"content": "read the code", "status": "completed"},
        {"content": "fix it", "status": "in_progress", "since": started},
    ]

    T.restore(snapshot)

    assert T.TODOS.snapshot("main")[1]["since"] == started
