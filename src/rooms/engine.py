"""The rooms engine: deterministic execution of a validated node graph.

The engine owns MACRO control — which node runs next — while the caller owns the
MICRO side effects (running a tool, an agent's bounded loop, prompting a human)
through a single injected ``run_node`` callback. This keeps the transition logic
pure and fully testable without an LLM or real tools.

Transitions are decided from typed state, not by the model directly:
  * ``if``   edge — taken when its condition holds;
  * ``while`` edge — taken while its condition holds AND its counter (kept in
    ``state.loops[edge.id]``) is below ``max``; the counter increments per hop,
    so a loop is always bounded;
  * ``else`` / unconditional — the fallback.

Terminal outcomes: an ``EXIT:<name>`` target ends the room with that name; a
``human_pause`` node that got no answer ends with ``human_pause`` (one that was
answered carries on along its edges); exhausting the step, token or time budget
ends with ``budget``; a node with no usable outgoing edge ends with ``stuck`` (the
validator makes this unreachable for admitted rooms). The higher room-graph
layer decides how ``budget``/``stuck`` route (typically to human/triage).

Parallel branches. A ``parallel`` node starts every arrow that leaves it at once; each branch
works on its OWN copy of the run state and the ``join`` node it names merges the copies back:
a field only one branch changed is taken as it is, a field several branches changed differently
is a conflict that the join's ``merge`` policy settles (stop / first / last / list, in the order
the arrows were drawn — never in the order the threads happened to finish). The step, token and
time budgets are the room's, shared by all branches. The journal sees the whole fork as one step.
"""

import concurrent.futures
import contextvars
import copy
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import run_scope
from src.rooms import condition
from src.rooms.models import Edge, Room, State

#: Pseudo-outcomes of a branch's walk: it got to its join / it was told to stop.
ARRIVED = "@join"
CANCELLED = "@cancelled"

_GONE = object()          # a field a branch deleted

#: Built-in counters are summed across branches, not treated as conflicts.
_COUNTER_SUFFIX = "_visits"


_PRODUCED_TEXT = 600      # chars of one produced value the screen may be told about
_PRODUCED_ITEMS = 12


def _show_value(value, depth=0):
    """A copy of what a node produced, small enough to be put on a screen."""
    if isinstance(value, str):
        return value if len(value) <= _PRODUCED_TEXT else value[:_PRODUCED_TEXT] + "…"
    if isinstance(value, (list, tuple)) and depth < 2:
        items = [_show_value(v, depth + 1) for v in value[:_PRODUCED_ITEMS]]
        if len(value) > _PRODUCED_ITEMS:
            items.append(f"… ещё {len(value) - _PRODUCED_ITEMS}")
        return items
    if isinstance(value, dict) and depth < 2:
        return {str(k): _show_value(v, depth + 1) for k, v in list(value.items())[:_PRODUCED_ITEMS]}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _show_value(str(value), depth)


def _produced(node, state) -> dict:
    """The fields a node has just put into the run state (what it `writes`, `sets`, saves or answers
    into), shortened for the screen. Only for showing: nothing reads it back."""
    names = []
    for n in list(getattr(node, "writes", None) or []) + list(getattr(node, "sets", None) or []) \
            + list(getattr(node, "outputs", None) or []):
        names.append(n)
    for attr in ("into", "save_as"):
        v = getattr(node, attr, None)
        if v:
            names.append(v)
    out = {}
    for name in names:
        if name in out or name not in state.data:
            continue
        value = state.data[name]
        if value in (None, "", [], {}):
            continue
        out[name] = _show_value(value)
    return out


@dataclass
class Outcome:
    exit: str                 # EXIT name, or "human_pause" | "budget" | "stuck"
    state: State
    steps: int
    reason: str = ""


class MergeConflict(RuntimeError):
    """Two branches wrote different values to the same field and the join's merge policy is `error`."""


@dataclass
class _Run:
    """What every walk of one `Engine.run` call shares: the room, the callbacks and the budget."""
    room: Room
    run_node: object
    journal: object
    on_event: object
    clock: object
    started: float
    max_steps: int
    node_by_id: dict
    outgoing: dict
    steps: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    emit_lock: threading.Lock = field(default_factory=threading.Lock)


class Engine:
    def run(self, room: Room, state: State, run_node, max_steps: int = None,
            journal=None, start_node: str = None, on_event=None, clock=None) -> Outcome:
        """Execute `room` from its entry (or `start_node`) until a terminal outcome.

        `run_node(node, state)` performs the node's side effects (mutating
        `state.data`). It is called for every node, including condition nodes
        (which usually no-op) and human_pause nodes (prompt the user).

        If `journal` is given, each node writes an intent record before its side
        effects and a commit record (with the chosen next target) after — see
        journal.py and plan_resume for crash-safe resume.

        `on_event(dict)` observes the run (node_start / node_end / room_end) for a
        progress display or a run report; it never influences the run, and an error
        in it is swallowed.

        The room's budget is enforced here: steps (`max_iterations`), the tokens
        the agent nodes spent in THIS room (`max_tokens`, measured in
        ``state.usage``) and, when set, wall-clock seconds (`max_seconds`).
        """
        if max_steps is None:
            max_steps = room.budget.max_iterations
        clock = clock or time.monotonic
        outgoing = defaultdict(list)
        for e in room.edges:
            outgoing[e.from_].append(e)
        run = _Run(room=room, run_node=run_node, journal=journal, on_event=on_event, clock=clock,
                   started=clock(), max_steps=max_steps,
                   node_by_id={n.id: n for n in room.nodes}, outgoing=outgoing)

        # What this room may not do (and what the room that called it already forbade) travels with the run.
        from src.rooms.rules import merge as merge_rules
        state.rules = merge_rules(state.rules, room.rules)

        return self._walk(run, state, start_node or room.entry, stop_at=None, branch=None,
                          depth=0, cancels=(), spent_base=0)

    # ── one walk: the main path of the room, or one branch of a fork ──────────────
    def _walk(self, run: _Run, state: State, current: str, *, stop_at, branch, depth: int,
              cancels: tuple, spent_base: int) -> Outcome:
        room, outgoing, node_by_id = run.room, run.outgoing, run.node_by_id
        main = branch is None
        journal = run.journal if main else None
        tokens_at_start = state.usage.get("tokens", 0)

        def _emit(event):
            if run.on_event is None:
                return
            if branch is not None:
                event = {**event, "branch": branch}
            with run.emit_lock:
                try:
                    run.on_event(event)
                except Exception:
                    pass

        def _commit(node_id, nxt, exit_name):
            if journal:
                journal.append({"kind": "commit", "room": room.room, "node": node_id,
                                "next": nxt, "exit": exit_name, "state": state.model_dump()})

        def _spent():
            return spent_base + state.usage.get("tokens", 0) - tokens_at_start

        def _over_budget():
            if _spent() >= room.budget.max_tokens:
                return f"token budget {room.budget.max_tokens} exhausted"
            limit = room.budget.max_seconds
            if limit and run.clock() - run.started >= limit:
                return f"time budget {limit}s exhausted"
            return ""

        def _end(outcome: Outcome) -> Outcome:
            return self._finish(room, outcome, _emit) if main else outcome

        while True:
            if stop_at is not None and current == stop_at:
                return Outcome(ARRIVED, state, run.steps, reason=stop_at)
            if any(ev.is_set() for ev in cancels):
                return Outcome(CANCELLED, state, run.steps, reason="stopped: another branch decided")
            if run.steps >= run.max_steps:
                return _end(Outcome("budget", state, run.steps, reason=f"step budget {run.max_steps} exhausted"))
            why = _over_budget()
            if why:
                return _end(Outcome("budget", state, run.steps, reason=why))

            node = node_by_id[current]
            # Built-in counters, for conditions like "this step has run 3 times" (a loop that does not end)
            state.data[f"{node.id}_visits"] = state.data.get(f"{node.id}_visits", 0) + 1
            state.data["_steps"] = run.steps + 1
            if journal:
                journal.append({"kind": "intent", "room": room.room, "node": current,
                                "state": state.model_dump()})
            if node.type == "human_pause":
                state.data.pop(f"{node.id}.output", None)   # an answer from an earlier visit is stale
            _emit({"type": "node_start", "room": room.room, "node": node.id, "kind": node.type,
                   "step": run.steps + 1})
            t0 = run.clock()
            failed_edge = None
            forked = None
            attempts = 1 + (node.retries or 0)
            for attempt in range(attempts):
                try:
                    run.run_node(node, state)
                    if node.type == "parallel" and not node.disabled:
                        forked = self._fork(run, node, state, depth=depth, cancels=cancels,
                                            spent_now=_spent(), emit=_emit)
                    break
                except Exception as exc:      # the node RAISED (a verdict "no" is data, not an exception)
                    _emit({"type": "node_error", "room": room.room, "node": node.id,
                           "error": str(exc)[:300], "attempt": attempt + 1})
                    if attempt + 1 < attempts:
                        continue              # retries: run it again
                    failed_edge = next((e for e in outgoing[current] if e.kind == "error"), None)
                    if failed_edge is None:
                        raise                 # nobody handles it: the run stops, the journal can resume it
                    state.data["error"] = str(exc)[:500]
                    state.data["error_node"] = node.id
            with run.lock:
                run.steps += 1
                steps = run.steps
            _emit({"type": "node_end", "room": room.room, "node": node.id, "kind": node.type,
                   "step": steps, "seconds": round(run.clock() - t0, 2), "failed": failed_edge is not None,
                   "tokens": _spent(), "produced": _produced(node, state)})

            if forked is not None and failed_edge is None:
                if forked.exit is not None:           # a branch ended the room (an exit, a pause, a budget)
                    _commit(current, None, forked.exit)
                    return _end(Outcome(forked.exit, state, run.steps, reason=forked.reason))
                _commit(current, node.join, None)
                current = node.join
                continue

            if failed_edge is None and node.type == "human_pause":
                answered = state.data.get(f"{node.id}.output") not in (None, "")
                if not (answered and outgoing[current]):
                    _commit(current, None, "human_pause")
                    return _end(Outcome("human_pause", state, run.steps,
                                        reason=f"human_pause node '{node.id}'"))

            edge = failed_edge or self._select_edge(outgoing[current], state)
            if edge is None:
                _commit(current, None, "stuck")
                return _end(Outcome("stuck", state, run.steps,
                                    reason=f"node '{current}' has no usable edge"))
            if edge.targets_exit():
                exit_name = edge.to.split(":", 1)[1]
                _commit(current, None, exit_name)
                return _end(Outcome(exit_name, state, run.steps))
            _commit(current, edge.to, None)
            current = edge.to

    # ── a fork: run the branches, merge what they did ──────────────────────────────
    def _fork(self, run: _Run, node, state: State, *, depth: int, cancels: tuple,
              spent_now: int, emit) -> "_Forked":
        """Run every branch of the `parallel` node `node` to its join and merge them into `state`.

        Returns a `_Forked`: `exit` is None when the branches were merged and the run goes on at the
        join, or the name of the outcome that ends the room (a branch's EXIT, a pause, a budget)."""
        from src.rooms.parallel import branch_starts
        starts = branch_starts(run.room, node)
        join = run.node_by_id[node.join]
        merge_mode = join.merge or "error"
        wait_any = (join.wait_for or "all") == "any"
        keep_going = (join.on_branch_fail or "fail") == "continue"
        workers = min(node.workers or len(starts), len(starts))

        base = copy.deepcopy(state.data)
        base_memory = list(state.memory)
        base_hints = copy.deepcopy(state.hints)
        base_usage = dict(state.usage)
        stop = threading.Event()
        branch_cancels = cancels + (stop,)
        states = []
        for _ in starts:
            states.append(State(data=copy.deepcopy(base), loops=dict(state.loops), usage=dict(base_usage),
                                rules=copy.deepcopy(state.rules), memory=list(base_memory),
                                hints=copy.deepcopy(base_hints)))
        emit({"type": "fork", "room": run.room.room, "node": node.id, "branches": list(starts),
              "join": join.id, "workers": workers})

        results = [None] * len(starts)          # (kind, outcome, error) per branch, in arrow order
        finished_order = []                      # branch indexes in the order they arrived (for `any`)
        order_lock = threading.Lock()

        def work(index):
            try:
                with run_scope.branch():
                    out = self._walk(run, states[index], starts[index], stop_at=join.id,
                                     branch=starts[index], depth=depth + 1, cancels=branch_cancels,
                                     spent_base=spent_now)
                result = ("arrived" if out.exit == ARRIVED else
                          "cancelled" if out.exit == CANCELLED else "ended", out, None)
            except Exception as exc:
                result = ("raised", None, exc)
            results[index] = result
            if result[0] == "arrived":
                with order_lock:
                    finished_order.append(index)
                if wait_any:
                    stop.set()
            elif result[0] in ("ended", "raised") and not keep_going and not wait_any:
                stop.set()                       # the room is going to stop anyway: do not start more work

        if workers <= 1:
            for index in range(len(starts)):
                if stop.is_set() or any(ev.is_set() for ev in cancels):
                    results[index] = ("cancelled", None, None)
                    continue
                work(index)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="room-branch") as pool:
                futures = []
                for index in range(len(starts)):
                    ctx = contextvars.copy_context()
                    futures.append(pool.submit(ctx.run, work, index))
                for future in futures:
                    future.result()

        for index, result in enumerate(results):         # branches never started because of a stop
            if result is None:
                results[index] = ("cancelled", None, None)

        arrived = [i for i, r in enumerate(results) if r[0] == "arrived"]
        broken = [i for i, r in enumerate(results) if r[0] in ("ended", "raised")]

        if wait_any:
            winners = finished_order[:1]
            broken_for_policy = [] if winners else broken
        else:
            winners = arrived
            broken_for_policy = broken

        # Spent tokens / steps of every branch count, whether or not its work is merged.
        for index, branch_state in enumerate(states):
            for key, value in branch_state.usage.items():
                delta = value - base_usage.get(key, 0)
                if isinstance(delta, (int, float)) and delta:
                    state.usage[key] = state.usage.get(key, 0) + delta

        emit({"type": "join", "room": run.room.room, "node": join.id,
              "arrived": [starts[i] for i in arrived],
              "failed": [starts[i] for i in broken]})

        if broken_for_policy and not keep_going:
            # the room ends here: keep what the branches that did arrive had done, for the report
            self._merge(state, base, base_memory, base_hints, states, arrived, "last", join, starts)
            first = broken_for_policy[0]
            kind, out, exc = results[first]
            if kind == "raised":
                raise exc
            return _Forked(exit=out.exit, reason=f"branch '{starts[first]}' of '{node.id}': {out.reason or out.exit}")
        if broken_for_policy and keep_going and not winners:
            kind, out, exc = results[broken_for_policy[0]]
            if kind == "raised":
                raise exc
            return _Forked(exit=out.exit, reason=f"every branch of '{node.id}' failed")

        self._merge(state, base, base_memory, base_hints, states, winners, merge_mode, join, starts)
        state.data[f"{join.id}_arrived"] = [starts[i] for i in winners]
        state.data[f"{join.id}_failed"] = [
            {"branch": starts[i], "why": (str(results[i][2])[:200] if results[i][0] == "raised"
                                          else (results[i][1].reason or results[i][1].exit))}
            for i in broken_for_policy]
        state.data[f"{join.id}.output"] = f"{len(winners)} of {len(starts)} branches arrived"
        return _Forked(exit=None)

    @staticmethod
    def _merge(state: State, base: dict, base_memory: list, base_hints: list, states: list,
               winners: list, mode: str, join, starts: list) -> None:
        """Fold the winning branches' changes into `state`, in the order the arrows were drawn."""
        changes = {}
        for index in sorted(winners):
            data = states[index].data
            for key, value in data.items():
                if key not in base or base[key] != value:
                    changes.setdefault(key, []).append((index, value))
            for key in base:
                if key not in data:
                    changes.setdefault(key, []).append((index, _GONE))
        for key, entries in changes.items():
            if key == "_steps":
                continue
            if key.endswith(_COUNTER_SUFFIX) and all(isinstance(v, (int, float)) for _, v in entries):
                before = base.get(key, 0) if isinstance(base.get(key, 0), (int, float)) else 0
                state.data[key] = before + sum(v - before for _, v in entries)
                continue
            values = [v for _, v in entries]
            if len(entries) > 1 and any(v != values[0] for v in values[1:]):
                if mode == "error":
                    who = [starts[i] for i, _ in entries]
                    raise MergeConflict(
                        f"join '{join.id}': branches {who} wrote different values to '{key}' "
                        "(set the join's merge to first, last or list, or give each branch its own field)")
                if mode == "first":
                    values = values[:1]
                elif mode == "last":
                    values = values[-1:]
                else:                                   # list: every value that was written, in arrow order
                    listed = [v for v in values if v is not _GONE]
                    state.data[key] = listed
                    continue
            chosen = values[0]
            if chosen is _GONE:
                state.data.pop(key, None)
            else:
                state.data[key] = chosen

        # while-edge / loop counters: the furthest any branch got
        for index in winners:
            for key, count in states[index].loops.items():
                if count > state.loops.get(key, 0):
                    state.loops[key] = count

        # what the branches' agents did, in arrow order, within the memory limit
        seen = set(base_memory)
        for index in sorted(winners):
            for entry in states[index].memory:
                if entry not in seen:
                    state.memory.append(entry)
                    seen.add(entry)
        from src.rooms.rules import Rules
        limit = Rules(state.rules).memory_chars
        while len(state.memory) > 1 and sum(len(m) + 1 for m in state.memory) > limit:
            state.memory.pop(0)

        # a hint one branch used up is used up; a hint a branch added is added (once)
        branch_states = [states[i] for i in sorted(winners)]
        state.hints[:] = [h for h in base_hints if all(h in b.hints for b in branch_states)]
        for b in branch_states:
            for h in b.hints:
                if h not in base_hints and h not in state.hints:
                    state.hints.append(h)

    @staticmethod
    def _finish(room: Room, outcome: Outcome, emit) -> Outcome:
        emit({"type": "room_end", "room": room.room, "exit": outcome.exit, "steps": outcome.steps,
              "reason": outcome.reason})
        return outcome

    @staticmethod
    def _select_edge(edges: list, state: State) -> Edge:
        """First matching conditional edge, else the else/unconditional fallback.
        A taken `while` edge increments its counter in state.loops."""
        fallback = None
        for e in edges:
            kind = e.kind
            if kind == "if":
                if condition.evaluate(e.if_, state.data):
                    return e
            elif kind == "while":
                count = state.loops.get(e.id, 0)
                if count < e.max and condition.evaluate(e.while_, state.data):
                    state.loops[e.id] = count + 1
                    return e
            elif kind == "error":
                continue              # taken only when the node raised, never as a normal route
            elif kind == "else":
                fallback = fallback or e
            else:  # unconditional
                fallback = fallback or e
        return fallback


@dataclass
class _Forked:
    """How a fork ended: `exit` None = merged, go on at the join; else the outcome that ends the room."""
    exit: object = None
    reason: str = ""


def plan_resume(room: Room, journal):
    """Decide where to resume `room` from its journal.

    Returns ``(start_node, state, status)`` where status is:
      * "fresh"        — nothing to resume; start at the entry with empty state;
      * "resume"       — continue at start_node with the recovered state;
      * "finished"     — the run already reached a terminal (start_node is None);
      * "indeterminate"— a non-idempotent node was interrupted mid-execution; the
        caller must route to a human instead of replaying it.
    """
    last = journal.last()
    if not last or last.get("room") != room.room:
        return room.entry, State(), "fresh"

    state = State(**last["state"])
    if last.get("kind") == "commit":
        nxt = last.get("next")
        if nxt is None:
            return None, state, "finished"
        return nxt, state, "resume"

    # An intent with no following commit: the node was interrupted.
    node = next((n for n in room.nodes if n.id == last.get("node")), None)
    if node is None:
        return room.entry, State(), "fresh"
    return last.get("node"), state, ("resume" if replayable(node, room) else "indeterminate")


def _may_change_things(tool: str) -> bool:
    """Whether a tool may change files, processes or the outside world (plan mode's own list, plus the
    shell, which plan mode polices command by command)."""
    try:
        from src.agent.plan_mode import _mutates
    except Exception:       # without the list assume anything may change something
        return True
    return tool == "run_command" or _mutates(tool)


def replayable(node, room: Room = None) -> bool:
    """Whether an interrupted node may simply be run again on resume. A condition and a human
    question can; an idempotent tool can; an agent can only when it was confined to tools that
    change nothing (a read-only analysis) — otherwise it may have half-applied a change.

    A `parallel` node can only when every step of every one of its branches can (the branches may have
    got half-way, and the journal does not look inside a fork); that needs the `room`."""
    if node.disabled:
        return True
    if node.type == "parallel":
        if room is None:
            return False
        from src.rooms.parallel import block_nodes
        by_id = {n.id: n for n in room.nodes}
        return all(replayable(by_id[i], room) for i in block_nodes(room, node) if i in by_id)
    if node.type in ("condition", "human_pause", "set", "notify", "foreach", "wait", "skill",
                     "loop", "break", "continue", "hint", "forget", "assert", "join"):
        return True               # only state, a question, a pause or a read: running it again changes nothing real
    if node.type in ("tool", "mcp", "plugin", "leeth", "power", "git", "http"):
        return bool(node.idempotent)
    if node.type == "agent":
        if node.tools is None:
            return False
        return not any(_may_change_things(t) for t in node.tools)
    return False
