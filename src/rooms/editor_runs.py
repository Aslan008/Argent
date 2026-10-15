"""Runs started from the room editor: a room (or one step of it) executed while the page watches.

The run goes through the same Engine and NodeRunner as every other run — the editor adds only
what a page needs around it: a record of what happened step by step (for the highlighting and
the "Прогоны" tab), and a way for a run that has to ask a person (a `human_pause` node, a step
with `confirm`) to ask THE PAGE and wait for the answer.

Nothing is persisted here: runs live in memory, the last few are kept. A run that is not
answered within `ASK_TIMEOUT` seconds is declined (a question nobody sees must never mean "yes").
"""

import json
import secrets
import threading
import time

from src.rooms.models import State

KEEP_RUNS = 20
ASK_TIMEOUT = 900.0
PREVIEW = 1500
#: state fields the run view leaves out (big, internal, or already shown elsewhere)
_HIDDEN = ("rooms_catalog",)


class RunStopped(BaseException):
    """Raised inside a run's thread when the person pressed Stop (not an Exception: a node's
    `on_error` edge must not swallow it)."""


def preview(value, limit: int = PREVIEW):
    """A value as short text for the page."""
    if isinstance(value, (list, dict)):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)     # JSON, so a list can be fed back as input
        except (TypeError, ValueError):
            text = repr(value)
    else:
        text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit})"


class Run:
    def __init__(self, run_id: str, room: str, kind: str, params: dict):
        self.id = run_id
        self.room = room
        self.kind = kind                    # "room" | "node"
        self.params = params
        self.status = "running"             # running | waiting | done | error | stopped
        self.started = time.time()
        self.finished = None
        self.events = []
        self.prompt = None                  # {"id", "kind": "confirm"|"ask", "node", "text"} while waiting
        self.outcome = None
        self.error = ""
        self.stop_requested = False
        self._lock = threading.RLock()
        self._answered = threading.Event()
        self._answer = None
        self._prompt_seq = 0

    def add(self, event: dict) -> None:
        with self._lock:
            event = dict(event)
            event["i"] = len(self.events)
            event["t"] = round(time.time() - self.started, 2)
            self.events.append(event)

    def summary(self) -> dict:
        return {"id": self.id, "room": self.room, "kind": self.kind, "status": self.status,
                "started": self.started, "seconds": round((self.finished or time.time()) - self.started, 1),
                "steps": sum(1 for e in self.events if e.get("type") == "node_end"),
                "exit": (self.outcome or {}).get("exit", ""), "error": self.error}

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            out = self.summary()
            out["events"] = self.events[since:]
            out["next"] = len(self.events)
            out["prompt"] = self.prompt
            out["outcome"] = self.outcome
            out["params"] = {k: preview(v, 200) for k, v in self.params.items()}
            return out


class RunManager:
    """Starts runs in threads and lets the page poll them.

    `runner_factory(human_prompt, confirm) -> NodeRunner` builds the production runner (or a
    fake one in tests); the manager never imports the model or the tools itself."""

    def __init__(self, runner_factory, keep: int = KEEP_RUNS, ask_timeout: float = ASK_TIMEOUT):
        self._factory = runner_factory
        self._keep = keep
        self._ask_timeout = ask_timeout
        self._runs = {}
        self._order = []
        self._lock = threading.RLock()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self, room, params: dict = None, *, node_id: str = None, ask_first: bool = False) -> Run:
        """Run `room` (a validated Room) from its entry, or just the step `node_id`."""
        params = dict(params or {})
        run = Run(secrets.token_hex(4), room.room, "node" if node_id else "room", params)
        with self._lock:
            self._runs[run.id] = run
            self._order.append(run.id)
            while len(self._order) > self._keep:
                old = self._runs.get(self._order[0])
                if old is not None and old.status in ("running", "waiting"):
                    break
                self._runs.pop(self._order.pop(0), None)
        thread = threading.Thread(target=self._work, args=(run, room, node_id, ask_first),
                                  name=f"rooms-editor-run-{run.id}", daemon=True)
        thread.start()
        run._thread = thread
        return run

    def get(self, run_id: str):
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list:
        with self._lock:
            return [self._runs[i].summary() for i in reversed(self._order) if i in self._runs]

    def answer(self, run_id: str, prompt_id: int, value) -> bool:
        run = self.get(run_id)
        if run is None:
            return False
        with run._lock:
            if run.prompt is None or run.prompt.get("id") != prompt_id:
                return False
            run._answer = value
            run._answered.set()
            return True

    def stop(self, run_id: str) -> bool:
        run = self.get(run_id)
        if run is None or run.status not in ("running", "waiting"):
            return False
        run.stop_requested = True
        run._answered.set()                 # wakes a pending question; the run then raises RunStopped
        return True

    # ── the work itself ──────────────────────────────────────────────────────
    def _ask(self, run: Run, kind: str, node, text: str):
        with run._lock:
            run._prompt_seq += 1
            run._answered.clear()
            run.prompt = {"id": run._prompt_seq, "kind": kind, "node": node.id, "text": text}
            run.status = "waiting"
        run.add({"type": "ask", "node": node.id, "kind": kind, "text": text})
        got = run._answered.wait(self._ask_timeout)
        with run._lock:
            answer = run._answer if got else None
            run._answer = None
            run.prompt = None
            if not run.stop_requested:
                run.status = "running"
        if run.stop_requested:
            raise RunStopped()
        return answer

    def _work(self, run: Run, room, node_id, ask_first) -> None:
        from src.rooms.engine import Engine
        from src.rooms.session import describe_call

        def human(node, state):
            question = node.prompt or node.title or "Ваш ввод:"
            answer = self._ask(run, "ask", node, question)
            if answer not in (None, ""):
                state.data[f"{node.id}.output"] = str(answer)

        def confirm(node, state):
            return bool(self._ask(run, "confirm", node, describe_call(node)))

        try:
            runner = self._factory(human, confirm)
            if hasattr(runner, "set_on_event"):
                runner.set_on_event(run.add)
            state = State(data=dict(run.params))

            def run_node(node, st):
                if run.stop_requested:
                    raise RunStopped()
                before = {k: repr(v) for k, v in st.data.items()}
                runner.run_node(node, st)
                changed = {k: preview(v, 400) for k, v in st.data.items()
                           if before.get(k) != repr(v) and k not in _HIDDEN}
                run.add({"type": "node_result", "room": room.room, "node": node.id,
                         "output": preview(st.data.get(f"{node.id}.output", "")), "changed": changed})

            if node_id:
                node = next((n for n in room.nodes if n.id == node_id), None)
                if node is None:
                    raise ValueError(f"no step '{node_id}' in the room")
                if ask_first:
                    from src.rooms.session import describe_call as _desc
                    if not self._ask(run, "confirm", node, f"Выполнить один шаг «{node.id}»: {_desc(node)}"):
                        run.outcome = {"exit": "declined", "steps": 0, "reason": "the step was not approved",
                                       "state": {}}
                        run.status = "done"
                        run.finished = time.time()
                        return
                    node = node.model_copy(update={"confirm": False})       # asked once, not twice
                run.add({"type": "node_start", "room": room.room, "node": node.id, "kind": node.type, "step": 1})
                t0 = time.monotonic()
                run_node(node, state)
                run.add({"type": "node_end", "room": room.room, "node": node.id, "kind": node.type, "step": 1,
                         "seconds": round(time.monotonic() - t0, 2), "failed": False})
                outcome = {"exit": "step", "steps": 1, "reason": ""}
            else:
                result = Engine().run(room, state, run_node, on_event=run.add)
                outcome = {"exit": result.exit, "steps": result.steps, "reason": result.reason}
            outcome["state"] = {k: preview(v, 800) for k, v in state.data.items() if k not in _HIDDEN}
            run.outcome = outcome
            run.status = "done"
        except RunStopped:
            run.status = "stopped"
            run.outcome = {"exit": "stopped", "steps": sum(1 for e in run.events if e.get("type") == "node_end"),
                           "reason": "stopped by the person", "state": {}}
        except Exception as e:
            run.status = "error"
            run.error = f"{type(e).__name__}: {str(e)[:400]}"
            run.add({"type": "run_error", "error": run.error})
        finally:
            run.finished = time.time()
