"""A conversation session: the agent behind a transport-agnostic event stream.

Two ways to drive a turn:

- handle(text): synchronous generator, yields events then {"type": "done"}.
  Used by tests and any non-interactive caller. Gated tool actions fall back to
  the terminal approval prompt.

- start(text) + get_event() + reply_approval(): the turn runs on a worker thread
  and gated actions surface as {"type": "approval_request", ...} events. The
  transport answers with reply_approval(id, decision) so a non-TTY client (the
  GUI) never deadlocks on a questionary prompt.
"""

import queue
import threading

from src.server.events import to_event


class AgentSession:
    def __init__(self, agent=None):
        self._agent = agent
        self._out: queue.Queue = queue.Queue()
        self._pending: dict = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._active = False

    @property
    def agent(self):
        if self._agent is None:
            from agent import ArgentAgent
            self._agent = ArgentAgent()
        return self._agent

    @property
    def busy(self) -> bool:
        return self._active

    # --- synchronous API (tests, non-interactive callers) ---------------------
    def handle(self, text: str):
        """Run one turn inline, yielding normalized events then a terminal 'done'."""
        for chunk in self.agent.process_user_input(text):
            event = to_event(chunk)
            if event is not None:
                yield event
        yield {"type": "done"}

    # --- threaded API with approvals-as-events (GUI transport) ----------------
    def start(self, text: str) -> None:
        """Run one turn on a worker thread; consume it with get_event()."""
        self._active = True
        threading.Thread(target=self._run_turn, args=(text,), daemon=True).start()

    def get_event(self) -> dict:
        """Block for the next event of the running turn."""
        ev = self._out.get()
        if ev.get("type") == "done":
            self._active = False
        return ev

    def _run_turn(self, text: str) -> None:
        import approval
        approval.set_approval_backend(self._approval_backend)
        try:
            for chunk in self.agent.process_user_input(text):
                event = to_event(chunk)
                if event is not None:
                    self._out.put(event)
        except Exception as e:  # a turn crash must not wedge the client
            self._out.put({"type": "error", "text": f"[System: turn failed: {e}]"})
        finally:
            approval.reset_approval_backend()
            self._out.put({"type": "done"})

    def _approval_backend(self, action: str, destructive: bool, grant_key):
        """Runs on the worker thread: emit a request event, block for the reply."""
        with self._lock:
            self._counter += 1
            aid = self._counter
            ev = threading.Event()
            self._pending[aid] = {"event": ev, "decision": "deny"}
        self._out.put({
            "type": "approval_request",
            "id": aid,
            "action": action,
            "destructive": bool(destructive),
            "grant_key": grant_key,
        })
        ev.wait()
        with self._lock:
            return self._pending.pop(aid, {}).get("decision", "deny")

    def reply_approval(self, approval_id, decision: str) -> None:
        """Answer a pending approval. decision: 'deny' | 'once' | 'always'."""
        with self._lock:
            entry = self._pending.get(approval_id)
            if entry is not None:
                entry["decision"] = decision if decision in ("deny", "once", "always") else "deny"
                entry["event"].set()

    def cancel(self) -> None:
        """Deny every pending approval — e.g. on client disconnect — so the
        worker thread unblocks instead of hanging forever."""
        with self._lock:
            for entry in self._pending.values():
                entry["decision"] = "deny"
                entry["event"].set()
