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
        self._stop_requested = False
        self.vibe = False

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
        self._stop_requested = False
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
        gen = self.agent.process_user_input(text)
        try:
            for chunk in gen:
                # Cooperative stop: takes effect between chunks. close()
                # raises GeneratorExit at the paused yield inside the turn.
                if self._stop_requested:
                    gen.close()
                    self._out.put({"type": "notice", "text": "[System: turn stopped by user]"})
                    break
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
        """Stop the running turn: deny every pending approval (so a blocked
        worker unblocks) and request a cooperative abort — the turn ends at
        the next chunk boundary instead of running to completion."""
        self._stop_requested = True
        with self._lock:
            for entry in self._pending.values():
                entry["decision"] = "deny"
                entry["event"].set()

    # --- GUI state ------------------------------------------------------------
    def state(self) -> dict:
        """Snapshot for the GUI header: model, provider, tier, context usage,
        vibe flag and the rewindable turn checkpoints (newest first)."""
        a = self.agent
        model = getattr(a, "model_name", "?")
        try:
            from agent import get_model_size_category
            tier = get_model_size_category(model)
        except Exception:
            tier = "?"
        try:
            usage = a.get_context_usage()
        except Exception:
            usage = {}
        try:
            from src.agent.checkpoints import list_checkpoints
            cps = [{"sha": c["sha"], "label": c["label"]} for c in list_checkpoints(10)]
        except Exception:
            cps = []
        return {
            "model": model,
            "provider": getattr(a, "provider", "?"),
            "tier": tier,
            "context": {"tokens": usage.get("tokens", 0), "max": usage.get("max", 0),
                        "percent": round(usage.get("percent", 0))},
            "vibe": self.vibe,
            "checkpoints": cps,
        }

    def set_vibe(self, enabled: bool) -> bool:
        """The GUI's vibe switch — same knobs as the terminal /vibe: safe
        actions auto-approved (destructive still prompt), checkpoints on."""
        import approval
        from src.agent.checkpoints import set_auto_checkpoint
        self.vibe = bool(enabled)
        approval.set_policy(approval.POLICY_AUTO if self.vibe else approval.POLICY_ASK)
        if self.vibe:
            set_auto_checkpoint(True)
        return self.vibe
