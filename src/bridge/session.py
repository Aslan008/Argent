"""A conversation session: the agent behind a transport-agnostic event stream.

Two ways to drive a turn:

- handle(text): synchronous generator, yields events then {"type": "done"}.
  Used by tests and any non-interactive caller. Gated tool actions fall back to
  the terminal approval prompt.

- start(text) + get_event() + reply_approval(): the turn runs on a worker thread
  and gated actions surface as {"type": "approval_request", ...} events. The
  transport answers with reply_approval(id, decision) so a non-TTY client (e.g.
  the Telegram bridge) never deadlocks on a questionary prompt.
"""

import queue
import threading
from typing import Optional

from src.bridge.events import to_event


class AgentSession:
    def __init__(self, agent=None):
        self._agent = agent
        self._out: queue.Queue = queue.Queue()
        self._event_sink = None
        self._pending: dict = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._active = False
        self._stop_requested = False
        self._turn_running = False
        self._last_tool_output = ""
        self.vibe = False

    def set_event_sink(self, sink) -> None:
        """Set a direct callback for events (e.g. PyO3 Rust channel sender)."""
        self._event_sink = sink

    def _emit(self, event: dict) -> None:
        """Deliver event to the active sink or fallback to internal queue."""
        # Attach an actionable hint to genuine failures, like the legacy CLI.
        if self._event_sink is not None and event.get("type") == "error":
            try:
                from error_help import explain_error
                hint = explain_error(str(event.get("text", "")))
                if hint and "hint" not in event:
                    event = dict(event, hint=hint)
            except Exception:
                pass
        if self._event_sink is not None:
            try:
                self._event_sink(event)
                return
            except Exception:
                pass
        self._out.put(event)

    @property
    def agent(self):
        if self._agent is None:
            from agent import ArgentAgent
            self._agent = ArgentAgent()
        return self._agent

    @property
    def busy(self) -> bool:
        return self._active

    @property
    def last_tool_output(self) -> str:
        """Result text of the most recent tool call in the last turn.

        The heartbeat protocol signals itself through a tool result
        (`[HEARTBEAT_REQUEST: ...]`), so the frontend needs access to it.
        """
        return self._last_tool_output

    # --- synchronous API (tests, non-interactive callers) ---------------------
    def handle(self, text: str):
        """Run one turn inline, yielding normalized events then a terminal 'done'."""
        for chunk in self.agent.process_user_input(text):
            event = to_event(chunk)
            if event is not None:
                yield event
        yield {"type": "done"}

    # --- threaded API with approvals-as-events --------------------------------
    def start(self, text: str) -> None:
        """Run one turn on a worker thread; consume it with get_event() or event_sink."""
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
        import time
        import approval
        approval.set_approval_backend(self._approval_backend)
        gen = self.agent.process_user_input(text)
        tool_start_times = {}
        turn_t0 = time.time()
        first_chunk_at = None
        tool_seconds = 0.0
        tool_calls = 0
        usage: dict = {}
        interrupted = False
        self._last_tool_output = ""
        try:
            for chunk in gen:
                if first_chunk_at is None:
                    first_chunk_at = time.time()
                if self._stop_requested:
                    gen.close()
                    # Same line the REPL prints on Ctrl+C (main.py).
                    self._emit({"type": "system",
                                "text": "\n[dim yellow]Ход прерван пользователем (Ctrl+C)[/dim yellow]"})
                    interrupted = True
                    break
                kind = chunk.get("type")
                if kind == "usage":
                    u = chunk.get("data")
                    if isinstance(u, dict):
                        # Accumulated into the session totals ONCE per turn in
                        # the finally block, like the REPL's snapshot does. (The
                        # per-chunk add that used to live here imported a module
                        # that does not exist, so the totals never moved.)
                        usage.update(u)
                elif kind == "tool_start":
                    t_name = chunk.get("name", "")
                    tool_start_times[t_name] = time.time()
                    tool_calls += 1
                elif kind == "tool_end":
                    t_name = chunk.get("name", "")
                    t_start = tool_start_times.pop(t_name, None)
                    if t_start is not None:
                        d = max(0.0, time.time() - t_start)
                        chunk["duration_secs"] = d
                        tool_seconds += d
                    res = chunk.get("result")
                    if isinstance(res, str) and res.strip():
                        self._last_tool_output = res

                event = to_event(chunk)
                if event is not None:
                    self._emit(event)
        except Exception as e:
            try:
                from src.cli.sound import play_sound
                play_sound("error")
            except Exception:
                pass
            self._emit({"type": "error", "text": f"[System: turn failed: {e}]"})
        finally:
            approval.reset_approval_backend()
            self._active = False
            if not self._stop_requested:
                try:
                    from src.cli.sound import play_sound
                    play_sound("done")
                except Exception:
                    pass
            # The stop may land while the generator is blocked inside the agent,
            # in which case the loop above never saw it: say so here, once.
            if self._stop_requested and not interrupted:
                interrupted = True
                self._emit({"type": "system",
                            "text": "\n[dim yellow]Ход прерван пользователем (Ctrl+C)[/dim yellow]"})
            # An interrupted turn prints no snapshot in the REPL (the Ctrl+C
            # handler skips it), so the TUI doesn't either.
            if not interrupted:
                snap = self._build_snapshot(
                    turn_t0, first_chunk_at, tool_seconds, tool_calls, usage)
                snap_event = {"type": "turn_snapshot", "data": snap}
                try:
                    snap_event.update(self._snapshot_markup(snap, usage))
                except Exception:
                    pass
                self._emit(snap_event)
            # 'done' is emitted exactly once per turn, here. cancel() sends
            # 'cancelling' instead, so the frontend never observes a turn as
            # finished while this worker is still unwinding (which used to let
            # a queued message start a second concurrent turn).
            self._emit({"type": "done"})
            with self._lock:
                self._turn_running = False
                # Consume the cancel so the next turn starts clean.
                self._stop_requested = False

    def _snapshot_markup(self, snap: dict, usage: dict) -> dict:
        """The REPL's own snapshot text for this turn, as Rich markup.

        Built by `cli_ui.turn_snapshot_lines` — the function the REPL prints —
        so the two frontends cannot drift apart. Records the turn's usage into
        the session totals first (exactly once), as the REPL does.
        """
        from src.cli.cli_ui import turn_snapshot_lines
        from usage_tracker import usage as session_usage
        from ui import context_usage_markup

        if usage:
            session_usage.add(usage)
        agent = self.agent
        lines = turn_snapshot_lines(
            agent,
            snap.get("total_s", 0.0),
            timing={"first": snap.get("ttft_s") or None, "tools": snap.get("tool_s", 0.0)},
            usage=usage,
            tool_calls=snap.get("tool_calls") or getattr(agent, "_turn_tool_calls", 0),
        )
        out = {"lines": lines}
        try:
            ctx = agent.get_context_usage()
            out["context_line"] = context_usage_markup(
                ctx["tokens"], ctx["max"], ctx["percent"])
        except Exception:
            pass
        return out

    def _build_snapshot(self, turn_t0, first_chunk_at, tool_seconds, tool_calls, usage) -> dict:
        """Per-turn metrics the TUI shows under the transcript.

        The legacy CLI computed these inline in `cli_ui`; the TUI cannot time
        the Python generator itself, so they are measured here instead.
        """
        import time
        now = time.time()
        snap = {
            "total_s": round(now - turn_t0, 3),
            "ttft_s": round(first_chunk_at - turn_t0, 3) if first_chunk_at else 0.0,
            "tool_s": round(tool_seconds, 3),
            "tool_calls": tool_calls,
        }
        for k_src, k_dst in (("prompt", "prompt_tokens"),
                             ("completion", "completion_tokens"),
                             ("cached", "cached_tokens")):
            v = usage.get(k_src)
            if isinstance(v, (int, float)):
                snap[k_dst] = v
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            snap["cost"] = round(float(cost), 6)
        try:
            ctx = self.agent.get_context_usage() or {}
            if isinstance(ctx, dict):
                for k in ("tokens", "max", "percent"):
                    if k in ctx:
                        snap[f"ctx_{k}"] = ctx[k]
        except Exception:
            pass
        return snap

    def _approval_backend(self, action: str, destructive: bool, grant_key):
        """Runs on the worker thread: emit a request event, block for the reply."""
        with self._lock:
            self._counter += 1
            aid = self._counter
            ev = threading.Event()
            self._pending[aid] = {"event": ev, "decision": "deny"}
        try:
            from src.cli.sound import play_sound
            play_sound("prompt")
        except Exception:
            pass
        from approval import approval_prompt_spec
        spec = approval_prompt_spec(grant_key, bool(destructive))
        self._emit({
            "type": "approval_request",
            "id": aid,
            "action": action,
            "destructive": bool(destructive),
            "grant_key": grant_key,
            # The terminal's own wording, so a frontend draws the same question.
            "question": spec["question"],
            "choices": spec["choices"],
            "decisions": spec["decisions"],
            "prompt_kind": spec["kind"],
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

    def ask_select(
        self,
        title: str,
        options: list,
        default: Optional[str] = None,
        multi: bool = False,
        initial_checked: Optional[list] = None,
    ) -> Optional[str]:
        """Request an interactive option selection from the TUI frontend."""
        with self._lock:
            self._counter += 1
            sid = f"sel_{self._counter}"
            ev = threading.Event()
            self._pending[sid] = {"event": ev, "value": None}
        self._emit({
            "type": "select_request",
            "id": sid,
            "title": str(title),
            "options": [str(o) for o in options],
            "default": str(default) if default else None,
            "multi": bool(multi),
            "initial_checked": [bool(b) for b in initial_checked] if initial_checked else None,
        })
        ev.wait()
        with self._lock:
            return self._pending.pop(sid, {}).get("value")

    def reply_select(self, select_id: str, value: Optional[str] = None) -> None:
        """Answer a pending interactive select request."""
        with self._lock:
            entry = self._pending.get(str(select_id))
            if entry is not None:
                entry["value"] = value if (value is not None and value != "") else None
                entry["event"].set()

    def cancel(self) -> None:
        """Stop the running turn."""
        self._stop_requested = True
        self._active = False
        if hasattr(self.agent, "request_stop"):
            self.agent.request_stop()
        else:
            self.agent._stop_requested = True
        with self._lock:
            for entry in self._pending.values():
                if "decision" in entry:
                    entry["decision"] = "deny"
                if "value" in entry:
                    entry["value"] = None
                entry["event"].set()
            was_running = self._turn_running
        # Tell the frontend the stop was acknowledged but NOT that the turn
        # ended — the worker thread owns the terminal 'done'.
        if was_running:
            self._emit({"type": "cancelling"})
        else:
            self._emit({"type": "done"})

    def state(self) -> dict:
        """Snapshot of state: model, provider, tier, context usage, vibe, checkpoints."""
        a = self.agent
        model = getattr(a, "model_name", "?")
        try:
            from config import get_model_size_category
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
        try:
            import config as _cfg
            cfg_max = _cfg.get_context_window()
        except Exception:
            cfg_max = 32768
        try:
            records = a._turn_records
            can_rewind = bool(records) and records[-1].get("messages") is not None
        except Exception:
            can_rewind = False
        ctx_max = usage.get("max") or cfg_max
        ctx_tokens = usage.get("tokens", 0)
        ctx_pct = usage.get("percent")
        if ctx_pct is None:
            ctx_pct = (ctx_tokens / ctx_max * 100) if ctx_max > 0 else 0.0
        return {
            "model": model,
            "provider": getattr(a, "provider", "?"),
            "tier": tier,
            "context": {"tokens": ctx_tokens, "max": ctx_max,
                        "percent": round(ctx_pct, 1)},
            "ctx_tokens": ctx_tokens,
            "ctx_max": ctx_max,
            "ctx_percent": round(ctx_pct, 1),
            "vibe": self.vibe,
            "checkpoints": cps,
            # The frontend shows its "back to here" button only when there is something to go back to.
            "can_rewind": can_rewind,
        }

    def set_vibe(self, enabled: bool) -> bool:
        import approval
        from src.agent.checkpoints import set_auto_checkpoint
        self.vibe = bool(enabled)
        approval.set_policy(approval.POLICY_AUTO if self.vibe else approval.POLICY_ASK)
        if self.vibe:
            set_auto_checkpoint(True)
        return self.vibe
