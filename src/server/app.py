"""FastAPI WebSocket adapter: the thin transport over AgentSession.

A client sends {"type": "message", "text": "..."} and receives the session's
event stream, ending with {"type": "done"}. Gated tool actions arrive as
{"type": "approval_request", "id", "action", "destructive", "grant_key"}; the
client answers with {"type": "approval_reply", "id", "decision"} where decision
is "deny" | "once" | "always". A {"type": "stop"} aborts the running turn:
pending approvals are denied and the turn ends at the next chunk boundary.

Header/state protocol:
- {"type": "get_state"}          -> {"type": "state", model, provider, tier,
  context:{tokens,max,percent}, vibe, checkpoints:[{sha,label}]}. The client
  asks once on connect; after every finished turn the server pushes a fresh
  state on its own.
- {"type": "vibe", "enabled"}    -> {"type": "vibe_state", "enabled"} — the
  GUI's /vibe switch (auto-approve safe actions + per-turn checkpoints).

Time-machine requests (idle only — refused while a turn is in flight):
- {"type": "undo_file", "file"}  -> {"type": "undo_result", "file", "ok", "text"}
  restores one file to its pre-edit snapshot (a diff card's Reject button).
- {"type": "rewind", "sha"}      -> {"type": "rewind_result", "sha", "ok", "text"}
  hard-resets the tree to a turn checkpoint (a timeline node click). The GUI
  confirms with the user BEFORE sending; mechanical safety lives in rewind_to.

The turn runs on a worker thread inside the session, so the single receive loop
here stays responsive to approval replies while the turn is in flight. The
session factory is injectable so the endpoint is tested with a fake agent.
"""

import asyncio
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.server.session import AgentSession

# CSWSH defense: a WebSocket is NOT covered by the browser Same-Origin Policy,
# so any website the user opens could otherwise connect to this localhost
# socket and drive the agent (run commands, read files). Browsers always send
# an Origin header they can't forge, so we allow only local origins and reject
# any real web page. Non-browser clients (the tests, a CLI) send no Origin and
# are allowed — they already have machine access, which is not the CSWSH threat.
_ALLOWED_ORIGIN_HOSTS = {"localhost", "127.0.0.1", "::1", "tauri.localhost"}


def _origin_allowed(origin: str | None) -> bool:
    if not origin:
        return True                       # native client / test — no browser origin
    try:
        parsed = urlparse(origin)
    except Exception:
        return False
    if parsed.scheme == "tauri":          # Tauri prod webview (tauri://localhost)
        return True
    return (parsed.hostname or "") in _ALLOWED_ORIGIN_HOSTS


def create_app(session_factory=None, scheduler=None) -> FastAPI:
    factory = session_factory or (lambda: AgentSession())

    # Ambient automations fire while the server is up. Events are broadcast to
    # every connected client, since a run belongs to the session-independent
    # background, not to whoever happens to be chatting.
    clients: set = set()

    def broadcast(event: dict):
        loop = getattr(app.state, "loop", None)
        if loop is None:
            return
        for ws in list(clients):
            asyncio.run_coroutine_threadsafe(_safe_send(ws, event), loop)

    async def _safe_send(ws, event):
        try:
            await ws.send_json(event)
        except Exception:
            clients.discard(ws)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.loop = asyncio.get_running_loop()
        sched = scheduler
        if sched is None:
            from src.automation.scheduler import AutomationScheduler
            sched = AutomationScheduler(on_event=broadcast)
        else:
            sched._on_event = broadcast
        app.state.scheduler = sched
        sched.start()
        try:
            yield
        finally:
            sched.stop()

    app = FastAPI(title="Argent backend", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/automations")
    def list_automations():
        from dataclasses import asdict
        from src.automation.store import load_automations, load_runs
        return {"automations": [asdict(a) for a in load_automations()],
                "recent_runs": load_runs(limit=20)}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        # Reject cross-site WebSocket hijacking before accepting the handshake.
        if not _origin_allowed(websocket.headers.get("origin")):
            await websocket.close(code=1008)      # policy violation
            return
        await websocket.accept()
        session = factory()
        loop = asyncio.get_running_loop()
        sender = None
        clients.add(websocket)

        async def send_state():
            state = await loop.run_in_executor(None, session.state)
            await websocket.send_json({"type": "state", **state})

        async def pump_events():
            # Drain the (blocking) event queue off-thread and forward each event
            # until the turn signals completion; then push the fresh state so
            # the header (context %, checkpoints) never goes stale.
            while True:
                event = await loop.run_in_executor(None, session.get_event)
                await websocket.send_json(event)
                if event.get("type") == "done":
                    await send_state()
                    return

        try:
            while True:
                data = await websocket.receive_json()
                kind = data.get("type")
                if kind == "message":
                    if session.busy:
                        continue                      # one turn at a time
                    session.start(data.get("text", ""))
                    sender = asyncio.create_task(pump_events())
                elif kind == "approval_reply":
                    session.reply_approval(data.get("id"), data.get("decision", "deny"))
                elif kind == "stop":
                    session.cancel()
                elif kind == "get_state":
                    await send_state()
                elif kind == "vibe":
                    enabled = session.set_vibe(bool(data.get("enabled")))
                    await websocket.send_json({"type": "vibe_state", "enabled": enabled})
                elif kind == "undo_file":
                    fp = data.get("file", "")
                    if session.busy:
                        await websocket.send_json({"type": "undo_result", "file": fp,
                                                   "ok": False, "text": "A turn is in flight — wait for it to finish."})
                    else:
                        from file_tracker import undo
                        text = await loop.run_in_executor(None, undo, fp)
                        await websocket.send_json({"type": "undo_result", "file": fp,
                                                   "ok": text.startswith("Restored"), "text": text})
                elif kind == "rewind":
                    sha = data.get("sha", "")
                    if session.busy:
                        await websocket.send_json({"type": "rewind_result", "sha": sha,
                                                   "ok": False, "text": "A turn is in flight — wait for it to finish."})
                    else:
                        from src.agent.checkpoints import CheckpointError, rewind_to
                        try:
                            text = await loop.run_in_executor(None, rewind_to, sha)
                            ok = True
                        except CheckpointError as e:
                            text, ok = str(e), False
                        await websocket.send_json({"type": "rewind_result", "sha": sha,
                                                   "ok": ok, "text": text})
        except WebSocketDisconnect:
            session.cancel()
            if sender is not None:
                sender.cancel()
        finally:
            clients.discard(websocket)

    return app


app = create_app()
