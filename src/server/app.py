"""FastAPI WebSocket adapter: the thin transport over AgentSession.

A client sends {"type": "message", "text": "..."} and receives the session's
event stream, ending with {"type": "done"}. Gated tool actions arrive as
{"type": "approval_request", "id", "action", "destructive", "grant_key"}; the
client answers with {"type": "approval_reply", "id", "decision"} where decision
is "deny" | "once" | "always". A {"type": "stop"} denies all pending approvals.

The turn runs on a worker thread inside the session, so the single receive loop
here stays responsive to approval replies while the turn is in flight. The
session factory is injectable so the endpoint is tested with a fake agent.
"""

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.server.session import AgentSession


def create_app(session_factory=None) -> FastAPI:
    app = FastAPI(title="Argent backend")
    factory = session_factory or (lambda: AgentSession())

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        session = factory()
        loop = asyncio.get_running_loop()
        sender = None

        async def pump_events():
            # Drain the (blocking) event queue off-thread and forward each event
            # until the turn signals completion.
            while True:
                event = await loop.run_in_executor(None, session.get_event)
                await websocket.send_json(event)
                if event.get("type") == "done":
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
        except WebSocketDisconnect:
            session.cancel()
            if sender is not None:
                sender.cancel()

    return app


app = create_app()
