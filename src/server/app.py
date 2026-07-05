"""FastAPI WebSocket adapter: the thin transport over AgentSession.

A client sends {"type": "message", "text": "..."} and receives the session's
event stream, ending with {"type": "done"}. The session factory is injectable so
the endpoint is tested with a fake agent — no provider, no network.

Brick-1 limitation: the agent turn runs synchronously on the event loop, so one
turn is handled at a time. Concurrent control messages (stop, approval replies)
move the turn onto a worker thread in the next step, alongside approvals-as-events.
"""

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
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") != "message":
                    continue
                for event in session.handle(data.get("text", "")):
                    await websocket.send_json(event)
        except WebSocketDisconnect:
            pass

    return app


app = create_app()
