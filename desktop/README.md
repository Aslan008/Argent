# Argent Desktop (Tauri)

The GUI client for Argent. It's a thin front-end — the real work runs in the
**same Python core** as the terminal version, exposed over a local WebSocket by
`argent_server.py`. Terminal and desktop are two clients of one engine.

```
┌──────────────┐     ws://127.0.0.1:8756/ws     ┌────────────────────┐
│ Tauri window │  ◀── JSON event stream ──────  │  argent_server.py  │
│  (React UI)  │  ──── {type:"message"} ─────▶  │  (ArgentAgent core)│
└──────────────┘                                └────────────────────┘
```

## Run

Two processes.

**1. Backend** (from the project root — same config/provider as the terminal Argent):
```bash
pip install -e ".[server]"      # once: fastapi + uvicorn + websockets
python argent_server.py         # listens on 127.0.0.1:8756
```

**2. Desktop app** (from `desktop/`):
```bash
npm install                     # once
npm run tauri dev               # first run compiles the Rust shell (a few minutes)
```
An **Argent** window opens; the header shows `connected` once it reaches the
backend. Type a message and watch the response stream in.

## Notes
- Needs Rust + Node (for `tauri dev`). The frontend alone builds with `npm run build`.
- The terminal version (`python main.py`) is untouched and needs none of this.
- Current limitation: a turn runs synchronously on the backend, so tool actions
  that require confirmation aren't wired to the GUI yet — approvals-as-events are
  the next backend step. For now the window is best for chat/answer turns.
