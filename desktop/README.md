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

## What the window gives you
- **Markdown answers** with code blocks; the reasoning stream folds into a
  collapsible 🧠 block.
- **Tool chips**: each tool call is a one-line chip (running/done) that expands
  to args + result.
- **Diff cards**: every file edit arrives as a card with +/− highlighting and
  Accept / Reject buttons — Reject restores that file's pre-edit snapshot.
- **Time machine**: turn checkpoints line up above the composer; click a node
  to rewind the whole tree to before that turn (uncommitted work is stashed).
- **Approvals**: gated tool actions pop a modal (Deny / Always allow / Approve);
  destructive ones are styled accordingly.
- **🌴 vibe toggle** in the header: auto-approve safe actions + checkpoint every
  turn — same switch as the terminal `/vibe`.
- **Stop** button aborts the running turn at the next chunk boundary.
- Header shows the active model, tier and live context usage.

## Notes
- Needs Rust + Node (for `tauri dev`). The frontend alone builds with `npm run build`.
- The terminal version (`python main.py`) is untouched and needs none of this.
- Model/provider selection still lives in the terminal (`/model`, `/provider`) —
  both clients share the same config, so switch there and restart the server.
