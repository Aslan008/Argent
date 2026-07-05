"""Transport-agnostic backend for Argent.

The terminal REPL and a future Tauri/web GUI are just two clients of the same
core. This package turns ArgentAgent's chunk generator into a stable JSON
event stream (events.py), wraps a conversation as a session (session.py), and
exposes it over WebSocket (app.py) — without touching the terminal version.
"""
