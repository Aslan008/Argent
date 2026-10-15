"""Runs the real Rust TUI against a fake bridge; every submitted prompt is appended to $FAKE_TUI_LOG.
Used by test_tui_paste_console.py (a console process, not imported)."""
import os
import sys

LOG = os.environ["FAKE_TUI_LOG"]


class Bridge:
    def set_event_sink(self, sink):
        self.sink = sink

    def get_ui_config(self):
        return "{}"

    def get_state(self):
        return {"model": "fake", "provider": "fake"}

    def get_command_completions(self):
        return [{"name": "/rooms", "help": "rooms"}, {"name": "/help", "help": "help"}]

    def get_history(self):
        return []

    def get_thinking_levels(self):
        return ["off", "low"]

    def start_turn(self, prompt):
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("PROMPT<<" + prompt.replace("\n", "\\n") + ">>\n")
        return None

    def turn_was_rejected(self):
        return False

    def cleanup(self):
        pass

    def cancel_turn(self):
        pass

    def render_block(self, *a, **k):
        return None


import argent_tui

argent_tui.run_tui(Bridge())
