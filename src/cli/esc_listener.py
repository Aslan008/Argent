"""Background ESC key listener for aborting active agent turns.

Runs a lightweight daemon thread during streaming / tool execution.
When ESC (0x1B) is detected, sets `agent._stop_requested = True`.
Safe against keystroke swallowing on Windows and arrow-key false triggers on POSIX.
"""

import os
import sys
import threading
import time


class EscInterruptListener:
    """Listens for the ESC key press in a background daemon thread."""

    def __init__(self, agent):
        self.agent = agent
        self._running = False
        self._paused = False
        self._thread = None

    def start(self):
        """Start listening for ESC keystrokes."""
        if self._running:
            return
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background listener."""
        self._running = False

    def pause(self):
        """Temporarily suspend ESC detection (e.g. during interactive prompts)."""
        self._paused = True

    def resume(self):
        """Resume ESC detection."""
        self._paused = False

    def __enter__(self):
        self.start()
        self._prev_listener = getattr(self.agent, "_esc_listener", None)
        try:
            self.agent._esc_listener = self
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        try:
            self.agent._esc_listener = getattr(self, "_prev_listener", None)
        except Exception:
            pass

    def _trigger_stop(self):
        if hasattr(self.agent, "request_stop"):
            self.agent.request_stop()
        else:
            self.agent._stop_requested = True

    def _listen_loop(self):
        if os.name == "nt":
            self._listen_windows()
        else:
            self._listen_posix()

    def _listen_windows(self):
        try:
            import ctypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            has_user32 = True
        except Exception:
            has_user32 = False

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            # Preferred Method: Win32 GetAsyncKeyState when Argent console has active focus.
            # This detects the physical ESC key WITHOUT consuming or modifying stdin!
            if has_user32:
                try:
                    console_hwnd = kernel32.GetConsoleWindow()
                    foreground_hwnd = user32.GetForegroundWindow()
                    # Only trigger if console window is active/focused (or console_hwnd==0 in unusual terminals)
                    if not console_hwnd or foreground_hwnd == console_hwnd:
                        # VK_ESCAPE is 0x1B; high-bit indicates currently pressed
                        if user32.GetAsyncKeyState(0x1B) & 0x8000:
                            self._trigger_stop()
                            break
                except Exception:
                    pass
            else:
                # Fallback: msvcrt with ungetch to prevent swallowing user input
                try:
                    import msvcrt
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch == b"\x1b":
                            self._trigger_stop()
                            break
                        else:
                            try:
                                msvcrt.ungetch(ch)
                            except Exception:
                                pass
                except Exception:
                    pass

            time.sleep(0.05)

    def _listen_posix(self):
        try:
            import select
            while self._running:
                if self._paused:
                    time.sleep(0.05)
                    continue

                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    try:
                        ch = sys.stdin.read(1)
                        if ch == "\x1b":
                            # Distinguish standalone ESC from arrow/function keys:
                            # Arrow keys emit \x1b[A, \x1b[B, etc. within microseconds.
                            # Check if subsequent bytes are immediately available:
                            seq, _, _ = select.select([sys.stdin], [], [], 0.03)
                            if seq:
                                # Multi-byte escape sequence (e.g. arrow keys), discard or pass through
                                sys.stdin.read(1)
                                continue
                            # Standalone ESC!
                            self._trigger_stop()
                            break
                    except Exception:
                        pass
                time.sleep(0.05)
        except Exception:
            pass
