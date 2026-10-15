"""Background user input queue and live keystroke listener during agent turns.

Allows users to type messages, edit them, and enqueue them while the agent is
actively generating responses, executing tools, or waiting for heartbeats.
"""

import os
import sys
import threading
import time
from typing import List, Optional, Tuple


def get_clipboard_text() -> str:
    """Safely retrieve text from system clipboard without external dependencies."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            CF_UNICODETEXT = 13
            user32.OpenClipboard.restype = wintypes.BOOL
            user32.OpenClipboard.argtypes = [wintypes.HWND]
            user32.CloseClipboard.restype = wintypes.BOOL
            user32.CloseClipboard.argtypes = []
            user32.GetClipboardData.restype = ctypes.c_void_p
            user32.GetClipboardData.argtypes = [wintypes.UINT]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalUnlock.restype = wintypes.BOOL
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]

            if user32.OpenClipboard(None):
                try:
                    handle = user32.GetClipboardData(CF_UNICODETEXT)
                    if handle:
                        ptr = kernel32.GlobalLock(handle)
                        if ptr:
                            try:
                                return ctypes.c_wchar_p(ptr).value or ""
                            finally:
                                kernel32.GlobalUnlock(handle)
                finally:
                    user32.CloseClipboard()
        except Exception:
            pass

    try:
        import pyperclip
        return pyperclip.paste() or ""
    except Exception:
        return ""


def set_clipboard_text(text: str) -> bool:
    """Safely copy text to system clipboard without external dependencies."""
    if not text:
        return False

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            GMEM_MOVEABLE = 0x0002
            CF_UNICODETEXT = 13

            kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
            kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
            kernel32.GlobalUnlock.restype = wintypes.BOOL
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            user32.OpenClipboard.restype = wintypes.BOOL
            user32.OpenClipboard.argtypes = [wintypes.HWND]
            user32.CloseClipboard.restype = wintypes.BOOL
            user32.CloseClipboard.argtypes = []
            user32.EmptyClipboard.restype = wintypes.BOOL
            user32.EmptyClipboard.argtypes = []
            user32.SetClipboardData.restype = wintypes.HANDLE
            user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]

            if user32.OpenClipboard(None):
                try:
                    user32.EmptyClipboard()
                    data = text.encode("utf-16le") + b"\x00\x00"
                    h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
                    if h_mem:
                        ptr = kernel32.GlobalLock(h_mem)
                        if ptr:
                            ctypes.memmove(ptr, data, len(data))
                            kernel32.GlobalUnlock(h_mem)
                            if user32.SetClipboardData(CF_UNICODETEXT, h_mem):
                                return True
                finally:
                    user32.CloseClipboard()
        except Exception:
            pass

    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        pass

    return False


class InputQueue:
    """Thread-safe queue of user messages waiting to be processed by the agent."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items: List[str] = []

    def push(self, text: str) -> int:
        """Add a message to the end of the queue. Returns new queue size."""
        if not text or not isinstance(text, str):
            return self.count()
        text = text.strip()
        if not text:
            return self.count()
        with self._lock:
            self._items.append(text)
            return len(self._items)

    def pop(self) -> Optional[str]:
        """Pop the first message in the queue (FIFO)."""
        with self._lock:
            if self._items:
                return self._items.pop(0)
            return None

    def pop_last(self) -> Optional[str]:
        """Pop the most recently added message (LIFO) for recall and editing."""
        with self._lock:
            if self._items:
                return self._items.pop()
            return None

    def remove(self, index: int) -> Optional[str]:
        """Remove and return an item by 0-based index."""
        with self._lock:
            if 0 <= index < len(self._items):
                return self._items.pop(index)
            return None

    def edit(self, index: int, new_text: str) -> bool:
        """Edit an existing item in the queue by 0-based index."""
        if not new_text or not isinstance(new_text, str):
            return False
        new_text = new_text.strip()
        if not new_text:
            return False
        with self._lock:
            if 0 <= index < len(self._items):
                self._items[index] = new_text
                return True
            return False

    def clear(self) -> List[str]:
        """Clear all messages from the queue and return them."""
        with self._lock:
            cleared = list(self._items)
            self._items.clear()
            return cleared

    def items(self) -> List[str]:
        """Return a snapshot copy of all queued messages."""
        with self._lock:
            return list(self._items)

    def count(self) -> int:
        """Return the number of messages in the queue."""
        with self._lock:
            return len(self._items)

    def has_items(self) -> bool:
        """Return True if the queue has at least one message."""
        with self._lock:
            return bool(self._items)


class LiveInputState:
    """State of the real-time input line currently being typed by the user."""

    def __init__(self, queue: Optional[InputQueue] = None):
        self._lock = threading.Lock()
        self.queue = queue if queue is not None else InputQueue()
        self.current_text: str = ""
        self.cursor_pos: int = 0

    def insert_char(self, ch: str):
        if not ch or not isinstance(ch, str):
            return
        with self._lock:
            self.cursor_pos = max(0, min(self.cursor_pos, len(self.current_text)))
            self.current_text = (
                self.current_text[: self.cursor_pos]
                + ch
                + self.current_text[self.cursor_pos :]
            )
            self.cursor_pos += len(ch)

    def insert_text(self, text: str):
        if not text or not isinstance(text, str):
            return
        with self._lock:
            self.cursor_pos = max(0, min(self.cursor_pos, len(self.current_text)))
            self.current_text = (
                self.current_text[: self.cursor_pos]
                + text
                + self.current_text[self.cursor_pos :]
            )
            self.cursor_pos += len(text)

    def backspace(self):
        with self._lock:
            if self.cursor_pos > 0:
                self.current_text = (
                    self.current_text[: self.cursor_pos - 1]
                    + self.current_text[self.cursor_pos :]
                )
                self.cursor_pos -= 1

    def delete(self):
        with self._lock:
            if self.cursor_pos < len(self.current_text):
                self.current_text = (
                    self.current_text[: self.cursor_pos]
                    + self.current_text[self.cursor_pos + 1 :]
                )

    def move_cursor_left(self):
        with self._lock:
            if self.cursor_pos > 0:
                self.cursor_pos -= 1

    def move_cursor_right(self):
        with self._lock:
            if self.cursor_pos < len(self.current_text):
                self.cursor_pos += 1

    def move_cursor_home(self):
        with self._lock:
            self.cursor_pos = 0

    def move_cursor_end(self):
        with self._lock:
            self.cursor_pos = len(self.current_text)

    def clear(self) -> str:
        """Clear current input buffer and return previous content."""
        with self._lock:
            prev = self.current_text
            self.current_text = ""
            self.cursor_pos = 0
            return prev

    def submit(self) -> Optional[str]:
        """Move current text into the queue. Returns the submitted text if any."""
        with self._lock:
            text = self.current_text.strip()
            self.current_text = ""
            self.cursor_pos = 0
        if text:
            self.queue.push(text)
            return text
        return None

    def recall_last(self) -> Optional[str]:
        """Pop the most recent queued message back into the input buffer for editing."""
        last_msg = self.queue.pop_last()
        if last_msg is not None:
            with self._lock:
                self.current_text = last_msg
                self.cursor_pos = len(last_msg)
            return last_msg
        return None

    def snapshot(self) -> Tuple[str, int, List[str]]:
        """Return (current_text, cursor_pos, queued_items) thread-safely."""
        with self._lock:
            return self.current_text, self.cursor_pos, self.queue.items()


class TurnInputListener:
    """Background keyboard listener active during agent turns.

    Captures typing and adds messages to InputQueue while agent executes.
    Also handles ESC (clears input, or stops agent if input is empty).
    """

    def __init__(self, agent, state: Optional[LiveInputState] = None):
        self.agent = agent
        self.state = state if state is not None else get_live_input_state()
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        global _ACTIVE_TURN_INPUT_LISTENER
        _ACTIVE_TURN_INPUT_LISTENER = self
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        global _ACTIVE_TURN_INPUT_LISTENER
        if _ACTIVE_TURN_INPUT_LISTENER is self:
            _ACTIVE_TURN_INPUT_LISTENER = None

    def pause(self):
        """Temporarily suspend keyboard capture (e.g. during interactive prompts/approval)."""
        self._paused = True

    def resume(self):
        """Resume keyboard capture."""
        self._paused = False

    def __enter__(self):
        self.start()
        # Save reference on agent for stop requests
        try:
            self.agent._turn_input_listener = self
            self.agent._esc_listener = self
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        try:
            if getattr(self.agent, "_turn_input_listener", None) is self:
                self.agent._turn_input_listener = None
            if getattr(self.agent, "_esc_listener", None) is self:
                self.agent._esc_listener = None
        except Exception:
            pass

    def _trigger_stop(self):
        """Stop agent execution when ESC is pressed on an empty input."""
        if hasattr(self.agent, "request_stop"):
            self.agent.request_stop()
        else:
            self.agent._stop_requested = True

    def _is_shift_pressed_windows(self) -> bool:
        """Check if Shift key is physically held down on Windows."""
        try:
            import ctypes
            VK_SHIFT = 0x10
            return bool(ctypes.windll.user32.GetKeyState(VK_SHIFT) & 0x8000)
        except Exception:
            return False

    def _is_alt_pressed_windows(self) -> bool:
        """Check if Alt key is physically held down on Windows."""
        try:
            import ctypes
            VK_MENU = 0x12
            return bool(ctypes.windll.user32.GetKeyState(VK_MENU) & 0x8000)
        except Exception:
            return False

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

        import msvcrt

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            # Verify console window has foreground focus before consuming keys
            if has_user32:
                try:
                    console_hwnd = kernel32.GetConsoleWindow()
                    foreground_hwnd = user32.GetForegroundWindow()
                    if console_hwnd and foreground_hwnd != console_hwnd:
                        time.sleep(0.05)
                        continue
                except Exception:
                    pass

            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()

                    # Two-stage scan codes (arrows, Del, Home, End, PageUp/Dn)
                    if ch in ("\x00", "\xe0"):
                        if msvcrt.kbhit():
                            code = msvcrt.getwch()
                            if code == "H":  # Up arrow
                                self.state.recall_last()
                            elif code == "K":  # Left arrow
                                self.state.move_cursor_left()
                            elif code == "M":  # Right arrow
                                self.state.move_cursor_right()
                            elif code == "G":  # Home
                                self.state.move_cursor_home()
                            elif code == "O":  # End
                                self.state.move_cursor_end()
                            elif code == "S":  # Delete
                                self.state.delete()
                        continue

                    # Enter key (\r)
                    if ch == "\r":
                        if self._is_shift_pressed_windows() or self._is_alt_pressed_windows():
                            # Shift+Enter or Alt+Enter -> Newline
                            self.state.insert_char("\n")
                        else:
                            # Plain Enter -> Submit to queue
                            self.state.submit()
                        continue

                    # Line Feed (\n) - can occur via Ctrl+J or some terminals
                    if ch == "\n":
                        self.state.insert_char("\n")
                        continue

                    # Backspace (\x08)
                    if ch == "\x08":
                        self.state.backspace()
                        continue

                    # Tab (\t)
                    if ch == "\t":
                        self.state.insert_text("    ")
                        continue

                    # Ctrl+V (\x16 = 22) -> Paste clipboard
                    if ch == "\x16":
                        clip = get_clipboard_text()
                        if clip:
                            self.state.insert_text(clip)
                        continue

                    # ESC (\x1b)
                    if ch == "\x1b":
                        current, _, _ = self.state.snapshot()
                        if current:
                            # Clear current line if user was typing
                            self.state.clear()
                        else:
                            # Interrupt agent if input was already empty
                            self._trigger_stop()
                            break
                        continue

                    # Ctrl+C (\x03)
                    if ch == "\x03":
                        current, _, _ = self.state.snapshot()
                        if current:
                            self.state.clear()
                        else:
                            self._trigger_stop()
                            break
                        continue

                    # Printable Unicode characters (Russian letters, English, symbols)
                    if ch.isprintable():
                        self.state.insert_char(ch)

            except Exception:
                pass

            time.sleep(0.02)

    def _listen_posix(self):
        try:
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while self._running:
                    if self._paused:
                        time.sleep(0.05)
                        continue

                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r:
                        ch = sys.stdin.read(1)
                        if ch == "\x1b":
                            # Check if part of escape sequence (e.g. arrow keys)
                            seq_avail, _, _ = select.select([sys.stdin], [], [], 0.03)
                            if seq_avail:
                                seq = sys.stdin.read(1)
                                if seq == "[":
                                    code = sys.stdin.read(1)
                                    if code == "A":  # Up arrow
                                        self.state.recall_last()
                                    elif code == "D":  # Left arrow
                                        self.state.move_cursor_left()
                                    elif code == "C":  # Right arrow
                                        self.state.move_cursor_right()
                                    elif code == "H":  # Home
                                        self.state.move_cursor_home()
                                    elif code == "F":  # End
                                        self.state.move_cursor_end()
                                    elif code == "3":  # Delete sequence \x1b[3~
                                        sys.stdin.read(1)  # '~'
                                        self.state.delete()
                                    elif code == "1":  # Kitty/xterm Shift+Enter \x1b[13;2u
                                        rest = sys.stdin.read(4)
                                        if "13;2" in rest or "u" in rest:
                                            self.state.insert_char("\n")
                                continue

                            # Standalone ESC
                            current, _, _ = self.state.snapshot()
                            if current:
                                self.state.clear()
                            else:
                                self._trigger_stop()
                                break
                        elif ch in ("\r", "\n"):
                            # On POSIX without Kitty protocol, LF vs CR
                            self.state.submit()
                        elif ch in ("\x7f", "\x08"):
                            self.state.backspace()
                        elif ch == "\t":
                            self.state.insert_text("    ")
                        elif ch == "\x03":  # Ctrl+C
                            current, _, _ = self.state.snapshot()
                            if current:
                                self.state.clear()
                            else:
                                self._trigger_stop()
                                break
                        elif ch.isprintable():
                            self.state.insert_char(ch)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


# ── Global singletons ────────────────────────────────────────────────────────

_GLOBAL_INPUT_QUEUE = InputQueue()
_GLOBAL_LIVE_INPUT_STATE = LiveInputState(_GLOBAL_INPUT_QUEUE)
_ACTIVE_TURN_INPUT_LISTENER: Optional[TurnInputListener] = None


def get_input_queue() -> InputQueue:
    """Return the global InputQueue instance."""
    return _GLOBAL_INPUT_QUEUE


def get_live_input_state() -> LiveInputState:
    """Return the global LiveInputState instance."""
    return _GLOBAL_LIVE_INPUT_STATE


def pause_active_input_listener() -> None:
    """Pause the active background input listener if one is running."""
    if _ACTIVE_TURN_INPUT_LISTENER is not None:
        _ACTIVE_TURN_INPUT_LISTENER.pause()


def resume_active_input_listener() -> None:
    """Resume the active background input listener if one is running."""
    if _ACTIVE_TURN_INPUT_LISTENER is not None:
        _ACTIVE_TURN_INPUT_LISTENER.resume()

