"""The stop switch of a running workflow.

`ArgentAgent.request_stop` (Esc, a closed window, /ultracode stop) sets it; the engine looks at it between two
polls of its workers and ends the run cleanly: workers stopped, state saved, `/ultracode resume` possible.
"""

import threading

_STOP = threading.Event()


def request_stop() -> None:
    _STOP.set()


def reset() -> None:
    _STOP.clear()


def stop_requested() -> bool:
    return _STOP.is_set()
