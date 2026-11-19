"""The name of the operating system, without asking the OS about it.

`platform.system()` builds the whole `uname()` record first, and on Windows that goes through a WMI query. When the
WMI service is stuck (it happens) the query never returns and Argent hangs while it assembles the first system
prompt. `sys.platform` knows the answer already and cannot block.
"""

import sys


def system() -> str:
    """'Windows', 'Darwin' or 'Linux' (anything else is reported as Linux, like the unix shells it behaves as)."""
    if sys.platform.startswith("win") or sys.platform == "cygwin":
        return "Windows"
    if sys.platform == "darwin":
        return "Darwin"
    return "Linux"
