"""Root conftest: runs before everything else, so it does two jobs that cannot wait.

1. Keep the run out of the developer's real home directory (``_isolate_user_home``).
2. Ensure the project root is on sys.path.

A third-party ``logger`` package (v1.4) is installed in user site-packages.
Without this file, when pytest collects the entire test suite the site-packages
package can win the ``import logger`` race and shadow the local ``logger.py``,
causing 137 collection errors (``cannot import name 'get_logger``).
"""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


def _isolate_user_home() -> None:
    """Make the developer's real home directory invisible to the test run.

    Argent keeps its state under the user's home — ~/.argent_coder_config.json
    (provider, API keys, MCP servers, disabled tools) and ~/.argent/{logs,
    sessions,file_history} — and resolves those paths once, at import time. A
    full run used to read all of it and write to part of it:

    * it rewrote the developer's real config (the file's mtime moved during a
      run), so a test that calls any ``set_*`` changed live settings;
    * every ReplBridge a test built started the developer's real MCP servers
      (about fifty processes per run), and at exit the leaked servers hung
      ``mcp_client.stop_all()``, so pytest never returned after its summary;
    * results depended on the machine: a ``disabled_tools`` entry in the real
      config failed the tool-profile tests on one box and not on another.

    This has to run before the first project import (``logger`` below computes
    its log directory at import), which is why it lives in the root conftest.
    Set ``ARGENT_TESTS_USE_REAL_HOME=1`` to opt out when debugging.
    """
    if os.environ.get("ARGENT_TESTS_USE_REAL_HOME") == "1":
        return
    real_home = Path.home()
    # Caches that are large, read-only and expensive to refill must survive the
    # move: models and tokenizers resolve through these, not through "~".
    for var, rel in (("HF_HOME", ".cache/huggingface"),
                     ("TORCH_HOME", ".cache/torch"),
                     ("XDG_CACHE_HOME", ".cache")):
        os.environ.setdefault(var, str(real_home / rel))
    # The owner's SAT solver (the "P = PN" project, outside this repo) is read-only too; src/leeth/arbitrator.py
    # looks for it under the home directory unless PN_SOLVER_PATH says where it is.
    os.environ.setdefault("PN_SOLVER_PATH", str(real_home / "Desktop" / "P = PN" / "sat.py"))
    # The routing models were trained on a state that names the working directory; the dispatcher builds it from
    # the home directory, which a test run has moved, so it is told the real one (LEETH_STATE_WORKDIR).
    os.environ.setdefault("LEETH_STATE_WORKDIR", (real_home / "Desktop" / "Argent").as_posix())
    # Tests that commit inside a tmp repo need an identity; reading it is harmless.
    gitconfig = real_home / ".gitconfig"
    if gitconfig.is_file():
        os.environ.setdefault("GIT_CONFIG_GLOBAL", str(gitconfig))
    os.environ["ARGENT_TEST_REAL_HOME"] = str(real_home)

    fake_home = tempfile.mkdtemp(prefix="argent-test-home-")
    atexit.register(shutil.rmtree, fake_home, ignore_errors=True)
    os.environ["USERPROFILE"] = fake_home   # Windows: what Path.home() reads
    os.environ["HOME"] = fake_home          # POSIX
    # Compaction archives the dropped messages under the PROJECT's .argent/; the project here is the developer's
    os.environ["ARGENT_TRANSCRIPT_DIR"] = os.path.join(fake_home, "transcripts")


_isolate_user_home()

# A ReplBridge built by a test starts the scheduler like the real one, and the
# project it runs in is the developer's own: its schedule must never fire here.
os.environ.setdefault("ARGENT_DISABLE_AUTOMATIONS", "1")

# A bridge attached by a test has no TUI to answer "С чего начать?"; waiting for
# one would hang the test. The start-up tests clear this and answer for themselves.
os.environ.setdefault("ARGENT_NO_STARTUP_PROMPTS", "1")

# A fake provider that raises a connection error must end the turn at once, not sit out the 15 minutes a real
# user would wait for the network. The tests of the waiting itself set their own patience.
os.environ.setdefault("ARGENT_NETWORK_PATIENCE", "0")

# Requests to a local model server queue in a lane and wait while the machine is busy (src/agent/governor.py).
# A test must not depend on how loaded the developer's PC is at that moment; the tests of the governor itself
# switch it back on.
os.environ.setdefault("ARGENT_GOVERNOR", "0")

# The summary a compaction will need is written ahead of time on a background thread (src/agent/summary_precompute.py);
# a test must not have a model call start by itself after a turn. Its own tests switch it back on.
os.environ.setdefault("ARGENT_PRECOMPUTE", "0")

# The render tests assert on colour escapes; a developer shell (or an IDE terminal) that exports NO_COLOR
# would turn them all red. The no-colour behaviour has its own tests, which set the variable themselves.
os.environ.pop("NO_COLOR", None)

_ROOT = str(Path(__file__).parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Pre-import the local logger so it is cached in sys.modules before any
# test module triggers the import.
import logger  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _fresh_compaction_history():
    """The compaction thrash guard remembers recent compactions per process. A test must start with a clean
    memory, or the third test that compacts a little is told the "conversation" is thrashing."""
    try:
        from src.agent import trimmer
        trimmer.reset_compaction_history()
    except Exception:
        pass
    yield


def pytest_configure(config):
    config.addinivalue_line("markers", "real_fast_dispatch: the test needs the real config.get_fast_dispatch_backend (tests turn the fast path off by default)")
