import pytest


@pytest.fixture(autouse=True)
def _offline_token_estimation(monkeypatch):
    """Keep the test suite offline and fast.

    estimate_tokens() queries the provider's tokenize endpoint (Ollama /
    KoboldCPP) for accuracy. In tests that hit a live endpoint for every unique
    message, which dragged the suite out to ~20 minutes. Stubbing the network
    call makes estimate_tokens fall back to tiktoken / the heuristic instantly,
    without changing the logic under test.
    """
    try:
        import src.agent.trimmer as trimmer
        monkeypatch.setattr(trimmer, "_query_tokenize_api", lambda *a, **k: None)
        # Reset the circuit breaker so state from a previous run never leaks.
        trimmer._API_TOKENIZE_BROKEN.clear()
        trimmer._TOKEN_CACHE.clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _no_auto_checkpoint():
    """Agent dispatch auto-commits a git checkpoint before the first edit of a
    turn. In tests the current directory is the Argent repo itself — a test
    driving an edit tool must never commit the developer's dirty tree. Tests
    that exercise the time machine re-enable it inside a tmp repo."""
    from src.agent import checkpoints
    checkpoints.set_auto_checkpoint(False)
    yield
    checkpoints.set_auto_checkpoint(True)


@pytest.fixture(scope="session")
def _leeth_feedback_sandbox(tmp_path_factory):
    """A throw-away feedback store for the whole run."""
    return str(tmp_path_factory.mktemp("leeth_feedback") / "feedback.jsonl")


@pytest.fixture(autouse=True)
def _leeth_feedback_off(request, monkeypatch, _leeth_feedback_sandbox):
    """Keep automated tests from polluting the Leeth active-learning store.

    Tests that specifically exercise the feedback loop (test_leeth_feedback.py)
    re-enable it with LEETH_FEEDBACK_FORCE=1 and an isolated LEETH_FEEDBACK_PATH.

    Disabling the feedback loop is not enough on its own: some writers (the
    dispatcher's skip log among them) do not consult the flag, and a full run once
    appended ~70 fixture queries ("выполни rm -rf /", "долго ещё?", …) to the
    user's real, git-tracked `src/leeth/data/leeth_feedback.jsonl`. Every test
    therefore also gets a sandbox store path, so whatever writes lands in a temp
    file.
    """
    import os
    if "test_leeth_feedback" in request.node.nodeid:
        yield
        return
    monkeypatch.setenv("LEETH_FEEDBACK_PATH", _leeth_feedback_sandbox)
    old = os.environ.get("LEETH_FEEDBACK_DISABLED")
    os.environ["LEETH_FEEDBACK_DISABLED"] = "1"
    yield
    if old is None:
        os.environ.pop("LEETH_FEEDBACK_DISABLED", None)
    else:
        os.environ["LEETH_FEEDBACK_DISABLED"] = old


@pytest.fixture(autouse=True)
def _fresh_user_settings():
    """Every test starts as a user who has never saved a setting.

    The root conftest points HOME at a scratch directory for the whole run, so a
    test can no longer touch the developer's real config — but one test's
    /provider or /model still landed in the same scratch file the next test read.
    test_tier_switching passed alone and failed in a full run for exactly that
    reason. Only a file inside the scratch home is ever removed."""
    import os
    from pathlib import Path

    import config

    scratch = os.environ.get("USERPROFILE", "")
    isolated = bool(os.environ.get("ARGENT_TEST_REAL_HOME")) and bool(scratch)
    path = Path(config.CONFIG_FILE)
    if isolated and scratch in str(path):
        try:
            path.unlink()
        except OSError:
            pass
        config._CONFIG_CACHE = None
        config._CONFIG_MTIME = None
    yield


@pytest.fixture(autouse=True)
def _restore_working_directory():
    """Restoring a session changes directory to where it was saved
    (src/startup.restore_session), and a test that does that without monkeypatch
    leaves the whole run in somebody else's folder: every later test that opens a
    relative path (the BPE tokenizer's data file) then fails, but only in a full run."""
    import os
    before = os.getcwd()
    yield
    try:
        os.chdir(before)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _isolate_tui_bridge_state():
    """ReplBridge.__init__ flips process-wide state: it sets ARGENT_TUI=1 and
    registers itself as ReplBridge.active_bridge. A test that builds a bridge
    would otherwise leave every later test believing a TUI is attached, so any
    prompt (ask_user_questions, questionary) blocks forever waiting for a modal
    nobody will answer. Restore both after each test, and stop the bridge's
    event monitor."""
    import os
    import sys
    old_env = os.environ.get("ARGENT_TUI")
    mod = sys.modules.get("src.bridge.repl_bridge")
    old_active = getattr(getattr(mod, "ReplBridge", None), "active_bridge", None)
    yield
    # The bridge also starts a thread that drains the process-global EVENT_QUEUE.
    # Left running it eats the events of whichever later test waits on that queue
    # (the heartbeat/sleeper tests failed only in a full run).
    mod = sys.modules.get("src.bridge.repl_bridge")
    built = getattr(getattr(mod, "ReplBridge", None), "active_bridge", None)
    if built is not None and built is not old_active:
        built.stop_event_monitor(join=True)
    if old_env is None:
        os.environ.pop("ARGENT_TUI", None)
    else:
        os.environ["ARGENT_TUI"] = old_env
    mod = sys.modules.get("src.bridge.repl_bridge")
    if mod is not None and hasattr(mod, "ReplBridge"):
        mod.ReplBridge.active_bridge = old_active


@pytest.fixture(autouse=True)
def _isolate_watches():
    """A bridge with a sink registers itself as the process-wide wake-up handler,
    which turns wait_heartbeat non-blocking for every later test. Reset it."""
    yield
    import sys
    mod = sys.modules.get("src.agent.watches")
    if mod is not None:
        mod.WATCHES.set_handler(None)


@pytest.fixture(autouse=True)
def _own_model_cache(tmp_path, monkeypatch):
    """The Claude model list is cached on disk and in memory. One test's catalogue must not be
    the next one's, and no test may read or write the developer's real cache."""
    monkeypatch.setenv("ARGENT_CACHE_DIR", str(tmp_path / "argent-cache"))
    import claude_catalog
    claude_catalog.forget()
    yield
    claude_catalog.forget()


@pytest.fixture(autouse=True)
def _isolate_working_state():
    """Plan mode and the todo list are process-wide; a test that leaves plan mode on would
    make every later test's file edits fail."""
    yield
    import sys
    plan = sys.modules.get("src.agent.plan_mode")
    if plan is not None:
        plan.PLAN.leave()
    todos = sys.modules.get("src.agent.todos")
    # A background thread of the test may still be importing the module: then there is nothing to clear yet.
    store = getattr(todos, "TODOS", None) if todos is not None else None
    if store is not None:
        store.clear()


@pytest.fixture(autouse=True)
def _no_fast_path_unless_asked(monkeypatch, request):
    """A working Leeth fast path answers "what is 40+2" or "привет" itself, so tests that script
    a provider never get asked. Whether it fires also depends on whether an earlier test happened
    to warm the router up — an order-dependent failure. Off by default; tests of the fast path
    itself patch get_fast_dispatch_backend (or call the dispatcher directly) as they need."""
    if request.node.get_closest_marker("real_fast_dispatch"):
        return
    import config
    monkeypatch.setattr(config, "get_fast_dispatch_backend", lambda: "off")
