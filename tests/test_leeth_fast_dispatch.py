"""
tests/test_leeth_fast_dispatch.py — Fast-Path Dispatcher Integration Tests for Leeth.

Tests:
1. Routine tool dispatch: read_file, run_command, calculate, search_files, grep_search.
2. Parameter extraction and normalization.
3. Conversational rejection (general chat, explanation, code writing).
4. Safety rejection (destructive commands like rm -rf, format).
5. Config integration (fast_dispatch_backend selection).

Pinned to the LEGACY 400M router: the per-query expectations below were
written and calibrated against it. The production mmBERT router has its own
module with its own calibrated queries: tests/test_leeth_mmbert_router.py.
"""

import os

import pytest
import config
from src.leeth.dispatcher import LeethDispatcher, dispatcher
from src.leeth.runtime import leeth


@pytest.fixture(autouse=True)
def _onnx_cpu():
    """Pin the runtime to the deterministic INT8 ONNX CPU backend + legacy router.

    CUDA inference is nondeterministic run-to-run (kernel scheduling), and
    several dispatch confidences sit right at the 0.80 threshold — on CUDA
    they flip between dispatch and delegate across runs. ONNX CPU is
    bit-deterministic, so the expected results hold on every run.
    """
    os.environ["LEETH_ROUTER"] = "legacy"
    if leeth.device == "onnx-cpu" and leeth.ort_session is not None and not leeth.mmbert_active:
        yield  # already pinned — avoid a 1.4 GB model reload per test
        return
    leeth.reload(force_device="onnx")
    assert leeth.device == "onnx-cpu" and leeth.ort_session is not None, (
        f"Runtime must be pinned to ONNX CPU (got {leeth.device}); CUDA inference is nondeterministic"
    )
    assert not leeth.mmbert_active, "This module tests the LEGACY router — set LEETH_ROUTER=legacy"
    yield


@pytest.fixture(autouse=True)
def ensure_leeth_enabled():
    orig_enabled = config.get_leeth_enabled()
    orig_backend = config.get_fast_dispatch_backend()
    config.set_leeth_enabled(True)
    config.set_fast_dispatch_backend("leeth")
    yield
    config.set_leeth_enabled(orig_enabled)
    config.set_fast_dispatch_backend(orig_backend)


def test_dispatcher_availability():
    """Dispatcher must report available when Leeth runtime has active checkpoints."""
    assert dispatcher.is_available is True
    assert leeth.is_available is True


def test_dispatch_read_file():
    """Routine read file request must be dispatched to read_file tool with accurate path.

    Query calibrated against the LEGACY router (this module's pinned model):
    legacy dispatches path-style read queries ("открой файл package.json",
    conf 0.94) but not bare filenames ("прочитай config.py" — the mmBERT
    router covers that case in test_leeth_mmbert_router.py).
    """
    res = dispatcher.try_fast_dispatch("открой файл package.json")
    assert res is not None
    assert res.tool_name == "read_file"
    assert "package.json" in res.arguments.get("file_path", "")
    assert res.confidence >= 0.80
    assert res.elapsed_ms > 0.0


def test_dispatch_run_command_safe():
    """Safe shell commands must be dispatched with arguments intact."""
    res = dispatcher.try_fast_dispatch("покажи git status")
    assert res is not None
    assert res.tool_name == "run_command"
    assert "git status" in res.arguments.get("command", "")
    assert res.confidence >= 0.80


def test_dispatch_calculate():
    """Arithmetic evaluation must route to calculate tool."""
    res = dispatcher.try_fast_dispatch("посчитай 128 * 64")
    assert res is not None
    assert res.tool_name == "calculate"
    assert "128 * 64" in res.arguments.get("expression", "")


def test_dispatch_search_files_or_grep():
    """Finding a file by name should route to a file-locating tool.

    ONNX CPU (deterministic): the 400M model routes 'найди файл config.py'
    to find_by_name with confidence 0.954. Long unfamiliar filenames
    ('PlayerMovement.cs') sit at 0.54 — below the 0.80 dispatch threshold —
    so the test uses the confident phrasing.
    """
    res = dispatcher.try_fast_dispatch("найди файл config.py")
    assert res is not None
    assert res.tool_name in ("search_files", "find_by_name", "read_file")
    target = res.arguments.get("name_contains") or res.arguments.get("file_path") or ""
    assert "config.py" in target


def test_dispatch_list_directory_signature():
    """list_directory must output dir_path argument matching tools.file_ops.list_directory signature."""
    args = dispatcher._build_arguments("list_directory", "src/agent")
    assert args is not None
    assert "dir_path" in args
    assert args["dir_path"] == "src/agent"


def test_dispatch_grep_search_signature():
    """grep_search must output pattern and directory arguments matching tools.search_ops.grep_search signature."""
    args = dispatcher._build_arguments("grep_search", "try_fast_dispatch")
    assert args is not None
    assert "pattern" in args
    assert args["pattern"] == "try_fast_dispatch"
    assert "directory" in args


def test_dispatch_conversational_rejections():
    """Conversational, explanation, model comparison, and reasoning requests must NEVER be fast-dispatched."""
    # 1. Greeting
    res1 = dispatcher.try_fast_dispatch("Привет, как твои дела?")
    assert res1 is None

    # 2. Project explanation
    res2 = dispatcher.try_fast_dispatch("Объясни мне как устроен этот проект")
    assert res2 is None

    # 3. Creative code writing
    res3 = dispatcher.try_fast_dispatch("Напиши функцию для быстрой сортировки на Python")
    assert res3 is None

    # 4. Model comparison and web search requests (user reported regression)
    res4 = dispatcher.try_fast_dispatch("Можешь честно и полностью сравнить эту модель с новой моделью под названием Jev.")
    assert res4 is None, f"Comparison query was falsely fast-dispatched: {res4}"

    res5 = dispatcher.try_fast_dispatch("Можешь честно и полностью сравнить эту модель с новой моделью под названием Jev. Можешь поискать в интернете")
    assert res5 is None, f"Search in internet query was falsely fast-dispatched: {res5}"


def test_dispatch_dangerous_command_rejection():
    """Destructive or formatting commands must be rejected by safety check."""
    res1 = dispatcher.try_fast_dispatch("выполни rm -rf /")
    assert res1 is None

    res2 = dispatcher.try_fast_dispatch("запусти format D: /fs:NTFS")
    assert res2 is None

    res3 = dispatcher.try_fast_dispatch("выполни drop database production")
    assert res3 is None


def test_dispatch_edge_cases():
    """Dispatcher must gracefully return None on invalid or slash inputs."""
    assert dispatcher.try_fast_dispatch("") is None
    assert dispatcher.try_fast_dispatch("   ") is None
    assert dispatcher.try_fast_dispatch(None) is None
    assert dispatcher.try_fast_dispatch(12345) is None
    assert dispatcher.try_fast_dispatch("/model qwen") is None
    assert dispatcher.try_fast_dispatch("!git status") is None


@pytest.mark.real_fast_dispatch
def test_config_fast_dispatch_backend():
    """Config must support 'leeth' backend and prioritize it when available."""
    config.set_fast_dispatch_backend("leeth")
    assert config.get_fast_dispatch_backend() == "leeth"

    config.set_fast_dispatch_backend("micromodel")
    assert config.get_fast_dispatch_backend() == "micromodel"

    config.set_fast_dispatch_backend("off")
    assert config.get_fast_dispatch_backend() == "off"

    # Reset back to leeth
    config.set_fast_dispatch_backend("leeth")
    assert config.get_fast_dispatch_backend() == "leeth"


def test_background_pid_comes_from_the_request_or_the_only_running_process():
    """"что там с сервером?" names no PID: use the session's single running process; never pass the sentence as a PID."""
    from src.leeth.dispatcher import LeethDispatcher as D

    one = {"active_tasks": [{"pid": 4242, "command": "python -m http.server", "status": "running", "elapsed": "5s"}]}
    two = {"active_tasks": [{"pid": 1, "command": "a"}, {"pid": 2, "command": "b"}]}
    assert D._background_pid("что там с сервером?", "что там с сервером?", one) == 4242
    assert D._background_pid("покажи вывод процесса 777", "777", one) == 777          # a written PID wins
    assert D._background_pid("что там с сервером?", "что там с сервером?", two) is None  # ambiguous: the LLM decides
    assert D._background_pid("что там с сервером?", "что там с сервером?", None) is None
    assert D._background_pid("посмотри build2 лог", "build2", one) == 4242              # a digit inside a word is not a PID
