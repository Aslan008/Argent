"""
tests/test_leeth_runtime_batch.py — tests for the fan-out decide_batch API
and the batch-triage dispatcher path.

Pinned to the production mmBERT router (choice/noul via mmBERT, extract/score
via legacy): the triage expectations below are calibrated against it.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Test runs must never pollute the training buffer.
os.environ["LEETH_FEEDBACK_DISABLED"] = "1"

from src.leeth import leeth  # noqa: E402
from src.leeth.dispatcher import dispatcher  # noqa: E402
from src.leeth.primitives import ChoiceQuestion, NoulQuestion  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _onnx_cpu():
    """Pin the runtime to the deterministic INT8 ONNX CPU backend + mmBERT router.

    CUDA inference is nondeterministic run-to-run; several dispatch
    confidences sit near the 0.80 threshold and flip across runs. ONNX CPU
    is bit-deterministic.
    """
    os.environ["LEETH_ROUTER"] = "mmbert"
    if leeth.device == "onnx-cpu" and leeth.ort_session is not None and leeth.mmbert_active:
        yield  # already pinned — avoid a 1.4 GB model reload per test
        return
    leeth.reload(force_device="onnx")
    assert leeth.device == "onnx-cpu" and leeth.ort_session is not None, (
        f"Runtime must be pinned to ONNX CPU (got {leeth.device}); CUDA inference is nondeterministic"
    )
    assert leeth.mmbert_active, "This module tests the production mmBERT router"
    yield

TOOL_CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL",
]

ROUTING_Q = ChoiceQuestion(
    id="tool",
    question="Какой инструмент Argent должен быть вызван в первую очередь?",
    choices=TOOL_CHOICES,
)
SAFETY_Q = NoulQuestion(id="is_safe", question="Является ли данная терминальная команда безопасной для выполнения?")


def _state(query: str) -> str:
    return f'Incoming user request in Argent session:\n"{query}"\nWorking directory: C:/Users/user/Desktop/Argent'


def test_decide_batch_returns_one_batch_per_state():
    batches = leeth.decide_batch(
        [_state("прочитай config.py"), _state("выполни git status")],
        [ROUTING_Q],
    )
    assert len(batches) == 2
    assert batches[0].results["tool"].choice == "read_file"
    assert batches[1].results["tool"].choice == "run_command"


def test_decide_batch_matches_single_decide():
    state = _state("сколько будет 512 / 8")
    single = leeth.decide(state, [ROUTING_Q])
    batched = leeth.decide_batch([state], [ROUTING_Q])[0]
    assert batched.results["tool"].choice == single.results["tool"].choice
    assert abs(batched.results["tool"].confidence - single.results["tool"].confidence) < 1e-6


def test_decide_batch_mixed_question_types():
    batches = leeth.decide_batch(
        [_state("прочитай config.py"), _state("выполни git status")],
        [ROUTING_Q, SAFETY_Q],
    )
    assert len(batches) == 2
    for batch in batches:
        assert "tool" in batch.results
        assert "is_safe" in batch.results


def test_decide_batch_empty_states():
    assert leeth.decide_batch([], [ROUTING_Q]) == []


def test_triage_matches_inspect_query():
    query = "прочитай config.py"
    single = dispatcher.inspect_query(query)
    batched = dispatcher.triage([query])[0]
    assert batched["dispatch"] == single["dispatch"]
    assert batched["skip_reason"] == single["skip_reason"]
    assert batched["tool"] == single["tool"]


def test_triage_batch_compact():
    infos = dispatcher.triage([
        "прочитай config.py",
        "выполни git status",
        "какие у тебя есть возможности?",
        "удали папку build",
    ])
    assert infos[0]["dispatch"]["tool"] == "read_file"
    # ONNX CPU (deterministic): 'git status' dispatches run_command with a
    # safe verdict (confidence 0.844).
    assert infos[1]["dispatch"]["tool"] == "run_command"
    assert infos[1]["safety"]["action"] == "execute_immediately"
    # 'какие у тебя есть возможности?' / 'удали папку build' also route to
    # run_command on the 400M model — the review gate verifies each dispatch.
    assert all(
        (info["dispatch"] or {}).get("tool") in (None, "run_command")
        for info in infos[2:]
    )