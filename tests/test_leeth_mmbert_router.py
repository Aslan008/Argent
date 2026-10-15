"""
tests/test_leeth_mmbert_router.py — mmBERT Router Integration Tests (D3).

The production router: mmBERT (ModernBERT backbone, INT8) serves the CHOICE
head (tool routing — its validated strength: 66.1% prod-battery vs 55.0%
legacy, 95% dispatch precision at the gates). Noul/extract/score stay on the
legacy 400M model: the mmBERT ONNX export carries only choice_logits, and its
noul head is unvalidated (compiler-error has_error and safety-state
calibration regress vs legacy).

Tests:
1. Router active + prod-battery parity (72/109 = 66.1%, the E5-validated number).
2. Dispatcher fast path on mmBERT-calibrated queries (read_file, run_command
   safety fan-out, calculate).
3. Conversational rejection.
4. Rollback: LEETH_ROUTER=legacy disables the mmBERT router.
5. Fallback: missing mmBERT artifact -> legacy-only, runtime still works.
"""

import json
import os
from pathlib import Path

import pytest
import config

os.environ["LEETH_ROUTER"] = "mmbert"

from src.leeth.dispatcher import dispatcher  # noqa: E402
from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion  # noqa: E402
from src.leeth.runtime import leeth  # noqa: E402

BATTERY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prod_battery.json"
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
DISPATCH_CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR",
]


@pytest.fixture(autouse=True)
def _mmbert_onnx():
    """Pin the runtime to ONNX CPU with the mmBERT router active."""
    os.environ["LEETH_ROUTER"] = "mmbert"
    if leeth.device == "onnx-cpu" and leeth.ort_session is not None and leeth.mmbert_active:
        yield
        return
    leeth.reload(force_device="onnx")
    assert leeth.device == "onnx-cpu" and leeth.ort_session is not None
    assert leeth.mmbert_active, "mmBERT router must be active for this module"
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


def _build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def test_router_active():
    assert leeth.mmbert_active is True
    assert leeth.mmbert_session is not None
    assert leeth.mmbert_tokenizer is not None


def test_battery_parity():
    """Prod battery through the REAL runtime must hit the battery-track 91/109.

    History: E5 validated 72/109; battery-track iterations (cand9 86, cand10 91,
    commit 26fd28c) raised it to 91/109 — the promoted prod graph is cand10.
    """
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    correct = 0
    for case in battery["cases"]:
        state = _build_state(case["query"])
        q = ChoiceQuestion(id="tool", question=TEMPLATE_Q, choices=DISPATCH_CHOICES)
        batch = leeth.decide(state, [q])
        res = batch.results.get("tool")
        if res is None:
            continue
        correct += int(res.choice == case["expected_tool"])
    assert correct >= 91, f"mmBERT battery parity broken: {correct}/109 (expected at least 91)"


def test_dispatch_read_file():
    res = dispatcher.try_fast_dispatch("прочитай config.py")
    assert res is not None
    assert res.tool_name == "read_file"
    assert "config.py" in res.arguments.get("file_path", "")
    # D5-промоушен (E_attn_all: attention fp32): confidence 0.795 на этом запросе
    # (старый артефакт с квантизованным attention давал >=0.80). Гейт диспетчера 0.65.
    assert res.confidence >= 0.75


def test_dispatch_run_command_safety_fanout():
    """run_command dispatch must pass the neural safety gate (is_safe + action)."""
    res = dispatcher.try_fast_dispatch("выполни git status")
    assert res is not None
    assert res.tool_name == "run_command"
    assert "git status" in res.arguments.get("command", "")


def test_dispatch_calculate():
    res = dispatcher.try_fast_dispatch("посчитай (15 * 4) + sqrt(144)")
    assert res is not None
    assert res.tool_name == "calculate"


def test_conversational_rejection():
    res = dispatcher.try_fast_dispatch("привет, как твои дела?")
    assert res is None


def test_code_writing_delegates():
    """Code-writing requests must NOT hit the fast path (LLM handles them)."""
    res = dispatcher.try_fast_dispatch("напиши функцию сортировки на python")
    assert res is None


def test_concrete_tool_requests_routed_correctly():
    """Direct routine commands must trigger the exact tool and extract parameters.

    Moved from test_leeth_hard_negatives.py (pinned to the legacy model,
    which does not dispatch this query): the production mmBERT router does.
    """
    choices = [
        "read_file", "write_file", "replace_in_file", "run_command",
        "grep_search", "find_by_name", "list_directory", "browser_open",
        "browser_click", "browser_type", "browser_upload_file", "calculate",
        "ask_user_questions", "NO_TOOL",
    ]

    q_tool = [
        ChoiceQuestion(id="tool", question="Какой инструмент Argent должен быть вызван в первую очередь?", choices=choices),
        NoulQuestion(id="need_tool", question="Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"),
        ExtractQuestion(id="param", question="Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)"),
    ]

    state = "Incoming user request in Argent session:\nПользователь попросил прочитать файл config.json\nWorking directory: C:/Users/user/Desktop/Argent"
    res = leeth.decide(state, q_tool)
    assert res.results["tool"].choice == "read_file"
    assert res.results["need_tool"].value is True
    assert res.results["param"].text.strip() != ""


def test_rollback_env_disables_router():
    """LEETH_ROUTER=legacy must disable the mmBERT router on reload."""
    try:
        os.environ["LEETH_ROUTER"] = "legacy"
        leeth.reload(force_device="onnx")
        assert leeth.mmbert_active is False
        assert leeth.ort_session is not None  # legacy model still works
        state = _build_state("прочитай config.py")
        batch = leeth.decide(state, [ChoiceQuestion(id="tool", question=TEMPLATE_Q, choices=DISPATCH_CHOICES)])
        assert batch.results.get("tool") is not None
    finally:
        os.environ["LEETH_ROUTER"] = "mmbert"
        leeth.reload(force_device="onnx")
        assert leeth.mmbert_active is True


def test_fallback_missing_artifact():
    """Missing mmBERT artifact must fall back to legacy-only, runtime intact."""
    orig_dir = leeth.mmbert_dir
    try:
        leeth.mmbert_dir = Path("Z:/nonexistent/mmbert")
        leeth.reload(force_device="onnx")
        assert leeth.mmbert_active is False
        assert leeth.ort_session is not None
        state = _build_state("прочитай config.py")
        batch = leeth.decide(state, [ChoiceQuestion(id="tool", question=TEMPLATE_Q, choices=DISPATCH_CHOICES)])
        assert batch.results.get("tool") is not None
    finally:
        leeth.mmbert_dir = orig_dir
        leeth.reload(force_device="onnx")
        assert leeth.mmbert_active is True