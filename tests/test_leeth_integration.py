"""
tests/test_leeth_integration.py — End-to-End Integration Tests for Leeth System One in Argent.

Tests:
1. Multi-question batch inference in a single parallel pass.
2. Integration with ModelStrategy.evaluate_command_safety (safe vs dangerous).
3. Integration with ModelStrategy.analyze_compiler_error (C#, C++, Rust, Python).
4. Config toggle behavior (get_leeth_enabled, set_leeth_enabled, get_leeth_device).
5. Dual backend switching (ONNX INT8 CPU vs CUDA/PyTorch).
"""

import os

import pytest
import config
from src.agent.strategy import get_model_strategy
from src.leeth import leeth
from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion, ScoreQuestion


# The compiler-error analysis below asserts the LEGACY noul head's has_error
# calibration (the mmBERT noul head is unvalidated and regresses on compiler
# states). Legacy choice/extract heads serve language/param here.
os.environ.setdefault("LEETH_ROUTER", "legacy")


@pytest.fixture(autouse=True)
def _legacy_router():
    """Pin the router mode to legacy for this module's calibrated expectations."""
    old = os.environ.get("LEETH_ROUTER")
    os.environ["LEETH_ROUTER"] = "legacy"
    if not (leeth.device == "onnx-cpu" and leeth.ort_session is not None and not leeth.mmbert_active):
        leeth.reload(force_device="onnx")
    assert not leeth.mmbert_active, "This module tests the LEGACY router"
    yield
    if old is None:
        os.environ.pop("LEETH_ROUTER", None)
    else:
        os.environ["LEETH_ROUTER"] = old


@pytest.fixture(autouse=True)
def ensure_leeth_enabled():
    orig_enabled = config.get_leeth_enabled()
    orig_dev = config.get_leeth_device()
    config.set_leeth_enabled(True)
    yield
    config.set_leeth_enabled(orig_enabled)
    config.set_leeth_device(orig_dev)


def test_leeth_availability_and_config():
    """Leeth must report available when checkpoints exist and respect config flags."""
    assert leeth.is_available is True

    # Test toggling
    config.set_leeth_enabled(False)
    assert config.get_leeth_enabled() is False

    config.set_leeth_enabled(True)
    assert config.get_leeth_enabled() is True

    # Test device setter
    config.set_leeth_device("onnx")
    assert config.get_leeth_device() == "onnx"

    config.set_leeth_device("auto")
    assert config.get_leeth_device() == "auto"


def test_strategy_command_safety_evaluation():
    """ModelStrategy must evaluate command safety using Leeth."""
    strategy = get_model_strategy("qwen2.5:7b", "ollama")

    # 1. Safe commands
    safe, conf, reason = strategy.evaluate_command_safety("git status")
    assert safe is True
    assert conf >= 0.90
    assert "verified" in reason.lower()

    safe_check, conf_check, _ = strategy.evaluate_command_safety("cargo check")
    assert safe_check is True
    assert conf_check >= 0.90

    # 2. Dangerous commands
    safe_rm, conf_rm, reason_rm = strategy.evaluate_command_safety("rm -rf /")
    assert safe_rm is False
    assert conf_rm <= 0.10
    assert "dangerous" in reason_rm.lower()

    safe_fmt, conf_fmt, _ = strategy.evaluate_command_safety("format D: /fs:NTFS /q")
    assert safe_fmt is False
    assert conf_fmt <= 0.10


def test_strategy_compiler_error_analysis():
    """ModelStrategy must analyze compiler output and extract language, severity, and files."""
    strategy = get_model_strategy("qwen2.5:7b", "ollama")

    cpp_log = (
        "Compiler stderr from clang++ / cmake build:\n"
        "src/renderer/vulkan_pipeline.cpp:118:5: error: undefined reference to 'vtable for RendererPipeline'\n"
        "Build target: //game:engine"
    )

    diag = strategy.analyze_compiler_error(cpp_log)
    assert diag.get("has_error") is True
    assert diag.get("error_probability", 0.0) >= 0.95
    assert diag.get("language") == "C++"
    assert diag.get("suggested_tool") == "read_file"
    assert "vulkan_pipeline.cpp" in diag.get("target_file", "")
    assert diag.get("severity_level") >= 3
    # 400M ONNX INT8 (seq ~120): ~52ms warm (the 50ms threshold was for the retired 24M model).
    assert diag.get("elapsed_ms", 999.0) < 400.0  # 400M model: ~52ms warm standalone, ~220ms under full-suite load


def test_multi_question_batching():
    """Evaluating 4 different primitives simultaneously should complete in a single forward pass."""
    state = "Build output for C# project:\n[ERROR] src/Server/PaymentService.cs(42,15): error CS0246"
    questions = [
        ChoiceQuestion(
            id="tool",
            question="Какой инструмент Argent оптимален для исправления данной проблемы?",
            choices=["read_file", "replace_in_file", "search_files", "run_command", "delete_file"],
        ),
        NoulQuestion(id="err", question="Содержит ли вывод компилятора критическую ошибку?"),
        ScoreQuestion(
            id="score",
            question="Оцените критичность и приоритет исправления данной проблемы по шкале 0..100",
            rubric=["Низкий", "Средний", "Высокий", "Критический", "Блокирующий"],
        ),
        ExtractQuestion(id="span", question="Укажите путь к проблемному файлу C++"),
    ]

    batch = leeth.decide(state, questions)
    assert len(batch.results) == 4
    assert batch.results["tool"].choice in ("read_file", "replace_in_file")
    assert batch.results["err"].value is True
    assert batch.results["score"].level >= 3
    # 400M model: ~140ms for a 4-question batch (was ~15ms on the 24M model).
    assert batch.elapsed_ms < 250.0


def test_disabled_leeth_graceful_fallback():
    """When disabled in config, strategy methods must degrade gracefully without errors."""
    config.set_leeth_enabled(False)
    strategy = get_model_strategy("qwen2.5:7b", "ollama")

    safe, conf, reason = strategy.evaluate_command_safety("git status")
    assert safe is True
    assert reason == "Leeth disabled"

    diag = strategy.analyze_compiler_error("error CS0246")
    assert diag == {}
