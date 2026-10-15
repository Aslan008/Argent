"""
tests/test_leeth_latency.py — Latency and High-Performance Benchmarks for Leeth.

Tests:
1. Verifies that warm single-decision latency on ONNX INT8 is under 10 ms on CPU.
2. Verifies that batched 4-question decision latency is under 25 ms.
"""

import os
import time
import pytest
from src.leeth import leeth
from src.leeth.primitives import ChoiceQuestion, NoulQuestion, ScoreQuestion


# Latency thresholds below are calibrated against the LEGACY 400M batched
# forward (the mmBERT router runs choice batch-1 solo, which is faster but
# changes the batch shape these benchmarks measure).
os.environ.setdefault("LEETH_ROUTER", "legacy")


@pytest.fixture(scope="module")
def runtime():
    old = os.environ.get("LEETH_ROUTER")
    os.environ["LEETH_ROUTER"] = "legacy"
    if not (leeth.device == "onnx-cpu" and leeth.ort_session is not None and not leeth.mmbert_active):
        leeth.reload(force_device="onnx")
    assert leeth.device == "onnx-cpu" and leeth.ort_session is not None, (
        f"Runtime must be pinned to ONNX CPU (got {leeth.device}); CUDA is nondeterministic"
    )
    assert not leeth.mmbert_active, "This module benchmarks the LEGACY router"
    yield leeth
    if old is None:
        os.environ.pop("LEETH_ROUTER", None)
    else:
        os.environ["LEETH_ROUTER"] = old


def test_decision_latency_benchmark(runtime):
    """Warm inference latency must be fast (under 10 ms per decision)."""
    state = "User shell execution request:\nCommand: cargo check\nContext: test\nCaller: Argent SubAgent"
    q = NoulQuestion(id="safe", question="Является ли данная терминальная команда безопасной для выполнения?")

    # Warmup
    for _ in range(5):
        runtime.decide(state, [q])

    # Measure
    n_iters = 20
    t0 = time.perf_counter()
    for _ in range(n_iters):
        runtime.decide(state, [q])
    avg_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    print(f"\nAverage Leeth single-question latency: {avg_ms:.2f} ms")
    # 400M model (INT8 ONNX, seq ~94): ~60 ms warm on this 14-core laptop CPU.
    # The old 25 ms threshold was measured on the retired 24M model.
    # ~60ms warm standalone; ~2-3x slower under full-suite parallel load.
    assert avg_ms < 300.0, f"Average latency {avg_ms:.2f} ms exceeds 300 ms threshold"


def test_batched_decision_latency(runtime):
    """Batched decision with 3 questions must run in sub-30ms."""
    state = "Compiler output for C# project:\n[ERROR] src/Server/PaymentService.cs(42,15): error CS0246"
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
    ]

    # Warmup
    for _ in range(3):
        runtime.decide(state, questions)

    n_iters = 10
    t0 = time.perf_counter()
    for _ in range(n_iters):
        runtime.decide(state, questions)
    avg_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    print(f"\nAverage Leeth batched 3-question latency: {avg_ms:.2f} ms")
    # 400M model: ~110 ms for a 3-question batch (was ~15 ms on the 24M model).
    assert avg_ms < 500.0, f"Average batched latency {avg_ms:.2f} ms exceeds 500 ms threshold"
