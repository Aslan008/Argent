"""Тесты DoPT: формат промпта (консистентность с кернелом), маркеры, runtime-интеграция.

Тяжёлые проверки (ONNX vs torch на реальной модели) — scripts/_dopt_verify.py,
здесь только лёгкие unit-тесты без загрузки весов.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.leeth.dopt_export import build_text as export_build_text
from src.leeth.dopt_export import find_marker_positions as export_find_markers
from src.leeth.dopt_solver import DoptResult, DoptSolver, build_text, find_marker_positions


# ---------------------------------------------------------------------------
# Формат промпта — байт-в-байт консистентность solver/export/кернел
# ---------------------------------------------------------------------------

def test_build_text_format_matches_kernel():
    """Формат идентичен kernel_dopt: [STATE] ... [QUESTION] ... [OPT] ... [OPT] ..."""
    text = build_text("змейка 6x6, H=голова, F=еда, B=тело\nH....\n", "Какой ход оптимален?", ["вверх", "вниз"])
    assert text == (
        "[STATE] змейка 6x6, H=голова, F=еда, B=тело\nH....\n"
        " [QUESTION] Какой ход оптимален? [OPT] вверх [OPT] вниз"
    )
    # solver и export дают байт-в-байт одинаковый текст
    assert export_build_text("s", "q", ["a", "b"]) == build_text("s", "q", ["a", "b"])


def test_find_marker_positions_subsequence():
    """Маркер ищется как подпоследовательность токенов, несколько вхождений."""
    marker = [7, 8, 9]
    ids = [1, 2, 7, 8, 9, 3, 7, 8, 9, 4]
    assert export_find_markers(ids, marker) == [2, 6]
    assert find_marker_positions(ids, marker) == [2, 6]  # solver == export
    assert find_marker_positions([7, 8], marker) == []  # обрезанный маркер не матчится


# ---------------------------------------------------------------------------
# DoptResult
# ---------------------------------------------------------------------------

def test_dopt_result_fields():
    r = DoptResult(choice="влево", confidence=0.87, probabilities={"влево": 0.87, "вправо": 0.13})
    assert r.choice == "влево"
    assert r.confidence == pytest.approx(0.87)
    assert "влево" in r.probabilities
    assert "DoptResult" in repr(r)


# ---------------------------------------------------------------------------
# DoptSolver: отсутствие артефактов
# ---------------------------------------------------------------------------

def test_solver_missing_assets_raises(tmp_path):
    """Ни ONNX, ни чекпоинт → FileNotFoundError (не тихая деградация)."""
    with pytest.raises(FileNotFoundError):
        DoptSolver(onnx_path=tmp_path / "no.onnx", ckpt_path=tmp_path / "no.pt")


# ---------------------------------------------------------------------------
# Runtime-интеграция: ленивая загрузка + негативный кэш + прокси
# ---------------------------------------------------------------------------

def _fresh_runtime():
    from src.leeth.runtime import LeethRuntime

    rt = LeethRuntime.__new__(LeethRuntime)  # без __init__: тестируем только dopt-часть
    rt.dopt_solver = None
    return rt


def test_runtime_decide_dopt_negative_cache(tmp_path, monkeypatch):
    """Неудачная загрузка солвера кэшируется негативно (не ретраится каждый вызов)."""
    rt = _fresh_runtime()
    calls = []

    def fake_solver():
        calls.append(1)
        raise FileNotFoundError("no assets")

    import src.leeth.dopt_solver as ds

    monkeypatch.setattr(ds, "DoptSolver", fake_solver)
    r1 = rt.decide_dopt("state", "q", ["a", "b"])
    r2 = rt.decide_dopt("state", "q", ["a", "b"])
    assert r1.choice == "" and r1.confidence == 0.0
    assert r2.choice == ""
    assert len(calls) == 1  # негативный кэш: DoptSolver создан один раз
    assert rt.dopt_solver is False


def test_runtime_decide_dopt_proxy():
    """decide_dopt проксирует в солвер и возвращает ChoiceResult."""
    from src.leeth.runtime import LeethRuntime

    rt = _fresh_runtime()
    captured = {}

    def fake_decide(state, question, options):
        captured["args"] = (state, question, list(options))
        return DoptResult(choice="влево", confidence=0.9, probabilities={"влево": 0.9, "вправо": 0.1})

    rt.dopt_solver = SimpleNamespace(decide=fake_decide)
    r = rt.decide_dopt("змейка 6x6", "Какой ход оптимален?", ["влево", "вправо"])
    assert captured["args"] == ("змейка 6x6", "Какой ход оптимален?", ["влево", "вправо"])
    assert r.choice == "влево" and r.confidence == pytest.approx(0.9)
    assert r.probabilities == {"влево": 0.9, "вправо": 0.1}


def test_runtime_decide_dopt_empty_options():
    """Пустой список опций → пустой результат без вызова солвера."""
    rt = _fresh_runtime()
    rt.dopt_solver = SimpleNamespace(decide=lambda *a: (_ for _ in ()).throw(AssertionError("must not be called")))
    r = rt.decide_dopt("state", "q", [])
    assert r.choice == "" and r.confidence == 0.0


def test_runtime_decide_dopt_solver_error_is_swallowed():
    """Ошибка инференса солвера → пустой ChoiceResult (не крашит прод-путь)."""
    rt = _fresh_runtime()

    def boom(*a):
        raise RuntimeError("onnx session dead")

    rt.dopt_solver = SimpleNamespace(decide=boom)
    r = rt.decide_dopt("state", "q", ["a"])
    assert r.choice == "" and r.confidence == 0.0


# ---------------------------------------------------------------------------
# Интеграционный (тяжёлый): реальный ONNX-солвер решает домены оптимально.
# Запускается только если артефакты на месте (локальный прогон, не CI).
# ---------------------------------------------------------------------------

def _artifacts_present() -> bool:
    # DoptSolver() loads its DEFAULT_ONNX — gate on exactly that file, not an older export.
    from src.leeth.dopt_solver import DEFAULT_ONNX

    return DEFAULT_ONNX.exists() and Path(".argent/models/mmbert-argent").exists()


@pytest.mark.skipif(not _artifacts_present(), reason="DoPT ONNX/base not present")
def test_solver_optimal_on_real_domains():
    import random
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "leeth"))
    from task_domains import GENERATORS

    solver = DoptSolver()
    assert solver.backend == "onnx"
    rng = random.Random(7)
    for domain, gen in GENERATORS.items():
        s = None
        while s is None:
            s = gen(rng)
        r = solver.decide(s.state_text, s.question, s.options)
        assert r.choice in s.options
        assert s.options.index(r.choice) in s.optimal_set, f"{domain}: choice {r.choice!r} not optimal"
        assert 0.0 <= r.confidence <= 1.0
        assert abs(sum(r.probabilities.values()) - 1.0) < 1e-4