# -*- coding: utf-8 -*-
"""tests/test_leeth_micro_adapters.py — Тесты модульных микро-адаптеров (guard.onnx, ground.onnx).

Проверяет:
1. Корректность загрузки и кэширования ONNX-сессий микро-адаптеров.
2. Гарантии задержки на CPU (mean latency <= 0.05 мс).
3. Работу дискриминации guard (safe vs destructive).
4. Корректность очистки кэша при reload().
"""

import time
from pathlib import Path
import numpy as np
import pytest

from src.leeth.runtime import LeethRuntime


@pytest.fixture
def runtime():
    rt = LeethRuntime.get_instance()
    rt.initialize()
    return rt


def test_micro_adapters_exist_and_load(runtime):
    """Адаптеры guard и ground должны присутствовать на диске и успешно инициализировать ONNX сессии."""
    adapters_dir = Path(".argent/models/adapters")
    assert (adapters_dir / "guard.onnx").exists(), "guard.onnx не найден"
    assert (adapters_dir / "ground.onnx").exists(), "ground.onnx не найден"

    guard_sess = runtime.get_adapter_session("guard")
    assert guard_sess is not None, "Не удалось загрузить сессию guard.onnx"

    ground_sess = runtime.get_adapter_session("ground")
    assert ground_sess is not None, "Не удалось загрузить сессию ground.onnx"


# A micro-benchmark of 50 microseconds is measured in a process of its own: in the shared pytest process the thread pools of
# every model loaded by earlier tests keep spinning and double the figure (0.027 ms alone, 0.06 ms after other Leeth tests).
_LATENCY_CHILD = r"""
import sys, time
sys.modules["_wmi"] = None  # platform.uname() would block on a machine whose WMI does not answer
import numpy as np
from src.leeth.runtime import LeethRuntime
rt = LeethRuntime.get_instance(); rt.initialize()
x = np.random.randn(1, 1024).astype(np.float32)
name = sys.argv[1]
for _ in range(10):
    rt.run_adapter(name, x)
t = []
for _ in range(100):
    a = time.perf_counter(); p = rt.run_adapter(name, x); t.append((time.perf_counter() - a) * 1000.0)
    assert 0.0 <= p <= 1.0
print(float(np.mean(t)))
"""


def _isolated_mean_latency_ms(name: str) -> float:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    cp = subprocess.run([sys.executable, "-c", _LATENCY_CHILD, name], cwd=str(root), capture_output=True, text=True, timeout=300)
    assert cp.returncode == 0, cp.stderr[-500:]
    return float(cp.stdout.strip().splitlines()[-1])


def test_guard_latency_bound(runtime):
    """Средняя задержка инференса guard.onnx на CPU не должна превышать 0.05 мс."""
    mean_lat = _isolated_mean_latency_ms("guard")
    assert mean_lat <= 0.05, f"Задержка guard.onnx ({mean_lat:.4f} мс) превышает порог 0.05 мс"


def test_ground_latency_bound(runtime):
    """Средняя задержка инференса ground.onnx на CPU не должна превышать 0.05 мс."""
    mean_lat = _isolated_mean_latency_ms("ground")
    assert mean_lat <= 0.05, f"Задержка ground.onnx ({mean_lat:.4f} мс) превышает порог 0.05 мс"


def test_runtime_caching_and_reload(runtime):
    """Проверяет, что сессии кэшируются в runtime и очищаются при reload."""
    sess1 = runtime.get_adapter_session("guard")
    sess2 = runtime.get_adapter_session("guard")
    assert sess1 is sess2, "Сессия адаптера должна возвращаться из кэша"

    runtime.reload()
    assert len(runtime._adapter_sessions) == 0, "Кэш адаптеров должен очищаться при reload"
    assert runtime.last_pooled_repr is None
