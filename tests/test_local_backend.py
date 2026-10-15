"""Which server a request goes to, and whether it costs this PC anything."""

import subprocess
import sys

import pytest

from src.agent import local_backend
from src.agent.local_backend import CLOUD, ServerTuner, classify, thread_budget


@pytest.fixture(autouse=True)
def _no_ollama_host(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


def test_default_ollama_is_local_and_has_a_lane():
    b = classify("ollama", "qwen3:8b")
    assert b.local and b.family == "ollama" and b.port == 11434
    assert b.lane_id == "localhost-11434"


def test_ollama_cloud_model_costs_nothing_here():
    assert classify("ollama", "glm-5:cloud") is CLOUD
    assert classify("ollama", "kimi-k2-cloud") is CLOUD


def test_ollama_on_another_computer_is_not_this_pc(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.20:11434")
    assert classify("ollama", "qwen3:8b") is CLOUD


def test_ollama_host_without_scheme_and_on_another_port(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11500")
    b = classify("ollama", "m")
    assert b.local and b.port == 11500


def test_loopback_spellings_share_one_lane():
    a = classify("x", "m", base_url="http://localhost:5001/v1")
    b = classify("x", "m", base_url="http://127.0.0.1:5001/v1")
    assert a.local and a.lane_id == b.lane_id


def test_server_families_by_port():
    assert classify("koboldcpp", "m", base_url="http://localhost:5001/v1").family == "koboldcpp"
    assert classify("custom", "m", base_url="http://localhost:1234/v1").family == "lmstudio"
    assert classify("custom", "m", base_url="http://localhost:8080/v1").family == "llama-server"
    assert classify("custom", "m", base_url="http://localhost:9999/v1").family == "local"


def test_a_cloud_endpoint_is_cloud():
    assert classify("openrouter", "m", base_url="https://openrouter.ai/api/v1") is CLOUD
    assert classify("deepseek", "m", base_url="") is CLOUD


def test_a_gateway_on_localhost_is_local_whatever_it_is_called():
    assert classify("openrouter", "m", base_url="http://localhost:4000/v1").local


def test_garbage_url_is_treated_as_cloud():
    assert classify("x", "m", base_url="http://[::bad") is CLOUD


def test_thread_budget_leaves_cores_for_the_person():
    cores = local_backend.physical_cores()
    assert 1 <= thread_budget() <= max(1, cores)
    assert thread_budget(reserve=0) == max(1, cores)
    assert thread_budget(reserve=999) == 1


def test_tuner_lowers_and_restores_a_process(monkeypatch):
    psutil = pytest.importorskip("psutil")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        monkeypatch.setattr(local_backend, "listener_pids", lambda port: [child.pid])
        before = psutil.Process(child.pid).nice()
        tuner = ServerTuner()
        changed = tuner.lower(local_backend.Backend(True, "ollama", "localhost", 11434))
        assert changed == [child.pid]
        assert psutil.Process(child.pid).nice() != before
        assert tuner.lower(local_backend.Backend(True, "ollama", "localhost", 11434)) == [], "no double lowering"
        assert tuner.restore() == 1
        assert psutil.Process(child.pid).nice() == before
    finally:
        child.kill()
        child.wait(5)


def test_tuner_ignores_cloud():
    assert ServerTuner().lower(CLOUD) == []
