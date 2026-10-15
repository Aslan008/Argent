"""/lanes, /doctor and the places that put themselves behind the governor."""

from unittest.mock import MagicMock

import pytest

import config
from src.agent import governor, lanes, pressure
from src.agent.lanes import P_WORKER
from src.agent.local_backend import Backend

LOCAL = Backend(True, "ollama", "localhost", 11434)


@pytest.fixture
def stored(monkeypatch, tmp_path):
    values = {}
    monkeypatch.setattr(config, "_get", lambda key, default=None: values.get(key, default))
    monkeypatch.setattr(config, "_set", lambda key, val: values.__setitem__(key, val))
    monkeypatch.setenv("ARGENT_LANES_DIR", str(tmp_path / "lanes"))
    lanes.forget_lanes()
    yield values
    lanes.forget_lanes()


def run(text):
    from src.cli.commands import CommandContext, dispatch_command
    return dispatch_command(text, CommandContext(agent=MagicMock(), user_input=text))


def test_defaults(stored):
    assert config.is_local_governor_enabled() is True
    assert config.get_local_parallelism() == 1
    assert config.is_lower_server_priority_enabled() is False
    assert config.is_limit_server_threads_enabled() is False


def test_parallelism_is_clamped(stored):
    config.set_local_parallelism(0)
    assert config.get_local_parallelism() == 1
    config.set_local_parallelism(99)
    assert config.get_local_parallelism() == 8
    stored["local_parallelism"] = "junk"
    assert config.get_local_parallelism() == 1


def test_command_switches(stored):
    assert run("/lanes off").handled
    assert stored["local_governor"] is False
    run("/lanes on")
    assert stored["local_governor"] is True
    run("/lanes parallel 3")
    assert stored["local_parallelism"] == 3
    run("/lanes nice on")
    assert stored["lower_model_server_priority"] is True
    run("/lanes threads on")
    assert stored["limit_server_threads"] is True


def test_nice_off_puts_the_server_priority_back(stored, monkeypatch):
    from src.agent import local_backend
    restored = []
    monkeypatch.setattr(local_backend.TUNER, "restore", lambda: restored.append(1) or 0)
    run("/lanes nice off")
    assert restored == [1]


def test_status_shows_who_holds_and_who_waits(stored, monkeypatch):
    calm = type("M", (), {"sample": lambda s: False, "last": None, "overloaded": False, "reason": ""})()
    monkeypatch.setattr(pressure, "monitor", lambda: calm)
    lane = governor.lane_for(LOCAL)
    holder = lane.acquire(0, "основная модель")
    import threading
    waiter = threading.Thread(target=lambda: lane.acquire(P_WORKER, "worker-1", timeout=2), daemon=True)
    waiter.start()
    import time
    time.sleep(0.3)
    text = "\n".join(governor.status_lines())
    holder.release()
    waiter.join(5)
    assert "основная модель" in text and "работает" in text
    assert "worker-1" in text and "ждёт" in text


def test_status_with_empty_lanes(stored, monkeypatch):
    calm = type("M", (), {"sample": lambda s: False, "last": None, "overloaded": False, "reason": ""})()
    monkeypatch.setattr(pressure, "monitor", lambda: calm)
    assert any("пусты" in line for line in governor.status_lines())


def test_a_bad_argument_does_not_change_anything(stored):
    run("/lanes parallel lots")
    assert "local_parallelism" not in stored


def test_doctor_reports_a_cloud_model_as_no_load(monkeypatch):
    import doctor
    monkeypatch.setattr(doctor, "get_provider", lambda: "deepseek")
    monkeypatch.setattr(doctor, "get_current_model", lambda: "deepseek-chat")
    status, detail = doctor._check_local_server()
    assert status == doctor.OK and "no local load" in detail


def test_doctor_warns_when_the_pc_is_busy(monkeypatch):
    import doctor
    monkeypatch.setattr(doctor, "get_provider", lambda: "ollama")
    monkeypatch.setattr(doctor, "get_current_model", lambda: "qwen3:8b")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    busy = type("M", (), {"sample": lambda s: True, "reason": "процессор занят"})()
    monkeypatch.setattr(pressure, "monitor", lambda: busy)
    status, detail = doctor._check_local_server()
    assert status == doctor.WARN and "процессор занят" in detail


# ── the other places that load a local server ────────────────────────────────

class TestEmbeddingsQueue:
    def _run(self, monkeypatch, texts):
        import rag_engine
        import requests
        from contextlib import contextmanager
        seen = []

        @contextmanager
        def fake_slot(backend, label, prio=None):
            seen.append((backend.family, label, prio))
            yield

        monkeypatch.setattr(governor, "slot", fake_slot)

        class Resp:
            def __init__(self, n):
                self.n = n

            def raise_for_status(self):
                pass

            def json(self):
                return {"embeddings": [[0.1, 0.2]] * self.n}

        monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: Resp(len(json["input"]) if isinstance(json["input"], list) else 1))
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        rag_engine.OllamaEmbeddingFunction("nomic-embed-text")(texts)
        return seen

    def test_indexing_is_background_work(self, monkeypatch):
        seen = self._run(monkeypatch, ["a", "b", "c"])
        assert seen and seen[0][0] == "ollama"
        assert seen[0][2] == governor.P_BACKGROUND

    def test_one_search_query_is_the_persons_own_request(self, monkeypatch):
        seen = self._run(monkeypatch, ["how does login work"])
        assert seen and seen[0][2] is None, "None = the priority of whoever asks (the interactive turn)"


class TestDelegateLocalQueue:
    def test_the_gguf_is_loaded_once_and_used_through_the_lane(self, monkeypatch, tmp_path):
        import sys
        import types
        import local_inference
        from contextlib import contextmanager

        loads = []

        class Llama:
            def __init__(self, **kw):
                loads.append(kw)

            def create_completion(self, prompt, **kw):
                return {"choices": [{"text": " answer "}]}

        fake = types.ModuleType("llama_cpp")
        fake.Llama = Llama
        monkeypatch.setitem(sys.modules, "llama_cpp", fake)
        monkeypatch.setattr(local_inference, "_gguf_cached", None)
        monkeypatch.setattr(local_inference.SystemResourceAuditor, "get_inference_device", staticmethod(lambda: "cpu"))
        used = []

        @contextmanager
        def fake_slot(backend, label, prio=None):
            used.append(backend.family)
            yield

        monkeypatch.setattr(governor, "slot", fake_slot)
        model = str(tmp_path / "m.gguf")
        assert local_inference._run_local_gguf("t", "c", model_path=model) == "answer"
        assert local_inference._run_local_gguf("t2", "c", model_path=model) == "answer"
        assert len(loads) == 1, "the weights are read once, not once per call"
        assert used == ["gguf", "gguf"]
