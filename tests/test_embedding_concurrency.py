"""Ollama embedding concurrency is bounded and configurable (weak-box safety)."""

import pytest

import config


class TestConfig:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        store = {}
        monkeypatch.setattr(config, "load_config", lambda: store)
        monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
        yield store

    def test_default_is_conservative(self):
        assert config.get_embedding_concurrency() == 4

    def test_reads_override(self, _isolate):
        _isolate["embedding_concurrency"] = 8
        assert config.get_embedding_concurrency() == 8

    def test_clamped_to_at_least_one(self, _isolate):
        _isolate["embedding_concurrency"] = 0
        assert config.get_embedding_concurrency() == 1

    def test_garbage_falls_back_to_default(self, _isolate):
        _isolate["embedding_concurrency"] = "lots"
        assert config.get_embedding_concurrency() == 4


class TestEmbeddingFunctionRespectsLimit:
    def test_worker_count_never_exceeds_config_or_input(self, monkeypatch):
        import rag_engine

        seen = {}
        # The embedding fn does `from concurrent.futures import ThreadPoolExecutor`
        # at call time, so patching the module attribute is enough.
        import concurrent.futures as cf

        class SpyPool(cf.ThreadPoolExecutor):
            def __init__(self, max_workers=None, **kw):
                seen["workers"] = max_workers
                super().__init__(max_workers=max_workers, **kw)

        monkeypatch.setattr(cf, "ThreadPoolExecutor", SpyPool)
        monkeypatch.setattr(config, "get_embedding_concurrency", lambda: 4)

        # Stub the network so no real Ollama call happens.
        import requests
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: type("R", (), {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"embeddings": [[0.1] * 768]},
            })(),
        )

        ef = rag_engine.OllamaEmbeddingFunction()
        ef(["one", "two"])                      # 2 texts, limit 4 -> 2 workers
        assert seen["workers"] == 2

        ef(["a", "b", "c", "d", "e", "f"])      # 6 texts, limit 4 -> 4 workers
        assert seen["workers"] == 4
