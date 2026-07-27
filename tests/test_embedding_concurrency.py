"""Ollama embeddings use native batching, with a bounded-concurrency fallback."""

import pytest

import config


class _Resp:
    def __init__(self, payload, boom=False):
        self._payload = payload
        self._boom = boom

    def raise_for_status(self):
        if self._boom:
            raise RuntimeError("HTTP 400")

    def json(self):
        return self._payload


@pytest.fixture
def isolated_config(monkeypatch):
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    return store


class TestConfig:
    def test_batch_default(self, isolated_config):
        assert config.get_embedding_batch_size() == 32

    def test_batch_override_and_clamp(self, isolated_config):
        isolated_config["embedding_batch_size"] = 8
        assert config.get_embedding_batch_size() == 8
        isolated_config["embedding_batch_size"] = 0
        assert config.get_embedding_batch_size() == 1

    def test_concurrency_default_is_conservative(self, isolated_config):
        assert config.get_embedding_concurrency() == 4

    def test_concurrency_garbage_falls_back(self, isolated_config):
        isolated_config["embedding_concurrency"] = "lots"
        assert config.get_embedding_concurrency() == 4


class TestNativeBatching:
    def test_one_request_per_batch_not_per_text(self, monkeypatch):
        import rag_engine
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(json["input"])
            return _Resp({"embeddings": [[0.5] * 768 for _ in json["input"]]})

        import requests
        monkeypatch.setattr(requests, "post", fake_post)
        monkeypatch.setattr(config, "get_embedding_batch_size", lambda: 32)

        ef = rag_engine.OllamaEmbeddingFunction()
        out = ef([f"text {i}" for i in range(5)])

        assert len(calls) == 1                      # ONE request for 5 texts
        assert calls[0] == [f"text {i}" for i in range(5)]
        assert len(out) == 5 and out[0][0] == 0.5

    def test_splits_into_bounded_batches(self, monkeypatch):
        import rag_engine
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(len(json["input"]))
            return _Resp({"embeddings": [[0.1] * 768 for _ in json["input"]]})

        import requests
        monkeypatch.setattr(requests, "post", fake_post)
        monkeypatch.setattr(config, "get_embedding_batch_size", lambda: 2)

        ef = rag_engine.OllamaEmbeddingFunction()
        out = ef([f"t{i}" for i in range(5)])
        assert calls == [2, 2, 1]
        assert len(out) == 5

    def test_order_is_preserved_across_batches(self, monkeypatch):
        import rag_engine
        counter = {"n": 0}

        def fake_post(url, json=None, timeout=None):
            vecs = []
            for _ in json["input"]:
                counter["n"] += 1
                vecs.append([float(counter["n"])] * 768)
            return _Resp({"embeddings": vecs})

        import requests
        monkeypatch.setattr(requests, "post", fake_post)
        monkeypatch.setattr(config, "get_embedding_batch_size", lambda: 2)

        ef = rag_engine.OllamaEmbeddingFunction()
        out = ef(["a", "b", "c"])
        assert [v[0] for v in out] == [1.0, 2.0, 3.0]

    def test_empty_input(self, monkeypatch):
        import rag_engine
        assert rag_engine.OllamaEmbeddingFunction()([]) == []


class TestFallbackPath:
    def test_batch_failure_degrades_to_per_text(self, monkeypatch):
        import rag_engine
        seen = []

        def fake_post(url, json=None, timeout=None):
            payload = json["input"]
            seen.append(payload)
            if isinstance(payload, list):          # the batch attempt
                return _Resp({}, boom=True)
            return _Resp({"embeddings": [[0.9] * 768]})   # per-text retry

        import requests
        monkeypatch.setattr(requests, "post", fake_post)
        monkeypatch.setattr(config, "get_embedding_batch_size", lambda: 32)
        monkeypatch.setattr(config, "get_embedding_concurrency", lambda: 4)

        ef = rag_engine.OllamaEmbeddingFunction()
        out = ef(["one", "two"])

        assert seen[0] == ["one", "two"]           # batch tried first
        assert seen[1:] == ["one", "two"]          # then individually
        assert all(v[0] == 0.9 for v in out)

    def test_short_response_triggers_fallback(self, monkeypatch):
        import rag_engine

        def fake_post(url, json=None, timeout=None):
            payload = json["input"]
            if isinstance(payload, list):
                return _Resp({"embeddings": [[0.2] * 768]})   # 1 vector for 2 texts
            return _Resp({"embeddings": [[0.7] * 768]})

        import requests
        monkeypatch.setattr(requests, "post", fake_post)
        monkeypatch.setattr(config, "get_embedding_batch_size", lambda: 32)

        ef = rag_engine.OllamaEmbeddingFunction()
        out = ef(["a", "b"])
        assert all(v[0] == 0.7 for v in out)       # recovered via per-text path

    def test_fallback_concurrency_is_bounded(self, monkeypatch):
        import concurrent.futures as cf
        import rag_engine
        seen = {}

        class SpyPool(cf.ThreadPoolExecutor):
            def __init__(self, max_workers=None, **kw):
                seen["workers"] = max_workers
                super().__init__(max_workers=max_workers, **kw)

        monkeypatch.setattr(cf, "ThreadPoolExecutor", SpyPool)
        monkeypatch.setattr(config, "get_embedding_concurrency", lambda: 4)

        import requests
        monkeypatch.setattr(requests, "post",
                            lambda url, json=None, timeout=None: _Resp({"embeddings": [[0.1] * 768]}))

        ef = rag_engine.OllamaEmbeddingFunction()
        ef._embed_individually(["a", "b"], "http://localhost:11434")
        assert seen["workers"] == 2                # never more workers than texts
