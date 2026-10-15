"""A model failing during context compression must never cost the session its history.

Whatever the compression model does (refuses, hangs, answers nonsense, dies half way) and
whatever breaks around it, `soft_trim_with_summarization` still returns a shorter history
that carries the facts a model-free summary can give, and never touches the original list.
"""

import copy

import providers
import src.agent.service_llm as sl
import src.agent.trimmer as trimmer
from providers import ProviderError
from tests.test_service_llm import FakeProvider, GOOD, provider_error


def _history(turns=25):
    msgs = [{"role": "system", "content": "System prompt"}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"Исправь ошибку номер {i} в файле module_{i}.py"})
        msgs.append({"role": "assistant", "content": f"Исправил ошибку номер {i}, тесты проходят."})
    return msgs


def _compress(msgs):
    return trimmer.soft_trim_with_summarization(msgs, "mock", 20, 30000)


def _service(monkeypatch, provider, primary="a/primary:free"):
    sl.clear_cooldowns()
    monkeypatch.setattr(sl, "_pause", lambda seconds, started: False)
    monkeypatch.setattr(providers, "create_service_provider", lambda: (provider, primary))


def _distillate(msgs):
    return next(m["content"] for m in msgs if "CONTEXT DISTILLATE" in m.get("content", ""))


def test_a_model_that_is_out_of_quota_hands_over_to_the_next_free_one(monkeypatch):
    p = FakeProvider({"a/primary:free": [provider_error(429, "Rate limit exceeded: free-models-per-day")],
                      "b/second:free": [GOOD]})
    _service(monkeypatch, p)
    monkeypatch.setattr(sl, "live_free_models", lambda provider: ["b/second:free"])
    out = _compress(_history())
    assert p.calls == ["a/primary:free", "b/second:free"]
    assert GOOD.strip()[:40] in _distillate(out)


def test_when_no_model_answers_the_summary_is_built_without_one(monkeypatch):
    dead = provider_error(429, "free-models-per-day")
    p = FakeProvider({"a/primary:free": [dead]})
    _service(monkeypatch, p)
    monkeypatch.setattr(sl, "live_free_models", lambda provider: [])
    msgs = _history()
    before = copy.deepcopy(msgs)
    out = _compress(msgs)
    assert msgs == before, "the history the caller holds is untouched"
    assert len(out) < len(msgs)
    d = _distillate(out)
    assert "Запросы" in d or "Ход работы" in d or "Промежуточная" in d, "the model-free facts are there"
    assert out[-1]["content"] == msgs[-1]["content"], "the latest turn survived"


def test_a_model_that_dies_mid_compression_costs_one_attempt_not_the_session(monkeypatch):
    class Dies(FakeProvider):
        def sync_chat(self, model, messages, timeout=None, **kw):
            self.calls.append(model)
            raise ConnectionResetError("connection reset mid-answer")

    p = Dies({})
    _service(monkeypatch, p)
    monkeypatch.setattr(sl, "live_free_models", lambda provider: [])
    out = _compress(_history())
    assert len(out) < 51 and any("CONTEXT DISTILLATE" in m.get("content", "") for m in out)


def test_an_exception_inside_the_distillation_still_compresses(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("distillation exploded")

    monkeypatch.setattr(trimmer, "_create_distillate", boom)
    out = _compress(_history())
    assert len(out) < 51 and any("CONTEXT DISTILLATE" in m.get("content", "") for m in out)


def test_failing_bookkeeping_cannot_undo_a_finished_compression(monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(trimmer, "_write_compact_boundary", boom)
    monkeypatch.setattr(trimmer, "_update_memory_from_messages", boom)
    monkeypatch.setattr(trimmer, "_summarize_with_llm", lambda *a, **k: "## Done\n- ok " + "x" * 80)
    out = _compress(_history())
    assert len(out) < 51 and any("CONTEXT DISTILLATE" in m.get("content", "") for m in out)


def test_the_summariser_hands_its_ui_to_the_runner(monkeypatch):
    seen = {}

    def fake_run(prompt, *, per_call_timeout, ui):
        seen.update(timeout=per_call_timeout, ui=ui)
        return sl.Result("the summary")

    monkeypatch.setattr(sl, "run", fake_run)
    marker = object()
    assert trimmer._summarize_with_llm("log", None, "", "m", timeout=77.0, ui=marker) == "the summary"
    assert seen == {"timeout": 77.0, "ui": marker}
    monkeypatch.setattr(sl, "run", lambda *a, **k: sl.Result(None, reason="none answered"))
    assert trimmer._summarize_with_llm("log", None, "", "m") is None


def test_the_compress_settings_default_sensibly_and_reject_nonsense(tmp_path, monkeypatch):
    import config
    assert config.get_compress_on_failure() == "auto"
    assert config.get_compress_retries() == 3
    assert config.get_compress_fallback_models() == []
    config.set_compress_on_failure("ask")
    config.set_compress_retries(99)
    config.set_compress_fallback_models(["x/a:free", " ", "y/b:free"])
    assert config.get_compress_on_failure() == "ask"
    assert config.get_compress_retries() == 6, "capped"
    assert config.get_compress_fallback_models() == ["x/a:free", "y/b:free"]
    config.set_compress_on_failure("whatever")
    assert config.get_compress_on_failure() == "auto"
