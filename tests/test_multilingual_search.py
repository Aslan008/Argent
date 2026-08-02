"""Multilingual research: reranker selection and per-topic query language.

The English-only reranker does not merely degrade on other languages — measured
on cross-encoder/ms-marco-MiniLM-L-6-v2:

    RU query vs RU passages   borscht recipe 7.06  >  relevant .NET doc 6.41
    mixed pool                relevant RU -0.33    vs  same text in EN +6.99

So it inverts the ranking outright, and in a mixed pool buries relevant
non-English text under mediocre English text. The integration test at the
bottom reproduces that against the real model; the rest run offline.
"""

import pytest

import config
from src.research import rerank as rerank_mod
from src.research.rerank import _DEFAULT_MODEL, MULTILINGUAL_MODEL, active_model_name, rerank


@pytest.fixture
def cfg(monkeypatch):
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    return store


class TestRerankerSelection:
    def test_defaults_to_the_small_english_model(self, cfg):
        assert active_model_name() == _DEFAULT_MODEL

    def test_configured_model_wins(self, cfg):
        config.set_reranker_model(MULTILINGUAL_MODEL)
        assert active_model_name() == MULTILINGUAL_MODEL

    def test_blank_setting_falls_back(self, cfg):
        config.set_reranker_model("   ")
        assert active_model_name() == _DEFAULT_MODEL

    def test_config_failure_is_survivable(self, monkeypatch):
        monkeypatch.setattr(config, "get_reranker_model",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert active_model_name() == _DEFAULT_MODEL

    def test_switching_the_setting_reloads_the_model(self, cfg, monkeypatch):
        """A cached encoder for the previous model must not be reused, or
        turning multilingual on would silently do nothing for the session."""
        built = []

        class FakeCE:
            def __init__(self, name):
                built.append(name)
            def predict(self, pairs):
                return [0.0] * len(pairs)

        import sentence_transformers
        monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCE)
        monkeypatch.setattr(rerank_mod, "_cross_encoder", None)
        monkeypatch.setattr(rerank_mod, "_cross_encoder_name", None)

        rerank_mod._get_cross_encoder()
        rerank_mod._get_cross_encoder()                 # cached — no reload
        assert built == [_DEFAULT_MODEL]

        config.set_reranker_model(MULTILINGUAL_MODEL)
        rerank_mod._get_cross_encoder()
        assert built == [_DEFAULT_MODEL, MULTILINGUAL_MODEL]


class TestRerankStillWorks:
    def test_orders_by_injected_scores(self):
        chunks = ["a", "b", "c"]
        out = rerank("q", chunks, top_n=2, _scorer=lambda pairs: [0.1, 0.9, 0.5])
        assert out == ["b", "c"]

    def test_empty_input(self):
        assert rerank("q", [], _scorer=lambda pairs: []) == []

    def test_scorer_failure_falls_back_to_input_order(self, monkeypatch):
        def boom(pairs):
            raise RuntimeError("model gone")
        monkeypatch.setattr(rerank_mod, "_bi_encoder_rank",
                            lambda q, c, n: (_ for _ in ()).throw(RuntimeError("also gone")))
        assert rerank("q", ["x", "y"], top_n=1, _scorer=boom) == ["x"]


class TestQueryLanguageRule:
    def _rule(self, monkeypatch, languages):
        import deep_research
        monkeypatch.setattr(config, "get_search_languages", lambda: languages)
        return deep_research._language_rule()

    def test_english_only_is_the_default_and_is_strict(self, monkeypatch):
        rule = self._rule(monkeypatch, ["en"])
        assert "ENGLISH even if the topic is stated in another" in rule

    def test_multiple_languages_switch_to_per_query_choice(self, monkeypatch):
        rule = self._rule(monkeypatch, ["en", "ru"])
        assert "PER QUERY" in rule and "Russian" in rule
        assert "Default to ENGLISH" in rule       # still the sane default
        assert "regional services" in rule        # and when to deviate

    def test_unknown_language_code_is_passed_through(self, monkeypatch):
        rule = self._rule(monkeypatch, ["en", "xx"])
        assert "xx" in rule

    def test_rule_reaches_the_prompt(self, monkeypatch):
        import deep_research
        monkeypatch.setattr(config, "get_search_languages", lambda: ["en", "ru"])
        captured = {}
        monkeypatch.setattr(deep_research, "_call_llm_sync",
                            lambda prompt, **kw: captured.setdefault("p", prompt) or '["a"]')
        deep_research._generate_queries("тема")
        assert "PER QUERY" in captured["p"]
        assert "Russian" in captured["p"]


class TestSearchLanguagesConfig:
    def test_default(self, cfg):
        assert config.get_search_languages() == ["en"]

    def test_set_list_and_csv(self, cfg):
        config.set_search_languages(["en", "ru"])
        assert config.get_search_languages() == ["en", "ru"]
        config.set_search_languages("en, de")
        assert config.get_search_languages() == ["en", "de"]

    def test_empty_falls_back_to_english(self, cfg):
        config.set_search_languages([])
        assert config.get_search_languages() == ["en"]


class TestDoctorFlagsTheMismatch:
    def test_warns_when_languages_outpace_the_reranker(self, monkeypatch):
        import doctor
        monkeypatch.setattr(config, "get_search_languages", lambda: ["en", "ru"])
        monkeypatch.setattr(config, "get_reranker_model", lambda: "")
        status, detail = doctor._check_web_search()
        assert status == doctor.WARN
        assert "English-only" in detail and "reranker_model" in detail

    def test_no_warning_when_they_match(self, monkeypatch):
        import doctor
        monkeypatch.setattr(config, "get_search_languages", lambda: ["en", "ru"])
        monkeypatch.setattr(config, "get_reranker_model", lambda: MULTILINGUAL_MODEL)
        status, _ = doctor._check_web_search()
        assert status == doctor.OK


@pytest.mark.integration
class TestRealModelLanguageRegression:
    """Reproduces the measurement that motivated all of the above. Needs the
    real weights, so it is excluded from the default run."""

    RU_QUERY = "как настроить сборку мусора в .NET"
    RELEVANT = "Настройка garbage collector в .NET: режимы Workstation и Server GC."
    IRRELEVANT = "Рецепт борща: свёкла, капуста, картофель, томатная паста."

    def _scores(self, model_name):
        from sentence_transformers import CrossEncoder
        m = CrossEncoder(model_name, max_length=512)
        return m.predict([(self.RU_QUERY, self.RELEVANT),
                          (self.RU_QUERY, self.IRRELEVANT)])

    def test_english_model_inverts_russian_ranking(self):
        relevant, irrelevant = self._scores(_DEFAULT_MODEL)
        assert irrelevant > relevant, "the documented failure no longer reproduces"

    def test_multilingual_model_ranks_russian_correctly(self):
        relevant, irrelevant = self._scores(MULTILINGUAL_MODEL)
        assert relevant > irrelevant
