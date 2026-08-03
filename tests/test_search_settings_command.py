"""/search — configuring web research from inside Argent.

The three settings are one decision in practice: query languages without a
multilingual reranker retrieve pages that are then scored too low to survive,
and a multilingual reranker with English-only queries has nothing to rerank.
The command therefore reports the mismatch rather than letting you set half.
"""

import pytest

import command_handler
import config
from src.research.rerank import MULTILINGUAL_MODEL, _DEFAULT_MODEL


@pytest.fixture
def cfg(monkeypatch):
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    return store


@pytest.fixture
def ui(monkeypatch):
    """Capture printed lines and script questionary answers."""
    printed = []
    monkeypatch.setattr(command_handler, "print_system", lambda m, *a, **k: printed.append(str(m)))
    monkeypatch.setattr(command_handler, "print_error", lambda m, *a, **k: printed.append(str(m)))
    return printed


def _answer(monkeypatch, select=None, password=None, checkbox=None, text=None):
    class _Ans:
        def __init__(self, value):
            self._value = value
        def ask(self):
            return self._value

    monkeypatch.setattr(command_handler.questionary, "select", lambda *a, **k: _Ans(select))
    monkeypatch.setattr(command_handler.questionary, "password", lambda *a, **k: _Ans(password))
    monkeypatch.setattr(command_handler.questionary, "checkbox", lambda *a, **k: _Ans(checkbox))
    monkeypatch.setattr(command_handler.questionary, "text", lambda *a, **k: _Ans(text))


class TestStatusReport:
    def test_shows_engines_languages_and_reranker(self, cfg, ui, monkeypatch):
        _answer(monkeypatch, select="Отмена")
        command_handler._handle_search_command()
        blob = "\n".join(ui)
        assert "DuckDuckGo" in blob and "GitHub" in blob
        assert "Языки запросов" in blob and "Reranker" in blob

    def test_warns_when_languages_outpace_the_reranker(self, cfg, ui, monkeypatch):
        config.set_search_languages(["en", "ru"])
        config.set_reranker_model("")                 # English-only
        _answer(monkeypatch, select="Отмена")
        command_handler._handle_search_command()
        assert any("отброшены при ранжировании" in line for line in ui)

    def test_no_warning_when_they_agree(self, cfg, ui, monkeypatch):
        config.set_search_languages(["en", "ru"])
        config.set_reranker_model(MULTILINGUAL_MODEL)
        _answer(monkeypatch, select="Отмена")
        command_handler._handle_search_command()
        assert not any("отброшены" in line for line in ui)


class TestBraveKey:
    def test_sets_the_key(self, cfg, ui, monkeypatch):
        _answer(monkeypatch, select="Brave API-ключ (второй независимый индекс поиска)",
                password="BSA-test-key")
        command_handler._handle_search_command()
        assert config.get_brave_api_key() == "BSA-test-key"

    def test_empty_disables(self, cfg, ui, monkeypatch):
        config.set_brave_api_key("old")
        _answer(monkeypatch, select="Brave API-ключ (второй независимый индекс поиска)",
                password="")
        command_handler._handle_search_command()
        assert config.get_brave_api_key() == ""

    def test_cancel_leaves_it_alone(self, cfg, ui, monkeypatch):
        config.set_brave_api_key("keep-me")
        _answer(monkeypatch, select="Brave API-ключ (второй независимый индекс поиска)",
                password=None)
        command_handler._handle_search_command()
        assert config.get_brave_api_key() == "keep-me"

    def test_key_is_asked_for_with_masked_input(self, cfg, ui, monkeypatch):
        """A visible key ends up in scrollback and screen shares, so the prompt
        must be questionary.password, never plain text."""
        used = []
        _answer(monkeypatch, select="Brave API-ключ (второй независимый индекс поиска)",
                password="k")

        class _Spy:
            def __init__(self, *a, **k):
                used.append("password")
            def ask(self):
                return "k"

        monkeypatch.setattr(command_handler.questionary, "password", _Spy)
        monkeypatch.setattr(command_handler.questionary, "text",
                            lambda *a, **k: pytest.fail("the key must not be typed in the clear"))
        command_handler._handle_search_command()
        assert used == ["password"]

    def test_key_is_never_printed(self, cfg, ui, monkeypatch):
        _answer(monkeypatch, select="Brave API-ключ (второй независимый индекс поиска)",
                password="SUPERSECRET")
        command_handler._handle_search_command()
        assert not any("SUPERSECRET" in line for line in ui)


class TestLanguages:
    def test_sets_the_selection(self, cfg, ui, monkeypatch):
        _answer(monkeypatch, select="Языки поисковых запросов", checkbox=["en", "ru"])
        command_handler._handle_search_command()
        assert config.get_search_languages() == ["en", "ru"]

    def test_empty_selection_is_refused(self, cfg, ui, monkeypatch):
        config.set_search_languages(["en", "ru"])
        _answer(monkeypatch, select="Языки поисковых запросов", checkbox=[])
        command_handler._handle_search_command()
        assert config.get_search_languages() == ["en", "ru"]     # unchanged

    def test_adding_a_language_points_at_the_reranker(self, cfg, ui, monkeypatch):
        config.set_reranker_model("")
        _answer(monkeypatch, select="Языки поисковых запросов", checkbox=["en", "ru"])
        command_handler._handle_search_command()
        assert any("мультиязычный" in line for line in ui)


class TestRerankerModel:
    def test_switch_to_multilingual(self, cfg, ui, monkeypatch):
        _answer(monkeypatch, select="Модель reranker'а")
        # The second select (inside the branch) returns the same scripted value,
        # so drive it explicitly.
        answers = iter(["Модель reranker'а",
                        f"Мультиязычный, ~490 МБ — {MULTILINGUAL_MODEL.split('/')[-1]}"])

        class _Seq:
            def ask(self):
                return next(answers)

        monkeypatch.setattr(command_handler.questionary, "select", lambda *a, **k: _Seq())
        command_handler._handle_search_command()
        assert config.get_reranker_model() == MULTILINGUAL_MODEL

    def test_switch_back_to_english_clears_the_setting(self, cfg, ui, monkeypatch):
        config.set_reranker_model(MULTILINGUAL_MODEL)
        answers = iter(["Модель reranker'а",
                        f"Английский, 92 МБ — {_DEFAULT_MODEL.split('/')[-1]}"])

        class _Seq:
            def ask(self):
                return next(answers)

        monkeypatch.setattr(command_handler.questionary, "select", lambda *a, **k: _Seq())
        command_handler._handle_search_command()
        assert config.get_reranker_model() == ""

    def test_custom_model_name(self, cfg, ui, monkeypatch):
        answers = iter(["Модель reranker'а", "Другая модель (ввести имя с HuggingFace)"])

        class _Seq:
            def ask(self):
                return next(answers)

        monkeypatch.setattr(command_handler.questionary, "select", lambda *a, **k: _Seq())
        monkeypatch.setattr(command_handler.questionary, "text",
                            lambda *a, **k: type("A", (), {"ask": lambda s: " BAAI/bge-reranker-base "})())
        command_handler._handle_search_command()
        assert config.get_reranker_model() == "BAAI/bge-reranker-base"
