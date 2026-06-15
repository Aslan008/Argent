import pytest


@pytest.fixture(autouse=True)
def _offline_token_estimation(monkeypatch):
    """Keep the test suite offline and fast.

    estimate_tokens() queries the provider's tokenize endpoint (Ollama /
    KoboldCPP) for accuracy. In tests that hit a live endpoint for every unique
    message, which dragged the suite out to ~20 minutes. Stubbing the network
    call makes estimate_tokens fall back to tiktoken / the heuristic instantly,
    without changing the logic under test.
    """
    try:
        import src.agent.trimmer as trimmer
        monkeypatch.setattr(trimmer, "_query_tokenize_api", lambda *a, **k: None)
        # Reset the circuit breaker so state from a previous run never leaks.
        trimmer._API_TOKENIZE_BROKEN.clear()
        trimmer._TOKEN_CACHE.clear()
    except Exception:
        pass
