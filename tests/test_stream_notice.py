"""System recovery notices on the 'error' stream channel must render as muted
system messages, not red 'Error:' — the self-healing narration (auto-continue,
salvage, loop-guard, no-action nudge, context trim) is not a failure.
"""

import src.cli.cli_ui as cli_ui
from src.cli.cli_ui import _render_stream_error


def _capture(monkeypatch):
    calls = {"system": [], "error": []}
    monkeypatch.setattr(cli_ui, "print_system", lambda t: calls["system"].append(t))
    monkeypatch.setattr(cli_ui, "print_error", lambda t: calls["error"].append(t))
    return calls


class TestRenderStreamError:
    def test_system_nudge_is_a_notice(self, monkeypatch):
        calls = _capture(monkeypatch)
        _render_stream_error(
            "\n[System: no answer or action produced — nudging the model to continue...]")
        assert calls["system"] and not calls["error"]

    def test_loop_guard_is_a_notice(self, monkeypatch):
        calls = _capture(monkeypatch)
        _render_stream_error("\n[Loop Guard]: повторяющийся цикл инструментов остановлен")
        assert calls["system"] and not calls["error"]

    def test_argent_recovery_is_a_notice(self, monkeypatch):
        calls = _capture(monkeypatch)
        _render_stream_error(
            "\n[Argent: prompt exceeded the model's context window — trimmed history and retrying...]")
        assert calls["system"] and not calls["error"]

    def test_genuine_error_stays_error(self, monkeypatch):
        calls = _capture(monkeypatch)
        _render_stream_error("Provider error: connection refused")
        assert calls["error"] and not calls["system"]

    def test_argent_tag_mid_string_is_still_an_error(self, monkeypatch):
        # A real failure whose message merely mentions [Argent:] later on must
        # not be downgraded — only a leading tag marks a notice.
        calls = _capture(monkeypatch)
        _render_stream_error(
            "HTTP 400: too many tokens\n[Argent: the prompt still won't fit after shrinking 3x...]")
        assert calls["error"] and not calls["system"]
