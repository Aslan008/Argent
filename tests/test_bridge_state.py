"""The TUI's hover data: where the context goes, and what the provider ignores."""
from src.bridge.repl_bridge import ReplBridge


class _Agent:
    model_name = "qwen3:8b"
    messages = []

    def get_context_breakdown(self):
        return {
            "system": 3100,
            "tools": 2400,
            "tool_count": 52,
            "history": 6800,
            "total": 12300,
            "max": 32768,
            "percent": 37.5,
        }


class _Session:
    agent = _Agent()

    def state(self):
        return {"provider": "koboldcpp", "model": "some-model", "context": {}}


def _bridge(session=None):
    bridge = object.__new__(ReplBridge)
    bridge.session = session or _Session()
    bridge.turn_counter = 0
    bridge.vibe_mode = False
    bridge.is_project_mode = False
    return bridge


def test_the_state_carries_a_context_breakdown_for_the_hover():
    state = ReplBridge.get_state(_bridge())

    text = state["ctx_breakdown"]
    assert "Системный промпт" in text and "3.1k tok" in text
    assert "Инструменты (52)" in text and "2.4k tok" in text
    assert "История разговора" in text and "6.8k tok" in text
    assert "Всего" in text and "12.3k / 32.8k" in text and "38%" in text


def test_the_state_carries_the_capability_note():
    state = ReplBridge.get_state(_bridge())

    assert state["inert_label"] == "n/a: thinking, depth"


def test_a_broken_breakdown_does_not_break_the_toolbar():
    class BadAgent(_Agent):
        def get_context_breakdown(self):
            raise RuntimeError("boom")

    class BadSession(_Session):
        agent = BadAgent()

    state = ReplBridge.get_state(_bridge(BadSession()))
    assert state["ctx_breakdown"] == ""
