"""Manual regression check for the OpenRouter tool-call serialization bug.

Reproduces the exact failing scenario: a message history where a previous
assistant turn carries a tool_call whose arguments are a dict (Argent's
internal form). Strict OpenAI-compatible backends (vLLM behind OpenRouter)
used to reject this with HTTP 400 — "arguments: Input should be a valid
string". With _normalize_outgoing_messages the dict is stringified on the
way out and the request succeeds.

Requires the active provider to be a configured OpenRouter. Usage:
    python scripts/check_openrouter_toolcall.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from providers import create_provider


def main():
    if config.get_provider() != "openrouter":
        print(f"Active provider is '{config.get_provider()}', not openrouter. "
              f"Switch with /provider first. Skipping.")
        return

    provider = create_provider()
    err = provider.validate_config()
    if err:
        print(f"Provider not ready: {err}")
        return

    model = config.get_current_model()
    # History with a dict-valued arguments (the shape that triggered HTTP 400).
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Найди погоду в Караганде."},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_demo_1", "type": "function",
            "function": {"name": "search_web", "arguments": {"query": "погода Караганда"}},
        }]},
        {"role": "tool", "tool_call_id": "call_demo_1",
         "content": "Karaganda: -3°C, clear."},
        {"role": "user", "content": "Спасибо, кратко скажи температуру."},
    ]

    print(f"Model: {model}")
    print("Sending history with dict-valued tool_call arguments...")
    try:
        got_text = ""
        for chunk in provider.stream_chat(model=model, messages=messages, tools=None, temperature=0.2):
            got_text += chunk.get("content", "")
        print(f"Response: {got_text.strip()[:200]}")
        print("PASS: OpenRouter accepted the history (no HTTP 400 on tool_call arguments).")
    except Exception as e:
        print(f"FAIL: {e}")


if __name__ == "__main__":
    main()
