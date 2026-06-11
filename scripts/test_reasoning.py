import json
import os
from typing import Dict, Any, Generator

# Mock Ollama for testing reasoning stream
class MockOllama:
    @staticmethod
    def chat(model, messages, tools=None, options=None, stream=False):
        # Simulated responses with thinking
        responses = [
            {"message": {"role": "assistant", "thinking": "Я думаю о том, как ответить на этот вопрос..."}},
            {"message": {"role": "assistant", "thinking": " Возможно, стоит упомянуть рекурсию."}},
            {"message": {"role": "assistant", "content": "Рекурсия — это процесс..."}},
            {"message": {"role": "assistant", "content": " когда функция вызывает саму себя."}}
        ]
        if stream:
            return iter(responses)
        return responses[0]

    @staticmethod
    def list():
        return {"models": []}

# Monkeypatch ollama
import sys
from types import ModuleType
m = ModuleType("ollama")
m.chat = MockOllama.chat
m.list = MockOllama.list
m.ResponseError = Exception
sys.modules["ollama"] = m

# Force provider to ollama for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
config.get_provider = lambda: "ollama"

from agent import ArgentAgent

def test_reasoning_capture():
    agent = ArgentAgent()
    print("Testing reasoning capture...")
    
    full_thinking = ""
    full_content = ""
    
    for chunk in agent.process_user_input("Что такое рекурсия?"):
        if chunk["type"] == "thinking_stream":
            full_thinking += chunk["content"]
        elif chunk["type"] == "content_stream":
            full_content += chunk["content"]
            
    print(f"Captured thinking: '{full_thinking}'")
    print(f"Captured content: '{full_content}'")
    
    # Verify thinking was saved to history
    last_msg = agent.messages[-1]
    print(f"Memory thinking: '{last_msg.get('thinking', 'MISSING')}'")
    
    if "Я думаю" in full_thinking and "Рекурсия" in full_content and last_msg.get('thinking'):
        print("PASS: Reasoning captured and stored correctly.")
    else:
        print("FAIL: Reasoning capture failed.")

if __name__ == "__main__":
    test_reasoning_capture()
