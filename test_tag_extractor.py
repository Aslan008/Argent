import json
import os
from typing import Dict, Any, Generator

# Mock Ollama for testing tag-based reasoning stream
class MockOllama:
    @staticmethod
    def chat(model, messages, tools=None, options=None, stream=False):
        # Simulated responses with tags
        responses = [
            {"message": {"role": "assistant", "content": "Привет! <thought>Я начинаю думать..."}},
            {"message": {"role": "assistant", "content": "Это сложный вопрос. </thought>Вот мой ответ."}}
        ]
        if stream:
            return iter(responses)
        return responses[0]

# Monkeypatch ollama
import sys
from types import ModuleType
m = ModuleType("ollama")
m.chat = MockOllama.chat
m.ResponseError = Exception
sys.modules["ollama"] = m

# Re-import ArgentAgent
sys.path.append(os.getcwd())
from agent import ArgentAgent

def test_tag_extraction():
    agent = ArgentAgent()
    print("Testing tag extraction...")
    
    full_thinking = ""
    full_answer = ""
    
    for chunk in agent.process_user_input("Напиши отзыв"):
        if chunk["type"] == "thinking_stream":
            full_thinking += chunk["content"]
        elif chunk["type"] == "content_stream":
            full_answer += chunk["content"]
            
    print(f"Extracted thinking: '{full_thinking}'")
    print(f"Final answer: '{full_answer}'")
    
    # Verify thinking was split correctly and tags removed
    # Expected thinking: "Я начинаю думать...Это сложный вопрос. "
    # Expected answer: "Привет! Вот мой ответ."
    
    if "Я начинаю" in full_thinking and "Привет!" in full_answer and "<thought>" not in full_answer:
        print("PASS: Tag-based reasoning extracted correctly.")
    else:
        print("FAIL: Tag-based reasoning extraction failed.")

if __name__ == "__main__":
    test_tag_extraction()
