from abc import ABC, abstractmethod
from typing import List, Dict, Any
from config import get_model_size_category
from src.agent.parser import parse_raw_tool_call
from src.agent.trimmer import hard_reset_with_memory, soft_trim_with_summarization, sliding_window_trim

class ModelStrategy(ABC):
    """Abstract base class representing strategy optimizations for a specific LLM tier."""

    @abstractmethod
    def get_max_history_messages(self, category: str) -> int:
        pass

    @abstractmethod
    def trim_history(self, messages: List[Dict[str, Any]], model_name: str, 
                     max_history_messages: int, max_context_tokens: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def parse_response(self, content: str) -> Dict[str, Any] | None:
        pass

    @abstractmethod
    def supports_native_tools(self) -> bool:
        pass


class TinyLocalStrategy(ModelStrategy):
    """Optimization strategy for extremely small local models (< 3B)."""

    def get_max_history_messages(self, category: str) -> int:
        return 6

    def trim_history(self, messages: List[Dict[str, Any]], model_name: str, 
                     max_history_messages: int, max_context_tokens: int) -> List[Dict[str, Any]]:
        # For extremely small models, keep sliding window or fallback
        return sliding_window_trim(messages, model_name, max_history_messages, max_context_tokens)

    def parse_response(self, content: str) -> Dict[str, Any] | None:
        # Fallback to parsing raw tool call JSON
        return parse_raw_tool_call(content)

    def supports_native_tools(self) -> bool:
        return False


class StandardLocalStrategy(ModelStrategy):
    """Optimization strategy for standard local models (3B - 14B, e.g. Qwen2.5-Coder)."""

    def get_max_history_messages(self, category: str) -> int:
        if category == "small":
            return 10
        return 20

    def trim_history(self, messages: List[Dict[str, Any]], model_name: str, 
                     max_history_messages: int, max_context_tokens: int) -> List[Dict[str, Any]]:
        return sliding_window_trim(messages, model_name, max_history_messages, max_context_tokens)

    def parse_response(self, content: str) -> Dict[str, Any] | None:
        return parse_raw_tool_call(content)

    def supports_native_tools(self) -> bool:
        return True


class CloudStrategy(ModelStrategy):
    """Strategy for large cloud models (Gemini, Z.AI, GPT) with large context windows."""

    def get_max_history_messages(self, category: str) -> int:
        if category == "medium":
            return 20
        elif category == "large":
            return 30
        return 40

    def trim_history(self, messages: List[Dict[str, Any]], model_name: str, 
                     max_history_messages: int, max_context_tokens: int) -> List[Dict[str, Any]]:
        # Cloud models handle parallel summarization API calls cleanly. Soft trim keeps detail.
        return soft_trim_with_summarization(messages, model_name, max_history_messages, max_context_tokens)

    def parse_response(self, content: str) -> Dict[str, Any] | None:
        return parse_raw_tool_call(content)

    def supports_native_tools(self) -> bool:
        return True


def get_model_strategy(model_name: str, provider: str) -> ModelStrategy:
    """Factory to retrieve the appropriate strategy for the active model and provider."""
    category = get_model_size_category(model_name)
    if provider == "zai" or category == "cloud":
        return CloudStrategy()
    elif category == "tiny":
        return TinyLocalStrategy()
    else:
        return StandardLocalStrategy()
