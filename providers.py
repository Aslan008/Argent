"""
Provider Abstraction Layer for Argent.
Normalizes LLM API communication across Ollama and Z.AI into a single interface.
"""

import json
import time
import random
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator

logger = logging.getLogger("argent.providers")


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False, original_error: Exception = None):
        super().__init__(message)
        self.retryable = retryable
        self.original_error = original_error


def with_retry(fn, max_retries=3, base_delay=1.0):
    retryable_codes = (429, 500, 502, 503)
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except ProviderError as e:
            if not e.retryable or attempt == max_retries - 1:
                raise
            last_error = e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning("Retrying in %.1fs (attempt %d/%d): %s", delay, attempt + 1, max_retries, e)
            time.sleep(delay)
        except Exception as e:
            error_str = str(e)
            is_retryable = any(str(c) in error_str for c in retryable_codes)
            if not is_retryable or attempt == max_retries - 1:
                raise ProviderError(str(e), retryable=is_retryable, original_error=e)
            last_error = e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning("Retrying in %.1fs (attempt %d/%d): %s", delay, attempt + 1, max_retries, e)
            time.sleep(delay)
    if last_error:
        raise last_error


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def validate_config(self) -> Optional[str]: ...

    @abstractmethod
    def list_models(self) -> List[str]: ...

    @abstractmethod
    def stream_chat(self, model, messages, tools=None, context_window=None, temperature=None): ...

    @abstractmethod
    def sync_chat(self, model, messages, temperature=0.3, json_format=False) -> str: ...

    @abstractmethod
    def format_tool_result(self, content: str, tool_call_id: str = None) -> dict: ...


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self):
        import ollama as _ollama
        self._ollama = _ollama

    @property
    def name(self) -> str:
        return "ollama"

    def validate_config(self) -> Optional[str]:
        try:
            self._ollama.list()
            return None
        except Exception as e:
            return f"Cannot connect to Ollama: {e}. Is Ollama running?"

    def list_models(self) -> List[str]:
        try:
            models_response = self._ollama.list()
            models = []
            if hasattr(models_response, 'models'):
                models_raw = models_response.models
            elif isinstance(models_response, dict):
                models_raw = models_response.get('models', [])
            else:
                models_raw = []
            for m in models_raw:
                if isinstance(m, dict):
                    models.append(m.get("model", m.get("name")))
                else:
                    models.append(getattr(m, 'model', getattr(m, 'name', str(m))))
            return models
        except Exception:
            return []

    def stream_chat(self, model, messages, tools=None, context_window=None, temperature=None):
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            
        options = {}
        if context_window:
            options["num_ctx"] = context_window
        if temperature is not None:
            options["temperature"] = temperature
            
        if options:
            kwargs["options"] = options

        try:
            response_stream = self._ollama.chat(**kwargs)
        except Exception as e:
            raise ProviderError(f"Ollama connection error: {e}", original_error=e)

        for chunk in response_stream:
            msg = chunk.get("message", {})
            tool_call_deltas = []
            if "tool_calls" in msg:
                for idx, tc in enumerate(msg["tool_calls"]):
                    args = tc.get("function", {}).get("arguments", {})
                    tool_call_deltas.append({
                        "index": idx,
                        "id": "",
                        "function_name_delta": tc.get("function", {}).get("name", ""),
                        "function_arguments_delta": json.dumps(args) if isinstance(args, dict) else str(args),
                    })
            yield {
                "content": msg.get("content", ""),
                "thinking": msg.get("thinking", ""),
                "tool_call_deltas": tool_call_deltas,
            }

    def sync_chat(self, model, messages, temperature=0.3, json_format=False) -> str:
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_format:
            kwargs["format"] = "json"

        def _call():
            try:
                response = self._ollama.chat(**kwargs)
                return response.get("message", {}).get("content", "")
            except Exception as e:
                raise ProviderError(f"Ollama error: {e}", original_error=e)

        return with_retry(_call, max_retries=2)

    def format_tool_result(self, content: str, tool_call_id: str = None) -> dict:
        return {"role": "tool", "content": content}


class OpenAICompatibleProvider(LLMProvider, ABC):
    """Base class for OpenAI-compatible providers."""

    def __init__(self, api_key: str, base_url: str):
        import openai
        self._openai = openai
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self):
        return self._openai.OpenAI(api_key=self._api_key, base_url=self._base_url)

    def stream_chat(self, model, messages, tools=None, context_window=None, temperature=None):
        client = self._get_client()
        openai_tools = None
        if tools:
            openai_tools = [{"type": "function", "function": t["function"]} for t in tools]

        kwargs = {
            "model": model,
            "messages": messages,
            "tools": openai_tools,
            "stream": True,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        # Pass max generation tokens to prevent premature truncation.
        # Without this, providers like KoboldCPP use their own default
        # (often 256-512 tokens), which causes the model to stop mid-response.
        from config import get_max_generation_tokens
        max_gen = get_max_generation_tokens()
        if max_gen:
            kwargs["max_tokens"] = max_gen

        try:
            response_stream = client.chat.completions.create(**kwargs)
        except self._openai.APIStatusError as e:
            self._handle_api_status_error(e)
        except Exception as e:
            raise ProviderError(f"{self.name.upper()} connection error: {e}", original_error=e)

        for chunk in response_stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            tool_call_deltas = []
            if getattr(delta, 'tool_calls', None):
                for tc in delta.tool_calls:
                    tool_call_deltas.append({
                        "index": tc.index,
                        "id": getattr(tc, 'id', '') or "",
                        "function_name_delta": getattr(tc.function, 'name', '') or "",
                        "function_arguments_delta": getattr(tc.function, 'arguments', '') or "",
                    })

            result = {
                "content": delta.content or "",
                "thinking": getattr(delta, 'reasoning_content', '') or "",
                "tool_call_deltas": tool_call_deltas,
            }

            # Detect forced truncation: finish_reason="length" means the model
            # was cut off by max_tokens, not because it finished naturally.
            finish_reason = getattr(choice, 'finish_reason', None)
            if finish_reason == "length":
                result["truncated"] = True

            yield result

    def sync_chat(self, model, messages, temperature=0.3, json_format=False) -> str:
        client = self._get_client()
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_format:
            kwargs["response_format"] = {"type": "json_object"}

        def _call():
            try:
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except self._openai.APIStatusError as e:
                self._handle_api_status_error(e)
            except Exception as e:
                raise ProviderError(f"{self.name.upper()} error: {e}", original_error=e)

        return with_retry(_call, max_retries=2)

    def format_tool_result(self, content: str, tool_call_id: str = None) -> dict:
        msg = {"role": "tool", "content": content}
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        return msg

    def _handle_api_status_error(self, e):
        raise ProviderError(
            f"{self.name.upper()} API Error (HTTP {e.status_code}): {e}",
            retryable=e.status_code >= 500, original_error=e
        )


class ZAIProvider(OpenAICompatibleProvider):
    """Z.AI cloud LLM provider (OpenAI-compatible API)."""

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "zai"

    def validate_config(self) -> Optional[str]:
        if not self._api_key:
            return "Z.AI API key is not set. Use /provider to configure it."
        return None

    def list_models(self) -> List[str]:
        return [
            "glm-5.1", "glm-5", "glm-5-turbo",
            "glm-4.7", "glm-4.7-flashx",
            "glm-4.6", "glm-4.5", "glm-4.5-x",
            "glm-4.5-air", "glm-4.5-airx",
            "glm-4-32b-0414-128k",
            "glm-4.7-flash", "glm-4.5-flash",
        ]

    def _handle_api_status_error(self, e):
        if e.status_code == 429:
            raise ProviderError(
                f"Z.AI: Insufficient balance or rate limited. Recharge at https://z.ai/manage-apikey/billing\n{e}",
                retryable=True, original_error=e
            )
        elif e.status_code == 401:
            raise ProviderError(
                f"Z.AI: Invalid API key. Use /provider to update your key.",
                original_error=e
            )
        else:
            super()._handle_api_status_error(e)


class KoboldCPPProvider(OpenAICompatibleProvider):
    """KoboldCPP local LLM provider (OpenAI-compatible API)."""

    def __init__(self, base_url: str):
        super().__init__(api_key="dummy_key", base_url=base_url)

    @property
    def name(self) -> str:
        return "koboldcpp"

    def validate_config(self) -> Optional[str]:
        try:
            client = self._get_client()
            client.models.list()
            return None
        except Exception as e:
            return f"Cannot connect to KoboldCPP at {self._base_url}: {e}. Is KoboldCPP running?"

    def list_models(self) -> List[str]:
        try:
            client = self._get_client()
            models_response = client.models.list()
            return [m.id for m in models_response.data]
        except Exception:
            return ["koboldcpp-model"]


def create_provider() -> LLMProvider:
    from config import (
        get_provider as _get_provider_name,
        get_zai_api_key,
        get_zai_endpoint,
        get_koboldcpp_url,
    )
    name = _get_provider_name()
    if name == "zai":
        return ZAIProvider(api_key=get_zai_api_key(), base_url=get_zai_endpoint())
    elif name == "koboldcpp":
        return KoboldCPPProvider(base_url=get_koboldcpp_url())
    return OllamaProvider()
