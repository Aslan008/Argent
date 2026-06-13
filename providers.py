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


def _parse_openai_usage(usage) -> dict:
    """Normalize an OpenAI usage object to {prompt, completion, total, cost?}.

    cost is non-standard: OpenRouter returns it (when requested) on the usage
    object, surfaced by the SDK via model_extra.
    """
    d = {
        "prompt": getattr(usage, "prompt_tokens", 0) or 0,
        "completion": getattr(usage, "completion_tokens", 0) or 0,
        "total": getattr(usage, "total_tokens", 0) or 0,
    }
    cost = getattr(usage, "cost", None)
    if cost is None:
        extra = getattr(usage, "model_extra", None)
        if isinstance(extra, dict):
            cost = extra.get("cost")
    if cost is not None:
        d["cost"] = cost
    return d


# Substrings marking a transient backend failure worth retrying / hinting about.
_TRANSIENT_HINTS = (
    "idle", "timeout", "timed out", "upstream", "overload", "unavailable",
    "502", "503", "504", "rate limit", "connection reset", "connection error",
)


def _is_transient_stream_error(e) -> bool:
    return any(h in str(e).lower() for h in _TRANSIENT_HINTS)


def _friendly_stream_error(provider_name: str, e) -> str:
    """One-line message for an error raised WHILE iterating the stream (the
    backend stopped sending mid-response). Adds an actionable hint for the
    common transient case so the user isn't left staring at a frozen panel."""
    msg = " ".join(str(e).split())
    if len(msg) > 200:
        msg = msg[:197] + "..."
    logger.warning("%s stream error: %s", provider_name.upper(), e)
    if _is_transient_stream_error(e):
        return (
            f"{provider_name.upper()}: {msg}. The model backend stopped responding "
            f"mid-generation — common on free/overloaded models. Retry, switch model "
            f"with /model, or split big outputs into smaller steps (create the file, "
            f"then append to it)."
        )
    return f"{provider_name.upper()} stream error: {msg}"


def _friendly_api_error(provider_name: str, e) -> str:
    """Turn an openai APIStatusError into a one-line, human-readable message.

    Strict backends (e.g. vLLM behind OpenRouter) return a deeply nested error
    payload; dumping it raw buries the user in JSON. This surfaces the salient
    message and the upstream provider, logs the full payload for /logs, and
    caps the length.
    """
    status = getattr(e, "status_code", "?")
    msg = None
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            meta = err.get("metadata")
            if isinstance(meta, dict):
                upstream = meta.get("provider_name")
                if upstream and msg:
                    msg = f"{msg} (upstream: {upstream})"
    if not msg:
        msg = getattr(e, "message", None) or str(e)
    msg = " ".join(str(msg).split())  # collapse whitespace/newlines
    if len(msg) > 280:
        msg = msg[:277] + "..."

    logger.warning("%s API error HTTP %s: %s", provider_name.upper(), status,
                   getattr(e, "body", None) or e)
    return f"{provider_name.upper()} error (HTTP {status}): {msg}"


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
    def stream_chat(self, model, messages, tools=None, context_window=None, temperature=None,
                    format_schema=None): ...

    def supports_constrained_decoding(self) -> bool:
        """Whether stream_chat can constrain output to a JSON schema (format_schema)."""
        return False

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

    def supports_constrained_decoding(self) -> bool:
        return True

    def stream_chat(self, model, messages, tools=None, context_window=None, temperature=None,
                    format_schema=None):
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        if format_schema:
            # Ollama >= 0.5 constrains generation to the JSON schema at the
            # decoder level: invalid step output becomes impossible.
            kwargs["format"] = format_schema

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
            result = {
                "content": msg.get("content", ""),
                "thinking": msg.get("thinking", ""),
                "tool_call_deltas": tool_call_deltas,
            }
            # The final Ollama chunk carries token counts (local = no cost).
            if chunk.get("done"):
                prompt = chunk.get("prompt_eval_count", 0) or 0
                completion = chunk.get("eval_count", 0) or 0
                result["usage"] = {"prompt": prompt, "completion": completion,
                                   "total": prompt + completion}
            yield result

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

    def __init__(self, api_key: str, base_url: str, default_headers: dict = None):
        import openai
        self._openai = openai
        self._api_key = api_key
        self._base_url = base_url
        self._default_headers = default_headers

    def _get_client(self):
        kwargs = {"api_key": self._api_key, "base_url": self._base_url}
        if self._default_headers:
            kwargs["default_headers"] = self._default_headers
        return self._openai.OpenAI(**kwargs)

    @staticmethod
    def _normalize_outgoing_messages(messages):
        """Make message history conform to the OpenAI tool-call spec.

        The spec requires tool_calls[].function.arguments to be a JSON *string*.
        Argent stores it as a dict internally (Ollama wants a dict, and the dict
        is what actually invokes the tool), so it must be stringified on the way
        out. Strict OpenAI-compatible backends (e.g. vLLM behind OpenRouter)
        reject a dict with HTTP 400; lenient ones (Z.AI) happen to accept it.
        Copies only the parts it changes — never mutates the agent's history.
        """
        out = []
        for m in messages:
            tcs = m.get("tool_calls")
            if not tcs:
                out.append(m)
                continue
            new_tcs = []
            for tc in tcs:
                fn = tc.get("function")
                if fn is not None and not isinstance(fn.get("arguments"), str):
                    fn = {**fn, "arguments": json.dumps(fn.get("arguments") or {}, ensure_ascii=False)}
                    tc = {**tc, "function": fn}
                new_tcs.append(tc)
            out.append({**m, "tool_calls": new_tcs})
        return out

    def stream_chat(self, model, messages, tools=None, context_window=None, temperature=None,
                    format_schema=None):
        # format_schema is ignored: OpenAI-compatible cloud endpoints use
        # native tool calling and don't need constrained step decoding.
        client = self._get_client()
        openai_tools = None
        if tools:
            openai_tools = [{"type": "function", "function": t["function"]} for t in tools]

        kwargs = {
            "model": model,
            "messages": self._normalize_outgoing_messages(messages),
            "tools": openai_tools,
            "stream": True,
            # Ask for a final usage-only chunk (token counts) at end of stream.
            "stream_options": {"include_usage": True},
        }
        kwargs.update(self._extra_create_kwargs())
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

        for chunk in self._guarded_stream(response_stream):
            # Usage may arrive either as a trailing usage-only chunk (no
            # choices) or attached to the final content chunk (OpenRouter/vLLM).
            chunk_usage = getattr(chunk, "usage", None)
            if not chunk.choices:
                if chunk_usage:
                    yield {"content": "", "thinking": "", "tool_call_deltas": [],
                           "usage": _parse_openai_usage(chunk_usage)}
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

            if chunk_usage:
                result["usage"] = _parse_openai_usage(chunk_usage)

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

    def _extra_create_kwargs(self) -> dict:
        """Provider-specific kwargs for chat.completions.create (override me)."""
        return {}

    def _guarded_stream(self, response_stream):
        """Iterate the response stream, converting mid-stream backend failures
        (idle/upstream timeouts, 5xx) into a friendly ProviderError instead of
        leaking a raw exception that surfaces as a frozen panel + double 'Error'."""
        try:
            for chunk in response_stream:
                yield chunk
        except self._openai.APIStatusError as e:
            self._handle_api_status_error(e)
        except Exception as e:
            raise ProviderError(
                _friendly_stream_error(self.name, e),
                retryable=_is_transient_stream_error(e), original_error=e,
            )

    def _handle_api_status_error(self, e):
        raise ProviderError(
            _friendly_api_error(self.name, e),
            retryable=getattr(e, "status_code", 0) >= 500, original_error=e
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


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter cloud aggregator (OpenAI-compatible API).

    Exposes models from many vendors (Anthropic, OpenAI, Google, Meta,
    DeepSeek, ...) behind one OpenAI-compatible endpoint. Model ids are
    vendor-prefixed slugs, e.g. 'anthropic/claude-3.5-sonnet'.
    """

    # Optional attribution headers OpenRouter uses for its app rankings.
    _ATTRIBUTION_HEADERS = {
        "HTTP-Referer": "https://github.com/Aslan008/Argent",
        "X-Title": "Argent",
    }

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key=api_key, base_url=base_url,
                         default_headers=self._ATTRIBUTION_HEADERS)

    def _extra_create_kwargs(self) -> dict:
        # Ask OpenRouter to include the request cost in the usage payload.
        return {"extra_body": {"usage": {"include": True}}}

    @property
    def name(self) -> str:
        return "openrouter"

    @staticmethod
    def is_free_model(model_id: str) -> bool:
        """OpenRouter marks zero-cost models with a ':free' slug suffix."""
        return model_id.endswith(":free")

    @classmethod
    def sort_free_first(cls, models: List[str]) -> List[str]:
        """Free models first, then alphabetical — so they're easy to spot."""
        return sorted(models, key=lambda m: (not cls.is_free_model(m), m))

    def validate_config(self) -> Optional[str]:
        if not self._api_key:
            return (
                "OpenRouter API key is not set. Use /provider to configure it. "
                "A free key from https://openrouter.ai/keys is enough to use the free "
                "(':free') models; add credits later only if you want the paid ones."
            )
        return None

    def list_models(self) -> List[str]:
        try:
            client = self._get_client()
            models_response = client.models.list()
            ids = [m.id for m in models_response.data]
            return sorted(ids) if ids else self._fallback_models()
        except Exception:
            return self._fallback_models()

    @staticmethod
    def _fallback_models() -> List[str]:
        # Shown when the live catalog can't be fetched (e.g. no key yet). The
        # exact set comes from the API once a key is set; the user can also type
        # any valid OpenRouter slug manually. Includes both free and paid tiers.
        return [
            "deepseek/deepseek-chat-v3-0324:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "qwen/qwen-2.5-coder-32b-instruct:free",
            "anthropic/claude-3.7-sonnet",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash-001",
            "deepseek/deepseek-chat",
        ]

    def _handle_api_status_error(self, e):
        if e.status_code == 401:
            raise ProviderError(
                "OpenRouter: Invalid API key. Use /provider to update it.",
                original_error=e
            )
        elif e.status_code == 402:
            raise ProviderError(
                "OpenRouter: Insufficient credits. Top up at https://openrouter.ai/credits",
                original_error=e
            )
        elif e.status_code == 429:
            raise ProviderError(
                "OpenRouter: Rate limited. Slow down or check your plan limits.",
                retryable=True, original_error=e
            )
        else:
            super()._handle_api_status_error(e)


def create_provider(provider_name: str = None) -> LLMProvider:
    """Build a provider. Defaults to the configured main provider; pass a name
    to build a specific one (used for the auxiliary service model)."""
    from config import (
        get_provider as _get_provider_name,
        get_zai_api_key,
        get_zai_endpoint,
        get_koboldcpp_url,
        get_openrouter_api_key,
        get_openrouter_url,
    )
    name = provider_name or _get_provider_name()
    if name == "zai":
        return ZAIProvider(api_key=get_zai_api_key(), base_url=get_zai_endpoint())
    elif name == "koboldcpp":
        return KoboldCPPProvider(base_url=get_koboldcpp_url())
    elif name == "openrouter":
        return OpenRouterProvider(api_key=get_openrouter_api_key(), base_url=get_openrouter_url())
    return OllamaProvider()


def create_service_provider():
    """Provider + model for lightweight service tasks (summarization, commit
    messages). Uses the auxiliary model when configured — e.g. a free local
    Ollama model while the main provider is a paid cloud one — otherwise falls
    back to the main provider and model. Returns (provider, model_name)."""
    from config import (
        get_auxiliary_model, get_auxiliary_provider,
        get_provider as _get_provider_name, get_current_model,
    )
    aux_model = get_auxiliary_model()
    if aux_model:
        aux_provider = get_auxiliary_provider() or _get_provider_name()
        return create_provider(aux_provider), aux_model
    return create_provider(), get_current_model()
