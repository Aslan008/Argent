"""
Provider Abstraction Layer for Argent.
Normalizes LLM API communication across Ollama and Z.AI into a single interface.
"""

import json
import time
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Was logging.getLogger("argent.providers"), which has no handler — so every
# retry, stream error and tool-channel fallback went nowhere. The same defect
# hid an MCP call that hung for ten minutes; a provider's warnings are exactly
# what you need when a turn dies and nobody knows why.
from logger import get_logger

logger = get_logger("providers")


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False, original_error: Exception = None):
        super().__init__(message)
        self.retryable = retryable
        self.original_error = original_error


class UpstreamBodyError(Exception):
    """A 200 response whose body is an error (OpenRouter passes an upstream failure on that
    way) or carries no choices. Shaped like an HTTP error so callers can classify it."""

    def __init__(self, message: str, status_code: Optional[int] = None, body: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body

    @property
    def retryable(self) -> bool:
        return self.status_code is None or self.status_code in (408, 429, 500, 502, 503, 504, 529)

    @classmethod
    def from_response(cls, provider_name: str, response) -> "UpstreamBodyError":
        err = getattr(response, "error", None)
        if err is None:
            extra = getattr(response, "model_extra", None)
            err = extra.get("error") if isinstance(extra, dict) else None
        if isinstance(err, dict):
            code = err.get("code")
            message = " ".join(str(err.get("message") or "provider returned an error").split())[:240]
            return cls(f"{provider_name.upper()} returned an error: {message}",
                       code if isinstance(code, int) else None, {"error": err})
        return cls(f"{provider_name.upper()} returned no choices (empty response)")


#: a dropped connection is usually a blink; it gets this many extra tries beyond what the caller asked for
_TRANSPORT_EXTRA_RETRIES = 2
_MAX_RETRY_DELAY = 15.0


def with_retry(fn, max_retries=3, base_delay=1.0):
    """Call `fn`, repeating it when the failure is the network's or the server's rather than the request's.

    Retryable: a ProviderError marked so, an HTTP code in the message (429/5xx/408), or a broken connection
    (`netstate.is_transport_error`) — the last used to be missed, so a one-second Wi-Fi blink failed the call.
    Waiting out a LONG outage is not this function's job: the agent loop does that (src/agent/netstate.py)."""
    from src.agent.netstate import is_transport_error
    max_retries = max(1, max_retries)
    base_delay = max(0.0, base_delay)
    retryable_codes = ("429", "500", "502", "503", "529", "408")
    last_error = None
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            transport = is_transport_error(e)
            if isinstance(e, ProviderError):
                retryable = e.retryable or transport
            else:
                retryable = any(c in str(e) for c in retryable_codes) or transport
            limit = max_retries + (_TRANSPORT_EXTRA_RETRIES if transport else 0)
            if not retryable or attempt >= limit - 1:
                if isinstance(e, ProviderError):
                    raise
                raise ProviderError(str(e), retryable=retryable, original_error=e)
            last_error = e
            delay = min(_MAX_RETRY_DELAY, base_delay * (2 ** attempt)) + random.uniform(0, 1)
            logger.warning("Retrying in %.1fs (attempt %d/%d): %s", delay, attempt + 1, limit, e)
            time.sleep(delay)
            attempt += 1


def _parse_openai_usage(usage) -> dict:
    """Normalize an OpenAI usage object to {prompt, completion, total, cost?, cached?}.

    Handles OpenAI (prompt_tokens_details.cached_tokens), DeepSeek
    (prompt_cache_hit_tokens), and OpenRouter / Anthropic (model_extra).
    cost is non-standard: OpenRouter returns it on the usage object.
    """
    if usage is None:
        return {}

    def _get(obj, key, default=0):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    prompt = _get(usage, "prompt_tokens", 0) or _get(usage, "prompt", 0) or 0
    completion = _get(usage, "completion_tokens", 0) or _get(usage, "completion", 0) or 0
    total = _get(usage, "total_tokens", 0) or _get(usage, "total", 0) or (prompt + completion)

    d = {
        "prompt": int(prompt),
        "completion": int(completion),
        "total": int(total),
    }

    cached = 0
    # 1. DeepSeek формат: prompt_cache_hit_tokens
    ds_hit = _get(usage, "prompt_cache_hit_tokens", 0)
    if ds_hit:
        cached = ds_hit

    # 2. OpenAI / OpenRouter / vLLM формат: prompt_tokens_details.cached_tokens
    if not cached:
        details = _get(usage, "prompt_tokens_details", None)
        if details:
            cached = _get(details, "cached_tokens", 0) or 0

    # 3. Прямое поле cached_tokens / cached на usage
    if not cached:
        cached = _get(usage, "cached_tokens", 0) or _get(usage, "cached", 0) or 0

    # 4. OpenRouter / Anthropic cache read via model_extra
    extra = getattr(usage, "model_extra", None)
    if isinstance(extra, dict):
        cache_read = extra.get("cache_read_input_tokens")
        if cache_read:
            try:
                cached = max(cached, int(cache_read))
            except (ValueError, TypeError):
                pass
        if not cached:
            prompt_details = extra.get("prompt_tokens_details")
            if isinstance(prompt_details, dict):
                cached = prompt_details.get("cached_tokens", 0) or 0

    if cached:
        try:
            d["cached"] = max(0, int(cached))
        except (ValueError, TypeError):
            pass

    # Стоимость
    cost = getattr(usage, "cost", None) if not isinstance(usage, dict) else usage.get("cost")
    if cost is None and isinstance(extra, dict):
        cost = extra.get("cost")
    if cost is not None:
        try:
            d["cost"] = max(0.0, float(cost))
        except (ValueError, TypeError):
            pass
    return d


# Substrings marking a transient backend failure worth retrying / hinting about.
_TRANSIENT_HINTS = (
    "idle", "timeout", "timed out", "upstream", "overload", "unavailable",
    "502", "503", "504", "529", "rate limit", "connection reset", "connection error",
)


def _is_transient_stream_error(e) -> bool:
    from src.agent.netstate import is_transport_error
    return any(h in str(e).lower() for h in _TRANSIENT_HINTS) or is_transport_error(e)


def _client_timeout():
    """httpx timeout for the cloud SDK clients.

    The SDKs default to ten minutes without a byte before they notice a dead socket (the Ollama client had
    this fixed already, the cloud ones did not): a cable pulled mid-answer froze the turn for ten minutes.
    The READ timeout is the gap between two chunks, so a slow but living generation is untouched; it is
    kept above 300 s because a reasoning model may stay silent while it thinks before the first token."""
    try:
        import httpx
        from config import get_stream_read_timeout
        return httpx.Timeout(connect=15.0, read=max(300.0, get_stream_read_timeout()), write=60.0, pool=15.0)
    except Exception:
        return None


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

    def resolve_model(self, text: str) -> Optional[str]:
        """The model id this provider means by `text`, spelled as it lists it; None when it is
        certain there is no such model. The default trusts the name (local and gateway providers
        take any tag)."""
        return text

    def suggest_models(self, text: str, n: int = 5) -> List[str]:
        return []

    def default_model(self) -> Optional[str]:
        """What to use when nothing is configured for this provider, or None to keep the global model."""
        return None

    @abstractmethod
    def sync_chat(self, model, messages, temperature=0.3, json_format=False, timeout=None) -> str: ...

    @abstractmethod
    def format_tool_result(self, content: str, tool_call_id: str = None) -> dict: ...


def _images_for_ollama(messages):
    """Translate the neutral image attachment into Ollama's dialect.

    History is stored provider-neutrally — {"images": [{"data", "media_type"}]}
    — because a conversation outlives the provider that started it: writing
    Ollama's spelling into the history would break the moment the user switches
    to OpenRouter mid-chat. Ollama wants a bare list of base64 strings.
    """
    out, touched = [], False
    for m in messages:
        images = m.get("images") if isinstance(m, dict) else None
        if not images:
            out.append(m)
            continue
        touched = True
        copy = {k: v for k, v in m.items() if k != "images"}
        copy["images"] = [img["data"] if isinstance(img, dict) else img for img in images]
        out.append(copy)
    return out if touched else messages


def _images_for_openai(messages):
    """The same attachment as OpenAI-style content parts (a data: URI)."""
    out, touched = [], False
    for m in messages:
        images = m.get("images") if isinstance(m, dict) else None
        if not images:
            out.append(m)
            continue
        touched = True
        parts = []
        text = m.get("content")
        if isinstance(text, str) and text:
            parts.append({"type": "text", "text": text})
        for img in images:
            data = img["data"] if isinstance(img, dict) else img
            media = img.get("media_type", "image/png") if isinstance(img, dict) else "image/png"
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{media};base64,{data}"}})
        copy = {k: v for k, v in m.items() if k != "images"}
        copy["content"] = parts
        out.append(copy)
    return out if touched else messages


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self):
        import ollama as _ollama

        # The bare module client has no read timeout, so a connection that dies
        # mid-generation is never noticed: the stream simply stops producing and
        # the turn waits forever. Observed — 948 seconds of "Выполнение…" ended
        # by Ctrl+C, with `lookup ollama.com: no such host` the moment the user
        # tried again. A READ timeout is the right one: it measures the gap
        # BETWEEN tokens, so a slow but living generation is untouched while a
        # dead socket is caught.
        try:
            import httpx
            from config import get_stream_read_timeout

            self._ollama = _ollama.Client(timeout=httpx.Timeout(
                connect=15.0, read=get_stream_read_timeout(), write=60.0, pool=15.0))
        except Exception as e:
            logger.warning("falling back to the default ollama client (%s); "
                           "a dropped connection will hang instead of failing", e)
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
            "messages": _images_for_ollama(messages),
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

        from config import (
            get_top_p,
            get_top_k,
            get_min_p,
            get_repeat_penalty,
        )
        top_p = get_top_p()
        if top_p is not None:
            options["top_p"] = top_p
        top_k = get_top_k()
        if top_k is not None:
            options["top_k"] = top_k
        min_p = get_min_p()
        if min_p is not None:
            options["min_p"] = min_p
        rep_pen = get_repeat_penalty()
        if rep_pen is not None:
            options["repeat_penalty"] = rep_pen

        if options:
            kwargs["options"] = options

        from src.agent.governor import server_options
        extra = server_options("ollama", model)
        if extra:
            kwargs.setdefault("options", {}).update(extra)

        try:
            response_stream = self._ollama.chat(**kwargs)
        except Exception as e:
            raise ProviderError(f"Ollama connection error: {e}", retryable=_is_transient_stream_error(e), original_error=e)

        # Ollama emits each tool call COMPLETE inside a chunk — it does not
        # stream them as deltas the way the OpenAI API does. So the index has to
        # count across the whole response: enumerate() restarted at 0 in every
        # chunk, and the consumer, which appends deltas per index, then merged
        # every call into slot 0. Observed with GLM 5.2 through Ollama cloud:
        #   read_file + read_file            -> "read_fileread_file"
        #   run_command + read_file + read_file -> "run_commandread_fileread_file"
        # with the arguments concatenated too, so the JSON failed to parse and
        # the call was executed with {} — silently losing what the model asked
        # for. Any model emitting more than one tool call per turn hit this.
        emitted_tool_calls = 0

        try:
            for chunk in response_stream:
                msg = chunk.get("message", {})
                tool_call_deltas = []
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"] or []:
                        args = tc.get("function", {}).get("arguments", {})
                        tool_call_deltas.append({
                            "index": emitted_tool_calls,
                            "id": "",
                            "function_name_delta": tc.get("function", {}).get("name", ""),
                            "function_arguments_delta": json.dumps(args) if isinstance(args, dict) else str(args),
                        })
                        emitted_tool_calls += 1
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
        except Exception as e:
            raise ProviderError(
                _friendly_stream_error("ollama", e),
                retryable=_is_transient_stream_error(e),
                original_error=e,
            )

    def sync_chat(self, model, messages, temperature=0.3, json_format=False, timeout=None) -> str:
        # timeout is accepted for interface parity; a local Ollama rarely hangs
        # and the daemon summarization thread is reclaimed at process exit.
        options = {}
        if temperature is not None:
            options["temperature"] = temperature

        from config import (
            get_top_p,
            get_top_k,
            get_min_p,
            get_repeat_penalty,
        )
        top_p = get_top_p()
        if top_p is not None:
            options["top_p"] = top_p
        top_k = get_top_k()
        if top_k is not None:
            options["top_k"] = top_k
        min_p = get_min_p()
        if min_p is not None:
            options["min_p"] = min_p
        rep_pen = get_repeat_penalty()
        if rep_pen is not None:
            options["repeat_penalty"] = rep_pen

        kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        from src.agent.governor import server_options
        options.update(server_options("ollama", model))
        if options:
            kwargs["options"] = options
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


def _is_openai_reasoning_model(model_name: str) -> bool:
    """Check if model is a reasoning model (o1, o3, deepseek-reasoner)
    which rejects temperature, top_p, and requires max_completion_tokens.
    """
    if not model_name:
        return False
    name = model_name.lower().split("/")[-1]
    return name.startswith("o1") or name.startswith("o3") or "deepseek-reasoner" in name


def _is_claude_model(model_name: str) -> bool:
    """Check if model is Anthropic Claude."""
    if not model_name:
        return False
    m = model_name.lower()
    return "claude" in m or "anthropic" in m


class OpenAICompatibleProvider(LLMProvider, ABC):
    """Base class for OpenAI-compatible providers."""

    #: Aggregators publish ids of their own ("anthropic/claude-sonnet-4.5"). What is typed is
    #: matched against the live list and returned spelled as listed; an id that is not listed is
    #: kept as typed, because the catalogue of a gateway is never the whole truth.
    corrects_model_spelling = False

    def resolve_model(self, text: str) -> Optional[str]:
        text = (text or "").strip()
        if not self.corrects_model_spelling or not text:
            return text
        from model_names import match_known
        try:
            ids = self.list_models()
        except Exception:
            return text
        return match_known(text, ids) or text

    def suggest_models(self, text: str, n: int = 5) -> List[str]:
        if not self.corrects_model_spelling:
            return []
        from model_names import suggest
        try:
            return suggest(text, self.list_models(), n)
        except Exception:
            return []

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
        timeout = _client_timeout()
        if timeout is not None:
            kwargs["timeout"] = timeout
        return self._openai.OpenAI(**kwargs)

    @staticmethod
    def _apply_prompt_cache(messages, model):
        """Mark the system prompt as a cacheable prefix for Anthropic models.

        Anthropic bills a cached prefix at a fraction of the input price and
        skips re-processing it, which cuts both cost and TTFT dramatically. The
        marker is an explicit ``cache_control`` block, unlike OpenAI/DeepSeek
        where caching is automatic and needs nothing from us. Argent's system
        prompt is the ideal breakpoint: it is large (rules + AGENTS.md + tool
        guidance) and already kept byte-stable across turns by
        _refresh_system_prompt, which is exactly what a prefix cache requires.

        Only the FIRST system message is marked (one breakpoint, well under
        Anthropic's limit of four), and only for models that understand the
        field — everything else is returned untouched.
        """
        name = (model or "").lower()
        if "claude" not in name and "anthropic" not in name:
            return messages

        out = []
        marked = False
        for m in messages:
            if not marked and m.get("role") == "system" and isinstance(m.get("content"), str):
                text = m["content"]
                # Below Anthropic's minimum cacheable prefix the marker is a
                # no-op at best, so don't bother rewriting the shape.
                if len(text) < 2000:
                    out.append(m)
                    continue
                out.append({**m, "content": [{
                    "type": "text",
                    "text": text,
                    "cache_control": {"type": "ephemeral"},
                }]})
                marked = True
            else:
                out.append(m)
        return out

    @staticmethod
    def _normalize_outgoing_messages(messages):
        """Make message history conform to OpenAI / Jinja chat template specs.

        1. Strict Jinja chat templates (llama-server, KoboldCPP, vLLM) require
           that role="system" may only appear as the very first message. Subsequent
           system messages (tool catalog, ephemeral repo context, objective anchor)
           cause HTTP 400 (raise_exception: "System message must be at the beginning...").
           Non-initial system messages are converted to role="user".
        2. OpenAI tool-call spec requires tool_calls[].function.arguments to be a
           JSON *string*. Argent stores it as a dict internally (Ollama wants a dict,
           and the dict is what actually invokes the tool), so it must be stringified
           on the way out.
        Copies only the parts it changes — never mutates the agent's history.
        """
        messages = _images_for_openai(messages)
        out = []
        for i, m in enumerate(messages):
            if i > 0 and m.get("role") == "system":
                m = {**m, "role": "user"}

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
            "messages": self._apply_prompt_cache(
                self._normalize_outgoing_messages(messages), model),
            "tools": openai_tools,
            "stream": True,
            # Ask for a final usage-only chunk (token counts) at end of stream.
            "stream_options": {"include_usage": True},
        }
        kwargs.update(self._extra_create_kwargs())

        # Pass max generation tokens to prevent premature truncation.
        # Without this, providers like KoboldCPP use their own default
        # (often 256-512 tokens), which causes the model to stop mid-response.
        from config import (
            get_max_generation_tokens,
            get_thinking_mode,
            get_reasoning_effort,
            get_thinking_budget,
            get_top_p,
            get_top_k,
            get_min_p,
            get_repeat_penalty,
        )

        is_reasoning = _is_openai_reasoning_model(model)
        is_claude = _is_claude_model(model)

        # Thinking / reasoning configuration on the fly
        thinking_mode = get_thinking_mode()
        effort = get_reasoning_effort()
        budget = get_thinking_budget()

        # Sampling parameter sanitization:
        # 1. OpenAI reasoning models (o1, o3-mini) reject temperature & top_p (HTTP 400).
        # 2. Claude with thinking mode enforces temperature == 1.0 (HTTP 400 otherwise).
        if not is_reasoning:
            if is_claude and (thinking_mode == "on" or (thinking_mode == "auto" and budget and budget > 0)):
                kwargs["temperature"] = 1.0
            elif temperature is not None:
                kwargs["temperature"] = temperature

        max_gen = get_max_generation_tokens()
        if max_gen:
            if is_reasoning:
                kwargs["max_completion_tokens"] = max_gen
            else:
                kwargs["max_tokens"] = max_gen

        if not is_reasoning:
            top_p = get_top_p()
            if top_p is not None:
                kwargs["top_p"] = top_p

        extra_body = {}
        chat_template_kwargs = {}
        if thinking_mode == "off":
            chat_template_kwargs["enable_thinking"] = False
            extra_body["reasoning_effort"] = "none"
        elif thinking_mode == "on":
            chat_template_kwargs["enable_thinking"] = True
            if effort:
                extra_body["reasoning_effort"] = effort
                chat_template_kwargs["reasoning_effort"] = effort
        elif thinking_mode == "auto":
            if effort:
                extra_body["reasoning_effort"] = effort
                chat_template_kwargs["reasoning_effort"] = effort

        if budget is not None and budget > 0:
            extra_body["reasoning_budget"] = budget

        if is_reasoning and effort and effort in ("low", "medium", "high"):
            kwargs["reasoning_effort"] = effort

        if not is_reasoning:
            top_k = get_top_k()
            if top_k is not None and not (is_claude and thinking_mode == "on"):
                extra_body["top_k"] = top_k

            min_p = get_min_p()
            if min_p is not None:
                extra_body["min_p"] = min_p

            rep_pen = get_repeat_penalty()
            if rep_pen is not None:
                extra_body["repetition_penalty"] = rep_pen

        if chat_template_kwargs:
            extra_body["chat_template_kwargs"] = chat_template_kwargs

        if extra_body:
            if "extra_body" in kwargs and isinstance(kwargs["extra_body"], dict):
                kwargs["extra_body"].update(extra_body)
            else:
                kwargs["extra_body"] = extra_body

        try:
            response_stream = client.chat.completions.create(**kwargs)
        except self._openai.APIStatusError as e:
            self._handle_api_status_error(e)
        except Exception as e:
            raise ProviderError(f"{self.name.upper()} connection error: {e}",
                                retryable=_is_transient_stream_error(e), original_error=e)

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

    def sync_chat(self, model, messages, temperature=0.3, json_format=False, timeout=None) -> str:
        client = self._get_client()
        kwargs = {
            "model": model,
            "messages": self._normalize_outgoing_messages(messages),
        }
        if not _is_openai_reasoning_model(model):
            kwargs["temperature"] = temperature
        if json_format:
            kwargs["response_format"] = {"type": "json_object"}
        if timeout is not None:
            # Bounds a hung remote provider so a caller running this on a worker
            # thread (e.g. context summarization) can't leak a thread forever.
            kwargs["timeout"] = timeout

        def _call():
            try:
                response = client.chat.completions.create(**kwargs)
                choices = getattr(response, "choices", None)
                if not choices:
                    raise UpstreamBodyError.from_response(self.name, response)
                return choices[0].message.content or ""
            except self._openai.APIStatusError as e:
                self._handle_api_status_error(e)
            except UpstreamBodyError as e:
                raise ProviderError(f"{self.name.upper()} error: {e}", retryable=e.retryable, original_error=e)
            except Exception as e:
                raise ProviderError(f"{self.name.upper()} error: {e}",
                                    retryable=_is_transient_stream_error(e), original_error=e)

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

    corrects_model_spelling = True

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

    def list_free_models(self, min_context: int = 16000) -> List[str]:
        """Free text models from the live catalogue, the longest context first. Models whose
        context is known to be shorter than `min_context` are left out: a compression prompt
        would not fit. The built-in shortlist stands in when the catalogue cannot be fetched."""
        try:
            data = self._get_client().models.list().data
        except Exception:
            data = []
        ranked = []
        for m in data:
            mid = str(getattr(m, "id", "") or "")
            if not self.is_free_model(mid):
                continue
            extra = getattr(m, "model_extra", None) or {}
            ctx = extra.get("context_length") or getattr(m, "context_length", None) or 0
            ctx = int(ctx) if isinstance(ctx, (int, float)) else 0
            if ctx and ctx < min_context:
                continue
            arch = extra.get("architecture")
            outputs = arch.get("output_modalities") if isinstance(arch, dict) else None
            if outputs and "text" not in outputs:
                continue
            ranked.append((-ctx, mid))
        if not ranked:
            return [m for m in self._fallback_models() if self.is_free_model(m)]
        return [mid for _, mid in sorted(ranked)]

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
            "anthropic/claude-sonnet-5.5",
            "anthropic/claude-opus-5.5",
            "anthropic/claude-haiku-5.5",
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
            # The body says WHICH limit (per minute, per day, one upstream) and when it
            # resets; callers that decide what to do next read it from original_error.
            raise ProviderError(
                f"OpenRouter: Rate limited. {_friendly_api_error(self.name, e)}",
                retryable=True, original_error=e
            )
        else:
            super()._handle_api_status_error(e)


class XKiroProvider(OpenAICompatibleProvider):
    """xKiro AI aggregator (OpenAI-compatible gateway at https://api.xkiro.com/v1)."""

    corrects_model_spelling = True

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "xkiro"

    def validate_config(self) -> Optional[str]:
        if not self._api_key:
            return (
                "xKiro API key is not set. Get one at https://xkiro.com "
                "and configure it with /provider or /key."
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
        return [
            "anthropic/claude-sonnet-5.5",
            "anthropic/claude-opus-5.5",
            "anthropic/claude-haiku-5.5",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.7-sonnet",
            "deepseek/deepseek-chat",
            "deepseek/deepseek-reasoner",
            "google/gemini-2.0-flash",
            "google/gemini-2.5-pro",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "qwen/qwen-2.5-72b-instruct",
        ]

    def _handle_api_status_error(self, e):
        if e.status_code == 401:
            raise ProviderError(
                "xKiro: Invalid API key. Use /provider or /key to update it.",
                original_error=e
            )
        elif e.status_code == 402:
            raise ProviderError(
                "xKiro: Insufficient credits or balance. Top up at https://xkiro.com",
                original_error=e
            )
        elif e.status_code == 429:
            raise ProviderError(
                "xKiro: Rate limited. Slow down or check your limits.",
                retryable=True, original_error=e
            )
        else:
            super()._handle_api_status_error(e)


class WormSoftProvider(OpenAICompatibleProvider):
    """ai.wormsoft.ru AI aggregator (OpenAI-compatible gateway at https://ai.wormsoft.ru/api/gpt)."""

    corrects_model_spelling = True

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "wormsoft"

    def validate_config(self) -> Optional[str]:
        if not self._api_key:
            return (
                "WormSoft API key is not set. Get one at https://ai.wormsoft.ru "
                "and configure it with /provider or /key."
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
        return [
            "wormsoft/code/high",
            "wormsoft/code/medium",
            "wormsoft/agent/high",
            "wormsoft/agent/medium",
            "anthropic/claude-sonnet-5.5",
            "anthropic/claude-opus-5.5",
            "anthropic/claude-haiku-5.5",
            "anthropic/claude-sonnet-5",
            "anthropic/claude-opus-5",
            "anthropic/claude-haiku-4.5",
            "openai/gpt-6-astra",
            "openai/gpt-5.6-sol",
            "openai/gpt-5.6-terra",
            "openai/gpt-5.6-luna",
            "openai/gpt-oss:120b",
            "deepseek-ai/deepseek-v4-pro",
            "deepseek-ai/deepseek-v4-flash",
            "kimi/kimi-k3",
            "kimi/kimi-k2.7-code",
            "qwen/qwen3.8:27b",
            "zai/glm-5.3",
            "google/gemma4:31b",
        ]

    def _handle_api_status_error(self, e):
        if e.status_code == 401:
            raise ProviderError(
                "WormSoft: Invalid API key. Get one at https://ai.wormsoft.ru and use /provider or /key to update it.",
                original_error=e
            )
        elif e.status_code == 402:
            raise ProviderError(
                "WormSoft: Insufficient credits or balance. Top up at https://ai.wormsoft.ru",
                original_error=e
            )
        elif e.status_code == 429:
            raise ProviderError(
                "WormSoft: Rate limited. Slow down or check your limits.",
                retryable=True, original_error=e
            )
        else:
            super()._handle_api_status_error(e)


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider (https://api.deepseek.com)."""

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "deepseek"

    def validate_config(self) -> Optional[str]:
        if not self._api_key:
            return (
                "DeepSeek API key is not set. Get one at https://platform.deepseek.com "
                "and configure it with /provider or /key."
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
        return [
            "deepseek-chat",
            "deepseek-reasoner",
        ]

    def _handle_api_status_error(self, e):
        if e.status_code == 401:
            raise ProviderError(
                "DeepSeek: Invalid API key. Get one at https://platform.deepseek.com and use /provider or /key to update it.",
                original_error=e
            )
        elif e.status_code == 402:
            raise ProviderError(
                "DeepSeek: Insufficient balance. Top up at https://platform.deepseek.com",
                original_error=e
            )
        elif e.status_code == 429:
            raise ProviderError(
                "DeepSeek: Rate limited or server busy. Slow down or check your limits.",
                retryable=True, original_error=e
            )
        elif e.status_code == 503:
            raise ProviderError(
                "DeepSeek: Service temporarily overloaded. Retrying...",
                retryable=True, original_error=e
            )
        else:
            super()._handle_api_status_error(e)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider (https://api.anthropic.com)."""

    def __init__(self, api_key: str = None, base_url: str = None, workspace_id: str = None):
        import anthropic
        self._anthropic = anthropic
        self._api_key = api_key
        self._base_url = base_url
        self._workspace_id = (workspace_id or "").strip() or None

    @property
    def name(self) -> str:
        return "claude"

    def _get_client(self):
        kwargs = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._workspace_id:
            kwargs["default_headers"] = {"anthropic-workspace-id": self._workspace_id}
        timeout = _client_timeout()
        if timeout is not None:
            kwargs["timeout"] = timeout
        return self._anthropic.Anthropic(**kwargs)

    def validate_config(self) -> Optional[str]:
        if not self._api_key:
            return (
                "Claude API key is not set. Get one at https://console.anthropic.com "
                "and configure it with /provider or /key."
            )
        return None

    # -- the model catalogue (see claude_catalog) ------------------------------------------

    def catalog(self, refresh: bool = False):
        """Every model this key can use, newest first, with what each can do. Read from Anthropic
        (all pages), kept on disk, and only as a last resort the built-in seed."""
        import claude_catalog
        import hashlib
        scope = (f"{self._base_url or ''}|{self._workspace_id or ''}|"
                 f"{hashlib.sha1((self._api_key or '').encode()).hexdigest()[:12]}")
        return claude_catalog.load(self._get_client, scope=scope, refresh=refresh)

    def list_models(self) -> List[str]:
        """Newest first. Opening the model menu is a deliberate act, so it asks Anthropic again:
        a model released this morning is in the list this morning."""
        self._last_catalog = self.catalog(refresh=True)
        return self._last_catalog.ids()

    def default_model(self) -> Optional[str]:
        import claude_catalog
        return claude_catalog.default_model(self.catalog().entries)

    def resolve_model(self, text: str) -> Optional[str]:
        """The id Anthropic lists for what was typed, or None.

        "Claude Sonnet 4.5", "claude-sonnet-4.5", "sonnet" all find the real id. When the list is
        only a seed or a saved copy, a well-formed Claude name is still accepted in the API's
        spelling: the list not showing a model does not prove it does not exist."""
        from model_names import canonical_claude, resolve_claude
        catalog = self.catalog()
        found = resolve_claude(text, catalog.entries)
        if found or catalog.authoritative:
            return found
        return canonical_claude(text)

    def suggest_models(self, text: str, n: int = 5) -> List[str]:
        from model_names import suggest
        return suggest(text, self.catalog().ids(), n)

    def _request_model(self, model: str) -> str:
        """The model id to put on the wire: what is configured, in the spelling the API accepts."""
        resolved = self.resolve_model(model)
        if resolved:
            return resolved
        similar = self.suggest_models(model)
        raise ProviderError(
            f"Claude: '{model}' не является моделью Claude, которую знает Anthropic. "
            f"Выберите модель командой /model (список берётся из Anthropic)."
            + (f" Похожие: {', '.join(similar)}." if similar else ""))

    @staticmethod
    def _sdk_accepts() -> frozenset:
        """The keyword arguments the installed SDK's messages.create takes. Anthropic removed the
        sampling parameters (temperature, top_p, top_k) from newer SDKs and models; sending one
        is a TypeError before any request leaves, so only what the SDK declares is sent."""
        import inspect
        try:
            from anthropic.resources.messages import Messages
            return frozenset(inspect.signature(Messages.create).parameters)
        except Exception:
            return frozenset({"temperature", "top_p", "top_k", "thinking", "output_config"})

    def format_tool_result(self, content: str, tool_call_id: str = None) -> dict:
        msg = {"role": "tool", "content": content}
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        return msg

    def _convert_messages(self, messages: List[Dict[str, Any]]):
        """Translate Argent message history to Anthropic Messages API format."""
        system_texts = []
        converted = []

        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role == "system":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    system_texts.append(c.strip())
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "text":
                            system_texts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            system_texts.append(part)
                continue

            blocks = []
            for img in m.get("images", []):
                data = img.get("data") if isinstance(img, dict) else img
                media = img.get("media_type", "image/png") if isinstance(img, dict) else "image/png"
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media,
                        "data": data,
                    }
                })

            if role == "assistant" and m.get("tool_calls"):
                if m.get("content"):
                    blocks.append({"type": "text", "text": str(m["content"])})
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except Exception:
                            args = {"raw": args}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id") or "tool_call",
                        "name": fn.get("name", ""),
                        "input": args if isinstance(args, dict) else {"value": args},
                    })
            elif role == "tool":
                cid = m.get("tool_call_id") or "tool_call"
                content = m.get("content") or ""
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": cid,
                    "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                })
            else:
                c = m.get("content")
                if c:
                    if isinstance(c, str):
                        blocks.append({"type": "text", "text": c})
                    elif isinstance(c, list):
                        blocks.extend(c)

            anthropic_role = "user" if role in ("user", "tool") else "assistant"
            if not blocks:
                blocks = [{"type": "text", "text": " "}]

            if converted and converted[-1]["role"] == anthropic_role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": anthropic_role, "content": blocks})

        if not converted:
            converted = [{"role": "user", "content": [{"type": "text", "text": " "}]}]
        elif converted[0]["role"] != "user":
            converted.insert(0, {"role": "user", "content": [{"type": "text", "text": " "}]})

        combined_system = "\n\n".join(system_texts)
        system_param = None
        if combined_system:
            if len(combined_system) >= 2000:
                system_param = [{
                    "type": "text",
                    "text": combined_system,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                system_param = combined_system

        return system_param, converted

    @staticmethod
    def _convert_tools(tools):
        if not tools:
            return None
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", t)
            name = fn.get("name")
            if not name:
                continue
            desc = fn.get("description", "")
            schema = fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}}
            anthropic_tools.append({
                "name": name,
                "description": desc,
                "input_schema": schema,
            })
        return anthropic_tools if anthropic_tools else None

    def stream_chat(self, model, messages, tools=None, context_window=None, temperature=None,
                    format_schema=None):
        client = self._get_client()
        system_param, conv_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        from config import (
            get_max_generation_tokens,
            get_reasoning_effort,
            get_thinking_mode,
            get_thinking_budget,
            get_top_p,
            get_top_k,
        )
        import claude_catalog

        model = self._request_model(model)
        entry = self.catalog().get(model)
        accepts = self._sdk_accepts()

        thinking_mode = get_thinking_mode()
        budget = get_thinking_budget()
        wants_thinking = thinking_mode == "on" or (thinking_mode == "auto" and bool(budget and budget > 0))
        style = claude_catalog.thinking_style(model, entry) if wants_thinking else None

        max_gen = get_max_generation_tokens() or 8192
        kwargs = {
            "model": model,
            "messages": conv_messages,
            "stream": True,
        }
        if system_param:
            kwargs["system"] = system_param
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        if style == "enabled" and "thinking" in accepts:
            thinking_budget = max(1024, budget if (budget and budget > 0) else 2048)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            max_gen = max(max_gen, thinking_budget + 1024)
        elif style == "adaptive" and "thinking" in accepts:
            kwargs["thinking"] = {"type": "adaptive"}

        # Reasoning depth for the models that take it (/effort), only at a level they list.
        effort = (get_reasoning_effort() or "").strip().lower()
        if effort and "output_config" in accepts and entry is not None and effort in (entry.effort_levels or ()):
            kwargs["output_config"] = {"effort": effort}

        if entry is not None and entry.max_output:
            max_gen = min(max_gen, entry.max_output)
        kwargs["max_tokens"] = max_gen

        # Sampling knobs: only where the installed SDK still has them. With extended thinking the
        # API wants the default temperature, so they are left out then as before.
        if "thinking" not in kwargs:
            if temperature is not None and "temperature" in accepts:
                kwargs["temperature"] = temperature
            top_p = get_top_p()
            if top_p is not None and "top_p" in accepts:
                kwargs["top_p"] = top_p
            top_k = get_top_k()
            if top_k is not None and "top_k" in accepts:
                kwargs["top_k"] = top_k

        try:
            response_stream = client.messages.create(**kwargs)
        except self._anthropic.APIStatusError as e:
            self._handle_api_status_error(e)
        except Exception as e:
            raise ProviderError(f"CLAUDE connection error: {e}", retryable=_is_transient_stream_error(e), original_error=e)

        prompt_tokens = 0
        cached_tokens = 0
        completion_tokens = 0
        active_tools = {}
        next_tool_call_idx = 0

        for event in self._guarded_stream(response_stream):
            etype = getattr(event, "type", "")
            if etype == "message_start":
                usage = getattr(getattr(event, "message", None), "usage", None)
                if usage:
                    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
                    cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            elif etype == "content_block_start":
                cb = getattr(event, "content_block", None)
                if cb and getattr(cb, "type", "") == "tool_use":
                    tc_idx = next_tool_call_idx
                    next_tool_call_idx += 1
                    tool_id = getattr(cb, "id", "") or ""
                    tool_name = getattr(cb, "name", "") or ""
                    active_tools[getattr(event, "index", 0)] = {
                        "id": tool_id,
                        "name": tool_name,
                        "tool_call_index": tc_idx,
                        "args_emitted": False,
                    }
                    yield {
                        "content": "",
                        "thinking": "",
                        "tool_call_deltas": [{
                            "index": tc_idx,
                            "id": tool_id,
                            "function_name_delta": tool_name,
                            "function_arguments_delta": "",
                        }],
                    }
            elif etype == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta:
                    dtype = getattr(delta, "type", "")
                    if dtype == "text_delta":
                        yield {
                            "content": getattr(delta, "text", "") or "",
                            "thinking": "",
                            "tool_call_deltas": [],
                        }
                    elif dtype == "thinking_delta":
                        yield {
                            "content": "",
                            "thinking": getattr(delta, "thinking", "") or "",
                            "tool_call_deltas": [],
                        }
                    elif dtype == "input_json_delta":
                        partial = getattr(delta, "partial_json", "") or ""
                        block_idx = getattr(event, "index", 0)
                        tc_info = active_tools.get(block_idx)
                        tc_idx = tc_info["tool_call_index"] if tc_info else 0
                        if tc_info:
                            tc_info["args_emitted"] = True
                        yield {
                            "content": "",
                            "thinking": "",
                            "tool_call_deltas": [{
                                "index": tc_idx,
                                "id": "",
                                "function_name_delta": "",
                                "function_arguments_delta": partial,
                            }],
                        }
            elif etype == "content_block_stop":
                block_idx = getattr(event, "index", 0)
                tc_info = active_tools.get(block_idx)
                if tc_info and not tc_info.get("args_emitted"):
                    yield {
                        "content": "",
                        "thinking": "",
                        "tool_call_deltas": [{
                            "index": tc_info["tool_call_index"],
                            "id": "",
                            "function_name_delta": "",
                            "function_arguments_delta": "{}",
                        }],
                    }
            elif etype == "message_delta":
                delta = getattr(event, "delta", None)
                stop_reason = getattr(delta, "stop_reason", None)
                truncated = (stop_reason == "max_tokens")
                usage = getattr(event, "usage", None)
                if usage:
                    completion_tokens = getattr(usage, "output_tokens", 0) or 0
                yield {
                    "content": "",
                    "thinking": "",
                    "tool_call_deltas": [],
                    "truncated": truncated,
                    "usage": {
                        "prompt": prompt_tokens,
                        "completion": completion_tokens,
                        "total": prompt_tokens + completion_tokens,
                        "cached": cached_tokens,
                    },
                }

    def sync_chat(self, model, messages, temperature=0.3, json_format=False, timeout=None) -> str:
        client = self._get_client()
        model = self._request_model(model)
        system_param, conv_messages = self._convert_messages(messages)
        kwargs = {
            "model": model,
            "messages": conv_messages,
            "max_tokens": 4096,
        }
        if system_param:
            if json_format:
                if isinstance(system_param, list):
                    kwargs["system"] = system_param + [{"type": "text", "text": "Respond with valid JSON only."}]
                else:
                    kwargs["system"] = f"{system_param}\nRespond with valid JSON only."
            else:
                kwargs["system"] = system_param
        elif json_format:
            kwargs["system"] = "Respond with valid JSON only."

        if temperature is not None and "temperature" in self._sdk_accepts():
            kwargs["temperature"] = temperature
        if timeout is not None:
            kwargs["timeout"] = timeout

        def _call():
            try:
                resp = client.messages.create(**kwargs)
                parts = []
                for block in resp.content:
                    if getattr(block, "type", "") == "text":
                        parts.append(getattr(block, "text", ""))
                return "".join(parts)
            except self._anthropic.APIStatusError as e:
                self._handle_api_status_error(e)
            except Exception as e:
                raise ProviderError(f"CLAUDE error: {e}", retryable=_is_transient_stream_error(e), original_error=e)

        return with_retry(_call, max_retries=2)

    def _guarded_stream(self, response_stream):
        try:
            for chunk in response_stream:
                yield chunk
        except self._anthropic.APIStatusError as e:
            self._handle_api_status_error(e)
        except Exception as e:
            raise ProviderError(
                _friendly_stream_error(self.name, e),
                retryable=_is_transient_stream_error(e),
                original_error=e,
            )

    def _handle_api_status_error(self, e):
        status = getattr(e, "status_code", 0)
        if status == 401:
            raise ProviderError(
                "Claude: Invalid API key. Get one at https://console.anthropic.com and use /provider or /key to update it.",
                original_error=e,
            )
        elif status == 402:
            raise ProviderError(
                "Claude: Insufficient balance or credits. Top up at https://console.anthropic.com/settings/billing",
                original_error=e,
            )
        elif status == 404:
            raise ProviderError(
                "Claude: модель не найдена (404). Выберите существующую командой /model — "
                "список берётся прямо из Anthropic.",
                original_error=e,
            )
        elif status == 429:
            raise ProviderError(
                "Claude: Rate limited. Slow down or check your limits.",
                retryable=True,
                original_error=e,
            )
        elif status == 529:
            raise ProviderError(
                "Claude: Anthropic API is temporarily overloaded (529). Retrying...",
                retryable=True,
                original_error=e,
            )
        elif status >= 500:
            raise ProviderError(
                f"Claude: Server error ({status}). Retrying...",
                retryable=True,
                original_error=e,
            )
        elif status == 400 and "workspace" in str(getattr(e, "message", None) or e).lower():
            raise ProviderError(
                "Claude: этот API-ключ не привязан к рабочему пространству (workspace). Укажите его ID: "
                "/key workspace <ID> (или переменная ANTHROPIC_WORKSPACE_ID), либо используйте ключ, "
                "созданный внутри workspace.",
                original_error=e,
            )
        else:
            msg = getattr(e, "message", None) or str(e)
            raise ProviderError(
                f"Claude error (HTTP {status}): {msg}",
                retryable=status >= 500,
                original_error=e,
            )


def create_provider(provider_name: str = None) -> LLMProvider:
    """Build a provider. Defaults to the configured main provider; pass a name
    to build a specific one (used for the auxiliary service model).

    Whatever runs on this machine comes back governed: its requests wait in the model server's lane (see
    src/agent/governor.py). Cloud models pass through untouched."""
    from src.agent.governor import govern
    return govern(_build_provider(provider_name))


def _build_provider(provider_name: str = None) -> LLMProvider:
    from config import (
        get_provider as _get_provider_name,
        get_zai_api_key,
        get_zai_endpoint,
        get_koboldcpp_url,
        get_openrouter_api_key,
        get_openrouter_url,
        get_xkiro_api_key,
        get_xkiro_url,
        get_wormsoft_api_key,
        get_wormsoft_url,
        get_deepseek_api_key,
        get_deepseek_url,
        get_claude_api_key,
        get_claude_url,
        get_claude_workspace_id,
    )
    name = provider_name or _get_provider_name()
    if name == "zai":
        return ZAIProvider(api_key=get_zai_api_key(), base_url=get_zai_endpoint())
    elif name == "koboldcpp":
        return KoboldCPPProvider(base_url=get_koboldcpp_url())
    elif name == "openrouter":
        return OpenRouterProvider(api_key=get_openrouter_api_key(), base_url=get_openrouter_url())
    elif name == "xkiro":
        return XKiroProvider(api_key=get_xkiro_api_key(), base_url=get_xkiro_url())
    elif name == "wormsoft":
        return WormSoftProvider(api_key=get_wormsoft_api_key(), base_url=get_wormsoft_url())
    elif name == "deepseek":
        return DeepSeekProvider(api_key=get_deepseek_api_key(), base_url=get_deepseek_url())
    elif name in ("claude", "anthropic"):
        return ClaudeProvider(api_key=get_claude_api_key(), base_url=get_claude_url(),
                              workspace_id=get_claude_workspace_id())
    elif name == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unknown provider: {name}")


def starting_model(provider_name: str, *candidates) -> Optional[str]:
    """The model to start a provider with: the first candidate that provider recognises, else its
    own default, else the first candidate as given.

    Switching to a provider used to inherit the previous provider's model name — Claude was handed
    "nvidia/nemotron-3-ultra-550b-a55b:free" and answered every request with a 404."""
    try:
        provider = create_provider(provider_name)
    except Exception:
        return next((c for c in candidates if c), None)
    for candidate in candidates:
        if candidate:
            try:
                resolved = provider.resolve_model(candidate)
            except Exception:
                resolved = candidate
            if resolved:
                return resolved
    try:
        fallback = provider.default_model()
    except Exception:
        fallback = None
    return fallback or next((c for c in candidates if c), None)


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
