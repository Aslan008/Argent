import os
import re
import threading
from typing import List, Dict, Any
from memory_manager import memory
from providers import create_provider, ProviderError
from logger import get_logger
from config import get_provider, get_context_window, get_model_size_category, get_strip_reasoning

# Was logging.getLogger("argent.agent.trimmer") — a name nothing ever attaches a
# handler to, so every record below vanished. That is the one place you would
# look to find out why a compacted session forgot the project: whether the
# summary was written, timed out, or was skipped for a hard reset.
log = get_logger("trimmer")

# estimate_tokens is on the hot path: _trim_history and get_context_usage call
# it for EVERY message EVERY turn. Without a cache that meant one HTTP request
# per message to the provider's tokenize endpoint \u2014 while the same provider was
# busy generating. The cache makes repeat estimates free; the circuit breaker
# stops querying an endpoint after its first failure.
_TOKEN_CACHE: Dict[tuple, int] = {}
_TOKEN_CACHE_MAX = 4096
_API_TOKENIZE_BROKEN: set = set()
_API_MAX_CHARS = 16000  # beyond this size heuristic precision is plenty


def _heuristic_tokens(text: str) -> int:
    # Cyrillic text is less compact (approx 1.8 chars/token)
    has_cyrillic = any(u'\u0400' <= char <= u'\u04FF' for char in text)
    if has_cyrillic:
        return int(len(text) / 1.8)
    return len(text) // 4


def _query_tokenize_api(text: str, model_name: str, provider: str) -> int | None:
    """Ask the provider to tokenize. Marks the provider broken on any failure
    so the session never waits on a dead endpoint twice."""
    try:
        import requests
        if provider == "ollama":
            ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            resp = requests.post(
                f"{ollama_host}/api/tokenize",
                json={"model": model_name, "prompt": text},
                timeout=1.0
            )
        elif provider == "koboldcpp":
            from config import get_koboldcpp_url
            base_url = get_koboldcpp_url()
            if base_url.endswith("/v1"):
                api_url = base_url[:-3] + "/api/v1/tokenize"
            elif base_url.endswith("/v1/"):
                api_url = base_url[:-4] + "/api/v1/tokenize"
            else:
                api_url = base_url.rstrip("/") + "/api/v1/tokenize"
            resp = requests.post(api_url, json={"prompt": text}, timeout=1.0)
        else:
            return None
        if resp.status_code == 200:
            return len(resp.json().get("tokens", []))
    except Exception:
        pass
    _API_TOKENIZE_BROKEN.add(provider)
    return None


def estimate_tokens(text: str, model_name: str, provider: str) -> int:
    """Calculate tokens accurately, querying the provider API if available,
    or falling back to tiktoken / a language-aware heuristic. Cached."""
    if not text:
        return 0

    key = (provider, model_name, len(text), hash(text))
    cached = _TOKEN_CACHE.get(key)
    if cached is not None:
        return cached

    count = None
    if provider not in _API_TOKENIZE_BROKEN and len(text) <= _API_MAX_CHARS:
        count = _query_tokenize_api(text, model_name, provider)

    if count is None:
        try:
            import tiktoken
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
            count = len(encoding.encode(text))
        except ImportError:
            count = _heuristic_tokens(text)

    if len(_TOKEN_CACHE) >= _TOKEN_CACHE_MAX:
        _TOKEN_CACHE.clear()
    _TOKEN_CACHE[key] = count
    return count


# What one image actually costs a vision model, to the right order of
# magnitude. The number that matters is that it is a CONSTANT: an attached
# picture is a fixed budget of visual tokens, not the length of its base64.
IMAGE_TOKEN_COST = 1400


def estimate_message_tokens(message, model_name: str, provider: str) -> int:
    """Tokens for one message, with images counted as images.

    Everything used to measure `str(message)`, which for an attachment means
    counting every base64 character. Measured on a real 1568px screenshot:
    192 KB of base64 estimated at 139,242 tokens against a true cost near
    1,400 — off by ninety times. The context meter read 300k after a single
    request, and worse, the SAME estimate drives trimming, so one screenshot
    would evict the entire conversation to make room for a cost that was never
    real.
    """
    if not isinstance(message, dict):
        return estimate_tokens(str(message), model_name, provider)

    images = message.get("images") or []
    if not images:
        return estimate_tokens(str(message), model_name, provider)

    without = {k: v for k, v in message.items() if k != "images"}
    return (estimate_tokens(str(without), model_name, provider)
            + IMAGE_TOKEN_COST * len(images))


def summarize_messages(msgs_to_summarize: List[Dict], model_name: str, timeout: float = 30.0) -> str:
    """Runs a synchronous LLM call to summarize old context.
    Protected by a timeout to prevent infinite blocking.
    """
    text_to_summarize = ""
    for m in msgs_to_summarize:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if not content and "tool_calls" in m:
            content = f"[Tool Calls: {m['tool_calls']}]"
        text_to_summarize += f"{role.upper()}: {content}\n\n"
        
    # A hard "1-2 paragraphs" cap optimised for brevity before completeness,
    # so whatever did not fit was lost for the rest of the session — and what
    # got dropped first was the open-problem list, the one thing the agent
    # cannot reconstruct by re-reading the code. Named buckets keep the
    # unrecoverable facts and let the recoverable ones go.
    summary_prompt = (
        "You are a context-compression engine. The conversation below is about to be "
        "dropped from the agent's context; your summary is all that survives.\n\n"
        "Capture completely, then compress:\n"
        "- **Goal**: what the user is trying to achieve, in their own terms.\n"
        "- **Decisions**: choices made and WHY, especially ones already rejected — "
        "without the reason, they get re-litigated.\n"
        "- **Changed**: files created/edited and what changed in each.\n"
        "- **Open**: bugs, failing tests and unfinished work, with the exact error text. "
        "This is the section that must not be shortened.\n"
        "- **Learned**: facts about this codebase discovered the hard way "
        "(commands, paths, gotchas) that a re-read would not reveal.\n\n"
        "Omit pleasantries, narration and tool output that can be re-fetched. "
        "Write nothing under a heading that has no content.\n\n"
        "Here is the log:\n\n" + text_to_summarize
    )
    
    result_container = [None]
    error_container = [None]
    
    def _do_summarize():
        try:
            # Use the auxiliary model when configured (a cheap/local model is
            # plenty for summarization), falling back to the main model.
            from providers import create_service_provider
            provider, svc_model = create_service_provider()
            validation_error = provider.validate_config()
            if validation_error:
                result_container[0] = None
                return
            # Pass the deadline down to the HTTP layer so a hung remote provider
            # ends the call (and this worker thread) instead of lingering past
            # the join() timeout below. Slightly under `timeout` so the network
            # gives up first.
            result_container[0] = provider.sync_chat(
                model=svc_model,
                messages=[{"role": "user", "content": summary_prompt}],
                timeout=max(1.0, timeout - 1.0),
            )
        except Exception as e:
            error_container[0] = e
    
    worker = threading.Thread(target=_do_summarize, daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    
    if worker.is_alive():
        log.warning("Context summarization timed out after %.0fs.", timeout)
        return None
    
    if error_container[0]:
        log.warning("Context summarization error: %s", error_container[0])
        return None
    
    return result_container[0] or "[Compression Failed]"


def update_memory_from_messages(messages: List[Dict[str, Any]]):
    """Extract key information from recent messages before clearing them."""
    last_user_msg = ""
    last_assistant_action = ""

    for m in reversed(messages):
        if m.get("role") == "user" and not last_user_msg:
            content = m.get("content", "")
            if content and not content.startswith("/"):
                last_user_msg = content
        elif m.get("role") == "assistant" and not last_assistant_action:
            content = m.get("content", "")
            if content:
                first_line = content.strip().split("\n")[0][:200]
                last_assistant_action = first_line

    if last_user_msg and not memory.data.get("objective"):
        memory.set_objective(last_user_msg)

    if last_assistant_action:
        memory.set_current_task(last_assistant_action)


def hard_reset_with_memory(messages: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    """Full context reset. Preserves system prompt and injects structured memory note."""
    update_memory_from_messages(messages)
    context_note = memory.build_context_note()

    system_content = messages[0]["content"]
    if context_note:
        sep = "\n\n=== PERSISTENT MEMORY (context was reset) ===\n"
        end = "\n=== END MEMORY ==="
        existing = system_content.find("=== PERSISTENT MEMORY")
        if existing != -1:
            end_marker = system_content.find("=== END MEMORY ===", existing)
            if end_marker != -1:
                system_content = system_content[:existing] + sep + context_note + end + system_content[end_marker + len("=== END MEMORY ==="):]
            else:
                system_content += sep + context_note + end
        else:
            system_content += sep + context_note + end
            
    last_user_msg = messages[-1] if messages and messages[-1].get("role") == "user" else None

    new_messages = [{"role": "system", "content": system_content}]
    if last_user_msg:
        new_messages.append(last_user_msg)
        
    log.info("Hard context reset performed (model=%s, category=%s)", model_name, get_model_size_category(model_name))
    return new_messages


def soft_trim_with_summarization(messages: List[Dict[str, Any]], model_name: str, max_history_messages: int, max_context_tokens: int) -> List[Dict[str, Any]]:
    """LLM-based summarization for history trimming."""
    pinned_indices = {0}
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            content = m.get("content", "")
            if any(x in content for x in ["PROJECT SPECIFICATION", "ARCHITECTURE MAP", "PREVIOUS CONTEXT MEMORY", "PERSISTENT MEMORY"]):
                pinned_indices.add(i)

    msgs_to_summarize = []
    indices_to_drop = []

    target_messages = max_history_messages // 2
    target_tokens = int(max_context_tokens * 0.5)

    current_tokens = sum(estimate_message_tokens(m, model_name, get_provider()) for m in messages[1:])
    for i in range(1, len(messages)):
        if i in pinned_indices:
            continue
        if len(messages) - len(indices_to_drop) <= target_messages and current_tokens <= target_tokens:
            break
        msgs_to_summarize.append(messages[i])
        indices_to_drop.append(i)
        current_tokens -= estimate_message_tokens(messages[i], model_name, get_provider())

    if not msgs_to_summarize:
        return messages

    from ui import console
    with console.status("[dim magenta]Оптимизация контекста...[/dim magenta]", spinner="dots"):
        summary = summarize_messages(msgs_to_summarize, model_name, timeout=120.0)
    
    if summary is None:
        log.info("Summarization failed, falling back to hard reset.")
        return hard_reset_with_memory(messages, model_name)
    
    memory_msg = {
        "role": "system",
        "content": f"=== PREVIOUS CONTEXT MEMORY ===\n{summary}\n=== END MEMORY ==="
    }

    new_messages = []
    memory_injected = False
    for i, m in enumerate(messages):
        if i in indices_to_drop:
            if not memory_injected:
                new_messages.append(memory_msg)
                memory_injected = True
            continue
        new_messages.append(m)

    return new_messages


def clean_messages_for_llm(messages: List[Dict[str, Any]], strip_reasoning: bool) -> List[Dict[str, Any]]:
    """Creates a clean copy of messages for token estimation or LLM API calls.
    - If strip_reasoning is False and a message has a 'thinking' block, wraps it in <think>...</think>
      and prepends/appends it to the 'content' so the model can see its past thoughts.
    - If strip_reasoning is True, strips <think>...</think> and unclosed <think>... from 'content'.
    - Removes the custom 'thinking' key to ensure OpenAI API schema compatibility.
    """
    cleaned = []
    for m in messages:
        m_copy = m.copy()
        
        # Always remove custom 'thinking' key for API compatibility,
        # but capture it first
        thinking = m_copy.pop("thinking", None)
        
        if m_copy.get("role") == "assistant":
            content = m_copy.get("content", "") or ""
            
            # If we want to keep reasoning, format it back into the content
            if not strip_reasoning and thinking:
                # Only add if it's not already in the content
                think_block = f"<think>\n{thinking.strip()}\n</think>"
                if think_block not in content:
                    content = f"{think_block}\n\n{content}".strip()
                m_copy["content"] = content
                
            elif strip_reasoning:
                # Strip reasoning blocks if enabled
                if content:
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                    content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
                    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
                    
                    content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
                    content = re.sub(r'<thought>.*', '', content, flags=re.DOTALL)
                    content = re.sub(r'<reasoning>.*', '', content, flags=re.DOTALL)
                    m_copy["content"] = content.strip()
                    
        cleaned.append(m_copy)

    # Last stop before the request is serialised and encoded. A lone surrogate
    # anywhere in the history makes the HTTP body unencodable, so the turn dies
    # with "'utf-8' codec can't encode ... surrogates not allowed" and so does
    # every turn after it — the chat stays bricked until /clear. Repairing on
    # the way out means an ALREADY poisoned history heals itself, which fixing
    # the producer alone cannot do.
    from text_safety import repair_structure
    return repair_structure(cleaned)


def compact_tool_history(messages: List[Dict[str, Any]], keep_recent: int = 2,
                         stub_over: int = 200) -> List[Dict[str, Any]]:
    """Shrink OLD tool-result messages to short stubs, keeping the last
    `keep_recent` full (they're usually the relevant ones).

    A single kept turn can still blow the context when it carries a few big tool
    dumps (a full file read, a long diff). Stubbing the older ones often makes it
    fit — a graceful step before nuking the whole history via hard reset. Returns
    the same list object unchanged when there's nothing worth compacting."""
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    to_stub = set(tool_indices[:-keep_recent]) if len(tool_indices) > keep_recent else set()
    if not to_stub:
        return messages

    changed = False
    out = []
    for i, m in enumerate(messages):
        content = m.get("content", "") or ""
        if i in to_stub and len(content) > stub_over:
            mc = m.copy()
            mc["content"] = content[:150] + f"\n…[tool output trimmed — {len(content)} chars]"
            out.append(mc)
            changed = True
        else:
            out.append(m)
    return out if changed else messages


def sliding_window_trim(messages: List[Dict[str, Any]], model_name: str,
                        max_history_messages: int, max_context_tokens: int) -> List[Dict[str, Any]]:
    """Keeps the last N complete chat turns, falling back to fewer turns or hard reset if they exceed context limits."""
    provider = get_provider()
    strip_enabled = get_strip_reasoning()
    
    # Estimate size of current messages (cleaned)
    cleaned_messages = clean_messages_for_llm(messages, strip_enabled)
    current_tokens = sum(estimate_message_tokens(m, model_name, provider) for m in cleaned_messages[1:])
    
    if len(messages) - 1 <= max_history_messages and current_tokens <= max_context_tokens:
        return messages

    # Find the indices of all 'user' messages (excluding the system message at index 0)
    user_indices = [i for i, m in enumerate(messages) if i > 0 and m.get("role") == "user"]
    if not user_indices:
        return hard_reset_with_memory(messages, model_name)

    # We try keeping 3 turns, then 2, then 1
    for turns in [3, 2, 1]:
        if turns > len(user_indices):
            continue
        
        target_user_idx = user_indices[-turns]
        candidate = [messages[0]] + messages[target_user_idx:]
        
        # Estimate candidate token size using cleaned versions
        cleaned_candidate = clean_messages_for_llm(candidate, strip_enabled)
        candidate_tokens = sum(estimate_message_tokens(m, model_name, provider) for m in cleaned_candidate[1:])

        if len(candidate) - 1 <= max_history_messages and candidate_tokens <= (max_context_tokens * 0.75):
            log.info("Sliding window trim successful, keeping last %d turns (%d messages)", turns, len(candidate))
            return candidate

        # Too large by tokens — try stubbing older tool dumps within this window
        # before giving up on it (a big file read shouldn't cost the whole turn).
        compacted = compact_tool_history(candidate)
        if compacted is not candidate:
            cc = clean_messages_for_llm(compacted, strip_enabled)
            ctoks = sum(estimate_message_tokens(m, model_name, provider) for m in cc[1:])
            if len(compacted) - 1 <= max_history_messages and ctoks <= (max_context_tokens * 0.75):
                log.info("Kept last %d turns after compacting old tool outputs (%d messages)", turns, len(compacted))
                return compacted

    # Fallback to hard reset if even 1 turn is too large
    log.info("Sliding window trim failed to fit context, falling back to hard reset.")
    return hard_reset_with_memory(messages, model_name)

