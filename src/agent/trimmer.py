import os
import re
import threading
import time
from typing import List, Dict, Any, Optional
from memory_manager import memory
from providers import create_provider, ProviderError
from logger import get_logger
from src.agent import compaction, summary_precompute
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
_TOKEN_CACHE_LOCK = threading.Lock()
_API_TOKENIZE_BROKEN: set = set()
_API_MAX_CHARS = 16000  # beyond this size heuristic precision is plenty


def _heuristic_tokens(text: str) -> int:
    if not text:
        return 0
    # Proportionally weigh Cyrillic (less compact, ~1.8 chars/tok) and Latin/other (~4.0 chars/tok)
    # to avoid skewing large codebases when a single Cyrillic comment is present.
    cyrillic_chars = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
    other_chars = len(text) - cyrillic_chars
    return int(cyrillic_chars / 1.8 + other_chars / 4.0)


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
    with _TOKEN_CACHE_LOCK:
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
        except Exception:
            count = _heuristic_tokens(text)

    with _TOKEN_CACHE_LOCK:
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


# ─── Progressive Distillation v3 ─────────────────────────────────────────
# Three-level context compression: In-Flight Aging → Distillation → Rehydration
# See implementation_plan.md for full architecture rationale.
# ─────────────────────────────────────────────────────────────────────────

import json as _json

AGE_THRESHOLD = 6  # tool results older than this many turns get aged


def _is_substantive(text: str) -> bool:
    """Is this a real user task, or just a confirmation / auto-impulse?"""
    return compaction.is_substantive(text)


def _find_active_goal(messages: List[Dict], active_window_start: int) -> str | None:
    """Find the LAST substantive user message OUTSIDE the active window.

    This is the "floating root contract": always the most recent real task,
    not a stale first message. Returns None when all substantive messages
    are already inside the active window (visible to the model verbatim)."""
    best = None
    for i in range(1, active_window_start):
        m = messages[i]
        if m.get("role") != "user":
            continue
        content = (m.get("content") or "").strip()
        if _is_substantive(content):
            best = content
    if best and len(best) > 500:
        best = best[:500] + "…"
    return best


def _find_active_window_start(messages: List[Dict], keep_cycles: int = 3,
                               model_name: str = "",
                               token_budget: int = 0) -> int:
    """Return the index where the protected active window begins.

    A 'cycle' runs from one user message to the next. Auto-impulses and
    heartbeat messages are not cycle boundaries.  When *keep_cycles* full
    cycles exceed *token_budget*, the window shrinks (3 → 2 → 1)."""
    user_indices = []
    for i, m in enumerate(messages):
        if i == 0:
            continue
        if m.get("role") == "user":
            content = (m.get("content") or "").strip()
            if not content.startswith(("[Режим Автоматизма]", "[Heartbeat")):
                user_indices.append(i)

    if not user_indices:
        # If there are no non-automated user messages (e.g. all turns are automated or heartbeat wakeups),
        # fall back to any user message boundaries or protect only the last few messages.
        all_user_indices = [i for i, m in enumerate(messages) if i > 0 and m.get("role") == "user"]
        if all_user_indices:
            user_indices = all_user_indices
        else:
            return max(1, len(messages) - 6)

    provider = get_provider()
    for cycles in range(keep_cycles, 0, -1):
        if cycles > len(user_indices):
            continue
        start = user_indices[-cycles]
        if token_budget > 0:
            window_tokens = sum(
                estimate_message_tokens(messages[j], model_name, provider)
                for j in range(start, len(messages))
            )
            if window_tokens > token_budget:
                continue
        return start

    # If even the last cycle exceeds token_budget, return the last user message
    # so the active window never begins on an orphaned tool or assistant message.
    return user_indices[-1]


# ── Pre-distillation ─────────────────────────────────────────────────────

#: how much text the summariser is given (characters). It used to be fixed per message — a user's 2 000-word brief
#: was cut to 500 characters before the model that had to summarise it ever saw it. Now the caps start generous
#: and tighten only as far as the budget demands.
_SUMMARIZER_BUDGET = 64000
_SUMMARIZER_LEVELS = (
    {"user": 2000, "assistant": 1200, "think": 600, "head": 300, "tail": 200, "err": 900, "key": 160},
    {"user": 1000, "assistant": 700, "think": 300, "head": 200, "tail": 120, "err": 500, "key": 100},
    {"user": 500, "assistant": 350, "think": 150, "head": 120, "tail": 80, "err": 250, "key": 80},
)
_ERROR_MARKERS = ("error", "traceback", "failed", "exception", "ошибка", "fatal", "denied", "not found")


def _render_for_summarizer(messages: List[Dict], caps: Dict[str, int]) -> List[str]:
    parts = []
    for m in messages:
        role = m.get("role", "unknown")
        content = compaction.message_text(m)

        if role == "tool":
            # command output and file bodies are recoverable (the archive keeps them whole); errors are not
            # worth losing, so they keep more
            low = content[:4000].lower()
            head, tail = ((caps["err"], caps["err"] // 2) if any(k in low for k in _ERROR_MARKERS)
                          else (caps["head"], caps["tail"]))
            if len(content) > head + tail + 50:
                content = f"{content[:head]}\n…[{len(content)} chars]\n{content[-tail:]}".strip()

        elif role == "assistant":
            tool_calls = m.get("tool_calls", [])
            if tool_calls:
                calls_desc = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = _json.loads(args)
                        except Exception:
                            args = {}
                    key = (args.get("file_path") or args.get("command")
                           or args.get("pattern") or "")
                    calls_desc.append(f"{name}({str(key)[:caps['key']]})")
                thinking = content[:caps["think"]] + "…" if len(content) > caps["think"] else content
                content = f"{thinking}\n→ [{'; '.join(calls_desc)}]"
            elif len(content) > caps["assistant"]:
                content = content[:caps["assistant"]] + "…"

        elif role == "user":
            if len(content) > caps["user"]:
                content = content[:caps["user"]] + "…"

        parts.append(f"{role.upper()}: {content}")
    return parts


def _prepare_for_summarizer(messages: List[Dict], budget_chars: int = _SUMMARIZER_BUDGET) -> str:
    """Dehydrate messages BEFORE sending to the LLM summariser.

    Strips recoverable content (file bodies, long command output) while keeping non-recoverable facts
    (decisions, errors, plans, what the person asked). The caps tighten level by level until the text fits
    `budget_chars`; if even the tightest does not, the middle of the conversation is dropped, not its ends."""
    for caps in _SUMMARIZER_LEVELS:
        parts = _render_for_summarizer(messages, caps)
        text = "\n\n".join(parts)
        if len(text) <= budget_chars:
            return text

    head_n = max(1, len(parts) // 8)
    used = sum(len(p) + 2 for p in parts[:head_n]) + 100
    tail: List[str] = []
    for p in reversed(parts[head_n:]):
        if used + len(p) + 2 > budget_chars:
            break
        tail.insert(0, p)
        used += len(p) + 2
    dropped = len(parts) - head_n - len(tail)
    marker = [f"[… пропущено сообщений из середины: {dropped} — они целиком в журнале …]"] if dropped else []
    return "\n\n".join(parts[:head_n] + marker + tail)


# ── LLM summarisation ────────────────────────────────────────────────────

def _summarize_with_llm(pre_distilled_text: str,
                         active_goal: str | None,
                         prior_context: str,
                         model_name: str,
                         timeout: float = 120.0,
                         ui=None) -> str | None:
    """Call the service LLM with a structured compression prompt.

    Returns the summary text, or None when no model could write one. A busy model is asked
    again, an exhausted free OpenRouter model is swapped for another (by itself or by the
    user's choice), and a model that dies mid-answer costs one attempt, not the session:
    see `service_llm`."""
    goal_line = (f"\nACTIVE GOAL (kept separately, do NOT repeat): {active_goal}\n"
                 if active_goal else "")

    prompt = (
        "You are a context-compression engine for an AI coding agent.\n"
        "The conversation below will be dropped from the agent's active window.\n"
        "What survives is: your summary, the user's own messages word for word (kept separately), the\n"
        "verified list of files/commands/errors (kept separately) and a file with the whole conversation.\n"
        "The agent must be able to continue working WITHOUT re-reading files, re-running commands, or\n"
        "asking the user to repeat anything.\n"
        f"{goal_line}{prior_context}\n"
        "Write a structured summary with EXACTLY these sections:\n\n"
        "## Intent\nWhat the user is trying to achieve overall, how it changed over the conversation, "
        "and what 'done' means to them.\n\n"
        "## Standing Rules\nInstructions, preferences and constraints from the user that must KEEP holding "
        "(do not commit, run only on X, answer in Russian, root fixes only...). Keep their wording. "
        "Skip if none.\n\n"
        "## Done\nWhat was accomplished. Each file changed and what changed.\n\n"
        "## Decisions\nChoices made and WHY. Especially rejected alternatives and dead ends, so they are "
        "not tried again.\n\n"
        "## Current State\n"
        "Where the work stopped. What was the agent about to do next?\n"
        "EXACT file names, line numbers, function names, running processes and what they are doing.\n"
        "THIS IS THE MOST CRITICAL SECTION.\n\n"
        "## Pending\nEverything the user asked for that is NOT finished yet (in the order asked), open "
        "questions, and the single next step. Skip if nothing is pending.\n\n"
        "## Errors\nUnresolved errors with exact text. Skip if none.\n\n"
        "## Key Facts\nExact identifiers needed to continue: paths, function/class names, commands, URLs, "
        "ids, numbers, config keys, and non-obvious project facts. Skip if none.\n\n"
        "Rules:\n"
        "- Be specific: file names, line numbers, variable names, exact values.\n"
        "- Skip content re-fetchable from disk.\n"
        "- Do NOT copy the user's messages; they are kept verbatim elsewhere. Do summarise what they ask.\n"
        "- If a PREVIOUS SUMMARY is given, update it: keep what is still true, add what is new, drop what "
        "is superseded. Never lose an earlier pending task or rule unless it was completed or withdrawn.\n"
        "- 'Current State' must tell the agent exactly what to do next.\n"
        "- Write NOTHING under empty headings.\n"
        "- Preserve the primary language used in the conversation log (e.g. if the conversation is in Russian, write the summary in Russian).\n\n"
        "Conversation log:\n\n" + pre_distilled_text
    )

    from src.agent import service_llm

    result = service_llm.run(prompt, per_call_timeout=timeout, ui=ui)
    if result.text is None:
        log.warning("Context summarization produced nothing (%s).", result.reason or "no reason given")
    return result.text


# Keep the old name as an alias so callers outside this file still work.
def summarize_messages(msgs_to_summarize: List[Dict], model_name: str,
                       timeout: float = 30.0) -> str | None:
    """Legacy wrapper — delegates to the new pre-distillation pipeline."""
    pre_distilled = _prepare_for_summarizer(msgs_to_summarize)
    return _summarize_with_llm(pre_distilled, None, "", model_name, timeout)


# ── Deterministic fact extraction (zero LLM tokens) ──────────────────────

def _extract_deterministic_facts(messages: List[Dict], prior: "compaction.Facts | None" = None,
                                 include_requests: bool = True) -> str:
    """Algorithmically extract structured facts from message history.

    Parses tool_calls and tool results to build a file/command/error summary, merged with the facts of earlier
    compactions (`prior`) so the second compaction does not forget what the first one knew.
    Cost: 0 LLM tokens — pure string parsing."""
    facts = compaction.collect_facts(messages)
    if prior is not None:
        facts = compaction.merge_facts(prior, facts)
    return compaction.render_facts(facts, include_requests=include_requests)


# ── File rehydration ─────────────────────────────────────────────────────

def _rehydrate_files(messages: List[Dict], max_files: int = 3,
                     max_chars_per_file: int = 6000) -> str:
    """Re-read the most recently edited files from disk.

    After compression the model loses the file contents it was working with.
    Rehydration restores the *current on-disk state* of the last few edited
    files so the model can continue editing without a redundant read_file.
    When fewer than `max_files` were edited, up to two of the most recently READ files fill the
    remaining places (the file just studied is what the next step most often needs again).
    Cost: 0 LLM tokens — only disk I/O."""
    edited_files: list = []
    read_files: list = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls", []):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            if name not in ("write_file", "replace_in_file", "read_file"):
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except Exception:
                    try:
                        args = _json.loads(args.replace('\\', '/'))
                    except Exception:
                        continue
            fp = args.get("file_path") or "" if isinstance(args, dict) else ""
            if fp:
                bucket = read_files if name == "read_file" else edited_files
                if fp in bucket:
                    bucket.remove(fp)
                bucket.append(fp)

    if not edited_files:
        try:
            edited_files = list(memory.data.get("files_modified", []))
        except Exception:
            pass

    recent = edited_files[-max_files:]
    room = min(2, max_files - len(recent))
    if room > 0:
        recent = [f for f in read_files if f not in edited_files][-room:] + recent
    parts = []
    for fp in recent:
        try:
            abs_path = os.path.abspath(fp)
            if not os.path.isfile(abs_path):
                continue
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > max_chars_per_file:
                half = max_chars_per_file // 2
                content = (content[:half]
                           + f"\n\n… [{len(content) - max_chars_per_file} символов пропущено] …\n\n"
                           + content[-half:])
            basename = os.path.basename(fp)
            parts.append(f"### {basename}\n```\n{content}\n```")
        except Exception as e:
            log.warning("Rehydration failed for %s: %s", fp, e)

    return "\n\n".join(parts) if parts else ""


# ── Compact boundary marker ──────────────────────────────────────────────

# ── Compaction metrics and the thrash guard ─────────────────────────────

from src.agent.compaction_log import (  # noqa: E402,F401  (re-exported: callers and tests use trimmer.<name>)
    THRASH_ADVICE, THRASH_MIN_SAVING, THRASH_RUNS, THRASH_WINDOW_SECONDS, _COMPACTIONS,
    describe_compaction, is_thrashing, last_compaction, record_compaction, reset_compaction_history,
)


def _count_tokens(messages: List[Dict], model_name: str) -> int:
    try:
        provider = get_provider()
        return sum(estimate_message_tokens(m, model_name, provider) for m in messages)
    except Exception:
        return 0


def _announce_compaction(entry: Dict[str, Any], thrashing: bool) -> None:
    try:
        from ui import print_system
        print_system(f"[dim]{describe_compaction(entry)}[/dim]")
        if thrashing:
            print_system(f"[yellow]{THRASH_ADVICE}[/yellow]")
    except Exception:
        pass


# ── Compact boundary marker ──────────────────────────────────────────────

def _write_compact_boundary(messages_before: List[Dict],
                            messages_after: List[Dict],
                            compression_id: int,
                            model_name: str = "", kind: str = "auto",
                            pre_tokens: Optional[int] = None):
    """Write a marker to .argent/compact_log.jsonl for future /rewind, with the token counts."""
    pre = pre_tokens if pre_tokens is not None else _count_tokens(messages_before, model_name)
    post = _count_tokens(messages_after, model_name)
    metrics = record_compaction(kind, compression_id, pre, post, model_name)
    try:
        from datetime import datetime
        from project_paths import project_root_or_cwd
        from pathlib import Path

        root = project_root_or_cwd() or Path.cwd()
        log_file = root / ".argent" / "compact_log.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "type": "compact_boundary",
            "compression_id": compression_id,
            "timestamp": datetime.now().isoformat(),
            "messages_before": len(messages_before),
            "messages_after": len(messages_after),
            "messages_dropped": len(messages_before) - len(messages_after),
            "kind": kind,
            "pre_tokens": metrics["pre_tokens"],
            "post_tokens": metrics["post_tokens"],
            "saving": metrics["saving"],
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("Could not write compact boundary: %s", e)


# ── Distillate assembly ──────────────────────────────────────────────────

def _get_compression_count(messages: List[Dict]) -> int:
    """How many compressions have already happened in this session."""
    count = 0
    for m in messages:
        if "CONTEXT DISTILLATE (compression #" in m.get("content", ""):
            count += 1
    return count


def _ledger_budget(context_tokens: int) -> int:
    """Characters the user ledger may take: a share of the window, so a small model is not buried by it."""
    return int(max(1200, min(compaction.LEDGER_BUDGET, context_tokens * 4 * 0.06)))


def _file_budget(context_tokens: int) -> int:
    return int(max(1500, min(6000, context_tokens * 4 * 0.05)))


def _latest_distillate(messages: List[Dict]) -> str | None:
    """The body of the newest CONTEXT DISTILLATE system message (regular or emergency), if any."""
    found = None
    for m in messages:
        content = compaction.message_text(m)
        if m.get("role") == "system" and content.startswith("=== CONTEXT DISTILLATE"):
            start_marker = content.find("===\n")
            end_marker = content.rfind("\n=== END")
            if start_marker > -1 and end_marker > start_marker:
                found = content[start_marker + 4:end_marker]
    return found


def _is_distillate(m: Dict) -> bool:
    return m.get("role") == "system" and compaction.message_text(m).startswith("=== CONTEXT DISTILLATE")


def _fallback_distillate(msgs_to_compress: List[Dict], existing_distillate: str | None,
                         archive_path: str | None) -> str:
    """What can still be written when the distillation itself failed: each part on its own, so one failing
    part does not take the others with it."""
    fresh = [m for m in msgs_to_compress if not _is_distillate(m)]
    prior = compaction.split_distillate(existing_distillate)

    def part(fn, default):
        try:
            return fn()
        except Exception:
            log.exception("fallback distillate: a part failed")
            return default

    texts = part(lambda: compaction.user_texts(fresh), [])
    return compaction.assemble(
        goal=None, summary=prior.summary or None,
        archive=prior.archive + ([archive_path] if archive_path else []),
        ledger=part(lambda: compaction.fit_ledger(
            compaction.merge_ledger(prior.ledger, compaction.ledger_entries(texts))), []),
        rules=part(lambda: compaction.merge_rules(prior.rules, compaction.extract_rules(texts)), []),
        facts=part(lambda: _extract_deterministic_facts(fresh, prior.facts, include_requests=False),
                   compaction.NO_ARTIFACTS))


def _previous_summary_note(summary: str) -> str:
    """The part of the summariser's prompt that carries an earlier summary to be updated."""
    if not summary:
        return ""
    return ("\nPREVIOUS SUMMARY (update it: keep what is still true, add what is new, drop what is "
            f"superseded; do not duplicate):\n{summary}\n")


def _create_distillate(msgs_to_compress: List[Dict],
                       all_messages: List[Dict],
                       active_window_start: int,
                       existing_distillate: str | None,
                       model_name: str,
                       ui=None,
                       context_tokens: int | None = None,
                       archive_path: str | None = None) -> str:
    """Two-stage distillation: LLM summary around what needs no model.

    Stage 1 — LLM summarisation of the dropped slice, UPDATING the previous summary (not re-summarising a
              blob that carries file bodies).
    Stage 2 — everything that must not depend on a model: the user's own messages word for word, their standing
              rules, the verified facts merged across compactions, a pointer to the whole transcript, and the
              current on-disk state of the files in play (see compaction.py).

    Guarantee: ALWAYS returns useful text, even on total network failure."""
    ctx = context_tokens or get_context_window()
    active_goal = _find_active_goal(all_messages, active_window_start)
    prior = compaction.split_distillate(existing_distillate)
    # an earlier distillate is carried by `prior` (parsed, so its file bodies are not fed back in)
    fresh = [m for m in msgs_to_compress if not _is_distillate(m)]

    texts = compaction.user_texts(fresh)
    ledger = compaction.fit_ledger(
        compaction.merge_ledger(prior.ledger, compaction.ledger_entries(texts)), _ledger_budget(ctx))
    rules = compaction.merge_rules(prior.rules, compaction.extract_rules(texts))
    facts = compaction.merge_facts(prior.facts, compaction.collect_facts(fresh))
    archive = prior.archive + ([archive_path] if archive_path else [])
    rehydration = _rehydrate_files(all_messages, max_chars_per_file=_file_budget(ctx))

    llm_summary = None
    # A summary written ahead of time in the background (M1b) stands for everything it covered: only what
    # came after it is left for the model, or nothing.
    ahead = summary_precompute.claim(fresh, prior.summary)
    if ahead is not None:
        written, newer = ahead
        llm_summary = written
        if newer:
            llm_summary = _summarize_with_llm(
                _prepare_for_summarizer(newer), active_goal, _previous_summary_note(written), model_name, ui=ui
            ) or written
        log.info("compaction used the summary written ahead (%d newer messages)", len(newer))
    else:
        llm_summary = _summarize_with_llm(
            _prepare_for_summarizer(fresh), active_goal, _previous_summary_note(prior.summary), model_name, ui=ui
        )
    if llm_summary:
        summary = llm_summary
    else:
        log.warning("LLM summarization failed — keeping the previous summary and the model-free parts")
        summary = prior.summary or (
            "[Сводку написать не удалось: модель сжатия недоступна. Опирайся на дословные сообщения "
            "пользователя, факты ниже и журнал сжатых сообщений.]")

    return compaction.assemble(
        goal=active_goal, summary=summary, archive=archive, ledger=ledger, rules=rules,
        facts=compaction.render_facts(facts, include_requests=False), files=rehydration)


# ── In-Flight Aging (Level 1: Prevention) ────────────────────────────────

def _infer_tool_context(messages: List[Dict], tool_msg_index: int) -> tuple:
    """Determine the tool name and key argument for a tool-result message."""
    tool_call_id = messages[tool_msg_index].get("tool_call_id")
    for i in range(tool_msg_index - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls", []):
            if tc.get("id") == tool_call_id:
                fn = tc.get("function", {})
                name = fn.get("name", "unknown")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except Exception:
                        args = {}
                key = (args.get("file_path") or args.get("command")
                       or args.get("pattern") or args.get("directory") or "")
                return name, str(key)[:100]
        break
    return "unknown", ""


def _build_aged_stub(tool_name: str, key_arg: str, content: str) -> str:
    """Build a compact metadata replacement for an old tool result."""
    char_count = len(content)
    line_count = content.count("\n") + 1

    if tool_name == "read_file":
        return (f"[aged: read_file({key_arg}) — {line_count} строк, "
                f"{char_count} символов. Содержимое доступно на диске.]")
    elif tool_name == "grep_search":
        match_count = content.count("\n")
        return (f"[aged: grep_search({key_arg}) — "
                f"{match_count} совпадений. Можно перезапустить.]")
    elif tool_name == "run_command":
        lines = content.strip().split("\n")
        first = lines[0][:120] if lines else ""
        exit_info = ""
        for line in lines:
            if "exit code" in line.lower():
                exit_info = f" ({line.strip()[:60]})"
                break
        return (f"[aged: run_command({key_arg}) — "
                f"{line_count} строк{exit_info}. Начало: {first}]")
    elif tool_name in ("write_file", "replace_in_file"):
        first_line = content.strip().split("\n")[0][:150]
        return f"[aged: {tool_name}({key_arg}) — {first_line}]"
    else:
        first_line = content.strip().split("\n")[0][:120]
        return f"[aged: {tool_name}({key_arg}) — {char_count} chars. {first_line}]"


def age_tool_results(messages: List[Dict],
                     age_threshold: int = AGE_THRESHOLD) -> List[Dict]:
    """Replace old tool results with compact metadata stubs.

    Called on EVERY turn, before checking the token budget.  This slows
    context growth and delays the full distillation trigger by ~50%.
    Cost: 0 LLM tokens, O(n) over messages."""
    # Compute the "age" of each message in user-turns from the end.
    user_count = 0
    msg_ages: Dict[int, int] = {}
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            user_count += 1
        msg_ages[i] = user_count

    changed = False
    out = []
    for i, m in enumerate(messages):
        if (m.get("role") != "tool"
                or msg_ages.get(i, 0) < age_threshold
                or len(m.get("content", "") or "") <= 200
                or (m.get("content") or "").startswith("[aged:")):
            out.append(m)
            continue

        tool_name, key_arg = _infer_tool_context(messages, i)
        stub = _build_aged_stub(tool_name, key_arg, m["content"])

        mc = m.copy()
        mc["content"] = stub
        out.append(mc)
        changed = True

    return out if changed else messages


def _update_memory_from_messages(messages: List[Dict[str, Any]]):
    """Extract and SAVE key info from messages before compression.

    The old version (update_memory_from_messages) extracted data into local
    variables that died with the call — pure dead code.  This version
    actually persists the findings."""
    last_user_msg = ""
    last_assistant_action = ""

    for m in reversed(messages):
        if m.get("role") == "user" and not last_user_msg:
            content = m.get("content", "")
            if content and _is_substantive(content):
                last_user_msg = content
        elif m.get("role") == "assistant" and not last_assistant_action:
            content = m.get("content", "")
            if content:
                first_line = content.strip().split("\n")[0][:200]
                last_assistant_action = first_line

    if last_user_msg:
        memory.set_current_task(last_user_msg[:300])
    if last_assistant_action:
        memory.add_completed(last_assistant_action)


# Keep the old public name for any external callers.
def update_memory_from_messages(messages: List[Dict[str, Any]]):
    """Legacy alias — delegates to _update_memory_from_messages."""
    _update_memory_from_messages(messages)


def hard_reset_with_memory(messages: List[Dict[str, Any]],
                           model_name: str) -> List[Dict[str, Any]]:
    """Emergency context reset with deterministic distillation + rehydration.

    Unlike the old version (which wiped everything and relied on stale
    memory.json), this one:
    - Finds the floating root contract (last real user task)
    - Keeps the user's own messages, their standing rules and the facts of earlier compactions
    - Writes the dropped messages to the transcript archive
    - Rehydrates last edited files from disk
    - Preserves the last user message
    """
    _update_memory_from_messages(messages)
    last_user = None
    last_user_idx = None
    for i in range(len(messages) - 1, 0, -1):
        if messages[i].get("role") == "user":
            last_user = messages[i]
            last_user_idx = i
            break

    search_end = last_user_idx if last_user_idx is not None else len(messages)
    active_goal = _find_active_goal(messages, search_end)
    if not active_goal and last_user:
        content = (last_user.get("content") or "").strip()
        if _is_substantive(content):
            active_goal = content

    ctx = get_context_window()
    rest = [m for i, m in enumerate(messages) if i > 0 and i != last_user_idx]
    prior = compaction.split_distillate(_latest_distillate(rest))
    texts = compaction.user_texts(rest)
    ledger = compaction.fit_ledger(
        compaction.merge_ledger(prior.ledger, compaction.ledger_entries(texts)), _ledger_budget(ctx))
    rules = compaction.merge_rules(prior.rules, compaction.extract_rules(texts))
    facts = compaction.merge_facts(prior.facts, compaction.collect_facts(rest))
    archived = compaction.archive_transcript(rest, "emergency")
    rehydration = _rehydrate_files(messages, max_files=3, max_chars_per_file=_file_budget(ctx))
    context_note = memory.build_context_note()

    body = compaction.assemble(
        goal=active_goal, summary=prior.summary or None, archive=prior.archive + ([archived] if archived else []),
        ledger=ledger, rules=rules, facts=compaction.render_facts(facts, include_requests=False),
        files=rehydration,
        extra=[f"{compaction.L_MEMORY}\n{context_note}"] if context_note else None)

    distillate_msg = {
        "role": "system",
        "content": (
            "=== CONTEXT DISTILLATE (emergency reset) ===\n"
            + body
            + "\n=== END DISTILLATE ==="
        )
    }

    result = [messages[0], distillate_msg]
    if last_user:
        result.append(last_user)

    _write_compact_boundary(messages, result, -1, model_name=model_name, kind="emergency")  # -1 = emergency
    log.info("Hard context reset performed: emergency reset with deterministic distillate + rehydration "
             "(model=%s)", model_name)
    return result


def _compression_zone(messages: List[Dict[str, Any]], model_name: str, max_context_tokens: int):
    """What a compaction would drop: (messages to compress, their indices, where the active window starts)."""
    # ── Pin immovable messages ──
    pinned: set = {0}
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            content = m.get("content", "")
            if any(tag in content for tag in [
                "PROJECT SPECIFICATION", "ARCHITECTURE MAP",
                "PREVIOUS CONTEXT MEMORY", "PERSISTENT MEMORY",
            ]):
                pinned.add(i)

    # ── The active window ──
    window_budget = int(max_context_tokens * 0.6)
    active_start = _find_active_window_start(
        messages, keep_cycles=3,
        model_name=model_name, token_budget=window_budget,
    )
    for i in range(active_start, len(messages)):
        pinned.add(i)

    # ── The compression zone ──
    msgs_to_compress: list = []
    indices_to_drop: list = []
    for i in range(1, len(messages)):
        if i not in pinned:
            msgs_to_compress.append(messages[i])
            indices_to_drop.append(i)
    return msgs_to_compress, indices_to_drop, active_start


def precompute_summary(messages: List[Dict[str, Any]], model_name: str, max_context_tokens: int) -> bool:
    """Start writing, in the background, the summary the next compaction will need (see summary_precompute).
    `max_context_tokens` is the budget the compaction will trim down to. True when a job was started."""
    if not summary_precompute.enabled() or is_thrashing():
        return False
    zone, _, active_start = _compression_zone(messages, model_name, max_context_tokens)
    if not zone:
        return False
    prior = compaction.split_distillate(_latest_distillate(zone))
    fresh = [m for m in zone if not _is_distillate(m)]
    if not summary_precompute.worth_writing(fresh):
        return False
    goal = _find_active_goal(messages, active_start)
    text = _prepare_for_summarizer(fresh)
    note = _previous_summary_note(prior.summary)
    return summary_precompute.start(
        fresh, prior.summary, lambda: _summarize_with_llm(text, goal, note, model_name, ui=None))


def soft_trim_with_summarization(
        messages: List[Dict[str, Any]], model_name: str,
        max_history_messages: int,
        max_context_tokens: int) -> List[Dict[str, Any]]:
    """Progressive Distillation v3.

    Replaces the old blind summarize-and-drop with a structured pipeline:

    1. Pin system messages and the active window (last K cycles).
    2. Find the floating root contract (last substantive user goal).
    3. Pre-distill the compression zone (50-100k → ~8k tokens).
    4. LLM summarisation → deterministic fallback on failure.
    5. Rehydrate last edited files from disk.
    6. Assemble the 4-layer context and write a compact boundary.
    """
    provider = get_provider()

    # Pre-check: if the entire conversation already fits the budget, no compression is needed
    strip_enabled = get_strip_reasoning()
    cleaned = clean_messages_for_llm(messages, strip_enabled)
    total_tokens = sum(
        estimate_message_tokens(m, model_name, provider)
        for m in cleaned[1:]
    )
    if len(messages) - 1 <= max_history_messages and total_tokens <= max_context_tokens:
        return messages

    # ── Steps 1-3: pin what must stay, find the active window, collect the compression zone ──
    msgs_to_compress, indices_to_drop, active_start = _compression_zone(messages, model_name, max_context_tokens)

    if not msgs_to_compress:
        if len(messages) - 1 > max_history_messages or total_tokens > max_context_tokens:
            return sliding_window_trim(messages, model_name, max_history_messages, max_context_tokens)
        return messages

    # ── Step 4: Find a prior distillate (incremental update) ──
    existing_distillate = _latest_distillate(msgs_to_compress)
    compression_id = _get_compression_count(messages) + 1

    # ── Step 5: Create the distillate ──
    # The dropped messages go to a file FIRST, whole: whatever the summary loses, the agent can read back.
    archive_path = compaction.archive_transcript(msgs_to_compress, compression_id)
    skip_model = is_thrashing()
    from ui import console
    from src.agent import service_llm
    with console.status(
        "[dim #58a6ff]Оптимизация контекста (Progressive Distillation)..."
        "[/dim #58a6ff]",
        spinner="dots",
    ) as status:
        try:
            if skip_model:
                # Compacting again and again without a gain: the summariser would be paid for nothing.
                distillate = _fallback_distillate(msgs_to_compress, existing_distillate, archive_path)
            else:
                distillate = _create_distillate(
                    msgs_to_compress, messages, active_start,
                    existing_distillate, model_name,
                    ui=service_llm.default_ui(status),
                    context_tokens=max_context_tokens,
                    archive_path=archive_path,
                )
        except Exception:
            # Nothing above should raise (it is built to fall back), but whatever does
            # must not cost the session its compression: the history is still intact
            # here, so settle for what needs no model.
            log.exception("Distillation failed; using the deterministic summary")
            distillate = _fallback_distillate(msgs_to_compress, existing_distillate, archive_path)

    # ── Step 6: Assemble new history ──
    distillate_msg = {
        "role": "system",
        "content": (
            f"=== CONTEXT DISTILLATE (compression #{compression_id}) ===\n"
            f"{distillate}\n"
            "=== END DISTILLATE ==="
        ),
    }

    drop_set = set(indices_to_drop)
    new_messages: list = []
    distillate_injected = False
    for i, m in enumerate(messages):
        if i in drop_set:
            if not distillate_injected:
                new_messages.append(distillate_msg)
                distillate_injected = True
            continue
        new_messages.append(m)

    # ── Step 7: Bookkeeping ──
    # Bookkeeping must not undo a compression that already succeeded.
    for step in (lambda: _write_compact_boundary(messages, new_messages, compression_id,
                                                 model_name=model_name, kind="auto", pre_tokens=total_tokens),
                 lambda: _update_memory_from_messages(messages)):
        try:
            step()
        except Exception:
            log.exception("Compression bookkeeping failed")

    log.info(
        "Progressive distillation #%d: %d messages → distillate + "
        "%d active window",
        compression_id, len(indices_to_drop),
        len(messages) - active_start,
    )
    latest = last_compaction()
    if latest is not None and latest["id"] == compression_id:
        _announce_compaction(latest, is_thrashing())
    return new_messages


def clean_messages_for_llm(messages: List[Dict[str, Any]], strip_reasoning: bool) -> List[Dict[str, Any]]:
    """Creates a clean copy of messages for token estimation or LLM API calls.
    - If strip_reasoning is False and a message has a 'thinking' block, wraps it in <think>...</think>
      and prepends/appends it to the 'content' so the model can see its past thoughts.
    - If strip_reasoning is True, strips <think>...</think> and unclosed <think>... from 'content'.
    - Removes the custom 'thinking' key to ensure OpenAI API schema compatibility.
    """
    # Before anything else: old image payloads never leave the machine again.
    # Done here rather than in the trimmer's fallback path because the cost is
    # per REQUEST — re-uploading a screenshot from twenty turns ago is waste
    # whether or not the history needs trimming — and because this is also the
    # copy the token estimate is computed from, so the meter matches the wire.
    cleaned = []
    for m in compact_image_history(messages):
        m_copy = m.copy()
        
        # Always remove custom 'thinking' and 'rethink' keys for API compatibility,
        # but capture them first
        thinking = m_copy.pop("thinking", None)
        rethink = m_copy.pop("rethink", None)
        
        if m_copy.get("role") == "assistant":
            content = m_copy.get("content", "") or ""
            
            # If we want to keep reasoning, format it back into the content
            if not strip_reasoning and (thinking or rethink):
                blocks = []
                if thinking:
                    draft_thinking = thinking
                    if rethink and rethink in thinking:
                        draft_thinking = thinking.replace(rethink, "").strip()
                    if draft_thinking:
                        blocks.append(f"<think>\n{draft_thinking}\n</think>")
                if rethink:
                    blocks.append(f"<rethink>\n{rethink.strip()}\n</rethink>")
                elif not blocks and thinking:
                    blocks.append(f"<think>\n{thinking.strip()}\n</think>")
                
                think_block = "\n\n".join(blocks)
                if think_block and think_block not in content:
                    content = f"{think_block}\n\n{content}".strip()
                m_copy["content"] = content
                
            elif strip_reasoning:
                # Strip reasoning blocks if enabled
                if content:
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                    content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
                    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
                    content = re.sub(r'<rethink>.*?</rethink>', '', content, flags=re.DOTALL)
                    content = re.sub(r'<reflection>.*?</reflection>', '', content, flags=re.DOTALL)
                    
                    content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
                    content = re.sub(r'<thought>.*', '', content, flags=re.DOTALL)
                    content = re.sub(r'<reasoning>.*', '', content, flags=re.DOTALL)
                    content = re.sub(r'<rethink>.*', '', content, flags=re.DOTALL)
                    content = re.sub(r'<reflection>.*', '', content, flags=re.DOTALL)
                    m_copy["content"] = content.strip()
                    
        cleaned.append(m_copy)

    # Ensure API contract: no orphaned tool messages without preceding assistant tool_calls
    valid_tool_ids = set()
    sanitized = []
    for m in cleaned:
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls", []):
                if tc.get("id"):
                    valid_tool_ids.add(tc["id"])
            sanitized.append(m)
        elif role == "tool":
            tid = m.get("tool_call_id")
            if not tid or tid in valid_tool_ids:
                sanitized.append(m)
            else:
                log.warning("Dropping orphaned tool message with tool_call_id=%s", tid)
        else:
            sanitized.append(m)
    cleaned = sanitized

    # Last stop before the request is serialised and encoded. A lone surrogate
    # anywhere in the history makes the HTTP body unencodable, so the turn dies
    # with "'utf-8' codec can't encode ... surrogates not allowed" and so does
    # every turn after it — the chat stays bricked until /clear. Repairing on
    # the way out means an ALREADY poisoned history heals itself, which fixing
    # the producer alone cannot do.
    from text_safety import repair_structure
    return repair_structure(cleaned)


# How many image-carrying messages keep their payload. Two covers the real
# working pattern — a screenshot, an action, a second screenshot to check, or a
# before/after comparison — while everything older keeps only its caption.
KEEP_RECENT_IMAGES = 2


def compact_image_history(messages: List[Dict[str, Any]],
                          keep_recent: int = KEEP_RECENT_IMAGES) -> List[Dict[str, Any]]:
    """Drop the pixels of images the model has already looked at.

    An attached screenshot is ~190 KB of base64 and it lived in the history
    forever, so every later request re-uploaded every picture ever taken — the
    one unbounded thing in a conversation that is otherwise trimmed. The model
    does not need the pixels twice: by the time the next turn starts it has
    already written what it saw, and that text is what the history is for.

    The caption stays, so "the screenshot you took" still refers to something.
    Returns the same list object when there is nothing to compact.
    """
    carriers = [i for i, m in enumerate(messages)
                if isinstance(m, dict) and m.get("images")]
    to_strip = set(carriers[:-keep_recent]) if len(carriers) > keep_recent else set()
    if not to_strip:
        return messages

    out = []
    for i, m in enumerate(messages):
        if i not in to_strip:
            out.append(m)
            continue
        stripped = {k: v for k, v in m.items() if k != "images"}
        count = len(m["images"])
        note = f"[{count} изобр. показано ранее — содержимое описано выше]"
        stripped["content"] = f"{(m.get('content') or '').strip()} {note}".strip()
        out.append(stripped)
    return out


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
    log.info("Summarization failed, falling back to hard reset.")
    return hard_reset_with_memory(messages, model_name)

