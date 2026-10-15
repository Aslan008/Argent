"""
Speculative CoT Lookahead Observer for Argent.

Observes real-time streaming reasoning (<thought> / thinking chunks) from the main LLM,
detects intended safe read-only actions, filters out doubts and refutations ("хотя нет, не надо"),
and speculatively pre-fetches tool results in background threads with 0 ms dispatch latency.
"""

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Set

from src.micromodels.runtime import runtime
from src.micromodels.specialists import extract_specialist_args
from src.micromodels.speculative_cache import (
    MAX_SPECULATIVE_FILE_SIZE_BYTES,
    SAFE_READONLY_TOOLS,
    speculative_cache,
)
from logger import get_logger

logger = get_logger("lookahead_observer")

class SpeculativeLookaheadObserver:
    """Watches the main LLM's thought stream and pre-fetches read-only data."""

    _instance: Optional["SpeculativeLookaheadObserver"] = None

    # Refutation regex: matches statements of doubt, retraction, or cancellation
    REFUTATION_REGEX = re.compile(
        r"\b(?:хотя\s+нет|не\s+(?:буду|надо|нужно|стоит|будем)|отставить|стоп|передумал|wait,\s+no|no\s+need|actually,\s+skip|nevermind)\b",
        re.IGNORECASE,
    )

    # Affirmation exception: phrases like "хотя бы" ("at least") are NOT refutations
    AFFIRMATION_EXCEPTION_REGEX = re.compile(
        r"\b(?:хотя\s+бы|at\s+least)\b",
        re.IGNORECASE,
    )

    def __init__(self, debounce_interval_sec: float = 0.20):
        self.debounce_interval_sec = debounce_interval_sec
        self._lock = threading.Lock()
        self._buffer: str = ""
        self._processed_sentences: Set[str] = set()
        self._active_prefetches: Set[str] = set()
        self._last_eval_time: float = 0.0
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cot_prefetch")

    @classmethod
    def get_instance(cls) -> "SpeculativeLookaheadObserver":
        if cls._instance is None:
            cls._instance = SpeculativeLookaheadObserver()
        return cls._instance

    def reset_turn(self) -> None:
        """Clears buffers and turn state when a new turn begins or finishes."""
        with self._lock:
            self._buffer = ""
            self._processed_sentences.clear()
            self._active_prefetches.clear()
            self._last_eval_time = 0.0
        speculative_cache.clear_turn()

    def is_refutation(self, text: str) -> bool:
        """Determines whether a sentence expresses doubt, cancellation, or retraction."""
        if not text:
            return False

        # If it matches an affirmative exception like "хотя бы", it is NOT a refutation
        if self.AFFIRMATION_EXCEPTION_REGEX.search(text):
            return False

        return bool(self.REFUTATION_REGEX.search(text))

    def feed(self, chunk: str) -> None:
        """Feeds a streaming thought chunk from the main LLM."""
        if not chunk or not isinstance(chunk, str):
            return

        with self._lock:
            self._buffer += chunk
            now = time.time()
            # Debounce evaluation to avoid overloading CPU during fast token streaming
            if now - self._last_eval_time < self.debounce_interval_sec:
                return
            self._last_eval_time = now
            text_snapshot = self._buffer

        self._evaluate_buffer(text_snapshot)

    def _evaluate_buffer(self, text: str) -> None:
        """Splits accumulated thoughts into sentences and evaluates intentions."""
        # Split on sentence boundaries: '.', '!', '?', ';', '\n'
        raw_sentences = re.split(r"(?<=[.!?;\n])\s+", text)
        if len(raw_sentences) <= 1:
            return

        # The last segment might be incomplete (still streaming), so evaluate completed sentences
        completed = raw_sentences[:-1]

        for sentence in completed:
            sentence_clean = sentence.strip()
            if len(sentence_clean) < 10 or sentence_clean in self._processed_sentences:
                continue

            self._processed_sentences.add(sentence_clean)

            # Check for refutation / retraction
            if self.is_refutation(sentence_clean):
                logger.debug(f"Refutation detected in thoughts: '{sentence_clean}' -> cancelling lookahead")
                speculative_cache.record_cancel()
                # Attempt to extract path and invalidate from active prefetches
                self._handle_refutation(sentence_clean)
                continue

            # Check if sentence signals a safe read-only tool intention
            self._evaluate_sentence_intent(sentence_clean)

    def flush(self) -> None:
        """Forces evaluation of any remaining buffered thoughts at the end of streaming."""
        with self._lock:
            text_snapshot = self._buffer
        if text_snapshot:
            if not text_snapshot.endswith((".", "!", "?", ";", "\n")):
                text_snapshot += ".\n"
            self._evaluate_buffer(text_snapshot)

    def _handle_refutation(self, sentence: str) -> None:
        """Handles retraction: extracts mentioned file and removes from cache if present."""
        args = extract_specialist_args("read_file", sentence)
        if args and "file_path" in args:
            target = args["file_path"]
            speculative_cache.invalidate(target)

    def _evaluate_sentence_intent(self, sentence: str) -> None:
        """Classifies intent of an affirmed thought sentence and triggers prefetch."""
        # Ultra fast pre-filter: does sentence mention file, code, or directory keywords?
        fast_keywords = (
            "файл", "file", "откро", "прочита", "глян", "чекн", "папк", "директор",
            "search", "grep", "lines", "строк", ".py", ".json", ".ts", ".cs", ".md",
        )
        if not any(kw in sentence.lower() for kw in fast_keywords):
            return

        intent, confidence, _, _, _ = runtime.predict_intent(sentence)

        # STRICT INVARIANT: Only safe read-only tools allowed
        if intent not in SAFE_READONLY_TOOLS:
            return

        if confidence < 0.65:
            return

        # Extract structured arguments via specialist
        args = extract_specialist_args(intent, sentence)
        if not args:
            return

        # Check deduplication
        prefetch_key = f"{intent}:{sorted(args.items())}"
        with self._lock:
            if prefetch_key in self._active_prefetches:
                return
            self._active_prefetches.add(prefetch_key)

        # Dispatch background safe pre-fetch
        self._executor.submit(self._execute_safe_prefetch, intent, args)

    def _execute_safe_prefetch(self, intent: str, args: Dict[str, Any]) -> None:
        """Executes the safe read-only tool in the background and populates SpeculativeCache."""
        try:
            target_path = None
            if intent == "read_file":
                from tools.file_ops import read_file
                fpath = args.get("file_path", "")
                target_path = fpath

                # File size check to prevent memory spikes on huge files
                if fpath and os.path.exists(fpath):
                    try:
                        if os.path.getsize(fpath) > MAX_SPECULATIVE_FILE_SIZE_BYTES:
                            logger.debug(f"Speculative skip: file '{fpath}' exceeds 512 KB")
                            return
                    except OSError:
                        pass

                result = read_file(**args)
                speculative_cache.put(
                    tool_name=intent,
                    arguments=args,
                    result=result,
                    target_path=target_path,
                )

            elif intent == "list_directory":
                from tools.file_ops import list_directory
                dpath = args.get("path", ".")
                target_path = dpath
                result = list_directory(**args)
                speculative_cache.put(
                    tool_name=intent,
                    arguments=args,
                    result=result,
                    target_path=target_path,
                )

            elif intent == "search_files":
                from tools.search_ops import search_files
                result = search_files(**args)
                speculative_cache.put(
                    tool_name=intent,
                    arguments=args,
                    result=result,
                )

            elif intent == "grep_search":
                from tools.search_ops import grep_search
                result = grep_search(**args)
                speculative_cache.put(
                    tool_name=intent,
                    arguments=args,
                    result=result,
                )

        except Exception as e:
            logger.debug(f"Speculative pre-fetch failed silently for {intent} ({args}): {e}")


# Global singleton
lookahead_observer = SpeculativeLookaheadObserver.get_instance()
