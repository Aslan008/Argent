"""
MicroModel Fast Dispatcher for Argent.

Connects MicroModelRuntime and micro-specialists to Argent's agent turn lifecycle.
Provides try_fast_dispatch() conforming to Argent's FastDispatchResult interface,
handles active learning telemetry on ambiguous queries, and normalizes tool parameters.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.micromodels.data_collector import collector
from src.micromodels.runtime import runtime
from logger import get_logger

logger = get_logger("micromodel_dispatcher")

@dataclass
class FastDispatchResult:
    """Result structure compatible with Argent's agent fast-path interception."""

    tool_name: str
    arguments: Dict[str, Any]
    confidence: float
    elapsed_ms: float
    raw_query: str


class MicroModelDispatcher:
    """Dispatcher for custom in-process MicroModels."""

    _instance: Optional["MicroModelDispatcher"] = None

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold
        self._runtime = runtime
        self._collector = collector

    @classmethod
    def get_instance(cls) -> "MicroModelDispatcher":
        if cls._instance is None:
            cls._instance = MicroModelDispatcher()
        return cls._instance

    @property
    def is_available(self) -> bool:
        return self._runtime.is_available

    def try_fast_dispatch(self, user_query: str) -> Optional[FastDispatchResult]:
        """Evaluates whether the user query represents an immediate single-step tool call.

        Returns FastDispatchResult if a routine tool call is recognized with high confidence.
        Returns None if query requires general LLM reasoning, multi-step actions, or code writing.
        """
        if not user_query or not isinstance(user_query, str):
            return None

        q_clean = user_query.strip()
        if len(q_clean) < 2 or q_clean.startswith("/") or q_clean.startswith("!"):
            return None

        # Delegate directly to runtime (handles direct neural router + zero-shot semantic fallback)
        raw_res = self._runtime.try_fast_dispatch(q_clean, threshold=self.confidence_threshold)

        if raw_res is None:
            # Check for active learning on ambiguous queries
            intent, confidence, all_probs, _, _ = self._runtime.predict_intent(q_clean)
            if 0.35 <= confidence < self.confidence_threshold:
                self._collector.log_uncertain_prediction(
                    query=q_clean,
                    predicted_intent=intent,
                    confidence=confidence,
                    probabilities=all_probs,
                )
            return None

        tool_name = raw_res.tool_name
        args = dict(raw_res.arguments)

        # Normalize intent/tools for Argent execution
        # 1. read_git_diff -> run_command with git diff
        if tool_name == "read_git_diff":
            tool_name = "run_command"
            target_file = args.get("file", "")
            args = {"command": f"git diff {target_file}".strip()}

        # 2. search_web query
        elif tool_name == "search_web":
            # Argent search_web takes query
            pass

        # 3. calculate expression
        elif tool_name == "calculate":
            # Argent calculate takes expression
            pass

        return FastDispatchResult(
            tool_name=tool_name,
            arguments=args,
            confidence=raw_res.confidence,
            elapsed_ms=raw_res.elapsed_ms,
            raw_query=user_query,
        )


# Global singleton
dispatcher = MicroModelDispatcher.get_instance()
