"""
Active Learning and Live Data Collector for Argent MicroModels.

Implements the user's requirement for active learning:
1. Logs uncertain predictions (confidence between 0.40 and 0.85) to active_learning_queue.jsonl.
2. Captures user query -> LLM tool call pairs from live sessions to live_dataset.jsonl.
3. Provides tools to review, curate, and trigger incremental model fine-tuning.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from logger import get_logger

logger = get_logger("active_learning_collector")

class MicroDataCollector:
    """Collects uncertain queries for Active Learning and pairs from live session execution."""

    _instance: Optional["MicroDataCollector"] = None

    def __init__(self, data_dir: Optional[str] = None):
        home = os.path.expanduser("~")
        self.storage_dir = Path(
            data_dir or os.path.join(home, ".argent", "micromodels")
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self.active_queue_file = self.storage_dir / "active_learning_queue.jsonl"
        self.live_dataset_file = self.storage_dir / "live_dataset.jsonl"

    @classmethod
    def get_instance(cls) -> "MicroDataCollector":
        if cls._instance is None:
            cls._instance = MicroDataCollector()
        return cls._instance

    def log_uncertain_prediction(
        self,
        query: str,
        predicted_intent: str,
        confidence: float,
        probabilities: Dict[str, float],
    ) -> None:
        """Logs queries where the model had moderate confidence (0.40 <= conf < 0.85) for review."""
        if not (0.35 <= confidence < 0.85):
            return

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "predicted_intent": predicted_intent,
            "confidence": round(confidence, 4),
            "top_candidates": sorted(
                probabilities.items(), key=lambda x: -x[1]
            )[:3],
        }

        self._append_record(self.active_queue_file, record, what="active learning record")

    def log_live_tool_call(
        self,
        user_query: str,
        tool_name: str,
        tool_arguments: Dict[str, Any],
    ) -> None:
        """Logs actual LLM tool executions corresponding to user requests for incremental training."""
        if not user_query or not tool_name:
            return
        if len(user_query.strip()) > 300 or user_query.strip().startswith("/"):
            return

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text": user_query.strip(),
            "intent": tool_name,
            "arguments": tool_arguments,
            "source": "live_user",
        }

        self._append_record(self.live_dataset_file, record, what="live tool call record")

    def _append_record(self, path: Path, record: Dict[str, Any], what: str) -> None:
        """Appends a record under the lock, skipping instead of deadlocking.

        The lock acquire carries a timeout: a stuck holder (e.g. a thread
        blocked on a wedged file handle from an earlier test/process) must
        never freeze the agent turn — losing one telemetry record is cheap,
        hanging the turn is not.
        """
        if not self._lock.acquire(timeout=5.0):
            logger.warning("MicroDataCollector lock busy; %s skipped", what)
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Failed to append {what}: {e}")
        finally:
            self._lock.release()

    def get_stats(self) -> Dict[str, int]:
        """Returns counts of collected items."""
        active_count = 0
        live_count = 0

        if not self._lock.acquire(timeout=5.0):
            logger.warning("MicroDataCollector lock busy; stats partial")
            return {"uncertain_queries_queued": 0,
                    "uncertain_queries_logged": 0,
                    "live_samples_recorded": 0}
        try:
            if self.active_queue_file.exists():
                with open(self.active_queue_file, "r", encoding="utf-8") as f:
                    active_count = sum(1 for line in f if line.strip())

            if self.live_dataset_file.exists():
                with open(self.live_dataset_file, "r", encoding="utf-8") as f:
                    live_count = sum(1 for line in f if line.strip())
        finally:
            self._lock.release()

        return {
            "uncertain_queries_queued": active_count,
            "uncertain_queries_logged": active_count,
            "live_samples_recorded": live_count,
        }

    def merge_and_retrain(self) -> Tuple[bool, str]:
        """Merges live samples into the main dataset and triggers retraining."""
        if not self.live_dataset_file.exists():
            return False, "No live collected samples found to merge."

        live_samples: List[Dict] = []
        with open(self.live_dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        live_samples.append(json.loads(line))
                    except Exception:
                        pass

        if not live_samples:
            return False, "Live dataset is empty."

        from src.micromodels.dataset_builder import DatasetBuilder
        builder = DatasetBuilder()
        builder.build()

        # Append live samples to training_pairs.jsonl
        main_pairs = builder.output_dir / "training_pairs.jsonl"
        with open(main_pairs, "a", encoding="utf-8") as f:
            for s in live_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # Trigger train_model
        from src.micromodels.train_router import train_model
        try:
            train_model(epochs=10)
            return True, f"Successfully merged {len(live_samples)} live samples and retrained model."
        except Exception as e:
            return False, f"Retraining failed: {e}"


collector = MicroDataCollector.get_instance()
