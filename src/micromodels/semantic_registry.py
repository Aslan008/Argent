"""
Semantic Tool Registry for Argent MicroModels.

Maintains semantic embeddings (192-dim) for all registered tools (both built-in and dynamic MCP/plugins).
Enables zero-shot routing to new tools without model retraining:
When a new tool or MCP server connects, its description is embedded in ~0.5ms and added to the registry.
User queries are matched against registered tools using cosine similarity in the shared semantic space.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from logger import get_logger

logger = get_logger("semantic_registry")

class SemanticToolRegistry:
    """In-memory semantic vector registry for dynamic tool matching."""

    _instance: Optional["SemanticToolRegistry"] = None

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(
            data_dir or (Path(__file__).parent / "data")
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.data_dir / "tool_embeddings.json"

        self._lock = threading.RLock()
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._matrix: Optional[np.ndarray] = None
        self._tool_keys: List[str] = []

    @classmethod
    def get_instance(cls) -> "SemanticToolRegistry":
        if cls._instance is None:
            cls._instance = SemanticToolRegistry()
        return cls._instance

    @property
    def total_tools(self) -> int:
        with self._lock:
            return len(self._tools)

    def register_tool(
        self,
        name: str,
        description: str,
        embedding: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, Any]] = None,
        is_dynamic: bool = False,
    ) -> None:
        """Registers a tool with its metadata and semantic vector."""
        if not name:
            return
        name = str(name).strip()
        with self._lock:
            self._tools[name] = {
                "name": name,
                "description": str(description or "").strip(),
                "parameters": parameters or {},
                "is_dynamic": is_dynamic,
            }

            if embedding is not None:
                emb_arr = np.asarray(embedding, dtype=np.float32).flatten()
                norm = np.linalg.norm(emb_arr)
                if norm > 1e-12 and not np.isnan(norm) and not np.isinf(norm):
                    vec = emb_arr / norm
                else:
                    vec = emb_arr
                self._embeddings[name] = vec.astype(np.float32)
                self._rebuild_matrix()

    def set_tool_embedding(self, name: str, embedding: np.ndarray) -> None:
        """Sets or updates the embedding vector for a tool."""
        name = str(name).strip()
        with self._lock:
            emb_arr = np.asarray(embedding, dtype=np.float32).flatten()
            norm = np.linalg.norm(emb_arr)
            if norm > 1e-12 and not np.isnan(norm) and not np.isinf(norm):
                vec = emb_arr / norm
            else:
                vec = emb_arr
            self._embeddings[name] = vec.astype(np.float32)
            self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        """Rebuilds the contiguous matrix for fast vectorized cosine similarity."""
        # Called within lock
        self._tool_keys = sorted(self._embeddings.keys())
        if not self._tool_keys:
            self._matrix = None
            return

        rows = [self._embeddings[k] for k in self._tool_keys]
        self._matrix = np.vstack(rows).astype(np.float32)

    def match(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.60,
        only_dynamic: bool = False,
    ) -> List[Tuple[str, float]]:
        """Finds closest tools by cosine similarity in the semantic embedding space.
        
        Validates input shape, finiteness, and optionally filters to dynamic tools only.
        """
        if query_embedding is None:
            return []

        q_arr = np.asarray(query_embedding, dtype=np.float32).flatten()
        if q_arr.size == 0 or np.isnan(q_arr).any() or np.isinf(q_arr).any():
            return []

        with self._lock:
            if self._matrix is None or len(self._tool_keys) == 0:
                return []

            expected_dim = self._matrix.shape[1]
            if q_arr.shape[0] != expected_dim:
                logger.warning(
                    f"Dimension mismatch in semantic match: expected {expected_dim}, got {q_arr.shape[0]}"
                )
                return []

            # Ensure query is normalized
            q_norm = np.linalg.norm(q_arr)
            if q_norm > 1e-12:
                q_vec = q_arr / q_norm
            else:
                q_vec = q_arr

            # Dot product with all normalized tool vectors: shape (num_tools,)
            sims = np.dot(self._matrix, q_vec.reshape(-1, 1)).flatten()

            # Filter candidates
            candidate_indices = range(len(self._tool_keys))
            if only_dynamic:
                candidate_indices = [
                    i for i in candidate_indices
                    if self._tools.get(self._tool_keys[i], {}).get("is_dynamic", False)
                ]
                if not candidate_indices:
                    return []

            # Sort candidate indices by descending similarity
            ranked = sorted(
                candidate_indices,
                key=lambda idx: float(sims[idx]),
                reverse=True,
            )

            results: List[Tuple[str, float]] = []
            for idx in ranked[:top_k]:
                score = float(sims[idx])
                if not np.isnan(score) and score >= min_similarity:
                    results.append((self._tool_keys[idx], score))

            return results

    def save(self) -> None:
        """Persists registered tools and embeddings to disk atomically with Windows lock retry."""
        import time

        with self._lock:
            data = {
                "tools": self._tools,
                "embeddings": {k: v.tolist() for k, v in self._embeddings.items()},
            }
            temp_file = self.data_dir / f".tool_embeddings_{threading.get_ident()}_{time.perf_counter_ns()}.tmp"
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())

                # Windows retry backoff if another reader currently holds file handle
                for attempt in range(10):
                    try:
                        os.replace(temp_file, self.registry_file)
                        break
                    except (PermissionError, OSError):
                        if attempt == 9:
                            raise
                        time.sleep(0.01 * (attempt + 1))
            finally:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass

    def load(self) -> bool:
        """Loads registered tools and embeddings from disk if available with retry."""
        import time

        with self._lock:
            if not self.registry_file.exists():
                return False

            data = None
            for attempt in range(10):
                try:
                    with open(self.registry_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    break
                except (PermissionError, json.JSONDecodeError):
                    if attempt == 9:
                        return False
                    time.sleep(0.01 * (attempt + 1))

            if data is None:
                return False

            try:
                self._tools = data.get("tools", {})
                raw_emb = data.get("embeddings", {})
                self._embeddings = {
                    k: np.array(v, dtype=np.float32) for k, v in raw_emb.items()
                }
                self._rebuild_matrix()
                logger.info(f"Loaded {len(self._tools)} tools into semantic registry.")
                return True
            except Exception as e:
                logger.warning(f"Failed to parse semantic registry: {e}")
                return False


# Global singleton
registry = SemanticToolRegistry.get_instance()
