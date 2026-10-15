"""
Speculative Cache for Argent MicroModels.

Thread-safe, in-memory LRU cache with mtime freshness validation,
TTL eviction, and size bounds for speculative read-only pre-fetching.
"""

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional
from logger import get_logger

logger = get_logger("speculative_cache")

# Safe Read-Only tools permitted for speculative pre-fetching
SAFE_READONLY_TOOLS = {
    "read_file",
    "list_directory",
    "search_files",
    "grep_search",
    "get_file_outline",
}

# Maximum file size allowed for speculative pre-fetching (512 KB)
MAX_SPECULATIVE_FILE_SIZE_BYTES = 512 * 1024


@dataclass
class SpeculativeEntry:
    """Stored result of a speculative pre-fetch."""

    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    created_at: float
    mtime: Optional[float] = None
    target_path: Optional[str] = None


class SpeculativeCache:
    """Thread-safe LRU cache for speculative tool results."""

    _instance: Optional["SpeculativeCache"] = None

    def __init__(self, max_entries: int = 16, ttl_seconds: float = 60.0):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, SpeculativeEntry] = OrderedDict()

        # Metrics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "prefetches": 0,
            "invalidations": 0,
            "cancels": 0,
        }

    @classmethod
    def get_instance(cls) -> "SpeculativeCache":
        if cls._instance is None:
            cls._instance = SpeculativeCache()
        return cls._instance

    @staticmethod
    def _make_key(tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generates a deterministic key from tool name and arguments."""
        # Normalize paths in arguments if present
        norm_args = {}
        for k, v in sorted(arguments.items()):
            if k in ("file_path", "path", "directory") and isinstance(v, str):
                norm_args[k] = v.replace("\\", "/").strip().rstrip("/")
            else:
                norm_args[k] = v
        import json
        return f"{tool_name}:{json.dumps(norm_args, sort_keys=True, ensure_ascii=False)}"

    def put(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        target_path: Optional[str] = None,
    ) -> None:
        """Stores a speculative pre-fetch result."""
        if tool_name not in SAFE_READONLY_TOOLS:
            logger.warning(f"Attempted to speculatively cache unsafe tool '{tool_name}'! Rejected.")
            return

        with self._lock:
            key = self._make_key(tool_name, arguments)
            now = time.time()

            # Capture mtime if target_path exists on disk
            mtime = None
            if target_path and os.path.exists(target_path):
                try:
                    mtime = os.path.getmtime(target_path)
                except OSError:
                    mtime = None

            # Evict oldest if full
            if len(self._cache) >= self.max_entries and key not in self._cache:
                self._cache.popitem(last=False)

            self._cache[key] = SpeculativeEntry(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                created_at=now,
                mtime=mtime,
                target_path=target_path,
            )
            self._cache.move_to_end(key)
            self.stats["prefetches"] += 1
            logger.debug(f"Speculative cache PUT: {key}")

    def get(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Retrieves a cached speculative result, verifying TTL and file freshness (mtime)."""
        with self._lock:
            key = self._make_key(tool_name, arguments)
            entry = self._cache.get(key)
            now = time.time()

            if entry is None:
                self.stats["misses"] += 1
                return None

            # 1. TTL Check
            if now - entry.created_at > self.ttl_seconds:
                logger.debug(f"Speculative cache EXPIRED: {key}")
                del self._cache[key]
                self.stats["misses"] += 1
                return None

            # 2. Disk Freshness (mtime) Check
            if entry.target_path and os.path.exists(entry.target_path):
                try:
                    current_mtime = os.path.getmtime(entry.target_path)
                    if entry.mtime is not None and current_mtime > entry.mtime:
                        logger.debug(f"Speculative cache STALE (file modified): {entry.target_path}")
                        del self._cache[key]
                        self.stats["misses"] += 1
                        return None
                except OSError:
                    pass

            # Cache HIT
            self._cache.move_to_end(key)
            self.stats["hits"] += 1
            logger.info(f"Speculative cache HIT for {tool_name}: {arguments}")
            return entry.result

    def invalidate(self, file_path: str) -> None:
        """Invalidates any cached entries that reference the modified file."""
        norm_target = file_path.replace("\\", "/").strip()
        with self._lock:
            keys_to_remove = []
            for k, entry in self._cache.items():
                if entry.target_path:
                    norm_entry = entry.target_path.replace("\\", "/").strip()
                    if norm_entry == norm_target or norm_entry.endswith(norm_target):
                        keys_to_remove.append(k)
                        continue
                # Check within arguments
                for arg_val in entry.arguments.values():
                    if isinstance(arg_val, str) and norm_target in arg_val.replace("\\", "/"):
                        keys_to_remove.append(k)
                        break

            for k in keys_to_remove:
                del self._cache[k]
                self.stats["invalidations"] += 1
                logger.debug(f"Invalidated speculative cache entry: {k}")

    def clear_turn(self) -> None:
        """Clears all speculative entries at the end of an agent turn."""
        with self._lock:
            self._cache.clear()

    def record_cancel(self) -> None:
        with self._lock:
            self.stats["cancels"] += 1


# Global singleton
speculative_cache = SpeculativeCache.get_instance()
