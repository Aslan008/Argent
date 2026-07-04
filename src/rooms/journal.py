"""Write-ahead journal for crash-safe room execution.

Each node step writes two records: an ``intent`` (with the pre-node state) before
side effects, then a ``commit`` (with the post-node state and the chosen next
target) after them. On resume:

  * last record is a ``commit`` -> that node finished; continue at its ``next``
    (or the run already reached a terminal if ``next`` is null);
  * last record is an ``intent`` -> the node was interrupted between its intent
    and its commit. If the node is safe to replay (a condition, a human_pause, or
    an idempotent tool) we re-run it from the intent's pre-node state; otherwise
    the outcome is indeterminate and the caller must route to a human rather than
    risk double-applying a side effect.

Append-only JSONL so a crash can at worst truncate the last line.
"""

import json
import time
from pathlib import Path


class Journal:
    def __init__(self, path):
        self.path = Path(path)

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = dict(record)
        rec.setdefault("ts", time.time())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def records(self) -> list:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass  # tolerate a torn final line from a crash mid-write
        return out

    def last(self):
        recs = self.records()
        return recs[-1] if recs else None

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
