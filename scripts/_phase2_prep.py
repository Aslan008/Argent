"""Phase 2 prep: purge poisoned feedback, derive trajectory verdicts, report.

Steps:
  1. Backup the feedback store.
  2. Purge heartbeat/system queries (maintenance noise, never user intent)
     and destructive-query auto-rejects without corrections (the
     self-reinforcing poison loop from _purge_poisoned_feedback.py).
  3. Run RLCD enrichment: derive implicit verdicts from trajectories.
  4. Report the resulting labeled/pending split.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FB = ROOT / "src/leeth/data/leeth_feedback.jsonl"
BACKUP = ROOT / "src/leeth/data/leeth_feedback_phase1_backup.jsonl"

HEARTBEAT_MARKS = ("heartbeat",)
DESTRUCTIVE_MARKS = ("удали", "очисти", "rm -rf", "снеси", "сотри",
                     "форматни", "убей", "стереть")


def main() -> None:
    shutil.copy(FB, BACKUP)
    records = [json.loads(l) for l in open(FB, encoding="utf-8") if l.strip()]
    kept, dropped_hb, dropped_poison = [], [], []
    for r in records:
        q = str(r.get("query", "")).lower()
        if any(m in q for m in HEARTBEAT_MARKS):
            dropped_hb.append(r["id"])
            continue
        is_poison = (r.get("verdict") == "reject"
                     and not (r.get("correction") or {}).get("tool")
                     and any(m in q for m in DESTRUCTIVE_MARKS))
        if is_poison:
            dropped_poison.append(r["id"])
            continue
        kept.append(r)
    with open(FB, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"purge: dropped {len(dropped_hb)} heartbeat, "
          f"{len(dropped_poison)} destructive-poison; kept {len(kept)} "
          f"(backup: {BACKUP.name})")

    from src.leeth import enrich
    stats = enrich.derive_verdicts()
    print("enrich:", json.dumps(stats, ensure_ascii=False, indent=2))

    from src.leeth import feedback
    labeled = feedback.labeled_records()
    print(f"final: labeled={len(labeled)} pending={feedback.pending_count()}")


if __name__ == "__main__":
    main()