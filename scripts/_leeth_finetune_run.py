"""Runner for a CPU fine-tune cycle (CUDA may be occupied by the main LLM)."""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.leeth.finetune import finetune_from_feedback  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    result = finetune_from_feedback(device="cpu")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))