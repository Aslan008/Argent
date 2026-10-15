"""Diagnose: what did the gates say about the user's compound message?"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dispatcher import LeethDispatcher

Q = "pytest крутит другой агент, не обращай на него внимания. А теперь проверька. Сколько будет 2 + 2 * 22?"
info = LeethDispatcher.get_instance().inspect_query(Q)
print(json.dumps(info, ensure_ascii=False, indent=2, default=str))