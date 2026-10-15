"""Debug: inspect the empty-param core."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.arbitrator import CDCLArbitrator

arb = CDCLArbitrator()
r = arb.arbitrate(tool="read_file", need_tool=True, param_text="", query="прочитай")
print("sat:", r.sat, "| core:", r.core, "| latency:", r.latency_ms)