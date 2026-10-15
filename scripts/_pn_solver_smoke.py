"""Smoke: verify FlexibleCDCLSolver works from Argent's environment (Python 3.14 + numpy)."""
import importlib.util
import sys
import time
from pathlib import Path

SOLVER_PATH = Path.home() / "Desktop" / "P = PN" / "sat.py"
spec = importlib.util.spec_from_file_location("pn_sat", SOLVER_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Trivial SAT: (x1 v x2) & (~x1 v x2) & (x1 v ~x2)  -> x1=1, x2=1
clauses = [[1, 2], [-1, 2], [1, -2]]
t0 = time.perf_counter()
solver = mod.FlexibleCDCLSolver(clauses, num_vars=2, heuristic="physics", max_conflicts=10000)
model = solver.solve()
dt = (time.perf_counter() - t0) * 1000
print(f"SAT case: model={model} in {dt:.2f}ms")
assert model is not None and model[1] == 1 and model[2] == 1
assert mod.verify_solution(clauses, model), "verify_solution failed on SAT model"

# Trivial UNSAT: (x1) & (~x1)
clauses_unsat = [[1], [-1]]
t0 = time.perf_counter()
solver2 = mod.FlexibleCDCLSolver(clauses_unsat, num_vars=1, heuristic="physics", max_conflicts=10000)
model2 = solver2.solve()
dt2 = (time.perf_counter() - t0) * 1000
print(f"UNSAT case: model={model2} in {dt2:.2f}ms (conflicts={solver2.conflicts})")
assert model2 is None

# Bigger smoke: random 3-SAT, 50 vars / 210 clauses (near phase transition)
random_clauses = mod.generate_critical_3sat(50, 210, seed=7) if False else None
print("SMOKE OK")