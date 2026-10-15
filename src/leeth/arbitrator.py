"""src/leeth/arbitrator.py — CDCL arbitrator for Leeth dispatch decisions.

Formal consistency check over the joint (need_tool, tool, param) proposal
against a hard-constraint base, per .argent/artifacts/cdcl_arbitrator_design.md.

Solver backend: the owner's FlexibleCDCLSolver (P = PN project) — full CDCL with
2WL, 1-UIP conflict analysis, LBD, VSIDS + the author's ChronoPhysics heuristic.
Loaded by path via importlib; pysat is NOT used.

MVP (shadow mode): the arbitrator OBSERVES and LOGS, never blocks. The proposal
is checked as-is; UNSAT yields a minimal conflict core (deletion-based
minimization over assumption+fact units) for escalation explanations.

Integration point: dispatcher.dispatch(), right after the UNFAMILIAR gate and
BEFORE the confidence/param gates — so shadow statistics capture real
disagreement between the symbolic base and the neural gates.
"""

import importlib.util
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from logger import get_logger

logger = get_logger("leeth_arbitrator")

DEFAULT_SOLVER_PATH = Path(os.environ.get(
    "PN_SOLVER_PATH", str(Path.home() / "Desktop" / "P = PN" / "sat.py")
))

# Must mirror dispatcher._build_state_and_questions choices exactly (order too).
TOOLS = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "read_background_command", "NO_TOOL", "UNFAMILIAR",
]
# Tools that touch the system and therefore require a non-empty in-query param.
DISPATCH_TOOLS = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "read_background_command",
]

_VAR_NAMES = {
    "NEED": "need_tool",
    "PARAM_NONEMPTY": "param_nonempty",
    "PARAM_IN_QUERY": "param_in_query",
    "DESTRUCTIVE": "destructive",
    "SAFE": "safety_allow",
    "EXEC": "execute_immediately",
    "P_PATH": "param_path_like",
    "P_ARITH": "param_arithmetic",
    "P_URL": "param_url_like",
}


# ── Conservative structural param checks (facts, not substring heuristics) ────
# The rule: reject only shapes that are CLEARLY wrong for the tool; bare words
# stay valid paths, domains stay valid URLs. False-UNSAT in shadow is noise,
# false-SAT is a missed signal — so the checks lean permissive.

def _looks_like_path(p: str) -> bool:
    if not p:
        return False
    if " " not in p:
        return True                      # bare word / single token: assume path
    if re.search(r"\.\w{1,5}$", p):      # trailing extension survives spaces
        return True
    if re.match(r"^([A-Za-z]:[\\/]|[\\/~]|\.\.?[/\\])", p):
        return True
    # Относительный путь с пробелами («my documents/file»): есть разделитель —
    # это форма пути. Политика lean-permissive: false-UNSAT = шум в shadow.
    if "/" in p or "\\" in p:
        return True
    return False


def _looks_like_arithmetic(p: str) -> bool:
    if not p or not re.search(r"\d", p):
        return False
    return bool(re.fullmatch(r"[\d\s+\-*/().,%^]+", p))


def _looks_like_url(p: str) -> bool:
    if not p:
        return False
    if p.startswith(("http://", "https://", "www.", "localhost")):
        return True
    # Conservative: reject only clearly-non-URL shapes (raw spaces, FS paths).
    return not (" " in p or p.startswith(("/", "\\", "./", "../")))


# ── Shadow JSONL log (queryable statistics; never raises) ───────────────────
_SHADOW_LOG = Path(".argent/artifacts/arbitrator_shadow.jsonl")
_SHADOW_LOG_MAX_BYTES = 5 * 1024 * 1024


def _append_shadow_log(record: dict) -> None:
    try:
        _SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
        if _SHADOW_LOG.exists() and _SHADOW_LOG.stat().st_size > _SHADOW_LOG_MAX_BYTES:
            _SHADOW_LOG.replace(_SHADOW_LOG.with_suffix(".old.jsonl"))
        with _SHADOW_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


@dataclass
class ArbitrationResult:
    sat: bool
    core: List[str]                 # readable literals of a minimal conflict (UNSAT only)
    model_tool: Optional[str]       # tool the model actually satisfies (SAT only)
    latency_ms: float
    conflicts: int
    decisions: int
    error: Optional[str] = None


class LeethCNFEncoder:
    """Fixed variable layout + hard constraint clauses (conservative MVP)."""

    def __init__(self):
        self.tool_vars = {t: i + 1 for i, t in enumerate(TOOLS)}
        base = len(TOOLS)
        self.NEED = base + 1
        self.PARAM_NONEMPTY = base + 2
        self.PARAM_IN_QUERY = base + 3
        self.DESTRUCTIVE = base + 4
        self.SAFE = base + 5
        self.EXEC = base + 6
        self.P_PATH = base + 7
        self.P_ARITH = base + 8
        self.P_URL = base + 9
        self.num_vars = base + 9

    def hard_clauses(self) -> List[List[int]]:
        cls: List[List[int]] = []
        tv = list(self.tool_vars.values())
        cls.append(list(tv))                              # at least one tool
        for i in range(len(tv)):                          # pairwise exclusivity
            for j in range(i + 1, len(tv)):
                cls.append([-tv[i], -tv[j]])
        t_no = self.tool_vars["NO_TOOL"]
        cls.append([-self.NEED, -t_no])                   # need_tool -> not NO_TOOL
        cls.append([self.NEED, t_no])                     # not need_tool -> NO_TOOL
        for t in DISPATCH_TOOLS:                          # dispatch needs a real param
            tv_t = self.tool_vars[t]
            cls.append([-tv_t, self.PARAM_NONEMPTY])
            cls.append([-tv_t, self.PARAM_IN_QUERY])
        # A destructive-looking string must never ride along as the parameter of
        # ANY dispatch tool (mirrors the dispatcher's static deny-list policy).
        for t in DISPATCH_TOOLS:
            cls.append([-self.DESTRUCTIVE, -self.tool_vars[t]])
        # Semantic param-tool coherence — NOT checked by the procedural gates.
        # These are the constraints that let the arbitrator disagree meaningfully.
        for t in ("read_file", "write_file", "replace_in_file"):
            cls.append([-self.tool_vars[t], self.P_PATH])
        cls.append([-self.tool_vars["calculate"], self.P_ARITH])
        cls.append([-self.tool_vars["browser_open"], self.P_URL])
        t_run = self.tool_vars["run_command"]
        cls.append([-t_run, self.SAFE])                   # run_command -> safety allow
        cls.append([-t_run, self.EXEC])                   # run_command -> execute_immediately
        return cls


class CDCLArbitrator:
    """Shadow-mode arbitrator over the owner's CDCL solver."""

    def __init__(self, solver_path: Optional[Path] = None, heuristic: str = "physics"):
        self._solver_path = Path(solver_path or DEFAULT_SOLVER_PATH)
        self._heuristic = heuristic
        self._mod = None
        self.encoder = LeethCNFEncoder()
        self.stats = {"calls": 0, "sat": 0, "unsat": 0, "errors": 0, "total_ms": 0.0}

    # ── Solver plumbing ──────────────────────────────────────────────────────
    def _load_solver_module(self):
        if self._mod is None:
            if not self._solver_path.exists():
                raise FileNotFoundError(f"Solver not found: {self._solver_path}")
            spec = importlib.util.spec_from_file_location("pn_sat", self._solver_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._mod = mod
        return self._mod

    def _solve(self, clauses: List[List[int]], num_vars: int):
        mod = self._load_solver_module()
        solver = mod.FlexibleCDCLSolver(
            clauses, num_vars=num_vars, heuristic=self._heuristic, max_conflicts=100000
        )
        return solver.solve(), solver

    # ── Facts (procedural evidence) ──────────────────────────────────────────
    @staticmethod
    def _is_destructive(text: str) -> bool:
        if not text:
            return False
        from src.leeth.dispatcher import DENY_PATTERNS  # deferred: breaks import cycle
        return any(p.search(text) for p in DENY_PATTERNS)

    # ── Core extraction (deletion-based, subset-minimal) ─────────────────────
    def _extract_core(self, hard: List[List[int]], candidates: Dict[int, bool], num_vars: int) -> Dict[int, bool]:
        working = dict(candidates)
        for v in list(working.keys()):
            trial = {k: val for k, val in working.items() if k != v}
            units = [[k if val else -k] for k, val in trial.items()]
            model, _ = self._solve(hard + units, num_vars)
            if model is None:                 # still UNSAT without it -> not needed
                working = trial
        return working

    def _core_readable(self, core_vars: Dict[int, bool]) -> List[str]:
        enc = self.encoder
        out = []
        for var, val in core_vars.items():
            if var in enc.tool_vars.values():
                out.append(f"tool={TOOLS[var - 1]}" if val else f"tool!={TOOLS[var - 1]}")
            elif var == enc.NEED:
                out.append(f"need_tool={'True' if val else 'False'}")
            elif var == enc.PARAM_NONEMPTY:
                out.append(f"param_nonempty={'True' if val else 'False'}")
            elif var == enc.PARAM_IN_QUERY:
                out.append(f"param_in_query={'True' if val else 'False'}")
            elif var == enc.DESTRUCTIVE:
                out.append(f"destructive={'True' if val else 'False'}")
            elif var == enc.SAFE:
                out.append(f"safety_allow={'True' if val else 'False'}")
            elif var == enc.EXEC:
                out.append(f"execute_immediately={'True' if val else 'False'}")
            elif var == enc.P_PATH:
                out.append(f"param_path_like={'True' if val else 'False'}")
            elif var == enc.P_ARITH:
                out.append(f"param_arithmetic={'True' if val else 'False'}")
            elif var == enc.P_URL:
                out.append(f"param_url_like={'True' if val else 'False'}")
        return out

    # ── Main entry ───────────────────────────────────────────────────────────
    def arbitrate(
        self,
        tool: str,
        need_tool: bool,
        param_text: str,
        query: str,
        safety_allow: Optional[bool] = None,
        execute_immediately: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
        log_shadow: bool = True,
    ) -> ArbitrationResult:
        """Check the joint (need_tool, tool, param) proposal against the base.

        Shadow mode: this NEVER gates dispatch; callers log the result.
        Fail-soft: any internal error yields an error result, not an exception.
        Every call appends one record to the shadow JSONL log unless
        log_shadow=False (safety GATES reuse the solver but are not dispatch
        observations — logging them would pollute shadow statistics).
        """
        t0 = time.perf_counter()
        self.stats["calls"] += 1
        try:
            res = self._arbitrate_inner(
                tool, need_tool, param_text, query, safety_allow, execute_immediately, t0
            )
        except Exception as exc:  # fail-soft: never break the dispatcher
            self.stats["errors"] += 1
            res = ArbitrationResult(
                False, [], None, (time.perf_counter() - t0) * 1000.0, 0, 0, error=str(exc)
            )
        if log_shadow:
            self._log_shadow(tool=tool, need_tool=need_tool, param_text=param_text,
                             query=query, res=res, context=context)
        return res

    def _arbitrate_inner(
        self, tool, need_tool, param_text, query, safety_allow, execute_immediately, t0
    ) -> ArbitrationResult:
        enc = self.encoder
        if tool not in enc.tool_vars:
            return ArbitrationResult(False, [], None, 0.0, 0, 0, error=f"unknown tool '{tool}'")

        param_clean = (param_text or "").strip().strip("'\"`")
        facts: Dict[int, bool] = {
            enc.PARAM_NONEMPTY: bool(param_clean),
            enc.PARAM_IN_QUERY: bool(param_clean) and param_clean.lower() in (query or "").lower(),
            enc.DESTRUCTIVE: self._is_destructive(param_clean),
            enc.P_PATH: _looks_like_path(param_clean),
            enc.P_ARITH: _looks_like_arithmetic(param_clean),
            enc.P_URL: _looks_like_url(param_clean),
        }
        if safety_allow is not None:
            facts[enc.SAFE] = safety_allow
        if execute_immediately is not None:
            facts[enc.EXEC] = execute_immediately

        assumptions: Dict[int, bool] = {
            enc.tool_vars[tool]: True,
            enc.NEED: bool(need_tool),
        }
        candidates = {**facts, **assumptions}
        hard = enc.hard_clauses()
        units = [[v if val else -v] for v, val in candidates.items()]

        model, solver = self._solve(hard + units, enc.num_vars)
        sat = model is not None
        core_vars: Dict[int, bool] = {}
        if not sat:
            core_vars = self._extract_core(hard, candidates, enc.num_vars)

        model_tool = None
        if sat:
            for name, var in enc.tool_vars.items():
                if model.get(var) == 1:
                    model_tool = name
                    break

        latency = (time.perf_counter() - t0) * 1000.0
        self.stats["sat" if sat else "unsat"] += 1
        self.stats["total_ms"] += latency
        return ArbitrationResult(
            sat=sat,
            core=self._core_readable(core_vars),
            model_tool=model_tool,
            latency_ms=round(latency, 2),
            conflicts=solver.conflicts,
            decisions=solver.decisions_count,
        )

    def _log_shadow(self, tool, need_tool, param_text, query, res, context) -> None:
        """Append one JSONL record; never raises, never blocks dispatch."""
        try:
            ctx = context or {}
            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "query": (query or "")[:200],
                "tool": tool,
                "need_tool": bool(need_tool),
                "param": (param_text or "")[:120],
                "sat": res.sat,
                "core": res.core,
                "model_tool": res.model_tool,
                "latency_ms": res.latency_ms,
                "conflicts": res.conflicts,
                "decisions": res.decisions,
                "gates": ctx.get("gates"),
                "confidence": ctx.get("confidence"),
                "enforce": False,
                "error": res.error,
            }
            _append_shadow_log(record)
        except Exception:
            pass


_INSTANCE: Optional[CDCLArbitrator] = None
_INSTANCE_LOCK = threading.Lock()


def get_arbitrator() -> CDCLArbitrator:
    """Thread-safe singleton: гон двух инстансов солвера недопустим."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:  # double-checked locking
                _INSTANCE = CDCLArbitrator()
    return _INSTANCE