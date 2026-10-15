"""
src/leeth/dispatcher.py — High-Speed System One Fast-Path Dispatcher for Argent.

Connects Leeth non-autoregressive decision model directly to Argent's agent turn loop.
Provides try_fast_dispatch() conforming to Argent's FastDispatchResult interface:
- Evaluates user queries in ~1.5 ms without invoking external LLM APIs.
- Filters out non-tool / conversational prompts and dangerous commands.
- Normalizes extracted parameters to Argent tool schemas.
"""

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion
from src.leeth.runtime import leeth
from logger import get_logger
from pathlib import Path

# The routing state names the working directory the models were trained on: the project's usual
# location under the current user's home (a constant on purpose, see _build_state).
_CANONICAL_WORKDIR = os.environ.get("LEETH_STATE_WORKDIR") or (Path.home() / "Desktop" / "Argent").as_posix()

logger = get_logger("leeth_dispatcher")

def _unwrap_quotes(s: str) -> str:
    """Снимает только СБАЛАНСИРОВАННЫЕ обёртки кавычек.

    Зачем: .strip('\"`') ломает команды вида '"/usr/local/bin/my app" --help'
    (снимает ведущую кавычку, теряя группировку аргументов shell).
    """
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"`":
        return s[1:-1].strip()
    return s

# ── Static Deny-List (belt-and-braces before the neural safety gate) ─────────
# Any match means: NEVER fast-dispatch this command. Worst case is a fallback to
# the LLM turn (which has its own approval flow) — conservative by design.
# Destructive verbs are anchored to the start of the command so that benign
# traps like `echo 'rm -rf /' > cleanup.sh` or `grep 'drop database' src/`
# are NOT caught; injection vectors (pipe-to-shell, -enc, iex) match anywhere.
DENY_PATTERNS = [
    # re.M: деструктив на второй строке многострочного param ('echo hi\nrm -rf /')
    # иначе обходил анкер ^.
    re.compile(r"^\s*rm\s+(?:-{1,2}[\w-]+\s+)*-[\w]*[rf]", re.I | re.M),
    # Long-form флаги: rm --recursive / --force (короткая форма выше их не ловит).
    re.compile(r"^\s*rm\s+(?:.*\s)?--(?:recursive|force)\b", re.I | re.M),
    re.compile(r"^\s*del\s+/[\w]*[fsq]", re.I | re.M),
    re.compile(r"^\s*(?:rd|rmdir)\s+/s\b", re.I | re.M),
    re.compile(r"^\s*format\s+\w:", re.I | re.M),
    # Disk-wipe tokens may appear mid-query ('format D: /fs:NTFS' after
    # 'запусти') — the extract head grabs only the flag span, so these
    # two variants are intentionally NOT anchored to the string start.
    re.compile(r"\bformat\s+\w:", re.I),
    re.compile(r"/fs:(?:NTFS|FAT|exFAT|ReFS)\b", re.I),
    re.compile(r"^\s*mkfs(?:\.\w+)?\b", re.I | re.M),
    re.compile(r"^\s*dd\s+if=", re.I | re.M),
    re.compile(r"^\s*shred\b|^\s*diskpart\b|^\s*vssadmin\b|^\s*wevtutil\b", re.I | re.M),
    re.compile(r"^\s*drop\s+(?:database|table)\b|^\s*truncate\s+table\b", re.I | re.M),
    re.compile(r"^\s*shutdown\s+(?:/[srta]\b|-\w)", re.I | re.M),
    re.compile(r"^\s*chmod\s+-R\s+777\b", re.I | re.M),
    re.compile(r"^\s*remove-item\s+[^|]*(?:-recurse|-force)", re.I | re.M),
    re.compile(r"^\s*reg\s+delete\b|^\s*net\s+stop\b", re.I | re.M),
    # Путь к интерпретатору тоже ловим: '| /bin/bash', '| /usr/bin/python3'.
    # Сегмент пути ограничен ({1,3} x {1,48}) — иначе \S* на payload из тысяч
    # '|' даёт квадратичный откат (ReDoS ~1с на 10KB).
    re.compile(r"\|\s*(?:[\w.\-]{0,48}[\\/]){0,3}(?:ba|z|da|k)?sh\b"),
    re.compile(r"\|\s*(?:[\w.\-]{0,48}[\\/]){0,3}(?:python3?|pwsh|powershell|perl|ruby|node)\b", re.I),
    re.compile(r"(?:^|\s)-enc(?:odedcommand)?\b", re.I),
    re.compile(r"\binvoke-expression\b|\biex\b", re.I),
    re.compile(r"\beval\s*\$\(", re.I),
    re.compile(r"\$\(\s*(?:curl|wget)\b", re.I),
    # Fork bomb: символьные классы вместо .* — линейное время (иначе ReDoS:
    # 10KB payload подвешивал матчинг на ~83 секунды).
    re.compile(r":\(\)\s*\{[^|]*\|[^&]*&[^}]*\}\s*;?"),
    # Chained destructive commands: 'git clean && rm -rf /var/lib' — the
    # destructive part rides after &&/;| so start-anchored patterns miss it.
    re.compile(r"(?:&&|;|\|)\s*rm\s+(?:-{1,2}[\w-]+\s+)*-[\w]*[rf]", re.I),
    re.compile(r"(?:&&|;|\|)\s*rm\s+(?:.*\s)?--(?:recursive|force)\b", re.I),
    re.compile(r"(?:&&|;|\|)\s*(?:rd|rmdir)\s+/s\b", re.I),
    re.compile(r"(?:&&|;|\|)\s*del\s+/[\w]*[fsq]", re.I),
    # System-critical process kill (svchost and friends) — never a routine op.
    # Порядок флагов не важен: /f /im X и /im X /f.
    re.compile(r"taskkill\s+(?:/\w+\s+)?/im\s+(?:svchost|csrss|wininit|services|lsass|smss|winlogon)\.exe", re.I),
    # Full docker wipe: -af, -a -f, -fa — любые флаги с 'a' перед/после f.
    re.compile(r"docker\s+system\s+prune\b[^&|;]*\s-[a-z]*a", re.I),
    # Disk free-space wipe (cipher /w) — hours of unrecoverable overwrite.
    re.compile(r"\bcipher\s+/w\b", re.I),
]

# Tools that may ride the fast path.
# Routine reads, file discovery, calculations, shell commands, and browser navigation.
ALLOWED_FAST_TOOLS = {
    "read_file", "run_command", "grep_search", "find_by_name",
    "search_files", "list_directory", "calculate", "browser_open",
    "read_background_command",
}

READ_FAST_TOOLS = {"read_file", "list_directory", "search_files", "find_by_name", "browser_open", "read_background_command"}
EXEC_FAST_TOOLS = {"run_command"}

# Reasoning/compound queries must never ride the fast path: they need the
# full LLM (planning, review, test-writing, analysis), and the choice head's
# known failure mode is confidently dispatching a tool on them (battery:
# expected=UNFAMILIAR -> best=run_command conf 0.92). Calibrated offline
# (scripts/_d4_reasoning_guard.py): battery false 2 -> 0, precision 92 -> 100%,
# one lost correct dispatch (a compound git-status query — borderline anyway).
REASONING_RE = re.compile(
    r"^(составь|спроектируй|проведи|сделай ревью|проанализируй|исследуй|сравни|"
    r"оптимизируй|улучши|ускорь|задокументируй|напиши тесты|добавь тесты|"
    r"напиши документацию|разберись|найди причину|почему|подготовь roadmap|"
    r"write tests|add tests|review|analyze|research|compare|optimize|refactor|document)\b",
    re.IGNORECASE)
COMPOUND_RE = re.compile(
    r"^(прочитай|открой|запусти|проверь|покажи|найди|посмотри|выполни)\b.+\bи\s+"
    r"(исправ|удали|обнови|найди|перепиши|почини|закрой|составь)",
    re.IGNORECASE)


@dataclass
class FastDispatchResult:
    """Result structure compatible with Argent's agent fast-path interception.

    steps: for multi-intent chains — the ordered sub-dispatches (each with its
    own decision_id for the review gate). None for a single-step dispatch.
    """

    tool_name: str
    arguments: Dict[str, Any]
    confidence: float
    elapsed_ms: float
    raw_query: str
    decision_id: Optional[str] = None
    steps: Optional[List["FastDispatchResult"]] = None


class LeethDispatcher:
    """System One Fast-Path Dispatcher powered by Leeth."""

    _instance: Optional["LeethDispatcher"] = None
    _instance_lock = threading.Lock()

    def __init__(self, confidence_threshold: float = 0.65, familiarity_threshold: Optional[float] = None):
        # 0.65 calibrated on the prod battery + held-out val routing split
        # (scripts/_d3_threshold_calib.py, mmBERT router): the need_tool gate
        # (mmBERT-noul, 99.5% val acc) does the precision work — the conf gate
        # is nearly flat on val (89-91% at every threshold). 0.65 vs 0.80:
        # +5 correct dispatches on the battery (18->23, 1 more false), +16 on
        # val (93->109, 1 more false). ALLOWED_FAST_TOOLS is read-only, the
        # deny list + run_command safety fan-out guard the risky ones.
        self.confidence_threshold = confidence_threshold
        # OOD gate: states with MLM NLL above this are delegated to the LLM
        # (None = gate off until calibrated on the live model).
        self.familiarity_threshold = familiarity_threshold
        self._runtime = leeth

    @classmethod
    def get_instance(cls) -> "LeethDispatcher":
        """Thread-safe singleton (double-checked locking)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = LeethDispatcher()
        return cls._instance

    @property
    def is_available(self) -> bool:
        return self._runtime.is_available

    def try_fast_dispatch(
        self,
        user_query: str,
        _depth: int = 0,
        trajectory_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[FastDispatchResult]:
        """Evaluates whether the user query represents an immediate single-step tool call.

        Returns:
            FastDispatchResult if a routine tool call is recognized with high confidence.
            For multi-intent queries (UNFAMILIAR + intent_count>=2 + a confident
            first-intent span) returns a chain result with .steps populated.
            None if the query requires LLM reasoning, code writing, or human approval.
        """
        if not user_query or not isinstance(user_query, str):
            return None

        q_clean = user_query.strip()
        # Skip slash commands and shell prefixes
        if len(q_clean) < 3 or q_clean.startswith("/") or q_clean.startswith("!"):
            return None
        # System notifications (heartbeat wakeups/interrupts) are LLM-turn
        # material: the fast path has nothing to dispatch, and logging them
        # polluted the feedback buffer (~25% of all records). Filter BEFORE
        # the neural pass — no decision record, no tokens spent.
        if q_clean.startswith("[Heartbeat") or q_clean.startswith("condition met"):
            return None

        if not self.is_available:
            return None

        t0 = time.perf_counter()

        # Step 1: Standalone evaluation. Self-contained queries (e.g. 'найди файл runtime.py',
        # 'посчитай 128*64') have complete intent in the text and avoid history bias.
        state, questions = self._build_state_and_questions(q_clean, context=None)
        batch = self._runtime.decide(state, questions)

        need_tool_res = batch.results.get("need_tool")
        tool_res = batch.results.get("tool")
        param_res = batch.results.get("param")

        # Step 2: If standalone evaluation did not yield a confident routine tool and real session
        # context exists (e.g. active background processes or anaphoric follow-up), re-evaluate with context.
        if (
            context
            and (
                not need_tool_res
                or not tool_res
                or tool_res.confidence < self.confidence_threshold
                or tool_res.choice in ("NO_TOOL", "UNFAMILIAR")
                or not need_tool_res.value
            )
        ):
            has_real_context = bool(
                context.get("active_tasks")
                or context.get("last_target")
                or context.get("last_tool")
                or context.get("prev_turn")
            )
            if has_real_context:
                ctx_state, ctx_questions = self._build_state_and_questions(q_clean, context=context)
                ctx_batch = self._runtime.decide(ctx_state, ctx_questions)
                ctx_tool = ctx_batch.results.get("tool")
                if ctx_tool and ctx_tool.confidence >= self.confidence_threshold:
                    state, questions, batch = ctx_state, ctx_questions, ctx_batch
                    need_tool_res = ctx_batch.results.get("need_tool")
                    tool_res = ctx_tool
                    param_res = ctx_batch.results.get("param")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # OOD gate: unfamiliar state -> delegate to the LLM without logging a
        # decision proposal (the fast path is not qualified to act here).
        if (
            self.familiarity_threshold is not None
            and batch.state_nll is not None
            and batch.state_nll > self.familiarity_threshold
        ):
            logger.info("State NLL %.3f above familiarity threshold %.3f for '%s' — delegating.",
                        batch.state_nll, self.familiarity_threshold, q_clean[:60])
            return None

        if not need_tool_res or not tool_res or not param_res:
            return None

        # ── Active-learning: log the decision (verdict attached later by reviewer) ──
        decision_id = None
        try:
            from src.leeth import feedback as leeth_feedback
            decision_id = leeth_feedback.log_decision(
                query=q_clean,
                state=state,
                proposal={
                    "need_tool": {"value": need_tool_res.value, "probability": round(need_tool_res.probability, 4)},
                    "tool": {"choice": tool_res.choice, "confidence": round(tool_res.confidence, 4)},
                    "param": param_res.text,
                },
                meta={"latency_ms": round(elapsed_ms, 2), "device": self._runtime.device},
                trajectory_id=trajectory_id,
            )
        except Exception as _fb_err:
            logger.debug("Feedback logging skipped: %s", _fb_err)

        # Phase 4: link the decision to the turn trajectory (both directions).
        # Skipped decisions are attached too — they are the RLCD correction signal.
        if decision_id:
            try:
                from src.leeth import trajectory as leeth_trajectory
                leeth_trajectory.attach_decision(decision_id, trajectory_id)
            except Exception as _traj_err:
                logger.debug("Trajectory attach skipped: %s", _traj_err)

        # If model decides this is conversational or reasoning, delegate to LLM
        if not need_tool_res.value or need_tool_res.probability < self.confidence_threshold:
            self._mark_skip(decision_id, "need_tool_false")
            return None

        tool_name = tool_res.choice
        confidence = tool_res.confidence

        # UNFAMILIAR gate: the model explicitly says this request is outside its
        # fast-path territory -> delegate to the LLM (checked before confidence).
        # EXCEPTION (depth 0 only): a multi-intent query — UNFAMILIAR on the
        # full query + intent_count>=2 + a confident first-intent span — is
        # decomposed into a chain of full-gate sub-dispatches instead.
        if tool_name == "UNFAMILIAR":
            if _depth == 0:
                chain = self._try_decomposition(q_clean, batch, elapsed_ms)
                if chain is not None:
                    return chain
            logger.info("Model flagged query as UNFAMILIAR for '%s' — delegating.", q_clean[:60])
            self._mark_skip(decision_id, "unfamiliar")
            return None

        # Differentiated risk-aware confidence threshold (Plan v2.1):
        # 1. Safe read tools: threshold relaxed to 0.60 ONLY under strict invariants
        #    (need_tool >= 0.85, non-reasoning, non-compound).
        # 2. Exec tools (run_command): strictly requires >= 0.80 before entering neural safety.
        # 3. Other tools (calculate): base confidence threshold (0.65).
        required_threshold = self.confidence_threshold
        if tool_name in READ_FAST_TOOLS:
            is_pure_read = (
                need_tool_res.probability >= 0.85
                and not REASONING_RE.search(q_clean)
                and not COMPOUND_RE.search(q_clean)
            )
            if is_pure_read:
                required_threshold = min(self.confidence_threshold, 0.60)
        elif tool_name in EXEC_FAST_TOOLS:
            required_threshold = max(self.confidence_threshold, 0.80)

        # Tool confidence gate: ensure the model is confident in the tool choice
        if confidence < required_threshold:
            logger.info("Tool confidence %.2f below threshold %.2f for '%s'", confidence, required_threshold, tool_name)
            self._mark_skip(decision_id, "low_tool_confidence")
            return None

        if tool_name not in ALLOWED_FAST_TOOLS:
            self._mark_skip(decision_id, "tool_not_allowed")
            return None

        extracted_param = _unwrap_quotes(param_res.text.strip())
        # Если ведущая/замыкающая кавычка пришла из промпта ("{normalized}"), а в запросе её нет — снимаем
        if extracted_param and extracted_param[0] in "'\"`" and not q_clean.startswith(extracted_param[0]):
            extracted_param = extracted_param[1:].strip()
        if extracted_param and extracted_param[-1] in "'\"`" and not q_clean.endswith(extracted_param[-1]):
            extracted_param = extracted_param[:-1].strip()

        # A background-output read needs a PID. "что там с сервером?" names none: the PID is the session's running
        # process, and only when that is unambiguous. Otherwise the LLM, which can look at the process list, decides.
        param_from_context = False
        if tool_name == "read_background_command":
            pid = self._background_pid(q_clean, extracted_param, context)
            if pid is None:
                self._mark_skip(decision_id, "no_background_pid")
                return None
            extracted_param, param_from_context = str(pid), True

        # Для browser_open параметр должен содержать целевой URL
        if tool_name == "browser_open":
            url_match = re.search(r"https?://[^\s\"']+", extracted_param)
            if not url_match:
                url_match = re.search(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s\"']*)?", extracted_param)
            if url_match:
                extracted_param = url_match.group(0).rstrip(".,;:\"'")

        if not extracted_param:
            self._mark_skip(decision_id, "empty_param")
            return None

        # Boundary check: Extracted parameter MUST be present within the actual user query,
        # preventing leakage or extraction from prompt boilerplate (e.g. 'Working directory:')
        param_in_query = extracted_param.lower() in q_clean.lower()
        param_in_state = extracted_param.lower() in state.lower()
        if not param_in_query and not param_in_state and not param_from_context:
            logger.info("Extracted param '%s' not present in user query or session state, rejecting fast-dispatch.", extracted_param)
            self._mark_skip(decision_id, "param_not_in_query_or_state")
            return None

        # Mid-word extraction guard: span pointers sometimes grab a mid-word
        # suffix (e.g. "down /s /t 0" out of "shutdown /s /t 0"). A legitimate
        # parameter starts at a word boundary within the query.
        _idx = q_clean.lower().find(extracted_param.lower())
        if _idx > 0 and not param_from_context and (q_clean[_idx - 1].isalnum() or q_clean[_idx - 1] in "_/."):
            logger.info("Extracted param '%s' starts mid-word in query, rejecting fast-dispatch.", extracted_param)
            self._mark_skip(decision_id, "mid_word_param")
            return None

        # Normalize find_by_name to search_files
        dispatch_tool = "search_files" if tool_name == "find_by_name" else tool_name

        # Build tool arguments
        arguments = self._build_arguments(dispatch_tool, extracted_param)
        if not arguments:
            self._mark_skip(decision_id, "arg_build_failed")
            return None

        # Static deny-list applies to ANY tool: a destructive-looking string must
        # never ride along as the parameter of a harmless-looking tool either
        # (e.g. a misroute of "shutdown /s" into calculate).
        if any(p.search(extracted_param) for p in DENY_PATTERNS):
            logger.info("Fast-dispatch denied by static deny-list: '%s'", extracted_param)
            self._mark_skip(decision_id, "deny_list")
            return None

        # Extra safety check for run_command: neural verification
        if dispatch_tool == "run_command":
            safety_q = [
                NoulQuestion(id="is_safe", question="Является ли данная терминальная команда безопасной для выполнения?"),
                ChoiceQuestion(
                    id="action",
                    question="Какое действие должен предпринять агент перед запуском этой команды?",
                    choices=["execute_immediately", "request_user_approval", "block_command", "sanitize_arguments"],
                ),
            ]
            safety_state = f"User shell execution request:\nCommand: {extracted_param}\nCaller: Argent Agent"
            safety_batch = self._runtime.decide(safety_state, safety_q)
            is_safe = safety_batch.results.get("is_safe")
            action = safety_batch.results.get("action")
            if decision_id:
                try:
                    from src.leeth import feedback as leeth_feedback
                    leeth_feedback.attach_safety(
                        decision_id,
                        safety_state,
                        is_safe.probability if is_safe else 0.0,
                        action.choice if action else None,
                    )
                except Exception:
                    pass

            if (
                not is_safe
                or not is_safe.value
                or is_safe.probability < 0.80
                or not action
                or action.choice != "execute_immediately"
            ):
                logger.info("Fast-dispatch skipped for command '%s' due to neural safety evaluation.", extracted_param)
                self._mark_skip(decision_id, "neural_safety")
                return None

            # Guard micro-adapter verification on safety state (latency <= 0.05 ms):
            guard_score = self._runtime.run_adapter("guard", question_id="is_safe")
            if guard_score is not None and guard_score < 0.20:
                logger.info("Fast-dispatch skipped for command '%s' due to guard micro-adapter (score=%.3f).", extracted_param, guard_score)
                self._mark_skip(decision_id, "guard_adapter_reject")
                return None

        # ── CDCL arbitrator (SHADOW): formal consistency check, never blocks ────
        # Positioned after all procedural gates: validates the constraint base
        # against would-be dispatches and measures solver latency in the hot path.
        try:
            from src.leeth.arbitrator import get_arbitrator
            _safety_allow = None
            _exec_imm = None
            if dispatch_tool == "run_command":
                _safety_allow = bool(is_safe and is_safe.value and is_safe.probability >= 0.80)
                _exec_imm = bool(action and action.choice == "execute_immediately")
            arb_res = get_arbitrator().arbitrate(
                tool=tool_name,
                need_tool=bool(need_tool_res.value),
                param_text=param_res.text,
                query=q_clean,
                safety_allow=_safety_allow,
                execute_immediately=_exec_imm,
                context={"gates": "dispatch", "confidence": round(confidence, 4)},
            )
            if arb_res.error:
                logger.debug("Arbitrator shadow error (fail-open): %s", arb_res.error)
            else:
                logger.info(
                    "Arbitrator[shadow] %s core=%s model_tool=%s (%.1fms) q='%s'",
                    "SAT" if arb_res.sat else "UNSAT",
                    arb_res.core or "-", arb_res.model_tool or "-",
                    arb_res.latency_ms, q_clean[:60],
                )
        except Exception as _arb_err:
            logger.debug("Arbitrator shadow failed (fail-open): %s", _arb_err)

        # Phase 2: persist the final dispatch payload on the decision record —
        # record_to_samples needs tool_name/arguments to build choice/extract targets.
        if decision_id:
            try:
                from src.leeth import feedback as leeth_feedback
                leeth_feedback.attach_dispatch(decision_id, dispatch_tool, arguments)
            except Exception as _disp_err:
                logger.debug("Dispatch attach skipped: %s", _disp_err)

        return FastDispatchResult(
            tool_name=dispatch_tool,
            arguments=arguments,
            confidence=confidence,
            elapsed_ms=elapsed_ms,
            raw_query=q_clean,
            decision_id=decision_id,
        )

    # ── Multi-intent decomposition (phase-2 triage) ─────────────────────────
    # Fires ONLY on UNFAMILIAR full-query verdicts: the phase-2 forward asks
    # intent_count (score, 1/2/3+) and first_intent (span). A confident
    # count>=2 + a valid span -> a chain of sub-dispatches, each re-triaged
    # through the FULL gate stack (try_fast_dispatch at depth 1: no nested
    # chains). Any sub-intent failing its gates aborts the whole chain ->
    # the caller delegates to the LLM (fail-safe).
    _DECOMP_COUNT_CONFIDENCE = 0.60
    _DECOMP_MAX_STEPS = 3

    def _try_decomposition(
        self, q_clean: str, batch: Any, elapsed_ms: float
    ) -> Optional[FastDispatchResult]:
        """Attempts to split an UNFAMILIAR multi-intent query into a chain."""
        try:
            from src.leeth.primitives import (
                INTENT_COUNT_QUESTION,
                FIRST_INTENT_QUESTION,
                ScoreQuestion,
                ExtractQuestion,
            )
            decomp_q = [
                ScoreQuestion(id="intent_count", question=INTENT_COUNT_QUESTION, levels=3),
                ExtractQuestion(id="first_intent", question=FIRST_INTENT_QUESTION, target_type="task"),
            ]
            decomp_state = self._build_state(q_clean)
            decomp_batch = self._runtime.decide(decomp_state, decomp_q)
        except Exception as _e:
            logger.debug("Decomposition phase-2 triage failed: %s", _e)
            return None

        count_res = decomp_batch.results.get("intent_count")
        span_res = decomp_batch.results.get("first_intent")
        if not count_res or not span_res:
            return None
        if count_res.level < 2 or count_res.confidence < self._DECOMP_COUNT_CONFIDENCE:
            return None

        raw_span = (span_res.text or "").strip().strip("'\"`")
        first_span = self._clean_span_to_query(raw_span, q_clean)
        if len(first_span) < 4 or first_span.lower() == q_clean.lower():
            return None
        if first_span.lower() not in q_clean.lower():
            logger.debug("Decomposition span '%s' not found in query, aborting.", first_span[:50])
            return None

        steps: List[FastDispatchResult] = []
        remaining = q_clean
        # the chain never exceeds the model's own intent count: meta-text that
        # MENTIONS a tool ("pytest крутит другой агент") must not add steps
        max_steps = min(self._DECOMP_MAX_STEPS, max(count_res.level, 2))
        for depth in range(max_steps):
            if depth == 0:
                span = first_span
            else:
                sub = self.inspect_query(remaining)
                sub_tool = (sub.get("tool") or {}).get("choice", "")
                if sub_tool != "UNFAMILIAR":
                    # the remainder triages as a single intent -> one more step
                    span = remaining.strip()
                else:
                    # the remainder still looks UNFAMILIAR (e.g. meta-text left
                    # over) — ask the phase-2 extract head for its first intent
                    span = self._phase2_first_intent(remaining)
                    if len(span) < 4:
                        break
            if len(span) < 4:
                break
            step_disp = self.try_fast_dispatch(span, _depth=1)
            if step_disp is None:
                logger.info(
                    "Decomposition aborted: sub-intent '%s' failed the gate stack.", span[:60]
                )
                return None
            steps.append(step_disp)
            remaining = self._strip_leading_span(remaining, span)
            if len(remaining.strip(" \t\n,.;:")) < 4:
                break

        if len(steps) < 2:
            return None
        logger.info(
            "Multi-intent decomposition: %d steps from '%s' (%s)",
            len(steps), q_clean[:50], " -> ".join(s.tool_name for s in steps),
        )
        return FastDispatchResult(
            tool_name=steps[0].tool_name,
            arguments=steps[0].arguments,
            confidence=min(s.confidence for s in steps),
            elapsed_ms=elapsed_ms,
            raw_query=q_clean,
            decision_id=None,
            steps=steps,
        )

    @staticmethod
    def _background_pid(query: str, extracted: str, context: Optional[Dict[str, Any]]) -> Optional[int]:
        """PID a background-output request means: one written in the request, else the session's only running process."""
        m = re.search(r"(?<![0-9A-Za-z_])[0-9]+(?![0-9A-Za-z_])", extracted or "") or re.search(r"(?<![0-9A-Za-z_])[0-9]+(?![0-9A-Za-z_])", query or "")
        if m:
            return int(m.group(0))
        tasks = (context or {}).get("active_tasks") or []
        pids = [t.get("pid") for t in tasks if str(t.get("pid", "")).isdigit()]
        return int(pids[0]) if len(pids) == 1 else None

    def _phase2_first_intent(self, text: str) -> str:
        """Phase-2 first_intent span extraction for an arbitrary text (read-only)."""
        try:
            from src.leeth.primitives import FIRST_INTENT_QUESTION, ExtractQuestion
            batch = self._runtime.decide(self._build_state(text), [
                ExtractQuestion(id="first_intent", question=FIRST_INTENT_QUESTION, target_type="task"),
            ])
            span_res = batch.results.get("first_intent")
            if not span_res:
                return ""
            return self._clean_span_to_query((span_res.text or "").strip().strip("'\"`"), text)
        except Exception as _e:
            logger.debug("Phase-2 first_intent extraction failed: %s", _e)
            return ""

    @staticmethod
    def _clean_span_to_query(span: str, query: str) -> str:
        """The extract pointer sometimes includes state boilerplate
        ('session: "прочитай config.py'). Keep the longest whitespace-delimited
        fragment of the span that actually occurs in the user query at a word
        boundary (mid-word matches like 'ли папку build' inside
        'удали папку build' are pointer artifacts, not real fragments)."""
        span = (span or "").strip().strip("'\"`")
        if not span or span.lower() in query.lower():
            return span
        tokens = span.split()
        best = ""
        ql = query.lower()
        n = len(tokens)
        for i in range(n):
            for j in range(n, i, -1):
                frag = " ".join(tokens[i:j]).strip().strip("'\"`").strip()
                if len(frag) <= len(best):
                    continue
                idx = ql.find(frag.lower())
                if idx < 0:
                    continue
                before_ok = idx == 0 or not (ql[idx - 1].isalnum() or ql[idx - 1] == "_")
                end_i = idx + len(frag)
                after_ok = end_i >= len(ql) or not (ql[end_i].isalnum() or ql[end_i] == "_")
                if before_ok and after_ok:
                    best = frag
                    break  # longest j for this i found
        return best

    @staticmethod
    def _strip_leading_span(text: str, span: str) -> str:
        """Removes the first occurrence of span and cleans connector debris."""
        idx = text.lower().find(span.lower())
        if idx < 0:
            return ""
        rem = text[:idx] + text[idx + len(span):]
        rem = re.sub(r"^[\s,.;:]+", "", rem)
        rem = re.sub(
            # Порядок: длинные коннекторы раньше коротких ('и ещё' до 'и'),
        # иначе 'и ещё X' обрезается до 'ещё X'.
        r"^(?:после этого|а также|а ещё|и ещё|потом|затем|также|плюс|и)\b[,:]?\s*",
            "", rem, flags=re.IGNORECASE,
        )
        rem = re.sub(r"[\s,.;:]+$", "", rem)
        return rem.strip()

    @staticmethod
    def _build_state(q_clean: str, context: Optional[Dict[str, Any]] = None) -> str:
        """The canonical routing state format (shared by phase 1 and phase 2).

        The first character is lowercased: sentence-initial capitalization is
        semantically meaningless, but the BPE tokenizer treats 'Что' and 'что'
        as different tokens, and the training corpus is lowercase-majority.
        Normalizing here makes live queries match the trained distribution.
        """
        normalized = q_clean[:1].lower() + q_clean[1:] if q_clean[:1].isalpha() else q_clean
        wd = _CANONICAL_WORKDIR
        has_real_context = bool(
            context and (
                context.get("active_tasks")
                or context.get("last_target")
                or context.get("last_tool")
                or context.get("prev_turn")
            )
        )
        if not has_real_context:
            return f"Incoming user request in Argent session:\n\"{normalized}\"\nWorking directory: {wd}"

        active_tasks = context.get("active_tasks") or []
        if active_tasks:
            active_str = "; ".join(
                f"PID {t.get('pid', '?')} ({t.get('command', '?')}, status={t.get('status', 'running')}, elapsed={t.get('elapsed', '?')})"
                for t in active_tasks
            )
        else:
            active_str = "None"

        last_tgt = context.get("last_target") or "None"
        last_tl = context.get("last_tool") or "None"
        prev_turn = context.get("prev_turn") or "None"

        return (
            f"Incoming user request in Argent session:\n"
            f'"{normalized}"\n'
            f"Active background processes: {active_str}\n"
            f"Recent session context:\n"
            f"- Last target: {last_tgt}\n"
            f"- Last tool: {last_tl}\n"
            f"- Previous turn: {prev_turn}\n"
            f"Working directory: {wd}"
        )

    @classmethod
    def _build_state_and_questions(cls, q_clean: str, context: Optional[Dict[str, Any]] = None):
        """Builds the routing state and the batched System One questions."""
        choices = [
            "read_file", "write_file", "replace_in_file", "run_command",
            "grep_search", "find_by_name", "list_directory", "browser_open",
            "browser_click", "browser_type", "browser_upload_file", "calculate",
            "ask_user_questions", "read_background_command", "NO_TOOL", "UNFAMILIAR"
        ]
        state = cls._build_state(q_clean, context=context)
        questions = [
            NoulQuestion(id="need_tool", question="Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"),
            ChoiceQuestion(id="tool", question="Какой инструмент Argent должен быть вызван в первую очередь?", choices=choices),
            ExtractQuestion(id="param", question="Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)"),
        ]
        return state, questions

    def inspect_query(self, user_query: str) -> Dict[str, Any]:
        """Dry-run inspection: what would the fast path do with this query?

        Read-only: no feedback logging, no skip marking, no execution.
        Returns the raw heads, the gate verdict (skip_reason) and, when every
        gate passes, the would-be dispatch (tool + arguments).
        """
        return self.triage([user_query])[0]

    def triage(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Batch-triage N queries through the fast-path gates in ONE fan-out pass.

        Read-only like inspect_query (no feedback logging, no execution), but
        all routing heads run in a single decide_batch forward, and all
        run_command safety evaluations run in one additional batched pass.
        """
        infos: List[Dict[str, Any]] = []
        for q in queries:
            info: Dict[str, Any] = {"query": q, "dispatch": None, "skip_reason": None}
            if not q or not isinstance(q, str):
                info["skip_reason"] = "empty_query"
            else:
                q_clean = q.strip()
                info["query"] = q_clean
                if len(q_clean) < 3 or q_clean.startswith("/") or q_clean.startswith("!"):
                    info["skip_reason"] = "slash_or_short"
                elif q_clean.startswith("[Heartbeat") or q_clean.startswith("condition met"):
                    info["skip_reason"] = "system_message"
                elif not self.is_available:
                    info["skip_reason"] = "runtime_unavailable"
            infos.append(info)

        live_idx = [i for i, inf in enumerate(infos) if inf["skip_reason"] is None]
        if not live_idx:
            return infos

        _, questions = self._build_state_and_questions(infos[live_idx[0]]["query"])
        state_strings = [self._build_state_and_questions(infos[i]["query"])[0] for i in live_idx]
        batch_list = self._runtime.decide_batch(state_strings, questions)

        safety_pending: List[tuple] = []  # (info_index, safety_state)
        for row, i in enumerate(live_idx):
            info = infos[i]
            batch = batch_list[row]
            q_clean = info["query"]
            info["elapsed_ms"] = batch.elapsed_ms
            info["state_nll"] = round(batch.state_nll, 4) if batch.state_nll is not None else None

            # OOD gate: unfamiliar state -> delegate instead of risking a
            # confident wrong dispatch (checked FIRST, before any head).
            if (
                self.familiarity_threshold is not None
                and batch.state_nll is not None
                and batch.state_nll > self.familiarity_threshold
            ):
                info["skip_reason"] = "state_unfamiliar"
                continue

            need_tool_res = batch.results.get("need_tool")
            tool_res = batch.results.get("tool")
            param_res = batch.results.get("param")
            if not need_tool_res or not tool_res or not param_res:
                info["skip_reason"] = "missing_heads"
                continue

            info["need_tool"] = {"value": need_tool_res.value, "probability": round(need_tool_res.probability, 4)}
            info["tool"] = {"choice": tool_res.choice, "confidence": round(tool_res.confidence, 4)}
            info["param_raw"] = param_res.text

            if not need_tool_res.value or need_tool_res.probability < self.confidence_threshold:
                info["skip_reason"] = "need_tool_false"
                continue

            tool_name = tool_res.choice
            if tool_name == "UNFAMILIAR":
                info["skip_reason"] = "unfamiliar"
                continue
            # Reasoning/compound guard: planning/review/test-writing/analysis
            # queries and two-verb compounds need the LLM, not a fast tool.
            if REASONING_RE.search(q_clean) or COMPOUND_RE.search(q_clean):
                info["skip_reason"] = "reasoning_task"
                continue
            if tool_res.confidence < self.confidence_threshold:
                info["skip_reason"] = "low_tool_confidence"
                continue
            if tool_name not in ALLOWED_FAST_TOOLS:
                info["skip_reason"] = "tool_not_allowed"
                continue

            extracted_param = param_res.text.strip().strip("'\"`")
            if not extracted_param:
                info["skip_reason"] = "empty_param"
                continue
            info["param"] = extracted_param

            if extracted_param.lower() not in q_clean.lower():
                info["skip_reason"] = "param_not_in_query"
                continue

            _idx = q_clean.lower().find(extracted_param.lower())
            if _idx > 0 and (q_clean[_idx - 1].isalnum() or q_clean[_idx - 1] in "_/."):
                info["skip_reason"] = "mid_word_param"
                continue

            dispatch_tool = "search_files" if tool_name == "find_by_name" else tool_name
            arguments = self._build_arguments(dispatch_tool, extracted_param)
            if not arguments:
                info["skip_reason"] = "arg_build_failed"
                continue

            if any(p.search(extracted_param) for p in DENY_PATTERNS):
                info["skip_reason"] = "deny_list"
                continue

            if dispatch_tool == "run_command":
                safety_state = f"User shell execution request:\nCommand: {extracted_param}\nCaller: Argent Agent"
                safety_pending.append((i, safety_state))
                continue

            info["dispatch"] = {"tool": dispatch_tool, "arguments": arguments, "confidence": tool_res.confidence}

        # Safety fan-out: one extra batched forward for ALL run_command candidates.
        if safety_pending:
            safety_q = [
                NoulQuestion(id="is_safe", question="Является ли данная терминальная команда безопасной для выполнения?"),
                ChoiceQuestion(
                    id="action",
                    question="Какое действие должен предпринять агент перед запуском этой команды?",
                    choices=["execute_immediately", "request_user_approval", "block_command", "sanitize_arguments"],
                ),
            ]
            safety_batches = self._runtime.decide_batch([sp[1] for sp in safety_pending], safety_q)
            for (i, _), safety_batch in zip(safety_pending, safety_batches):
                info = infos[i]
                is_safe = safety_batch.results.get("is_safe")
                action = safety_batch.results.get("action")
                info["safety"] = {
                    "is_safe": round(is_safe.probability, 4) if is_safe else None,
                    "action": action.choice if action else None,
                }
                if (
                    not is_safe
                    or not is_safe.value
                    or is_safe.probability < 0.80
                    or not action
                    or action.choice != "execute_immediately"
                ):
                    info["skip_reason"] = "neural_safety"
                else:
                    info["dispatch"] = {
                        "tool": "run_command",
                        "arguments": {"command": info["param"]},
                        "confidence": info["tool"]["confidence"],
                    }
        return infos

    @staticmethod
    def _mark_skip(decision_id: Optional[str], reason: str) -> None:
        """Records why a logged decision did not result in a dispatch (active-learning metadata)."""
        if not decision_id:
            return
        try:
            from src.leeth import feedback as leeth_feedback
            leeth_feedback.mark_skipped(decision_id, reason)
        except Exception:
            pass

    @staticmethod
    def _is_path_like(param: str) -> bool:
        """A path names a place: one token, or it carries a separator, an extension, a drive or ~.  The extract head sometimes returns the request
        itself ('открой его') or a span that runs into the next line of the state; neither is a path, and reading it would only fail."""
        if re.search(r"[\r\n]", param):
            return False
        return not (re.search(r"\s", param) and not re.search(r"[\\/.~]", param))

    def _build_arguments(self, tool_name: str, param: str) -> Optional[Dict[str, Any]]:
        """Normalizes extracted text to tool parameters matching real Argent tool signatures."""
        if tool_name in ("read_file", "list_directory") and not self._is_path_like(param):
            return None
        if tool_name == "read_file":
            return {"file_path": param}
        elif tool_name == "run_command":
            # A command starts with an executable name or a path, never with a Russian word: that is the request itself handed back as a "command"
            # ('проверь сеть', 'пересобери zip архив'), and the shell would only answer "not recognised".
            if re.match(r"\s*[А-Яа-яЁё]", param):
                return None
            return {"command": param}
        elif tool_name == "grep_search":
            # Match search_ops.grep_search(directory=..., pattern=...)
            return {"pattern": param, "directory": "."}
        elif tool_name in ("find_by_name", "search_files"):
            # Fast-path search_files is for finding files by name.
            # Reject full sentences, phrases with spaces > 3, or conversational phrasing.
            words = param.split()
            if len(words) > 3 or len(param) > 80:
                return None
            conversational_markers = ("можешь", "поищи", "сравни", "интернет", "расскажи", "объясни")
            if any(m in param.lower() for m in conversational_markers):
                return None
            return {"name_contains": param}
        elif tool_name == "list_directory":
            # Match file_ops.list_directory(dir_path=...)
            return {"dir_path": param}
        elif tool_name == "calculate":
            # Arithmetic expressions must contain digits or standard math functions
            param_lower = param.lower().strip()
            if "\\" in param or re.search(r"[{}]", param):      # LaTeX (\cdot, \sqrt{..}, \frac{..}{..}) is not an expression the calculator evaluates
                return None
            has_digit = any(c.isdigit() for c in param)
            has_math_func = any(
                f in param_lower
                for f in ("sqrt", "pi", "sin", "cos", "tan", "log", "round", "abs", "max", "min", "factorial")
            )
            if not (has_digit or has_math_func):
                return None
            # Reject shell flags or destructive command tokens mistakenly passed to calculate
            shell_verbs = (
                "rm", "del", "format", "curl", "wget", "bash", "sh", "chmod", "kill",
                "shutdown", "reg", "mkfs", "dd", "shred", "diskpart", "truncate",
                "drop", "net", "vssadmin", "wevtutil", "invoke-expression", "iex",
                "remove-item", "eval", "sudo",
            )
            if any(tok.startswith(shell_verbs) for tok in param_lower.split()):
                return None
            return {"expression": param}
        elif tool_name == "browser_open":
            url = param.strip()
            if not url.startswith(("http://", "https://")):
                if "." in url and not url.startswith(("/", "\\")):
                    url = f"https://{url}"
                else:
                    return None
            return {"url": url}
        elif tool_name == "browser_click":
            return {"selector": param}
        elif tool_name == "browser_type":
            return {"text": param}
        elif tool_name == "browser_upload_file":
            return {"file_path": param}
        elif tool_name == "ask_user_questions":
            return {"question": param}
        elif tool_name == "read_background_command":
            pid_match = re.search(r"\b\d+\b", param)
            pid = int(pid_match.group(0)) if pid_match else param
            return {"pid": pid}
        return {}


dispatcher = LeethDispatcher.get_instance()
