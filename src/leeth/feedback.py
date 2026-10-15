"""
src/leeth/feedback.py — Active-Learning Feedback Loop for Leeth.

Records every System One decision (proposal) made by the fast-path dispatcher,
collects verdicts (accept / reject / partial + correction) from a reviewer
(the main Argent agent via an LLM review call, or the human via a terminal
prompt), and converts labeled decisions into LeethSample objects consumable
by the existing training pipeline (train.py / finetune.py).

Design notes:
- JSONL store with atomic rewrite on updates (small file, single process).
- Verdicts are always explicit. Unlabeled ("pending") decisions are NEVER
  used for training — no silent self-labeling, no "делает вид" learning.
- Corrections outside the known choice taxonomy are logged but skipped for
  choice targets (the choice head size must stay compatible with the
  existing checkpoint; head resizing is a separate retraining step).
- Logging is auto-disabled under pytest and can be disabled with
  LEETH_FEEDBACK_DISABLED=1 or config.set_leeth_feedback_enabled(False).
"""

import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.leeth.dataset_generator import LeethSample
from logger import get_logger

logger = get_logger("leeth_feedback")

DEFAULT_STORE_PATH = "src/leeth/data/leeth_feedback.jsonl"

NEED_TOOL_Q = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"
TOOL_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
PARAM_Q = "Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)"
SAFE_Q = "Является ли данная терминальная команда безопасной для выполнения?"
ACTION_Q = "Какое действие должен предпринять агент перед запуском этой команды?"

TOOL_CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL",
]
ACTION_CHOICES = ["execute_immediately", "request_user_approval", "block_command", "sanitize_arguments"]

_lock = threading.Lock()


def _store_path() -> Path:
    return Path(os.environ.get("LEETH_FEEDBACK_PATH", DEFAULT_STORE_PATH))


def _is_enabled() -> bool:
    if os.environ.get("LEETH_FEEDBACK_DISABLED") == "1":
        return False
    if os.environ.get("LEETH_FEEDBACK_FORCE") == "1":
        return True
    # Keep automated test runs from polluting the training buffer.
    if "pytest" in sys.modules:
        return False
    try:
        import config
        return config.get_leeth_feedback_enabled()
    except Exception:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Store primitives
# ─────────────────────────────────────────────────────────────────────────────

def _load_all() -> List[Dict[str, Any]]:
    path = _store_path()
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt feedback line in %s", path)
    return records


def _save_all(records: List[Dict[str, Any]]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _update_record(decision_id: str, mutate) -> bool:
    with _lock:
        records = _load_all()
        for rec in records:
            if rec.get("id") == decision_id:
                mutate(rec)
                _save_all(records)
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Decision logging (called by the dispatcher)
# ─────────────────────────────────────────────────────────────────────────────

def log_decision(
    query: str,
    state: str,
    proposal: Dict[str, Any],
    dispatch: Optional[Dict[str, Any]] = None,
    safety: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    trajectory_id: Optional[str] = None,
) -> Optional[str]:
    """Logs a Leeth decision (proposal) with verdict=None (pending review)."""
    if not _is_enabled():
        return None
    record = {
        "id": f"fb_{int(time.time())}_{uuid.uuid4().hex[:8]}",
        "ts": time.time(),
        "query": query,
        "state": state,
        "proposal": proposal,
        "dispatch": dispatch,
        "safety": safety,
        "skip_reason": None,
        "verdict": None,
        "correction": None,
        "verdict_source": None,
        "reason": "",
        "trajectory_id": trajectory_id,
        "outcome": None,
        "meta": meta or {},
    }
    with _lock:
        path = _store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record["id"]


def mark_skipped(decision_id: str, reason: str) -> None:
    """Records why a logged decision did not result in a dispatch."""
    if not decision_id:
        return

    def _mutate(rec: Dict[str, Any]) -> None:
        rec["skip_reason"] = reason

    try:
        _update_record(decision_id, _mutate)
    except Exception as exc:
        logger.debug("mark_skipped failed: %s", exc)


def attach_trajectory(decision_id: str, trajectory_id: str) -> bool:
    """Links a decision record to its agent-turn trajectory (Phase 4)."""

    def _mutate(rec: Dict[str, Any]) -> None:
        rec["trajectory_id"] = trajectory_id

    try:
        return _update_record(decision_id, _mutate)
    except Exception as exc:
        logger.debug("attach_trajectory failed: %s", exc)
        return False


def attach_dispatch(decision_id: str, tool_name: str, arguments: Dict[str, Any]) -> bool:
    """Attaches the final dispatch payload (tool + normalized args) to a decision.

    Written by the dispatcher right before a FastDispatchResult is returned:
    record_to_samples needs it to build choice/extract training targets.
    """

    def _mutate(rec: Dict[str, Any]) -> None:
        rec["dispatch"] = {"tool_name": tool_name, "arguments": arguments}

    try:
        return _update_record(decision_id, _mutate)
    except Exception as exc:
        logger.debug("attach_dispatch failed: %s", exc)
        return False


def record_outcome(
    decision_id: str,
    outcome_type: str,
    detail: Optional[Dict[str, Any]] = None,
) -> bool:
    """Records the real-world outcome of a dispatched decision (Phase 4).

    outcome_type: executed_ok | executed_error | approval_denied |
    approval_granted | command_exit | delegated_llm.
    """

    def _mutate(rec: Dict[str, Any]) -> None:
        rec["outcome"] = {"type": outcome_type, "ts": time.time(), **(detail or {})}

    try:
        return _update_record(decision_id, _mutate)
    except Exception as exc:
        logger.debug("record_outcome failed: %s", exc)
        return False


def attach_safety(decision_id: str, safety_state: str, is_safe_prob: float, action: Optional[str]) -> None:
    """Attaches the run_command neural-safety evaluation to a decision record."""

    def _mutate(rec: Dict[str, Any]) -> None:
        rec["safety"] = {
            "state": safety_state,
            "is_safe_prob": round(float(is_safe_prob), 4),
            "action": action,
        }

    try:
        _update_record(decision_id, _mutate)
    except Exception as exc:
        logger.debug("attach_safety failed: %s", exc)


def submit_verdict(
    decision_id: str,
    verdict: str,
    correction_tool: Optional[str] = None,
    correction_param: Optional[str] = None,
    source: str = "agent",
    reason: str = "",
) -> bool:
    """Attaches a reviewer verdict (accept/reject/partial) to a decision record."""
    if verdict not in ("accept", "reject", "partial"):
        raise ValueError(f"Invalid verdict '{verdict}'")
    correction = None
    if correction_tool or correction_param:
        correction = {"tool": correction_tool, "param": correction_param}

    def _mutate(rec: Dict[str, Any]) -> None:
        rec["verdict"] = verdict
        rec["correction"] = correction
        rec["verdict_source"] = source
        rec["reason"] = reason or rec.get("reason", "")

    ok = _update_record(decision_id, _mutate)
    if ok:
        logger.info("Feedback verdict '%s' recorded for %s (source=%s)", verdict, decision_id, source)
    return ok


def get_record(decision_id: str) -> Optional[Dict[str, Any]]:
    for rec in _load_all():
        if rec.get("id") == decision_id:
            return rec
    return None


def load_records() -> List[Dict[str, Any]]:
    return _load_all()


def labeled_records() -> List[Dict[str, Any]]:
    return [rec for rec in _load_all() if rec.get("verdict")]


def pending_count() -> int:
    return sum(1 for rec in _load_all() if not rec.get("verdict"))


def stats() -> Dict[str, Any]:
    records = _load_all()
    by_verdict: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    for rec in records:
        v = rec.get("verdict") or "pending"
        by_verdict[v] = by_verdict.get(v, 0) + 1
        if rec.get("verdict"):
            s = rec.get("verdict_source") or "unknown"
            by_source[s] = by_source.get(s, 0) + 1
    return {"total": len(records), "by_verdict": by_verdict, "by_source": by_source, "store": str(_store_path())}


# ─────────────────────────────────────────────────────────────────────────────
# Reviewer helpers
# ─────────────────────────────────────────────────────────────────────────────

FAST_TOOLS_HINT = "read_file, run_command, grep_search, search_files, list_directory, calculate"


def format_review_prompt(record: Dict[str, Any]) -> str:
    proposal = record.get("proposal") or {}
    dispatch = record.get("dispatch") or {}
    tool = dispatch.get("tool_name") or (proposal.get("tool") or {}).get("choice")
    conf = (proposal.get("tool") or {}).get("confidence")
    need = (proposal.get("need_tool") or {}).get("probability")
    param = dispatch.get("arguments") or proposal.get("param")
    return (
        f"Leeth (fast-path decision model) proposes to execute a tool call WITHOUT the LLM.\n"
        f"Review it and teach it.\n\n"
        f"User query: \"{record.get('query', '')}\"\n"
        f"Proposal:\n"
        f"- Tool: {tool}\n"
        f"- Arguments: {json.dumps(param, ensure_ascii=False)}\n"
        f"- Tool confidence: {conf}\n"
        f"- need_tool probability: {need}\n\n"
        f"Fast-path tools: {FAST_TOOLS_HINT}\n\n"
        "Respond with ONLY a JSON object:\n"
        '{"verdict": "accept|reject|partial", "correction_tool": "<tool or null>", '
        '"correction_param": "<string or null>", "reason": "<short>"}\n'
        "- accept: right tool, right arguments — execute.\n"
        "- partial: a tool IS needed but the tool or argument is wrong — provide correction_tool and/or correction_param.\n"
        "- reject: this query must NOT be fast-dispatched (conversational, ambiguous, or dangerous) — it should go to the LLM."
    )


def parse_review_response(text: str) -> Optional[Dict[str, Any]]:
    """Parses the reviewer's JSON verdict; returns None when unparseable."""
    if not text:
        return None
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    verdict = str(data.get("verdict", "")).lower().strip()
    if verdict not in ("accept", "reject", "partial"):
        return None
    return {
        "verdict": verdict,
        "correction_tool": data.get("correction_tool") or None,
        "correction_param": data.get("correction_param") or None,
        "reason": str(data.get("reason", ""))[:300],
    }


def human_review(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Interactive terminal review (questionary). Returns the same verdict dict."""
    try:
        import questionary
    except ImportError:
        logger.warning("questionary not available; human review unavailable.")
        return None
    print("\n" + "=" * 70)
    print(format_review_prompt(record))
    choice = questionary.select("Вердикт по предложению Leeth:", choices=["accept", "partial", "reject", "skip"]).ask()
    if choice in (None, "skip"):
        return None
    correction_tool = None
    correction_param = None
    if choice in ("partial", "reject"):
        correction_tool = questionary.text("Правильный инструмент (Enter = оставить предложенный):").ask() or None
        correction_param = questionary.text("Правильный параметр (Enter = оставить предложенный):").ask() or None
    reason = questionary.text("Причина (коротко):").ask() or ""
    return {
        "verdict": choice,
        "correction_tool": (correction_tool or None),
        "correction_param": (correction_param or None),
        "reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feedback → LeethSample conversion (consumed by finetune.py / train.py)
# ─────────────────────────────────────────────────────────────────────────────

_PARAM_KEYS = ("file_path", "command", "pattern", "name_contains", "dir_path", "expression", "url")


def _known_classes() -> Dict[str, int]:
    path = Path("src/leeth/data/choice_classes.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def record_to_samples(record: Dict[str, Any]) -> List[LeethSample]:
    """Converts one labeled decision into LeethSample objects for training."""
    samples: List[LeethSample] = []
    verdict = record.get("verdict")
    if not verdict:
        return samples
    correction = record.get("correction") or {}
    dispatch = record.get("dispatch") or {}
    proposal = record.get("proposal") or {}
    state = record.get("state") or ""
    if not state:
        return samples

    class_to_idx = _known_classes()

    # 1. need_tool (noul): accept/partial → a tool was needed; reject → it was not.
    # Exception: a trajectory-derived "accept" on a SKIPPED decision means the
    # skip itself was confirmed (the LLM made no tool calls) → need_tool=0.0.
    skip_confirmed = bool(record.get("skip_reason")) and verdict == "accept"
    need_target = 0.0 if (verdict == "reject" or skip_confirmed) else 1.0
    samples.append(LeethSample(
        state=state, question_type="noul", question=NEED_TOOL_Q, target_noul=need_target,
    ))

    # 2. tool choice + param extract — only when the correct tool is known.
    # Fallback: an accepted proposal IS the correct tool even when the dispatch
    # payload was never attached (older records) — the reviewer approved it.
    proposal_tool = (proposal.get("tool") or {}).get("choice")
    target_tool = correction.get("tool") or dispatch.get("tool_name") or (
        proposal_tool if verdict == "accept" and not skip_confirmed else None
    )
    if verdict == "reject" and not correction.get("tool"):
        target_tool = "NO_TOOL"
    if target_tool:
        if target_tool in class_to_idx:
            samples.append(LeethSample(
                state=state, question_type="choice", question=TOOL_Q,
                choices=TOOL_CHOICES, target_choice=target_tool,
            ))
            args = dispatch.get("arguments") or {}
            proposed_param = next(iter(args.values()), None) if args else proposal.get("param")
            target_param = correction.get("param") or proposed_param
            if verdict != "reject" and target_param:
                samples.append(LeethSample(
                    state=state, question_type="extract", question=PARAM_Q,
                    target_span_text=str(target_param),
                ))
        else:
            logger.warning("Correction tool '%s' is outside the trained taxonomy; choice target skipped.", target_tool)

    # 3. run_command safety heads. Only unambiguous verdicts teach safety:
    # accept = the executed command was safe; reject = it was not. A 'partial'
    # verdict corrects the TOOL choice and says nothing about command safety,
    # so no safety samples are generated for it.
    safety = record.get("safety")
    if safety and safety.get("state"):
        if verdict == "accept":
            samples.append(LeethSample(state=safety["state"], question_type="noul", question=SAFE_Q, target_noul=1.0))
            samples.append(LeethSample(
                state=safety["state"], question_type="choice", question=ACTION_Q,
                choices=ACTION_CHOICES, target_choice="execute_immediately",
            ))
        elif verdict == "reject":
            samples.append(LeethSample(state=safety["state"], question_type="noul", question=SAFE_Q, target_noul=0.0))
            samples.append(LeethSample(
                state=safety["state"], question_type="choice", question=ACTION_Q,
                choices=ACTION_CHOICES, target_choice="block_command",
            ))
    return samples


def build_training_samples(val_ratio: float = 0.2, seed: int = 7, correction_boost: int = 1) -> Tuple[List[LeethSample], List[LeethSample], Dict[str, Any]]:
    """Builds train/val LeethSample lists from labeled feedback records.

    correction_boost repeats samples from 'partial' (corrective) verdicts so a
    handful of corrections is not diluted by the bulk of accept/reject records.
    """
    records = labeled_records()
    all_samples: List[LeethSample] = []
    skipped = 0
    for rec in records:
        rec_samples = record_to_samples(rec)
        if rec_samples:
            boost = max(1, correction_boost) if rec.get("verdict") == "partial" else 1
            all_samples.extend(rec_samples * boost)
        else:
            skipped += 1
    if not all_samples:
        return [], [], {"labeled_records": len(records), "samples": 0, "skipped_records": skipped}

    # Deterministic split by record id so val never leaks into train across runs.
    train, val = [], []
    step = max(2, round(1 / max(val_ratio, 0.01)))
    for i, s in enumerate(all_samples):
        (val if (i % step == step - 1 and len(all_samples) >= 10) else train).append(s)
    info = {"labeled_records": len(records), "samples": len(all_samples), "train": len(train), "val": len(val), "skipped_records": skipped}
    return train, val, info