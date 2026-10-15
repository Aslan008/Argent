"""
src/leeth/enrich.py — Trajectory-derived verdicts for pending decisions (Phase 2).

The agent-turn trajectory holds the ground truth about what happened after a
fast-path decision. Enrichment joins feedback records with their trajectories
and derives implicit verdicts where the evidence is unambiguous:

  executed + approval denied                         → reject
  executed + command_exit == 0                       → accept (+ outcome executed_ok)
  executed + command_exit != 0                       → outcome executed_error (no verdict:
                                                       a failed command does not prove
                                                       the choice itself was wrong)
  skipped + LLM used an in-taxonomy tool             → partial + correction (RLCD target:
                                                       the LLM behavior is the teacher)
  skipped + LLM used only out-of-taxonomy tools      → partial (a tool was needed)
  skipped (need_tool_false) + LLM made no tool calls → accept (skip confirmed)

Attribution is trusted only for single-decision trajectories; multi-decision
turns need credit assignment (Phase 5). Heartbeat/system queries are excluded
entirely — they are maintenance noise, not user intent.
"""

from typing import Any, Dict

from src.leeth import feedback, trajectory
from logger import get_logger

logger = get_logger("leeth_enrich")

_HEARTBEAT_MARKS = ("heartbeat",)


def _is_heartbeat(text: Any) -> bool:
    return any(m in str(text or "").lower() for m in _HEARTBEAT_MARKS)


def derive_verdicts() -> Dict[str, Any]:
    """Derives implicit verdicts for pending feedback records from trajectories.

    Never overwrites an explicit verdict. Returns a stats dict for reporting.
    """
    stats = {
        "scanned": 0, "already_labeled": 0, "heartbeat_skipped": 0,
        "no_trajectory": 0, "multi_decision": 0,
        "reject_denied": 0, "accept_exit0": 0, "outcome_error": 0,
        "partial_llm_tool": 0, "partial_llm_other": 0,
        "accept_skip_confirmed": 0, "ambiguous": 0,
    }
    classes = feedback._known_classes()
    trajs = {t.get("id"): t for t in trajectory._load_all()}

    for rec in feedback.load_records():
        stats["scanned"] += 1
        if rec.get("verdict"):
            stats["already_labeled"] += 1
            continue
        if _is_heartbeat(rec.get("query")):
            stats["heartbeat_skipped"] += 1
            continue
        tid = rec.get("trajectory_id")
        traj = trajs.get(tid) if tid else None
        if not traj:
            stats["no_trajectory"] += 1
            continue
        if len(traj.get("decision_ids") or []) != 1:
            stats["multi_decision"] += 1
            continue

        events = traj.get("events") or []
        llm_calls = [e for e in events if e.get("type") == "llm_tool_exec"]
        llm_tools = [e.get("tool") for e in llm_calls if e.get("tool")]
        denied = any(e.get("type") == "approval" and e.get("approved") is False
                     for e in events)
        exits = [e for e in events if e.get("type") == "command_exit"]

        did = rec["id"]
        if rec.get("skip_reason"):
            if llm_tools:
                in_tax = [t for t in llm_tools if t in classes]
                if in_tax:
                    tool = in_tax[0]
                    args = next((e.get("args") for e in llm_calls
                                 if e.get("tool") == tool and e.get("args")), None)
                    param = next(iter(args.values()), None) if isinstance(args, dict) and args else None
                    feedback.submit_verdict(
                        did, "partial", correction_tool=tool, correction_param=param,
                        source="trajectory",
                        reason="RLCD: LLM used this tool after the fast-path skip",
                    )
                    stats["partial_llm_tool"] += 1
                else:
                    feedback.submit_verdict(
                        did, "partial", source="trajectory",
                        reason="RLCD: LLM acted with tools outside the fast-path taxonomy",
                    )
                    stats["partial_llm_other"] += 1
            elif rec.get("skip_reason") == "need_tool_false":
                feedback.submit_verdict(
                    did, "accept", source="trajectory",
                    reason="Skip confirmed: the LLM made no tool calls",
                )
                stats["accept_skip_confirmed"] += 1
            else:
                stats["ambiguous"] += 1
            continue

        # Executed decision (no skip_reason): outcome evidence.
        if denied:
            feedback.submit_verdict(did, "reject", source="trajectory",
                                    reason="Approval denied for the dispatched action")
            stats["reject_denied"] += 1
        elif exits:
            code = exits[-1].get("code")
            if code == 0:
                feedback.record_outcome(did, "executed_ok", {"exit_code": 0})
                feedback.submit_verdict(did, "accept", source="trajectory",
                                        reason="Command exited 0")
                stats["accept_exit0"] += 1
            else:
                feedback.record_outcome(did, "executed_error", {"exit_code": code})
                stats["outcome_error"] += 1
        else:
            stats["ambiguous"] += 1
    return stats