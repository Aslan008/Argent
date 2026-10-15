"""Phase 4: trajectory store analysis — decisions per turn, outcomes, approvals.

Reads src/leeth/data/leeth_trajectories.jsonl + leeth_feedback.jsonl and prints
a compact summary: turn statuses, decisions per turn, outcome distribution,
approval rate, command exit codes, and LLM-path correction signals.
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth import trajectory as T  # noqa: E402
from src.leeth import feedback as F  # noqa: E402


def main() -> None:
    trajs = T.load_trajectories()
    decisions = {d["id"]: d for d in F.load_records()}
    if not trajs:
        print("no trajectories recorded yet")
        return

    statuses = Counter(t.get("status") or "open" for t in trajs)
    events = Counter()
    outcomes = Counter()
    approvals_ok = approvals_total = 0
    exit_codes = Counter()
    llm_tools = Counter()
    linked = 0

    for t in trajs:
        ids = t.get("decision_ids") or []
        linked += len(ids)
        for did in ids:
            d = decisions.get(did)
            if d and d.get("outcome"):
                outcomes[d["outcome"].get("type", "?")] += 1
        for ev in t.get("events") or []:
            et = ev.get("type", "?")
            events[et] += 1
            if et == "approval":
                approvals_total += 1
                approvals_ok += 1 if ev.get("approved") else 0
            elif et == "command_exit":
                exit_codes[ev.get("code")] += 1
            elif et == "llm_tool_exec":
                llm_tools[ev.get("tool", "?")] += 1

    n = len(trajs)
    per_turn = [len(t.get("decision_ids") or []) for t in trajs]
    print(f"trajectories: {n}  |  statuses: {dict(statuses)}")
    print(f"decisions linked: {linked} (avg {sum(per_turn) / n:.2f}/turn, max {max(per_turn) if per_turn else 0})")
    print(f"decision outcomes: {dict(outcomes) or '-'}")
    print(f"events: {dict(events)}")
    if approvals_total:
        print(f"approvals: {approvals_ok}/{approvals_total} granted ({100 * approvals_ok / approvals_total:.0f}%)")
    if exit_codes:
        print(f"exit codes: {dict(sorted(exit_codes.items(), key=lambda x: -x[1]))}")
    if llm_tools:
        print(f"llm-path tools: {dict(llm_tools.most_common(8))}")

    # Turns where a decision was logged AND the LLM then acted — RLCD gold:
    # the LLM's actual behavior is the correction target for the proposal.
    gold = 0
    for t in trajs:
        ids = t.get("decision_ids") or []
        has_llm_action = any(ev.get("type") == "llm_tool_exec" for ev in t.get("events") or [])
        if ids and has_llm_action:
            gold += 1
    print(f"RLCD-gold turns (decision + LLM follow-up): {gold}")


if __name__ == "__main__":
    main()