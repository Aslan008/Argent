"""What the person — and the model that started the workflow — is told when it ends or when they ask."""

from __future__ import annotations

from typing import List, Optional

from src.agent.workflow.calibration import LEVEL_NAMES
from src.agent.workflow.plan import DONE, FAILED, PENDING, RUNNING, SKIPPED
from src.agent.workflow.store import RunState

_MARK = {DONE: "✓", FAILED: "✗", SKIPPED: "–", RUNNING: "▸", PENDING: "·"}
_STATUS = {"planning": "составляется план", "scheduled": "запланирован (ночной запуск)", "running": "выполняется", "stopped": "остановлен (можно продолжить)",
           "done": "завершён", "failed": "завершён с ошибками", "cancelled": "отменён"}


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} мин" if seconds >= 60 else f"{seconds:.0f} с"


def render(state: RunState, final_log: Optional[List[str]] = None, brief: bool = False) -> str:
    plan = state.get_plan()
    lines = [f"Ultracode {state.id}: {_STATUS.get(state.status, state.status)}",
             f"Цель: {state.goal}",
             f"Уровень {state.level} — {LEVEL_NAMES.get(state.level, '?')}; затрачено {_minutes(state.spent_seconds)}; "
             f"обращений к советнику: {state.advisor_calls}"]
    if plan is None:
        lines.append(state.result or "План ещё не составлен.")
        return "\n".join(lines)

    lines.append("")
    for s in plan.steps:
        tail = []
        if s.attempts > 1:
            tail.append(f"попыток: {s.attempts}")
        if s.verdict:
            tail.append(f"критик: {s.verdict}")
        if s.changed:
            tail.append("файлы: " + ", ".join(s.changed[:5]) + (f" и ещё {len(s.changed) - 5}" if len(s.changed) > 5 else ""))
        lines.append(f"{_MARK.get(s.status, '?')} {s.id}. {s.title}" + (f" ({'; '.join(tail)})" if tail else ""))
        if s.status in (FAILED, SKIPPED) and s.error:
            lines.append(f"    причина: {s.error.splitlines()[0][:200]}")

    if not brief:
        for s in plan.steps:
            if s.status == DONE and s.report:
                lines += ["", f"[{s.id}] {s.title}:", s.report.strip()[:1200]]
    if final_log:
        lines += ["", "Итоговые проверки:", *final_log]
    for note in (state.advisor or {}).get("notes", []):
        lines += ["", note]
    unfinished = [s for s in plan.steps if s.status in (PENDING, RUNNING)]
    if unfinished and state.status in ("stopped", "running"):
        lines += ["", f"Осталось шагов: {len(unfinished)}. Продолжить: /ultracode resume."]
    return "\n".join(lines)
