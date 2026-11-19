"""Markdown/JSON reports for a bench run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from argent.bench.runner import CaseResult


def to_markdown(results: Iterable[CaseResult]) -> str:
    results = list(results)
    passed = sum(1 for r in results if r.passed)
    lines = [
        "# Agent bench",
        "",
        f"Итого: **{passed}/{len(results)}** кейсов прошли.",
        "",
        "| Кейс | Результат | Время | События |",
        "|---|---|---:|---:|",
    ]
    for r in results:
        mark = "✅" if r.passed else "❌"
        lines.append(f"| {r.case_id} | {mark} | {r.duration_s:.1f}s | {r.events} |")
    lines.append("")
    for r in results:
        lines.append(f"## {r.case_id}")
        if r.setup_error:
            lines.append(f"**Setup:** {r.setup_error}")
        if r.agent_error:
            lines.append(f"**Agent:** {r.agent_error}")
        if r.timed_out:
            lines.append("**Timeout:** агент не уложился в лимит.")
        for c in r.checks:
            mark = "✅" if c.passed else "❌"
            detail = c.error or f"exit={c.exit_code}"
            lines.append(f"- {mark} `{c.cmd}` — {detail}")
            if not c.passed and c.output:
                lines.append("  ```")
                lines.extend("  " + line for line in c.output.splitlines()[-15:])
                lines.append("  ```")
        if r.diff:
            lines.append("**Diff:**")
            lines.append("```")
            lines.append(r.diff)
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def to_json(results: Iterable[CaseResult]) -> str:
    return json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2)


def write_report(results: List[CaseResult], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(results), encoding="utf-8")
    return path
