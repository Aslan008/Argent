"""A run's report written to a file — by the runner, not by the model.

"Write the summary to reports/daily.md" used to mean giving an unattended model a
file-writing tool. Having the runner write what the run reported needs no tool at
all, so a read-only job can still leave a paper trail, and the places a job can
write to are exactly the ones its owner named, nothing the model picks.

The destination is confined to the project and to plain-text formats: a scheduled
job must not be able to overwrite a source file, a script, or Argent's own state.
"""

import json
import re
from datetime import datetime
from pathlib import Path

ALLOWED_SUFFIXES = (".md", ".txt", ".log", ".jsonl")

# Past this a report is rotated aside: an every-5-minutes job would otherwise
# grow one file forever.
ROTATE_AT_BYTES = 2_000_000

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_PROTECTED = {"automations.json", "automation_state.json", "automation_runs.jsonl",
              "automation_memory.json"}


class ReportPathError(ValueError):
    """The report destination is not acceptable."""


def expand(template: str, name: str, now: datetime) -> str:
    values = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H-%M"),
        "datetime": now.strftime("%Y-%m-%d_%H-%M"),
        "name": re.sub(r"[^\w.-]+", "_", name).strip("_") or "automation",
        "weekday": now.strftime("%a").lower(),
    }

    def _sub(match):
        key = match.group(1)
        if key not in values:
            raise ReportPathError(
                f"в пути отчёта неизвестная подстановка {{{key}}} (можно: "
                f"{', '.join('{' + k + '}' for k in values)})")
        return values[key]

    return _PLACEHOLDER.sub(_sub, template)


def resolve_report_path(template: str, name: str, now: datetime, root: Path) -> Path:
    """The absolute destination, or ReportPathError."""
    text = expand((template or "").strip(), name, now)
    if not text:
        raise ReportPathError("путь отчёта пуст")
    root = Path(root).resolve()
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    try:
        path = path.resolve()
        path.relative_to(root)
    except (ValueError, OSError):
        raise ReportPathError(f"отчёт должен лежать внутри проекта ({root}): {template}") from None
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ReportPathError(
            f"отчёт можно писать только в {', '.join(ALLOWED_SUFFIXES)}: {path.name}")
    parts = {p.lower() for p in path.relative_to(root).parts}
    if ".git" in parts or (".argent" in parts and path.name.lower() in _PROTECTED):
        raise ReportPathError("в служебные файлы Argent и .git отчёт писать нельзя")
    return path


def append_report(path: Path, name: str, status: str, finished: datetime,
                  summary: str, trigger: str = "schedule") -> None:
    """Add one run to the report file (creating folders as needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists() and path.stat().st_size > ROTATE_AT_BYTES:
            path.replace(path.with_name(f"{path.stem}.old{path.suffix}"))
    except OSError:
        pass
    stamp = finished.strftime("%Y-%m-%d %H:%M")
    if path.suffix.lower() == ".jsonl":
        line = json.dumps({"time": finished.isoformat(timespec="seconds"), "name": name,
                           "status": status, "trigger": trigger, "summary": summary},
                          ensure_ascii=False)
        text = line + "\n"
    else:
        text = f"## {stamp} — {name} [{status}]\n\n{summary.strip()}\n\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)
