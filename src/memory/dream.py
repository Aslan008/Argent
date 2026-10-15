"""Consolidation of the memory ("dream"): what the recent sessions taught, written into notes while the PC is idle.

The model does not edit files. It reads the index, the notes that look related and an excerpt of the sessions
saved since the last run, and answers with a short JSON list of operations. Everything else is code:

  gate        enabled, a cloud model only with the person's consent, nobody else running, the PC on mains power
              (or the person allowed battery) and not used for a few minutes
  snapshot    a copy of the notes first, so one command puts everything back
  validate    names, sizes, secrets removed, a note written by the person is never touched (it becomes a
              suggestion in the report), at most a few operations and a few removals per run
  report      what was done, what was refused and why, what the model suggests — a file next to the notes

The request goes out at the background priority: it waits in the model server's queue behind everything the
person asks for, and for a quiet PC (src/agent/governor.py).

    python -m src.memory.dream [--force] [--dry-run] [--idle-minutes N]
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from logger import get_logger
from src.memory.store import MemoryRefused, Node, NodeStore, TYPES, normalize_name

log = get_logger("memory")

MAX_OPS = 8
MAX_FORGETS = 3
EXCERPT_CHARS = 12000          # what one run reads of the sessions (a small local model has a small window)
PER_SESSION_CHARS = 4000
TURN_CHARS = 600
RELATED_NODES = 4
RELATED_CHARS = 1200
IDLE_MINUTES = 5
LOCK_STALE_SECONDS = 2 * 60 * 60
STATE_NAME = ".dream.json"
REPORTS_DIR = ".reports"
REPORTS_KEPT = 30


# --- gate ------------------------------------------------------------------------------------------------------

@dataclass
class Gate:
    ok: bool
    reason: str = ""


def idle_seconds() -> Optional[float]:
    """Seconds since the last key or mouse input (Windows). None when it cannot be told."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        class _Info(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = _Info()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        tick = ctypes.windll.kernel32.GetTickCount()
        return ((tick - info.dwTime) & 0xFFFFFFFF) / 1000.0
    except Exception:
        return None


def on_battery() -> bool:
    try:
        import psutil
        battery = psutil.sensors_battery()
        return bool(battery is not None and battery.power_plugged is False)
    except Exception:
        return False


def _lock_path(root: Path) -> Path:
    return root / ".argent" / "run" / "dream.lock"


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def take_lock(root: Path) -> bool:
    path = _lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stat = path.stat()
        pid = int(path.read_text(encoding="utf-8").strip() or 0)
        if _pid_alive(pid) and time.time() - stat.st_mtime < LOCK_STALE_SECONDS and pid != os.getpid():
            return False
    except (OSError, ValueError):
        pass
    try:
        path.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        return True               # a lock that cannot be written must not stop the run
    return True


def release_lock(root: Path) -> None:
    try:
        _lock_path(root).unlink()
    except OSError:
        pass


def check_gate(target_cloud: bool = False, force: bool = False, idle_minutes: float = IDLE_MINUTES,
               idle: Callable[[], Optional[float]] = idle_seconds,
               battery: Callable[[], bool] = on_battery) -> Gate:
    import config
    if not config.is_memory_enabled():
        return Gate(False, "память выключена (/memory on)")
    if target_cloud and not config.is_memory_cloud_consented():
        return Gate(False, "модель облачная, а согласия на отправку выдержек из диалогов нет (/memory dream consent on)")
    if force:
        return Gate(True)
    if battery() and not config.is_dream_on_battery_allowed():
        return Gate(False, "ноутбук работает от батареи")
    seconds = idle()
    if seconds is not None and seconds < idle_minutes * 60:
        return Gate(False, f"компьютером пользуются (последний ввод {int(seconds)} с назад)")
    return Gate(True)


# --- state -----------------------------------------------------------------------------------------------------

def state_path(store: NodeStore) -> Path:
    return store.root / STATE_NAME


def load_state(store: NodeStore) -> dict:
    try:
        data = json.loads(state_path(store).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(store: NodeStore, state: dict) -> None:
    from atomic_io import atomic_write_text
    store.root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(state_path(store), json.dumps(state, ensure_ascii=False, indent=2))


# --- what the sessions say -------------------------------------------------------------------------------------

def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content
                         if isinstance(part, dict) and part.get("type") in (None, "text"))
    return ""


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def excerpt_of(messages: list, limit: int = PER_SESSION_CHARS) -> str:
    """What the person asked and how each turn ended, as short lines (the most recent turns win the budget)."""
    turns, ask, last = [], None, ""
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        text = _text_of(message.get("content"))
        if role == "user" and text.strip():
            if ask is not None:
                turns.append((ask, last))
            ask, last = text, ""
        elif role == "assistant" and text.strip():
            last = text
    if ask is not None:
        turns.append((ask, last))
    lines = []
    for ask, answer in turns:
        lines.append(f"USER: {_clip(ask, TURN_CHARS)}")
        if answer:
            lines.append(f"ASSISTANT: {_clip(answer, TURN_CHARS // 2)}")
    out, used = [], 0
    for line in reversed(lines):
        if used + len(line) + 1 > limit:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(reversed(out))


@dataclass
class Gathered:
    text: str = ""
    sessions: int = 0
    newest: str = ""           # saved_at of the newest session read: the next run starts after it


def gather(project_root: Path, since: str = "", limit: int = EXCERPT_CHARS) -> Gathered:
    import session as sessions_mod
    from src.agent.secret_redactor import redact

    root = os.path.normcase(os.path.abspath(str(project_root)))
    chosen = []
    for meta in sessions_mod.list_sessions() or []:
        cwd = meta.get("cwd") or ""
        saved = str(meta.get("saved_at") or "")
        try:
            inside = os.path.normcase(os.path.abspath(cwd)).startswith(root)
        except Exception:
            inside = False
        if inside and saved and saved > since:
            chosen.append((saved, meta))
    chosen.sort(key=lambda s: s[0])
    parts, used, newest, count = [], 0, "", 0
    for saved, meta in chosen:
        data = sessions_mod.load_session(meta.get("id") or "")
        if not data:
            continue
        body = excerpt_of(data.get("messages") or [])
        newest = saved
        if not body:
            continue
        block = f"=== session {saved[:16]} ===\n{redact(body)}"
        if used + len(block) > limit:
            # keep the newest: drop from the front until it fits
            while parts and used + len(block) > limit:
                used -= len(parts.pop(0)) + 1
            count = len(parts)
            if len(block) > limit:
                block = block[-limit:]
        parts.append(block)
        used += len(block) + 1
        count = len(parts)
    return Gathered(text="\n".join(parts), sessions=count, newest=newest)


# --- the request and the answer --------------------------------------------------------------------------------

_RULES = """You maintain the long-term memory of a coding assistant: short Markdown notes read in later sessions.
You receive the current index, a few related notes and excerpts of recent sessions. Decide what, if anything,
deserves to be remembered.

Write a note only for what will still matter in a later session and cannot be read from the code or git:
- who the user is and how they like to work, corrections they made ("feedback"),
- decisions and their reason, goals, constraints that are not obvious ("project"),
- where things live outside the repository ("reference").
Do NOT write: progress of a task, what a file contains, anything the code shows, temporary states, secrets, keys,
passwords. Turn relative dates into absolute ones. Prefer updating a related note to creating a near-duplicate.
If two notes contradict each other, update the older one and say so in "conflicts". Notes marked [user] are
written by the person: never update or forget them, put the idea into "suggestions" instead.

Answer with ONE JSON object and nothing else:
{"ops":[
  {"op":"create","name":"short-name","type":"user|feedback|project|reference","description":"one line","body":"text, [[links]] allowed","reason":"why"},
  {"op":"update","name":"existing-name","description":"one line","body":"the full new text","reason":"why"},
  {"op":"forget","name":"existing-name","reason":"why (obsolete, wrong, duplicate)"}],
 "conflicts":["..."],"suggestions":["..."]}
At most 8 operations, at most 3 forgets. Write notes in the language the user speaks. If nothing is worth
saving: {"ops":[]}"""

_RULES_GLOBAL = """
GLOBAL notes (the section "GLOBAL INDEX") are read in EVERY project of this user. Each operation may carry
"scope":"global" to act on one; without it the operation is about the project's notes. Rules for global notes:
- only what is true of the PERSON in any project: who they are, how they like to work, corrections ("user" or
  "feedback" notes). Never a path, a module, a decision about this code or anything that names this project.
- a project note and a global note that contradict each other: put it in "conflicts" and correct the one that is
  wrong (the project's note wins inside this project; the global one is for what holds everywhere).
- at most 3 operations on global notes per run."""


def build_messages(store: NodeStore, gathered: Gathered, global_store: Optional[NodeStore] = None) -> list:
    def listing(s: NodeStore) -> str:
        return "\n".join(f"- {n.name} [{n.source}] ({n.type}): {n.description}" for n in s.all()) or "(no notes yet)"

    def related_of(s: NodeStore, tag: str = "") -> list:
        found = []
        for _, node in s.search(gathered.text[:3000], limit=RELATED_NODES):
            found.append(f"--- {tag}{node.name} [{node.source}] ---\n{node.body[:RELATED_CHARS]}")
        return found

    related = related_of(store)
    user = f"## INDEX\n{listing(store)}\n\n"
    if global_store is not None:
        user += f"## GLOBAL INDEX\n{listing(global_store)}\n\n"
        related += related_of(global_store, "global: ")
    user += f"## RELATED NOTES\n{chr(10).join(related) or '(none)'}\n\n## RECENT SESSIONS\n{gathered.text}"
    rules = _RULES + (_RULES_GLOBAL if global_store is not None else "")
    return [{"role": "system", "content": rules}, {"role": "user", "content": user}]


@dataclass
class Op:
    op: str
    name: str
    description: str = ""
    body: str = ""
    type: str = "project"
    reason: str = ""
    scope: str = "project"                # "global": a note read in every project


MAX_GLOBAL_OPS = 3
GLOBAL_TYPES = ("user", "feedback")        # what is true of the person; project facts do not belong there


def parse_answer(text: str) -> tuple:
    """(ops, conflicts, suggestions) from the model's text. Raises ValueError when there is no JSON object."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in the answer")
    depth, in_str, esc, end = 0, False, False, -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise ValueError("the JSON object is not closed")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("the answer is not an object")
    ops = []
    for raw in data.get("ops") or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("op", "")).strip().lower()
        if kind not in ("create", "update", "forget"):
            continue
        ops.append(Op(op=kind, name=str(raw.get("name", "")), description=str(raw.get("description", "") or ""),
                      body=str(raw.get("body", "") or ""), type=str(raw.get("type", "") or "project"),
                      reason=str(raw.get("reason", "") or ""),
                      scope="global" if str(raw.get("scope", "")).strip().lower() in ("global", "глобальная")
                      else "project"))
    def strings(key):
        return [str(x).strip() for x in (data.get(key) or []) if str(x).strip()][:10]
    return ops, strings("conflicts"), strings("suggestions")


# --- applying --------------------------------------------------------------------------------------------------

@dataclass
class Report:
    applied: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    skipped: str = ""                      # why nothing ran
    sessions: int = 0
    snapshot: str = ""
    global_snapshot: str = ""
    dry_run: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.applied) and not self.dry_run

    def text(self) -> str:
        if self.skipped:
            return f"Консолидация памяти пропущена: {self.skipped}."
        head = f"Консолидация памяти: диалогов прочитано {self.sessions}, изменений {len(self.applied)}"
        if self.dry_run:
            head += " (пробный прогон, ничего не записано)"
        lines = [head + "."]
        lines += [f"  + {x}" for x in self.applied]
        lines += [f"  ! отклонено: {x}" for x in self.refused]
        lines += [f"  ~ противоречие: {x}" for x in self.conflicts]
        lines += [f"  ? предложение: {x}" for x in self.suggestions]
        if self.snapshot:
            lines.append("Вернуть как было: /memory undo")
        return "\n".join(lines)


def apply_ops(store: NodeStore, ops: List[Op], report: Report, now: Optional[datetime] = None,
              global_store: Optional[NodeStore] = None) -> None:
    forgets = 0
    global_ops = 0
    for index, op in enumerate(ops):
        if index >= MAX_OPS:
            report.refused.append(f"{op.op} {op.name}: больше {MAX_OPS} операций за один раз")
            continue
        try:
            name = normalize_name(op.name)
        except MemoryRefused as e:
            report.refused.append(f"{op.op} {op.name!r}: {e}")
            continue
        target = store
        if op.scope == "global":
            if global_store is None:
                report.refused.append(f"{op.op} global:{name}: глобальная консолидация выключена "
                                      f"(/memory dream global on)")
                continue
            if global_ops >= MAX_GLOBAL_OPS:
                report.refused.append(f"{op.op} global:{name}: больше {MAX_GLOBAL_OPS} операций с глобальными "
                                      f"заметками за один раз")
                continue
            if op.op == "create" and op.type not in GLOBAL_TYPES:
                report.refused.append(f"create global:{name}: в глобальную память идёт только то, что верно о "
                                      f"вас в любом проекте (тип {' или '.join(GLOBAL_TYPES)}), а не «{op.type}»")
                continue
            global_ops += 1
            target = global_store
        label = f"global:{name}" if target is not store else name
        existing = target.get(name)
        if existing is not None and existing.is_user_owned and op.op in ("create", "update", "forget"):
            report.suggestions.append(f"заметка '{label}' написана вами, модель предлагает ({op.op}): {op.reason or '—'}")
            continue
        try:
            if op.op == "forget":
                if forgets >= MAX_FORGETS:
                    report.refused.append(f"forget {label}: больше {MAX_FORGETS} удалений за один раз")
                    continue
                if existing is None:
                    report.refused.append(f"forget {label}: такой заметки нет")
                    continue
                if not report.dry_run:
                    target.forget(name, by="dream", now=now)
                forgets += 1
                report.applied.append(f"забыта «{label}» ({op.reason or 'без причины'}) — в корзине, "
                                      f"/memory restore {label}")
                continue
            if op.op == "update" and existing is None:
                op.op = "create"
            if op.op == "create" and op.type not in TYPES:
                op.type = "project"
            if not report.dry_run:
                result = target.put(name, op.description, op.body, type=op.type if op.op == "create" else
                                    (existing.type if existing else op.type), source="dream", now=now)
                if not result.changed:
                    continue
                report.applied.append(f"{'создана' if result.created else 'обновлена'} «{label}» — "
                                      f"{result.node.description}")
            else:
                report.applied.append(f"{'создать' if existing is None else 'обновить'} «{label}»")
        except MemoryRefused as e:
            report.refused.append(f"{op.op} {label}: {e}")


def _write_report(store: NodeStore, report: Report, now: datetime) -> Optional[Path]:
    from atomic_io import atomic_write_text
    folder = store.root / REPORTS_DIR
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{now.strftime('%Y%m%d-%H%M%S')}.md"
        atomic_write_text(path, report.text() + "\n")
        NodeStore._prune(folder, REPORTS_KEPT)
        return path
    except OSError:
        return None


def run(store: NodeStore, ask: Callable[[list], str], project_root: Path, force: bool = False, dry_run: bool = False,
        target_cloud: bool = False, idle_minutes: float = IDLE_MINUTES, now: Optional[datetime] = None,
        gate: Optional[Callable[..., Gate]] = None, global_store: Optional[NodeStore] = None) -> Report:
    """One consolidation. Never raises: a failure is a report that says what happened."""
    now = now or datetime.now()
    report = Report(dry_run=dry_run)
    verdict = (gate or check_gate)(target_cloud=target_cloud, force=force, idle_minutes=idle_minutes)
    if not verdict.ok:
        report.skipped = verdict.reason
        return report
    if not take_lock(project_root):
        report.skipped = "консолидация уже идёт в другом процессе"
        return report
    try:
        state = load_state(store)
        gathered = gather(project_root, since=str(state.get("processed_until") or ""))
        report.sessions = gathered.sessions
        if not gathered.text.strip():
            report.skipped = "нет новых диалогов"
            return report
        try:
            answer = ask(build_messages(store, gathered, global_store))
            ops, conflicts, suggestions = parse_answer(answer)
        except Exception as e:
            log.warning("memory consolidation: no usable answer (%s)", e)
            report.skipped = f"модель не дала годного ответа ({e})"
            return report
        report.conflicts, report.suggestions = conflicts, suggestions
        snapshot = None if dry_run else store.snapshot(now=now, allow_empty=True)
        report.snapshot = str(snapshot) if snapshot else ""
        if global_store is not None and not dry_run and any(o.scope == "global" for o in ops):
            shot = global_store.snapshot(now=now, allow_empty=True)
            report.global_snapshot = str(shot) if shot else ""
        apply_ops(store, ops, report, now=now, global_store=global_store)
        if not dry_run:
            state.update({"processed_until": gathered.newest or state.get("processed_until", ""),
                          "last_run": now.isoformat(timespec="seconds"),
                          "last_snapshot": report.snapshot, "last_global_snapshot": report.global_snapshot,
                          "last_changes": len(report.applied)})
            save_state(store, state)
            _write_report(store, report, now)
        return report
    finally:
        release_lock(project_root)


# --- the model -------------------------------------------------------------------------------------------------

def resolve_target() -> tuple:
    """(provider, model, is_cloud) for the consolidation."""
    import config
    from src.agent.local_backend import classify
    provider = (config.get_dream_provider() or config.get_provider() or "").strip()
    model = (config.get_dream_model() or config.get_current_model(provider) or "").strip()
    return provider, model, not classify(provider, model).local


def default_ask(provider: str, model: str, timeout: float = 600.0) -> Callable[[list], str]:
    def ask(messages: list) -> str:
        from providers import create_provider
        from src.agent import governor
        with governor.priority(governor.P_BACKGROUND):
            return create_provider(provider).sync_chat(model, messages, temperature=0.2, timeout=timeout) or ""
    return ask


def main(argv: Optional[list] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="python -m src.memory.dream")
    parser.add_argument("--force", action="store_true", help="ignore idle/power checks")
    parser.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    parser.add_argument("--idle-minutes", type=float, default=IDLE_MINUTES)
    args = parser.parse_args(argv)

    os.environ.setdefault("ARGENT_LANE_PRIORITY", "3")        # background: behind every request of the person
    try:
        from src.agent import governor
        governor.lower_process_priority(os.getpid())
    except Exception:
        pass
    from src.memory import hub
    store = hub.store_for("project")
    from project_paths import project_root_or_cwd
    import config
    provider, model, cloud = resolve_target()
    report = run(store, default_ask(provider, model), Path(project_root_or_cwd()), force=args.force,
                 dry_run=args.dry_run, target_cloud=cloud, idle_minutes=args.idle_minutes,
                 global_store=hub.store_for("global") if config.is_dream_global_enabled() else None)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(report.text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
