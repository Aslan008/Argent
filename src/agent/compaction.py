"""What a compaction must not lose.

The summary a model writes is the lossy part of a compaction: it paraphrases, and every paraphrase drops
something. This module holds the parts that are NOT lossy and that the summary must not be trusted with:

* the **transcript archive** — the dropped messages, whole, in a file the agent can read back with
  `read_file` when the summary is not enough (the exact wording, a command's full output);
* the **user ledger** — what the person actually said, verbatim, in order, accumulated across compactions
  (the thing a paraphrase corrupts first);
* the **standing rules** — "never commit", "run only on X": instructions that must keep holding, picked out
  of the person's own sentences so that a summary that forgot them cannot lose them;
* the **verified facts** — files touched, commands run, errors seen, merged across compactions
  (they used to be rebuilt from the compressed slice only, so the second compaction forgot the first);
* `split_distillate` / `assemble` — a distillate is written and read back by the same code, so the next
  compaction updates the previous summary instead of re-summarising a blob that carries file bodies.

Pure Python, no model calls: everything here works with the network down.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── what the distillate looks like ───────────────────────────────────────────

SEP = "\n\n---\n"
GOAL = "**ACTIVE GOAL:**"
L_ARCHIVE = "**Полный журнал сжатых сообщений:**"
L_LEDGER = "**Сообщения пользователя (по порядку, дословно):**"
L_RULES = "**Правила и предпочтения пользователя:**"
L_FACTS = "**Верифицированные факты:**"
L_FILES = "**Актуальное содержимое ключевых файлов:**"
L_MEMORY = "**Постоянная память:**"

HEADER = (
    "Это сводка ранее сжатой части диалога: от неё остались только эта сводка, дословные сообщения "
    "пользователя и журнал в файле. Продолжай работу с раздела «Current State» и не пересказывай сводку "
    "пользователю. Если нужна точная формулировка или полный вывод команды — прочитай журнал "
    "(read_file по пути ниже), а не угадывай."
)

NO_ARTIFACTS = "[Промежуточная работа без значимых артефактов]"


def message_text(m: Dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in c)
    return "" if c is None else str(c)


def clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"


# ── is this a real request from the person? ──────────────────────────────────

_TRIVIAL = {
    "ок", "ok", "да", "yes", "продолжай", "continue", "дальше", "go", "го", "давай", "хорошо", "понял", "ага",
    "угу", "принято", "делай", "запускай", "ладно", "плиз", "please", "работаем", "давай дальше", "плюс", "к", "же",
}
#: texts the harness itself puts in the "user" role (nudges, continuations, impulses): not the person's words
_INJECTED = (
    "[Режим Автоматизма]", "[Heartbeat", "[Argent:", "[System:", "/",
    "Your file write was cut off", "Your tool call was cut off", "Your previous response was cut off",
    "Your preliminary reasoning was captured", "You emitted a placeholder", "You stopped after thinking",
)


def is_substantive(text: str) -> bool:
    """A real task or instruction, not a confirmation or an auto-impulse."""
    if not text or len(text.strip()) < 6:
        return False
    if text.startswith(("[Режим Автоматизма]", "[Heartbeat", "/")):
        return False
    words = set(re.findall(r"\b\w+\b", text.lower().strip()))
    return not (words and words.issubset(_TRIVIAL))


def user_texts(messages: List[Dict]) -> List[str]:
    """The person's own messages, in order (harness-injected 'user' messages and bare confirmations left out)."""
    out = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        text = message_text(m).strip()
        if text and not text.startswith(_INJECTED) and is_substantive(text):
            out.append(text)
    return out


# ── the ledger: what the person said ─────────────────────────────────────────

ENTRY_CAP = 900
LEDGER_BUDGET = 7000


def ledger_entries(texts: List[str]) -> List[str]:
    return [clip(" ⏎ ".join(line.strip() for line in t.splitlines() if line.strip()), ENTRY_CAP) for t in texts]


def _size(entries: List[str]) -> int:
    return sum(len(e) + 6 for e in entries)


def fit_ledger(entries: List[str], budget: int = LEDGER_BUDGET) -> List[str]:
    """Shorten the OLD entries (never the original task or the newest few) until the ledger fits; if that is not
    enough, drop the middle but keep the first two and say how many were dropped."""
    entries = list(entries)
    if _size(entries) <= budget:
        return entries
    fresh = 4
    for cap in (360, 160, 80):
        entries = [e if i == 0 or i >= len(entries) - fresh else clip(e, cap) for i, e in enumerate(entries)]
        if _size(entries) <= budget:
            return entries
    head, rest = entries[:2], entries[2:]
    room = budget - _size(head) - 80
    tail: List[str] = []
    for e in reversed(rest):
        if _size(tail) + len(e) + 6 > room:
            break
        tail.insert(0, e)
    dropped = len(rest) - len(tail)
    return head + ([f"… пропущено сообщений: {dropped} …"] if dropped > 0 else []) + tail


def merge_ledger(prior: List[str], new: List[str]) -> List[str]:
    out = list(prior)
    for e in new:
        if not (out and out[-1] == e):
            out.append(e)
    return out


def render_ledger(entries: List[str]) -> str:
    lines = [f"- {e}" if e.startswith("…") else f"- «{e}»" for e in entries]
    return L_LEDGER + "\n" + "\n".join(lines)


def _bullets(block: str) -> List[str]:
    return [ln[2:].rstrip() for ln in block.splitlines()[1:] if ln.startswith("- ")]


def parse_ledger(block: str) -> List[str]:
    return [e[1:-1] if e.startswith("«") and e.endswith("»") else e for e in _bullets(block)]


# ── standing rules ───────────────────────────────────────────────────────────

_RULE = re.compile(
    r"(?iu)\b(всегда|никогда|нельзя|запрещ\w*|обязательно|исключительно|запомни|на\s+будущее|в\s+дальнейшем|"
    r"по\s+умолчанию|не\s+(?:делай|трогай|меняй|коммить\w*|пуш\w*|запускай|удаляй|добавляй|используй|пиши|"
    r"создавай|спавни\w*|скачивай)|always|never|must\s+not|do\s+not|don't|only\s+use|from\s+now\s+on|"
    r"remember\s+to|make\s+sure)\b")
RULES_LIMIT = 14


def extract_rules(texts: List[str]) -> List[str]:
    """Sentences of the person's messages that read as a standing instruction. Crude on purpose: it is the net
    under the model's own '## Standing Rules', for the day that model forgets one."""
    found: List[str] = []
    for t in texts:
        for sentence in re.split(r"(?<=[.!?…])\s+|\n+", t):
            sentence = sentence.strip(" -•*\t")
            if 12 <= len(sentence) <= 260 and _RULE.search(sentence):
                found.append(clip(sentence, 240))
    return found


def merge_rules(prior: List[str], new: List[str], limit: int = RULES_LIMIT) -> List[str]:
    seen, out = set(), []
    for r in reversed(prior + new):                       # the latest wording of a rule wins
        key = re.sub(r"\W+", " ", r.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return list(reversed(out[:limit]))


def render_rules(rules: List[str]) -> str:
    return L_RULES + "\n" + "\n".join(f"- {r}" for r in rules)


def parse_rules(block: str) -> List[str]:
    return _bullets(block)


# ── verified facts ───────────────────────────────────────────────────────────

@dataclass
class Facts:
    written: List[str] = field(default_factory=list)
    read: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    requests: List[str] = field(default_factory=list)


def _tool_args(tc: Dict) -> Dict:
    args = (tc.get("function") or {}).get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return {}
    return args if isinstance(args, dict) else {}


def collect_facts(messages: List[Dict]) -> Facts:
    f = Facts()
    for m in messages:
        role, content = m.get("role", ""), message_text(m)
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                name = (tc.get("function") or {}).get("name", "")
                args = _tool_args(tc)
                fp = args.get("file_path") or args.get("path") or ""
                if name == "read_file" and fp:
                    f.read.append(fp)
                elif name in ("write_file", "replace_in_file") and fp:
                    f.written.append(fp)
                elif name == "run_command" and (args.get("command") or ""):
                    f.commands.append(args["command"][:120])
            if content.strip():
                summary = " ".join([ln.strip() for ln in content.strip().split("\n") if ln.strip()][:2])[:200]
                if len(summary) > 30:
                    f.steps.append(summary)
        elif role == "tool":
            low = content.lower()
            if any(k in low for k in ("error:", "failed", "exception", "traceback", "ошибка")):
                lines = [ln.strip() for ln in content.strip().split("\n") if ln.strip()]
                err = lines[-1][:200] if "traceback" in low and len(lines) > 1 else (lines[0][:200] if lines else "")
                if err:
                    f.errors.append(err)
        elif role == "user":
            text = content.strip()
            if text and not text.startswith(_INJECTED) and is_substantive(text):
                f.requests.append(text[:200])
    return f


def _union(old: List[str], new: List[str]) -> List[str]:
    seen, out = set(), []
    for x in old + new:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def merge_facts(prior: Facts, new: Facts) -> Facts:
    return Facts(
        written=_union(prior.written, new.written), read=_union(prior.read, new.read),
        commands=(prior.commands + new.commands)[-8:], errors=(prior.errors + new.errors)[-3:],
        steps=(prior.steps + new.steps)[-5:], requests=(prior.requests + new.requests)[-3:])


def render_facts(f: Facts, include_requests: bool = True) -> str:
    parts = []
    written = sorted(set(f.written))
    if written:
        parts.append("**Изменённые файлы:** " + ", ".join(written[:40]))
    readable = sorted(set(f.read) - set(f.written))[:10]
    if readable:
        parts.append("**Прочитанные файлы:** " + ", ".join(readable))
    if f.commands:
        parts.append("**Команды:** " + "; ".join(f.commands[-8:]))
    if f.errors:
        parts.append("**Ошибки:**\n" + "\n".join(f"- {e}" for e in f.errors[-3:]))
    if f.steps:
        parts.append("**Ход работы:**\n" + "\n".join(f"- {s}" for s in f.steps[-5:]))
    if include_requests and f.requests:
        parts.append("**Запросы:**\n" + "\n".join(f"- {d}" for d in f.requests[-3:]))
    return "\n".join(parts) if parts else NO_ARTIFACTS


def parse_facts(text: str) -> Facts:
    f = Facts()
    current: Optional[List[str]] = None
    for line in text.splitlines():
        for label, target, sep in (("**Изменённые файлы:**", f.written, ", "), ("**Прочитанные файлы:**", f.read, ", "),
                                   ("**Команды:**", f.commands, "; ")):
            if line.startswith(label):
                target.extend(x.strip() for x in line[len(label):].split(sep) if x.strip())
                current = None
                break
        else:
            if line.startswith("**Ошибки:**"):
                current = f.errors
            elif line.startswith("**Ход работы:**"):
                current = f.steps
            elif line.startswith("**Запросы:**"):
                current = f.requests
            elif line.startswith("- ") and current is not None:
                current.append(line[2:].strip())
    return f


# ── the transcript archive ───────────────────────────────────────────────────

MESSAGE_CAP = 40000
ARCHIVE_KEEP = 60
ARCHIVE_MAX_BYTES = 200 * 1024 * 1024
ARCHIVE_PATHS_KEPT = 6


def _cap(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n * 2 // 3] + f"\n… [{len(text) - n} символов пропущено] …\n" + text[-(n // 3):]


def render_transcript(messages: List[Dict]) -> str:
    lines = [f"# Argent: сжатый фрагмент диалога — {len(messages)} сообщений, {datetime.now():%Y-%m-%d %H:%M}", ""]
    for i, m in enumerate(messages, 1):
        head = f"## [{i}] {str(m.get('role', '?')).upper()}"
        if m.get("tool_call_id"):
            head += f" (ответ на вызов {m['tool_call_id']})"
        lines += [head, _cap(message_text(m), MESSAGE_CAP)]
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments", {})
            args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            lines.append(f"→ вызов {fn.get('name', '?')}: {_cap(args, 4000)}")
        if m.get("images"):
            lines.append(f"[изображений: {len(m['images'])}]")
        lines.append("")
    return "\n".join(lines)


def _prune(folder: Path, keep_path: Path) -> None:
    try:
        files = sorted((p for p in folder.glob("compact-*.md") if p != keep_path),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        total = keep_path.stat().st_size
        for i, p in enumerate(files):
            total += p.stat().st_size
            if i + 1 >= ARCHIVE_KEEP or total > ARCHIVE_MAX_BYTES:
                p.unlink(missing_ok=True)
    except OSError:
        pass


def archive_transcript(messages: List[Dict], compression_id: int, root: Optional[Path] = None) -> Optional[str]:
    """Write the messages that are about to be dropped to `.argent/transcripts/` and return the path
    (None when it could not be written: a read-only folder must not cost the compaction)."""
    # an earlier distillate is already summarised AND archived; copying it in would only bury the real messages
    rows = [m for m in messages if "CONTEXT DISTILLATE" not in message_text(m)[:80]]
    if not rows:
        return None
    try:
        override = os.environ.get("ARGENT_TRANSCRIPT_DIR")
        if root is not None:
            folder = Path(root) / ".argent" / "transcripts"
        elif override:
            folder = Path(override)
        else:
            from project_paths import project_root_or_cwd
            folder = Path(project_root_or_cwd() or Path.cwd()) / ".argent" / "transcripts"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"compact-{datetime.now():%Y%m%d-%H%M%S}-{compression_id}.md"
        from atomic_io import atomic_write_text
        atomic_write_text(path, render_transcript(rows))
        _prune(folder, path)
        return str(path)
    except Exception:
        return None


def render_archive(paths: List[str]) -> str:
    return L_ARCHIVE + "\n" + "\n".join(f"- {p}" for p in paths[-ARCHIVE_PATHS_KEPT:])


def parse_archive(block: str) -> List[str]:
    return _bullets(block)


# ── writing and reading back a distillate ────────────────────────────────────

@dataclass
class Parts:
    summary: str = ""
    archive: List[str] = field(default_factory=list)
    ledger: List[str] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    facts: Facts = field(default_factory=Facts)


def split_distillate(text: Optional[str]) -> Parts:
    """The structured contents of a distillate written by `assemble` (or by the older, unstructured format,
    from which only the summary and the facts are recovered). The rehydrated file bodies are dropped: they are
    re-read from disk at the next compaction, and feeding them to the summariser was pure bloat."""
    parts = Parts()
    if not text:
        return parts
    summary: List[str] = []
    for block in (b.strip() for b in text.split(SEP)):
        if block.startswith(GOAL):
            # `assemble` gives the goal a block of its own; the older format glued it to the summary
            block = block.split("\n\n", 1)[1].strip() if "\n\n" in block else ""
        if not block or block == HEADER:
            continue
        if block.startswith(L_ARCHIVE):
            parts.archive = parse_archive(block)
        elif block.startswith(L_LEDGER):
            parts.ledger = parse_ledger(block)
        elif block.startswith(L_RULES):
            parts.rules = parse_rules(block)
        elif block.startswith(L_FACTS):
            parts.facts = parse_facts(block[len(L_FACTS):])
        elif block.startswith((L_FILES, L_MEMORY)):
            continue                 # both are rebuilt from disk each time
        else:
            summary.append(block.replace(HEADER, "").strip())
    parts.summary = "\n\n".join(s for s in summary if s)
    return parts


def assemble(*, goal: Optional[str], summary: Optional[str], archive: List[str], ledger: List[str],
             rules: List[str], facts: str, files: str = "", extra: Optional[List[str]] = None) -> str:
    blocks = [HEADER]
    if goal:
        blocks.append(f"{GOAL} {goal}")
    if summary:
        blocks.append(summary.strip())
    if archive:
        blocks.append(render_archive(archive))
    if ledger:
        blocks.append(render_ledger(ledger))
    if rules:
        blocks.append(render_rules(rules))
    blocks.append(f"{L_FACTS}\n{facts}")
    if files:
        blocks.append(f"{L_FILES}\n{files}")
    blocks.extend(b for b in (extra or []) if b)
    return SEP.join(blocks)
