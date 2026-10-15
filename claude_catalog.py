"""The list of Claude models, read from Anthropic instead of typed into the source.

A hand-written list is wrong the day a model ships — and was wrong here: it held ids that
never existed ("claude-5-5-sonnet-latest"), missed the models released since, and listed
alphabetically, so the newest model sat wherever its name sorted. `GET /v1/models` is the
authority: it returns every model the key can use, with its release date, its context window,
its output ceiling and what it can do (thinking, effort). This module reads all of it —
every page, not just the first twenty — keeps it newest-first, and remembers the last good
answer on disk, so a new model appears without anyone editing anything and a flaky network
does not empty the menu.

The tiers, in the order they are tried:
    live API  →  last good answer on disk (any age)  →  a small built-in seed
The seed is only what is needed to start from nothing (no key yet, no network); it is not
a catalogue and says so ("source": "seed").
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from logger import get_logger
from model_names import ClaudeName, parse_claude_name

log = get_logger("providers")

CACHE_TTL_SECONDS = 6 * 3600

#: What is known with no network and no catalogue — enough to start from. Spelled as the API spells
#: them; capabilities are unknown (None), so requests fall back to the version-based rules below.
SEED_IDS = ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5")


@dataclass(frozen=True)
class ModelEntry:
    id: str
    display_name: str = ""
    created: float = 0.0
    max_input: Optional[int] = None
    max_output: Optional[int] = None
    thinking_enabled: Optional[bool] = None       # `thinking: {type: enabled, budget_tokens}` accepted
    thinking_adaptive: Optional[bool] = None      # `thinking: {type: adaptive}` accepted
    effort_levels: Optional[tuple] = None         # levels `output_config.effort` accepts; () = none, None = unknown

    @property
    def known_capabilities(self) -> bool:
        return self.thinking_enabled is not None or self.thinking_adaptive is not None


def _flag(obj, *path) -> Optional[bool]:
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    value = getattr(obj, "supported", obj)
    return bool(value) if isinstance(value, bool) else None


def entry_from_api(model) -> Optional[ModelEntry]:
    """A ModelEntry from the SDK's ModelInfo (or anything shaped like it)."""
    mid = str(getattr(model, "id", "") or "").strip()
    if not mid:
        return None
    created = getattr(model, "created_at", None)
    try:
        stamp = float(created.timestamp()) if created is not None else 0.0
    except Exception:
        stamp = 0.0
    caps = getattr(model, "capabilities", None)
    effort = getattr(caps, "effort", None)
    levels = None
    if effort is not None and isinstance(getattr(effort, "supported", None), bool):
        levels = tuple(name for name in ("low", "medium", "high", "xhigh", "max")
                       if effort.supported and _flag(effort, name))

    def num(value):
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None

    return ModelEntry(
        id=mid, display_name=str(getattr(model, "display_name", "") or ""), created=max(0.0, stamp),
        max_input=num(getattr(model, "max_input_tokens", None)),
        max_output=num(getattr(model, "max_tokens", None)),
        thinking_enabled=_flag(caps, "thinking", "types", "enabled"),
        thinking_adaptive=_flag(caps, "thinking", "types", "adaptive"),
        effort_levels=levels,
    )


def sort_newest_first(entries: List[ModelEntry]) -> List[ModelEntry]:
    """Newest release first; where dates tie (or are unknown) the higher version, then the id."""
    def key(e: ModelEntry):
        p = parse_claude_name(e.id)
        return (e.created, p.version if p else (), e.id)
    return sorted(entries, key=key, reverse=True)


def seed_entries() -> List[ModelEntry]:
    return [ModelEntry(id=i) for i in SEED_IDS]


# -- the disk copy --------------------------------------------------------------------------

def cache_path() -> Path:
    base = os.environ.get("ARGENT_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".argent"
    return root / "model_cache" / "claude_models.json"


def _read_cache(path: Path) -> Tuple[List[ModelEntry], float]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = []
        for raw in data.get("models", []):
            if isinstance(raw.get("effort_levels"), list):
                raw["effort_levels"] = tuple(raw["effort_levels"])
            entries.append(ModelEntry(**raw))
        return entries, float(data.get("fetched_at", 0))
    except (OSError, ValueError, TypeError):
        return [], 0.0


def _write_cache(path: Path, entries: List[ModelEntry]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fetched_at": time.time(), "models": [asdict(e) for e in entries]},
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        log.info("could not save the Claude model cache: %s", e)


# -- reading it -----------------------------------------------------------------------------

@dataclass
class Catalog:
    entries: List[ModelEntry]
    source: str            # "live" | "cache" | "seed"
    problem: str = ""      # why the live list could not be read, when it could not

    @property
    def authoritative(self) -> bool:
        """Whether "not in the list" means "does not exist" (a live answer) or only "not seen"."""
        return self.source == "live"

    def ids(self) -> List[str]:
        return [e.id for e in self.entries]

    def get(self, model_id: str) -> Optional[ModelEntry]:
        low = (model_id or "").lower()
        return next((e for e in self.entries if e.id.lower() == low), None)


def fetch_live(client) -> List[ModelEntry]:
    """Every model the key can use, all pages."""
    page = client.models.list(limit=1000)
    items = list(page.auto_paging_iter()) if hasattr(page, "auto_paging_iter") else list(getattr(page, "data", page))
    entries = [e for e in (entry_from_api(m) for m in items) if e]
    # Only chat models belong in a chat menu; the endpoint is Claude-only today, but a proxy may mix.
    return sort_newest_first([e for e in entries if parse_claude_name(e.id) or e.id.lower().startswith("claude")])


def _reason(error: Exception) -> str:
    """One line a person can act on."""
    text = " ".join(str(getattr(error, "message", None) or error).split())
    return text[:300] or type(error).__name__


_memo: dict = {}          # scope -> (valid until, Catalog)
_RETRY_AFTER_FAILURE = 60


def forget() -> None:
    """Drop the in-process copy (tests, and after the key or URL changes)."""
    _memo.clear()


def load(get_client: Callable[[], object], scope: str = "", refresh: bool = False,
         path: Optional[Path] = None) -> Catalog:
    """The catalogue: live when possible, else the disk copy, else the seed.

    A fresh answer is reused for CACHE_TTL_SECONDS (in memory, and on disk across runs), so the
    menu does not wait on the network each time; `refresh` forces a new read. `scope` keeps
    different keys / base URLs from sharing one in-memory answer."""
    now = time.time()
    path = path or cache_path()
    memo = _memo.get(scope)
    if memo and not refresh and now < memo[0]:
        return memo[1]

    cached, fetched_at = _read_cache(path)
    if cached and not refresh and now - fetched_at < CACHE_TTL_SECONDS:
        catalog = Catalog(sort_newest_first(cached), "cache")
        _memo[scope] = (fetched_at + CACHE_TTL_SECONDS, catalog)
        return catalog

    problem = ""
    try:
        live = fetch_live(get_client())
    except Exception as e:
        log.info("Claude model list unavailable (%s); using the saved copy", e)
        live, problem = [], _reason(e)
    if live:
        _write_cache(path, live)
        catalog = Catalog(live, "live")
        _memo[scope] = (now + CACHE_TTL_SECONDS, catalog)
        return catalog

    catalog = (Catalog(sort_newest_first(cached), "cache", problem) if cached
               else Catalog(seed_entries(), "seed", problem))
    # Do not pin a failure for hours: look again soon.
    _memo[scope] = (now + _RETRY_AFTER_FAILURE, catalog)
    return catalog


# -- what a model can do --------------------------------------------------------------------

def thinking_style(model_id: str, entry: Optional[ModelEntry]) -> Optional[str]:
    """How this model takes extended thinking: "enabled" (a token budget), "adaptive", or None.

    From the catalogue when it said; otherwise from the version — thinking arrived with 3.7,
    and every 4-and-later model has it."""
    if entry is not None and entry.known_capabilities:
        if entry.thinking_enabled:
            return "enabled"
        if entry.thinking_adaptive:
            return "adaptive"
        return None
    parsed = parse_claude_name(model_id)
    if parsed is None or not parsed.version:
        return None
    return "enabled" if parsed.version >= (3, 7) else None


def _known_entries() -> List[ModelEntry]:
    """Whatever catalogue is already in memory or on disk. Never touches the network: this is called
    to draw a toolbar, and a toolbar must not wait for Anthropic."""
    for _, catalog in list(_memo.values()):
        if catalog.entries:
            return catalog.entries
    return _read_cache(cache_path())[0]


def display_name(model_id: str, entries: Optional[List[ModelEntry]] = None) -> str:
    """"Claude Sonnet 5.5" for `claude-sonnet-5-5`: Anthropic's display name when the catalogue has
    one, else the name built from the id. Not a Claude id: returned unchanged."""
    from model_names import pretty_claude
    if not model_id:
        return model_id
    low = model_id.lower()
    for e in (entries if entries is not None else _known_entries()):
        if e.id.lower() == low and e.display_name.strip():
            return e.display_name.strip()
    return pretty_claude(model_id)


def label_for(provider_name: str, model_id: str) -> str:
    """What to show for a model of this provider. Only Claude ids are rewritten; other providers'
    ids are the names their users know."""
    return display_name(model_id) if provider_name in ("claude", "anthropic") else model_id


def default_model(entries: List[ModelEntry]) -> Optional[str]:
    """What to use when nothing is chosen: the newest Sonnet (the working horse), else the newest model."""
    for e in entries:
        p = parse_claude_name(e.id)
        if p and p.family == "sonnet":
            return e.id
    return entries[0].id if entries else None


def is_claude_family(name: str) -> bool:
    return parse_claude_name(name) is not None


__all__ = ["Catalog", "ModelEntry", "ClaudeName", "load", "forget", "thinking_style", "default_model",
           "sort_newest_first", "is_claude_family", "seed_entries", "SEED_IDS"]
