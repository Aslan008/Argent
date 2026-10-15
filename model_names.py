"""Model names as people type them, and as the API spells them.

The same model is written a dozen ways: "Claude Sonnet 4.5", "claude-sonnet-4.5",
"claude_sonnet_4_5", "anthropic/claude-sonnet-4.5", "sonnet 4.5", "claude-3.5-sonnet".
The API accepts exactly one spelling of each (`claude-sonnet-4-5`, `claude-3-5-sonnet-latest`),
and answers anything else with a 404 that reads like the model does not exist. Making the
person remember which dots become dashes, and which family word goes where, is the bug.

This module turns what was typed into the spelling the provider really lists:

* `resolve_claude(text, entries)` — for Anthropic's own API. It understands the name (family and
  version), not just its characters, so "claude 3.5 sonnet" finds `claude-3-5-sonnet-20241022`
  and "opus" finds the newest Opus. It never invents a model that is not in `entries`.
* `match_known(text, ids)` — for aggregators (OpenRouter, ...) whose slugs are their own
  ("anthropic/claude-sonnet-4.5" really has a dot there): it finds the listed id the text means
  and returns it exactly as listed.
* `canonical_claude(text)` — with no list at all (offline, no key yet): the spelling the API uses.

Pure functions, no network: the catalogue lives in claude_catalog.
"""

import difflib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

#: Families Anthropic has used. A name with another family word is still understood when the
#: catalogue lists it (resolve_claude collects the families from `entries`).
KNOWN_FAMILIES = ("opus", "sonnet", "haiku", "fable", "instant")

_SEPARATORS = re.compile(r"[\s._/\\:\-]+")
_NOISE = {"claude", "anthropic", "latest", "model", "models", "the"}
_DATE = re.compile(r"^(20\d{6})$")


@dataclass(frozen=True)
class ClaudeName:
    family: Optional[str]
    version: Tuple[int, ...]          # as written: (4, 0) stays (4, 0)
    date: Optional[str] = None

    @property
    def key(self) -> Tuple[Optional[str], Tuple[int, ...]]:
        """Family and version with trailing zeros dropped: 4.0 is 4, 5.0 is 5."""
        version = list(self.version)
        while len(version) > 1 and version[-1] == 0:
            version.pop()
        return self.family, tuple(version)


def fold(text: str) -> str:
    """Case, spaces, dots, underscores, slashes and dashes folded into single dashes."""
    return _SEPARATORS.sub("-", (text or "").strip().lower()).strip("-")


def _clean(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"^.*?(?=claude)", "", s) if "claude" in s else s      # "us.anthropic.claude-…", "anthropic/claude-…"
    s = re.sub(r"-v\d+:\d+$", "", s)                                    # Bedrock revision suffix
    s = re.sub(r":[a-z0-9_-]+$", "", s)                                 # OpenRouter variants (":thinking", ":beta")
    return s


def parse_claude_name(text: str, families: Iterable[str] = KNOWN_FAMILIES) -> Optional[ClaudeName]:
    """Family, version and date out of any spelling; None when it is not a Claude name."""
    s = _clean(text)
    tokens = [t for t in _SEPARATORS.split(s) if t]
    if not tokens:
        return None
    families = {f.lower() for f in families} | set(KNOWN_FAMILIES)
    family, version, date = None, [], None
    for token in tokens:
        if token in _NOISE:
            continue
        if _DATE.match(token):
            date = token
        elif token.isdigit() and len(token) <= 2:
            version.append(int(token))
        elif token in families and family is None:
            family = token
        else:
            return None                      # a word that is not part of a Claude name: not one of ours
    if family is None:
        return None                          # "4.5" alone does not say which model
    return ClaudeName(family, tuple(version), date)


def canonical_claude(text: str, families: Iterable[str] = KNOWN_FAMILIES) -> Optional[str]:
    """The API's spelling of the name, built from its parts (no catalogue needed).

    Claude 4 and later put the family first (`claude-sonnet-4-5`); Claude 3 and earlier put it
    last and need either a date or `-latest` (`claude-3-5-sonnet-latest`). A bare family
    ("sonnet") has no canonical id without a catalogue and returns None.
    """
    parsed = parse_claude_name(text, families)
    if parsed is None or not parsed.version:
        return None
    ver = "-".join(str(n) for n in parsed.version)
    if parsed.version[0] <= 3:
        return f"claude-{ver}-{parsed.family}-{parsed.date or 'latest'}"
    return f"claude-{parsed.family}-{ver}" + (f"-{parsed.date}" if parsed.date else "")


def _version_sort(name: ClaudeName) -> Tuple[int, ...]:
    return name.version + (0,) * (4 - len(name.version))


def pretty_claude(model_id: str) -> str:
    """How a person writes the model: "Claude Sonnet 5.5", "Claude 3.5 Sonnet". Built from the id
    when Anthropic's own display name is not at hand; anything that is not a Claude id is returned as is.
    Claude 3 and earlier put the version before the family, 4 and later after it, as Anthropic does."""
    parsed = parse_claude_name(model_id)
    if parsed is None or not parsed.version or "/" in (model_id or ""):
        return model_id                       # a gateway slug ("anthropic/claude-…") is that gateway's name
    version = ".".join(str(n) for n in parsed.version)
    family = parsed.family.capitalize()
    return f"Claude {version} {family}" if parsed.version[0] <= 3 else f"Claude {family} {version}"


def families_in(ids: Iterable[str]) -> set:
    """The family words the ids themselves use, so a model family new to this file still parses."""
    found = set(KNOWN_FAMILIES)
    for i in ids:
        for token in _SEPARATORS.split(_clean(i)):
            if token and token.isalpha() and token not in _NOISE:
                found.add(token)
    return found


def resolve_claude(text: str, entries: Sequence) -> Optional[str]:
    """The id in `entries` that `text` means, or None.

    `entries` need `.id` and `.created` (a timestamp; 0 when unknown). A bare family such as
    "opus" means the newest of that family; "sonnet 4.5" the newest snapshot of that version,
    preferring the undated alias when the catalogue lists one.
    """
    if not (text or "").strip() or not entries:
        return None
    ids = [e.id for e in entries]
    lowered = {i.lower(): i for i in ids}
    wanted = text.strip().lower()
    if wanted in lowered:
        return lowered[wanted]
    folded = {fold(i): i for i in ids}
    if fold(text) in folded:
        return folded[fold(text)]

    families = families_in(ids)
    asked = parse_claude_name(text, families)
    if asked is None:
        return None

    parsed = [(e, parse_claude_name(e.id, families)) for e in entries]
    pool = [(e, p) for e, p in parsed if p is not None and p.family == asked.family]
    if asked.version:
        pool = [(e, p) for e, p in pool if p.key == asked.key]
        if asked.date:
            pool = [(e, p) for e, p in pool if p.date == asked.date]
    if not pool:
        return None

    def newest(item):
        entry, p = item
        # newest snapshot first; an undated alias ranks with the newest snapshot, ahead of older ones
        return (_version_sort(p), getattr(entry, "created", 0) or 0, p.date or "99999999")

    pool.sort(key=newest, reverse=True)
    best_version = pool[0][1].key
    same = [(e, p) for e, p in pool if p.key == best_version]
    if not asked.date:
        undated = [e for e, p in same if p.date is None]
        if undated:
            return undated[0].id
    return same[0][0].id


def suggest(text: str, ids: Sequence[str], n: int = 5) -> List[str]:
    """The listed ids that look most like `text`, for "did you mean"."""
    by_fold = {fold(i): i for i in ids}
    close = difflib.get_close_matches(fold(text), list(by_fold), n=n, cutoff=0.5)
    return [by_fold[c] for c in close]


def match_known(text: str, ids: Sequence[str]) -> Optional[str]:
    """The listed id `text` means, spelled exactly as listed — for providers whose ids are their own.

    Exact (any case), then equal after folding separators, then a unique id that ends with what
    was typed ("claude-sonnet-4.5" for "anthropic/claude-sonnet-4.5"). Ambiguity is None, not a guess.
    """
    wanted = (text or "").strip()
    if not wanted or not ids:
        return None
    for i in ids:
        if i == wanted:
            return i
    lowered = [i for i in ids if i.lower() == wanted.lower()]
    if len(lowered) == 1:
        return lowered[0]
    key = fold(wanted)
    equal = [i for i in ids if fold(i) == key]
    if len(equal) == 1:
        return equal[0]
    tail = [i for i in ids if fold(i).endswith("-" + key) or fold(i.split("/", 1)[-1]) == key]
    return tail[0] if len(tail) == 1 else None
