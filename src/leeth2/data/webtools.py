"""Browser element ranking (Mind2Web) and function calling (Glaive) converters."""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from ..schema import Sample
from . import banks
from .common import clean, make_choice, make_extract, pick
from .kinds import Spec

# ---------------------------------------------------------------------------------------------- Mind2Web
_ATTR_KEYS = ["aria_label", "aria-label", "title", "placeholder", "name", "alt", "value", "id", "class", "href", "type", "role"]


def _node_text(html: str, node_id: str, max_chars: int = 80) -> str:
    """Visible text of the element carrying backend_node_id=node_id in the cleaned HTML (nesting-aware)."""
    m = re.search(r"<(\w+)[^>]*backend_node_id=\"%s\"[^>]*>" % re.escape(str(node_id)), html)
    if not m:
        return ""
    tag = m.group(1)
    i, depth = m.end(), 1
    open_re = re.compile(r"<%s[\s>]" % tag)
    close_re = re.compile(r"</%s>" % tag)
    end = len(html)
    pos = i
    while depth and pos < len(html):
        o, c = open_re.search(html, pos), close_re.search(html, pos)
        if c is None:
            break
        if o is not None and o.start() < c.start():
            depth += 1
            pos = o.end()
        else:
            depth -= 1
            pos = c.end()
            if depth == 0:
                end = c.start()
    inner = re.sub(r"<[^>]+>", " ", html[i:end])
    return re.sub(r"\s+", " ", inner).strip()[:max_chars]


def describe_candidate(cand: Any, html: str) -> str:
    c = json.loads(cand) if isinstance(cand, str) else cand
    attrs = c.get("attributes") or {}
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except json.JSONDecodeError:
            attrs = {}
    parts: List[str] = [f"<{c.get('tag', 'element')}>"]
    text = _node_text(html, c.get("backend_node_id") or attrs.get("backend_node_id", "")) if html else ""
    if text:
        parts.append(f"text='{text}'")
    for k in _ATTR_KEYS:
        v = attrs.get(k)
        if v:
            v = clean(str(v), 60)
            if k == "class":
                v = " ".join(v.split()[:3])
            parts.append(f"{k}='{v}'")
        if len(parts) >= 6:
            break
    return " ".join(parts)[:220]


def conv_mind2web(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    task = clean(row.get("confirmed_task"))
    pos = row.get("pos_candidates") or []
    neg = row.get("neg_candidates") or []
    html = row.get("cleaned_html") or ""
    if not task or not pos or len(neg) < 3:
        return []
    k = rng.randint(6, 30)
    negs = rng.sample(neg, min(len(neg), k - 1))
    cands = [pos[0]] + negs
    texts = [describe_candidate(c, html) for c in cands]
    order = list(range(len(texts)))
    rng.shuffle(order)
    options = [texts[i] for i in order]
    if len(set(options)) != len(options):
        return []
    target = order.index(0)
    prev = row.get("action_reprs") or []
    idx = row.get("target_action_index")
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = len(prev)
    history = [clean(a, 120) for a in prev[max(0, idx - 3) : idx]]
    op = row.get("operation")
    try:
        op = json.loads(op) if isinstance(op, str) else (op or {})
    except json.JSONDecodeError:
        op = {}
    ql = "en" if rng.random() < 0.85 else "ru"
    state = f"Website: {row.get('website', '')} ({row.get('domain', '')})\nTask: {task}\n"
    state += "Previous steps: " + (" | ".join(history) if history else "none")
    if op.get("op"):
        state += f"\nNext operation: {op.get('op')}" + (f" with value '{clean(op.get('value'), 80)}'" if op.get("value") else "")
    q = "Which page element should the next operation be applied to?" if ql == "en" else "К какому элементу страницы нужно применить следующую операцию?"
    return [s for s in [make_choice(spec.id, spec.family, spec.lang, state, q, options, target)] if s]


# ---------------------------------------------------------------------------------------------- Glaive function calling
_FN_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _json_objects(text: str) -> List[Dict[str, Any]]:
    """Top-level {...} objects in a string (the functions listed in a system prompt), brace-matched."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    out.append(json.loads(text[start : i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return out


def conv_glaive(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    system, chat = row.get("system") or "", row.get("chat") or ""
    fns = [o for o in _json_objects(system) if isinstance(o, dict) and "name" in o]
    m = re.search(r"USER:\s*(.+?)\s*\n+\s*ASSISTANT:\s*(.*?)(?:<\|endoftext\|>|\n\nFUNCTION RESPONSE|\Z)", chat, re.S)
    if not fns or not m:
        return []
    user, asst = clean(m.group(1), 1500), m.group(2).strip()
    if not user:
        return []
    called = None
    args: Dict[str, Any] = {}
    if asst.startswith("<functioncall>"):
        mm = _FN_NAME.search(asst)
        if not mm:
            return []
        called = mm.group(1)
        am = re.search(r"\"arguments\"\s*:\s*'(\{.*?\})'", asst, re.S)
        if am:
            try:
                args = json.loads(am.group(1))
            except json.JSONDecodeError:
                args = {}
    names = [f["name"] for f in fns]
    if called is not None and called not in names:
        return []
    rng.shuffle(fns)
    style = rng.choice(["name", "both", "both"])
    options = [f["name"] if style == "name" else f"{f['name']} — {clean(f.get('description', ''), 140)}" for f in fns]
    options.append(rng.choice(["no function needed — answer directly", "none of the functions", "just reply"]))
    if len(set(options)) != len(options):
        return []
    tgt = len(options) - 1 if called is None else [f["name"] for f in fns].index(called)
    q = "Which function should be called, if any?"
    out = [Sample(spec.id, spec.family, "en", "choice", user, q, options, tgt)] if len(options) >= 2 else []
    if called is not None and args:
        k, v = rng.choice(sorted(args.items()))
        if isinstance(v, str) and v and v in user:
            qe = f"Extract the value for the argument “{k}” of {called}."
            s = make_extract(spec.id, spec.family, "en", user, qe, v, user.find(v))
            if s:
                out.append(s)
    return out
