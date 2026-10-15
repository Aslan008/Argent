"""
tools/leeth_bulk_tools.py -- bulk and form tools for the main model: it states WHAT, Leeth does it for many items / many fields.

  leeth_map          one question about many texts, in one batched pass (filter / categorise a list)
  leeth_scan_page    the same for the clickable elements of the current browser page (read-only)
  leeth_click_many   click the elements a scan returned, re-finding each one by label + link; destructive ones need confirm=True
  leeth_fill_form    fill a whole form (text fields, dropdowns, checkboxes) from one instruction, verify, never press a payment button

Leeth answers with calibrated probabilities; what it is not sure about comes back to the caller (the 'unsure' lists, the 'skipped'
fields) instead of being guessed.
"""

import json
import os
import re
import threading
import time

from logger import get_logger

logger = get_logger("leeth_bulk_tools")

_V2_LOCK = threading.Lock()
_V2 = None
_LAST_SCAN: dict = {}  # session -> {element index: {"label", "href"}} of the last leeth_scan_page: leeth_click_many acts only on those


def _leeth2():
    """The Leeth 2 runtime: the dispatcher's own when that backend is selected, otherwise loaded here on first use.
    These tools ask typed questions that only Leeth 2 answers (batched yes/no, choices of many options, multi-field extraction)."""
    global _V2
    try:
        from src.leeth.runtime import LeethRuntime

        v2 = LeethRuntime.get_instance()._use_v2()
        if v2 is not None:
            return v2
    except Exception:
        pass
    with _V2_LOCK:
        if _V2 is None:
            from pathlib import Path

            from src.leeth2.runtime import Leeth2Runtime

            default = Path(__file__).resolve().parents[1] / ".argent" / "models" / "leeth2" / "final"
            _V2 = Leeth2Runtime.load(os.environ.get("LEETH2_MODEL_DIR") or default)
        return _V2


def _as_list(value, split_lines: bool = True) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    s = str(value).strip()
    if not s:
        return []
    if s[0] in "[{":
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return [ln for ln in (s.splitlines() if split_lines else [s]) if ln.strip()]


def leeth_map(items, criterion: str, mode: str = "yesno", categories=None, accept_above: float = 0.7, reject_below: float = 0.3,
              max_items: int = 500, positive_examples=None, negative_examples=None) -> str:
    """Answer ONE question about MANY texts at once (milliseconds each, no API tokens): filter or categorise a list.

    Use it for the routine part of "go through these posts / bookmarks / files / tickets and pick those that ...". State the criterion
    once; items whose answer is uncertain (probability between reject_below and accept_above) come back for you to judge.

    Args:
        items: the texts, as a JSON list or one per line.
        criterion: for mode='yesno' a yes/no question about one item ("Is this post a complaint about delivery?");
            for mode='categories' an optional question ("What is this bookmark about?").
        mode: 'yesno' or 'categories'.
        categories: for mode='categories', the allowed categories (2 or more, as a JSON list or one per line).
        accept_above: yesno -- probability from which an item counts as 'yes'.
        reject_below: yesno -- probability up to which an item counts as 'no'; between the two: 'unsure'.
        max_items: refuse longer lists (split them).
        positive_examples, negative_examples: for mode='yesno', a few texts (2-3 of each) that DO / do NOT satisfy the criterion,
            as a JSON list or one per line. Strongly recommended for content questions ("is this post about X"): the cut between
            yes and no is then learnt from them, which lifts recall from ~0.3 to ~0.7 at ~2% false yes. Without them the
            fixed accept_above / reject_below are used and many clear matches land in 'unsure' (read it from the top: sorted by p).

    Returns:
        JSON: counts, index lists ('yes' / 'no' or per category) and the unsure items with their text.
    """
    from src.leeth import bulk

    rows = _as_list(items)
    if not rows:
        return "Error: no items given."
    if len(rows) > int(max_items):
        return f"Error: {len(rows)} items; the limit is {int(max_items)} per call -- split the list."
    mode = (mode or "yesno").strip().lower()
    cal: dict = {}
    t0 = time.perf_counter()
    try:
        rt = _leeth2()
        if mode == "yesno":
            if not criterion or not criterion.strip():
                return "Error: criterion (a yes/no question) is required for mode='yesno'."
            res = bulk.classify_yesno(rt, rows, criterion, accept=float(accept_above), reject=float(reject_below),
                                      positives=_as_list(positive_examples), negatives=_as_list(negative_examples), meta=cal)
        elif mode == "categories":
            res = bulk.classify_categories(rt, rows, _as_list(categories), question=criterion or None)
        else:
            return "Error: mode must be 'yesno' or 'categories'."
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("leeth_map error: %s", e)
        return f"Error in leeth_map: {e}"
    rep = bulk.summarize(res, rows, mode)
    rep.update(cal)
    _hint_unsure(rep, len(rows), bool(cal))
    rep["ms"] = round((time.perf_counter() - t0) * 1000)
    return json.dumps(rep, ensure_ascii=False)


def _hint_unsure(rep: dict, total: int, calibrated: bool) -> None:
    """A content question without examples leaves most clear matches in 'unsure' (conservative absolute probabilities): say how
    to get a sharper answer instead of letting the caller think the model does not know."""
    n = len(rep.get("unsure") or [])
    if not calibrated and total >= 6 and n * 2 >= total:
        rep["hint"] = ("many items are 'unsure': for content questions the absolute probability is conservative while the ranking is "
                       "reliable -- review 'unsure' from the top (sorted by p), or pass positive_examples / negative_examples "
                       "(2-3 texts of each kind) so the cut is learnt from them")


def _scan_items(elements: list, include: str, ctx: dict = None) -> list:
    """The clickable elements worth judging: {index, label, href, text, ctx}. ``ctx`` maps an element index to the beginning of
    the post / row that contains it, which tells identical buttons ("Delete") of different rows apart."""
    inc = (include or "links").strip().lower()
    ctx = ctx or {}
    out = []
    for el in elements:
        tag = str(el.get("tag") or "").lower()
        base = tag.split("[", 1)[0]
        if inc == "links" and base != "a":
            continue
        if inc == "buttons" and not (base == "button" or "type=submit" in tag or "type=button" in tag):
            continue
        label = str(el.get("label") or "").strip()
        href = str(el.get("href") or "").strip()
        if not label and not href:
            continue
        idx = int(el.get("idx"))
        out.append({"index": idx, "label": label, "href": href, "text": str(el.get("text") or "").strip(), "ctx": ctx.get(idx, "")})
    return out


_POST_BUTTONS = 8  # buttons / links of one post that are reported (and made clickable) next to it
_SCAN_SCROLL_PX = 700


def _post_items(blocks: list, elements: list, ctx: dict) -> list:
    """Text blocks (posts, cards, rows) as scan items: {index: None, label: <text>, href, buttons: [<element item>, ...]}."""
    by_idx = {int(e.get("idx")): e for e in elements if e.get("idx") is not None}
    out = []
    for b in blocks:
        text = str(b.get("text") or "").strip()
        if not text:
            continue
        btns = []
        for i in b.get("idxs") or []:
            el = by_idx.get(int(i))
            if el is None:
                continue
            tag = str(el.get("tag") or "").lower().split("[", 1)[0]
            label = str(el.get("label") or "").strip()
            href = str(el.get("href") or "").strip()
            if tag not in ("a", "button", "input") or not (label or href):
                continue
            btns.append({"index": int(i), "label": label, "href": href, "text": str(el.get("text") or "").strip(), "ctx": ctx.get(int(i), text[:80])})
            if len(btns) >= _POST_BUTTONS:
                break
        out.append({"index": None, "label": _strip_ui_tail(text, [x["label"] for x in btns]), "href": str(b.get("href") or "").strip(),
                    "text": "", "ctx": text[:80], "buttons": btns})
    return out


def _strip_ui_tail(text: str, labels: list) -> str:
    """A post's text ends with its own buttons' captions ("... Нравится В закладки"): they are not content and would only pull the
    judgement towards the UI. The post's context key is taken from the untouched text before this."""
    labs = sorted({l.strip() for l in labels if l and l.strip()}, key=len, reverse=True)
    t = text.rstrip()
    changed = True
    while changed and t:
        changed = False
        for l in labs:
            if t.endswith(l) and len(t) > len(l):
                t = t[: -len(l)].rstrip()
                changed = True
    return t or text


def _item_key(it: dict) -> tuple:
    return (it["label"][:120], it["href"], it.get("ctx", ""))


_ENGINE_ELEMENT_CAP = 600  # browser_engine's per-frame cap of the interactive elements it lists (MAX_ELEMENTS in its page script)


async def _gather(session: str, include: str, pages: int, limit: int, flags: dict = None) -> list:
    """The items of the current page and, with ``pages`` > 0, of what appears as the page is scrolled down (infinite feeds):
    de-duplicated, in the order met. Stops at ``limit`` items, at the end of the page or after two scrolls that brought nothing."""
    from browser_engine import browser_engine

    inc = (include or "links").strip().lower()
    items, seen, stale = [], set(), 0
    for rnd in range(max(0, int(pages)) + 1):
        els = await browser_engine.get_interactive_elements_raw(session)
        if flags is not None and len(els) >= _ENGINE_ELEMENT_CAP:
            flags["page_truncated"] = True  # the engine stopped listing: the tail of a long page is not in this scan
        ctx = await browser_engine.get_block_contexts(session)
        if inc == "posts":
            blocks = await browser_engine.get_text_blocks(session)
            snap = _post_items(blocks, els, ctx)
        else:
            snap = _scan_items(els, inc, ctx)
        new = 0
        for it in snap:
            k = _item_key(it)
            if k in seen:
                continue
            seen.add(k)
            items.append(it)
            new += 1
        if len(items) >= limit or rnd >= pages:
            break
        stale = stale + 1 if new == 0 else 0
        if stale >= 2:
            break
        await browser_engine.scroll("down", amount=_SCAN_SCROLL_PX, session=session, return_state=False)
    return items[:limit]


def leeth_scan_page(criterion: str, mode: str = "yesno", categories=None, include: str = "links", accept_above: float = 0.7,
                    reject_below: float = 0.3, max_items: int = 300, scroll_pages: int = 0, positive_examples=None, negative_examples=None,
                    session: str = "default") -> str:
    """Read-only: classify the clickable elements -- or the POSTS -- of the CURRENT browser page against a criterion, in one batched pass.

    For "which of these links / bookmarks / posts ...": returns the matching items (label, link, probability) and the unsure ones.
    It changes nothing on the page (except scrolling when scroll_pages > 0). Acting on the result is a separate step
    (leeth_click_many with the indices of the buttons it reports, or browser_open on a link).

    include='links' | 'buttons' | 'all' judge the clickable elements (the text judged is the label plus the link; for delete / like
    buttons the label carries the item's title). include='posts' judges the repeated text blocks of the page (posts, cards, rows,
    comments): their text, up to 600 characters; each reported post lists its own buttons / links with indices for leeth_click_many.

    Args:
        criterion: a yes/no question about one item ("Is this a recipe?"), or, for mode='categories', the question.
        mode: 'yesno' or 'categories'.
        categories: for mode='categories', the categories.
        include: 'links', 'buttons', 'all' or 'posts'.
        accept_above, reject_below: the yes / no bounds of the probability; between them an item is 'unsure'.
        max_items: how many items are judged at most.
        scroll_pages: scroll the page down up to this many times (700 px each) and judge what appears too, for endless feeds (max 30).
        positive_examples, negative_examples: for mode='yesno', 2-3 texts (as they look on the page: a post's text, a link's label)
            that DO / do NOT satisfy the criterion. Strongly recommended for content questions: the yes / no cut is learnt from them
            (recall ~0.3 -> ~0.7 at ~2% false yes); without them many clear matches stay 'unsure' (sorted by p, read from the top).
        session: browser session.
    """
    from browser_engine import browser_engine
    from src.leeth import bulk

    mode = (mode or "yesno").strip().lower()
    inc = (include or "links").strip().lower()
    if inc not in ("links", "buttons", "all", "posts"):
        return "Error: include must be 'links', 'buttons', 'all' or 'posts'."
    pages = max(0, min(int(scroll_pages or 0), 30))
    limit = max(1, int(max_items))
    flags: dict = {}
    try:
        items = browser_engine.run(_gather(session, inc, pages, limit, flags))
    except Exception as e:
        return f"Error reading the page: {e}"
    if not items:
        what = "posts / text blocks" if inc == "posts" else "elements"
        return f"No {what} to judge on the page (open it with browser_open first" + ("" if inc == "posts" else ", or use include='all'") + ")."
    texts = [(f"{it['label']} — {it['href']}" if it["href"] and it["label"] and inc != "posts" else it["label"] or it["href"]) for it in items]
    cal: dict = {}
    t0 = time.perf_counter()
    try:
        rt = _leeth2()
        if mode == "yesno":
            if not criterion or not criterion.strip():
                return "Error: criterion (a yes/no question) is required for mode='yesno'."
            res = bulk.classify_yesno(rt, texts, criterion, accept=float(accept_above), reject=float(reject_below),
                                      positives=_as_list(positive_examples), negatives=_as_list(negative_examples), meta=cal)
        elif mode == "categories":
            res = bulk.classify_categories(rt, texts, _as_list(categories), question=criterion or None)
        else:
            return "Error: mode must be 'yesno' or 'categories'."
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("leeth_scan_page error: %s", e)
        return f"Error in leeth_scan_page: {e}"

    # what leeth_click_many may act on: scan-local handles (an element index of an earlier scroll position can be reused by another
    # element later, so a handle is never taken from the engine twice)
    scan: dict = {}
    used: set = set()

    def handle(el: dict, want=None) -> int:
        h = int(want) if want is not None and int(want) not in used else (max(used) + 1 if used else 1)
        used.add(h)
        scan[h] = {"label": el["label"], "href": el["href"], "text": el.get("text", ""), "ctx": el.get("ctx", "")}
        return h

    def brief(r):
        it = items[r["i"]]
        if inc == "posts":
            d = {"post": r["i"] + 1, "text": it["label"][:160]}
            if it["href"]:
                d["href"] = it["href"][:160]
            d["buttons"] = [{"index": handle(b, b["index"]), "label": (b["label"] or b["href"])[:60]} for b in it.get("buttons", [])]
        else:
            d = {"index": handle(it, it["index"]), "label": it["label"][:100]}
            if it["href"]:
                d["href"] = it["href"][:120]
        if mode == "categories":
            d["category"] = r["best"]
        d["p"] = r["p"]
        return d

    if mode == "yesno":
        yes = [brief(r) for r in res if r["decision"] == "yes"]
        unsure = [brief(r) for r in res if r["decision"] == "unsure"]
        rep = {"mode": "yesno", "judged": len(items), "yes": yes, "unsure": unsure, "no_count": sum(r["decision"] == "no" for r in res)}
    else:
        groups: dict = {}
        for r in res:
            groups.setdefault(r["best"] if r["category"] else "(unsure)", []).append(brief(r))
        rep = {"mode": "categories", "judged": len(items), "groups": groups}
    _LAST_SCAN[session] = scan
    rep.update(cal)
    if mode == "yesno":
        _hint_unsure(rep, len(items), bool(cal))
    if flags.get("page_truncated"):
        rep["page_truncated"] = ("the page has more interactive elements than the browser lists at once (600): the tail was not "
                                 "judged -- scroll_pages brings the rest in, or act on this part first and scan again")
    if len(items) >= limit:
        rep["limit_reached"] = f"only the first {limit} items were judged (max_items)"
    if pages:
        rep["scrolled"] = pages
    rep["ms"] = round((time.perf_counter() - t0) * 1000)
    rep["note"] = "indices are handles of this scan; act with leeth_click_many(indices) before the page changes"
    return json.dumps(rep, ensure_ascii=False)


def leeth_click_many(indices, session: str = "default", confirm: bool = False) -> str:
    """Click the elements the last leeth_scan_page returned, one after another, re-finding each one on the page before its click
    (a delete re-renders a list and shifts the indices -- the click goes by label + link, never by a stale number).

    Elements that look destructive (delete, pay, confirm, purchase ...) are NOT clicked unless confirm=True; ask the user first
    (request_user_approval) and say how many items there are. Elements that are no longer on the page, or are ambiguous, are skipped
    and reported.

    Args:
        indices: element indices from the last leeth_scan_page (JSON list or comma-separated).
        session: browser session.
        confirm: the user has approved clicking destructive elements.
    """
    from browser_engine import browser_engine
    from src.leeth import webforms

    raw = _as_list(indices, split_lines=False)
    if len(raw) == 1 and isinstance(raw[0], str):
        raw = [x for x in re.split(r"[,\s]+", raw[0]) if x]
    try:
        wanted = [int(x) for x in raw]
    except (TypeError, ValueError):
        return "Error: indices must be integers."
    scan = _LAST_SCAN.get(session)
    if not scan:
        return "Error: nothing to click -- run leeth_scan_page first (its indices are the only ones accepted)."
    unknown = [i for i in wanted if i not in scan]
    targets = [(i, scan[i]) for i in wanted if i in scan]
    # the label alone is not enough: a delete button carries its item's title as the label ("Рецепт борща") and "Удалить" only as its text
    danger = [(i, t["label"] or t.get("text", "")) for i, t in targets if webforms.is_dangerous_label(t["label"]) or webforms.is_dangerous_label(t.get("text", ""))]
    if danger and not confirm:
        shown = "; ".join(f"[{i}] {lab[:40]}" for i, lab in danger[:5]) + (" ..." if len(danger) > 5 else "")
        return (f"Not clicked: {len(danger)} of {len(targets)} elements look destructive ({shown}). "
                "Ask the user for approval, then call again with confirm=True.")

    async def _go():
        done, skipped, failed = [], [], []
        for i, t in targets:
            els = await browser_engine.get_interactive_elements_raw(session)
            same = [e for e in els if str(e.get("label") or "").strip() == t["label"] and str(e.get("href") or "").strip() == t["href"]]
            if len(same) > 1 and t.get("ctx"):  # identical buttons of different posts: the post's own text tells them apart
                ctxs = await browser_engine.get_block_contexts(session)
                same = [e for e in same if ctxs.get(int(e["idx"]), "") == t["ctx"]]
            if len(same) != 1:
                skipped.append({"index": i, "label": t["label"][:60], "reason": "gone from the page" if not same else "ambiguous: several identical elements"})
                continue
            res = await browser_engine.click(index=int(same[0]["idx"]), session=session, return_state=False)
            if str(res).lstrip().lower().startswith("error"):
                failed.append({"index": i, "label": t["label"][:60], "error": str(res)[:120]})
            else:
                done.append(i)
        return done, skipped, failed

    try:
        done, skipped, failed = browser_engine.run(_go())
    except Exception as e:
        logger.exception("leeth_click_many error: %s", e)
        return f"Error in leeth_click_many: {e}"
    _LAST_SCAN.pop(session, None)  # the page has changed: the old indices mean nothing now
    return json.dumps({"clicked": done, "skipped": skipped, "failed": failed, "unknown_indices": unknown}, ensure_ascii=False)


def _is_error(result) -> bool:
    return str(result).lstrip().lower().startswith("error")


def leeth_fill_form(instruction: str, session: str = "default", dry_run: bool = False, submit: bool = False, min_confidence: float = 0.55) -> str:
    """Fill the form on the CURRENT browser page from one instruction in a single pass: every field, dropdown and checkbox at once.

    "Введи в логин Валера, рост 175 см, возраст 29 лет, выбери город Гонконг" -> Leeth reads the form (labels, dropdown options),
    finds the value of every field in the instruction (nothing is invented: a field with no value in the instruction stays empty and
    is reported), fills them in one batch and re-reads the page to verify. A destructive / payment button is never pressed.

    Args:
        instruction: what to enter, in the user's own words.
        session: browser session.
        dry_run: only show the plan, change nothing.
        submit: also press the submit button -- only when it is not a payment / delete / confirm button, otherwise it is reported.
        min_confidence: a value below it is not entered (reported instead).
    """
    from browser_engine import browser_engine
    from src.leeth import webforms

    if not instruction or not instruction.strip():
        return "Error: empty instruction."

    async def _go():
        import asyncio

        els = await browser_engine.get_interactive_elements_raw(session)
        fields = webforms.parse_fields(els)
        if not fields:
            return {"error": "no fillable fields on the page (open the page with browser_open first)"}
        sel = [f.index for f in fields if f.kind == "select"]
        opts = await browser_engine.get_select_options(sel, session) if sel else {}
        rt = _leeth2()
        plan = await asyncio.to_thread(webforms.plan_fill, rt, instruction, fields, opts, float(min_confidence))
        report = {"plan": plan.to_dict(), "fields_on_page": len(fields), "executed": not dry_run}
        if not dry_run:
            results = []
            for s in plan.steps:
                if s.kind == "type":
                    r = await browser_engine.fill_input(index=s.index, text=s.value, session=session, return_state=False)
                elif s.kind == "select":
                    r = await browser_engine.select_option(index=s.index, label=s.value, session=session, return_state=False)
                else:  # check / uncheck: a click toggles
                    r = await browser_engine.click(index=s.index, session=session, return_state=False)
                row = {"label": s.label, "ok": not _is_error(r)}
                if _is_error(r):
                    row["error"] = str(r)[:120]
                results.append(row)
            report["results"] = results
            # verify against a fresh read of the page, by label (the indices may have moved)
            after = webforms.parse_fields(await browser_engine.get_interactive_elements_raw(session))
            by_label: dict = {}
            for f in after:
                by_label.setdefault(f.label, []).append(f)
            verified = []
            for s in plan.steps:
                cand = by_label.get(s.label) or []
                f = cand[0] if len(cand) == 1 else None
                if f is None:
                    ok = None
                elif s.kind == "type":
                    ok = f.value.strip() == s.value.strip() or f.value.strip() == s.value.strip()[:50]
                elif s.kind == "select":
                    ok = webforms._norm(f.value) == webforms._norm(s.value)
                else:
                    ok = bool(f.checked) == (s.kind == "check")
                verified.append({"label": s.label, "verified": ok})
            report["verified"] = verified
        if submit:
            pick = await asyncio.to_thread(webforms.pick_submit, rt, instruction, els, plan.language)
            if pick.get("index") is None:
                report["submit"] = {"pressed": False, "reason": pick.get("reason")}
            elif pick.get("dangerous"):
                report["submit"] = {"pressed": False, "button": pick["label"],
                                    "reason": "the button looks like a payment / delete / confirm action: needs the user's approval"}
            elif dry_run:
                report["submit"] = {"pressed": False, "would_press": pick["label"]}
            else:
                r = await browser_engine.click(index=int(pick["index"]), session=session, return_state=False)
                report["submit"] = {"pressed": not _is_error(r), "button": pick["label"]}
        return report

    try:
        rep = browser_engine.run(_go())
    except Exception as e:
        logger.exception("leeth_fill_form error: %s", e)
        return f"Error in leeth_fill_form: {e}"
    return json.dumps(rep, ensure_ascii=False)
