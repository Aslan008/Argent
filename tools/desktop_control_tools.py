"""`desktop_inspect` and `desktop_act`: another program's controls as text, and pressing them.

The loop is observe -> act -> check: `desktop_inspect` returns the controls of a window with ids, `desktop_act`
uses those ids and answers with the window as it looks AFTER the action, so the next step is made on what is
there, not on what was expected. Every rule about who may act where lives in src/automation/desktop_guard.py.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

from logger import get_logger

log = get_logger("tools")

_LAST: Dict[int, "object"] = {}              # hwnd -> the latest Observation of that window


def _unavailable() -> Optional[str]:
    from src.automation import desktop_guard, uia
    if os.name != "nt":
        return "Error: desktop control works on Windows only."
    if not uia.available():
        return "Error: the 'uiautomation' package is not installed (pip install uiautomation)."
    if not desktop_guard.enabled():
        return ("Error: desktop control is off. Only the person can switch it on (/desktop on) and choose the "
                "programs it may act in (/desktop allow <program.exe>).")
    return None


def _store(obs) -> None:
    _LAST[obs.window.hwnd] = obs


def desktop_inspect(action: str = "windows", window: str = "", query: str = "", role: str = "",
                    timeout: int = 10) -> str:
    from src.automation import desktop_guard, uia
    action = (action or "windows").strip().lower()
    problem = _unavailable()
    if problem:
        return problem
    try:
        if action == "windows":
            allowed = desktop_guard.allowed_apps()
            lines = ["Windows (acting is possible only in the ones marked [allowed]):"]
            for w in uia.windows():
                mark = "[allowed]" if w.process.lower() in allowed else "[not allowed]"
                lines.append(f"- '{w.title}' — {w.process or 'pid ' + str(w.pid)} {mark}")
            return "\n".join(lines)

        if action not in ("tree", "find", "wait"):
            return "Error: action must be one of: windows, tree, find, wait."
        if not window.strip():
            return "Error: window is required (part of its title; see action='windows')."

        deadline = time.monotonic() + max(1, min(int(timeout or 10), 60))
        info = uia.find_window(window)
        while info is None and action == "wait" and time.monotonic() < deadline:
            time.sleep(0.4)
            info = uia.find_window(window)
        if info is None:
            return f"Error: no visible window matching '{window}'."
        verdict = desktop_guard.check_window(info.process, info.title)
        if not verdict:
            desktop_guard.audit("observe_refused", window=info.title, process=info.process, reason=verdict.reason)
            return f"Refused: {verdict.reason}."

        if action == "wait" and query.strip():
            while True:
                obs = uia.observe(info)
                hits = uia.match(obs, query, role)
                if hits or time.monotonic() >= deadline:
                    break
                time.sleep(0.4)
        else:
            obs = uia.observe(info)
        _store(obs)
        desktop_guard.audit("observe", window=info.title, process=info.process, controls=len(obs.elements))

        if action == "find" or (action == "wait" and query.strip()):
            hits = uia.match(obs, query, role)
            if not hits:
                return (f"Nothing matches '{query}' in '{info.title}' (observation #{obs.number}, "
                        f"{len(obs.elements)} controls).")
            shown = uia.Observation(number=obs.number, window=obs.window, elements=hits[:40],
                                    truncated=len(hits) > 40)
            return uia.render(shown)
        return uia.render(obs)
    except uia.UiaUnavailable as e:
        return f"Error: {e}"
    except Exception as e:
        log.warning("desktop_inspect failed: %s", e)
        return f"Error: {e}"


_DESCRIBE = {
    "click": "нажать", "double_click": "двойной щелчок по", "right_click": "правый щелчок по",
    "type": "ввести текст в", "set_value": "записать значение в", "hotkey": "нажать сочетание клавиш в окне",
    "focus": "вывести на передний план окно", "scroll": "прокрутить", "drag": "перетащить",
}
_DIRECTION_WORD = {"up": "вверх", "down": "вниз", "left": "влево", "right": "вправо"}


def _label(element, fallback: str = "точку на экране") -> str:
    return f"«{element.name or element.automation_id or element.role}»" if element is not None else fallback


def _describe(action: str, element, window, text: str, keys: str, extra: str = "") -> str:
    from src.agent.secret_redactor import redact
    what = f"{_DESCRIBE.get(action, action)} "
    if action == "focus":
        return f"{what}«{window.title}» ({window.process})"
    if action == "hotkey":
        return f"{what}«{window.title}» ({window.process}): {keys}"
    if action in ("scroll", "drag"):
        return f"{what}{_label(element, 'содержимое окна' if action == 'scroll' else 'точку на экране')} " \
               f"{extra} в окне «{window.title}» ({window.process})".replace("  ", " ")
    target = _label(element)
    extra = f": «{redact(text)[:80]}»" if text else ""
    return f"{what}{target}{extra} в окне «{window.title}» ({window.process})"


def _inside(rect, point) -> bool:
    wx, wy, ww, wh = rect
    return wx <= point[0] <= wx + ww and wy <= point[1] <= wy + wh


def desktop_act(action: str = "click", window: str = "", element: str = "", text: str = "", keys: str = "",
                x: int = -1, y: int = -1, direction: str = "down", amount: int = 3, to_element: str = "",
                to_x: int = -1, to_y: int = -1) -> str:
    from approval import request_approval
    from src.automation import desktop_guard, uia

    action = (action or "click").strip().lower()
    problem = _unavailable()
    if problem:
        return problem
    if action not in _DESCRIBE:
        return f"Error: action must be one of: {', '.join(_DESCRIBE)}."
    if not window.strip():
        return "Error: window is required (part of its title)."

    # Auto mode approves by itself; the mouse and keyboard are not something to hand out that way.
    import approval
    if approval.get_policy() == approval.POLICY_AUTO:
        return ("Refused: auto mode approves actions without the person looking, and desktop control never "
                "runs that way. The person switches to manual approval (/vibe off) first.")

    series = desktop_guard.SERIES
    try:
        info = uia.find_window(window)
        if info is None:
            return f"Error: no visible window matching '{window}'."
        obs = _LAST.get(info.hwnd)
        elem = None
        point = None
        target = None                      # drag: the element it is dropped on
        drop = None                        # drag: the point it is dropped at
        if action == "scroll":
            direction = (direction or "down").strip().lower()
            if direction not in uia._DIRECTIONS:
                return f"Error: direction must be one of: {', '.join(uia._DIRECTIONS)}."
            if element.strip():
                elem = obs.get(element) if obs is not None else None
                if elem is None:
                    return (f"Error: element '{element}' is not in the latest observation of '{info.title}'. "
                            f"Call desktop_inspect(action='tree' or 'find', window=...) and use the ids it returns.")
            elif x >= 0 and y >= 0:
                point = (int(x), int(y))
                if not _inside(info.rect, point):
                    return "Error: the point is outside the window."
        elif action == "drag":
            if not (element.strip() or (x >= 0 and y >= 0)):
                return "Error: give the element to drag (element='e3') or its point (x, y)."
            if not (to_element.strip() or (to_x >= 0 and to_y >= 0)):
                return "Error: give where to drop it: to_element='e5' or to_x, to_y."
            if element.strip():
                elem = obs.get(element) if obs is not None else None
                if elem is None:
                    return (f"Error: element '{element}' is not in the latest observation of '{info.title}'. "
                            f"Call desktop_inspect(action='tree' or 'find', window=...) and use the ids it returns.")
            else:
                point = (int(x), int(y))
            if to_element.strip():
                target = obs.get(to_element) if obs is not None else None
                if target is None:
                    return f"Error: element '{to_element}' is not in the latest observation of '{info.title}'."
            else:
                drop = (int(to_x), int(to_y))
            for p in (point, drop):
                if p is not None and not _inside(info.rect, p):
                    return "Error: the point is outside the window."
        elif action in ("click", "double_click", "right_click", "type", "set_value"):
            if element.strip():
                elem = obs.get(element) if obs is not None else None
                if elem is None:
                    desktop_guard.audit("refused", action=action, window=info.title, process=info.process,
                                        reason="unknown element")
                    return (f"Error: element '{element}' is not in the latest observation of '{info.title}'. "
                            f"Call desktop_inspect(action='tree' or 'find', window=...) and use the ids it returns.")
            elif action in ("click", "double_click", "right_click") and x >= 0 and y >= 0:
                point = (int(x), int(y))
                if obs is not None:
                    wx, wy, ww, wh = info.rect
                    if not (wx <= point[0] <= wx + ww and wy <= point[1] <= wy + wh):
                        return "Error: the point is outside the window."
            elif action in ("type", "set_value"):
                focused = next((e for e in (obs.elements if obs else []) if e.focused), None)
                if focused is None:
                    return "Error: name the element to type into (element='e3'): nothing in the window has focus."
                elem = focused
            else:
                return "Error: give an element id (from desktop_inspect) or x and y for a click."
        if action in ("type", "set_value") and not text:
            return "Error: text is required."
        if action == "hotkey":
            try:
                uia.parse_keys(keys)
            except ValueError as e:
                return f"Error: {e}."

        foreground = uia.foreground_window()
        # keys and typed text land where the focus is: a focused password field is as forbidden as a named one
        focus_is_password = bool(obs and any(e.focused and e.password for e in obs.elements))
        decision = desktop_guard.check_action(
            action=action, process=info.process, title=info.title,
            # a drop onto «Корзина» deletes as surely as pressing «Удалить»: both names are looked at
            element_name=" ".join(n for n in ((elem.name if elem is not None else ""),
                                              (target.name if target is not None else "")) if n),
            is_password=bool(elem and elem.password) or (action in ("type", "hotkey") and focus_is_password),
            typing=action in ("type", "hotkey"), observed_number=obs.number if obs else None,
            current_number=obs.number if obs else None,
            observation_age=(time.monotonic() - obs.taken_at) if obs else None,
            foreground_ok=foreground is not None, series=series)
        if not decision:
            desktop_guard.audit("refused", action=action, window=info.title, process=info.process,
                                element=elem.name if elem else "", reason=decision.reason)
            return f"Refused: {decision.reason}."

        extra = ""
        if action == "scroll":
            extra = f"{_DIRECTION_WORD.get(direction, direction)} ×{max(1, min(int(amount or 1), uia.MAX_SCROLL_STEPS))}"
        elif action == "drag":
            extra = "на " + (_label(target) if target is not None else f"точку {drop[0]},{drop[1]}")
        description = _describe(action, elem, info, text, keys, extra)
        approved = request_approval(f"{desktop_guard.ACTION_PREFIX}{description}", destructive=decision.destructive,
                                    grant_key=f"desktop:{info.process.lower()}")
        if not approved:
            desktop_guard.audit("denied", action=action, window=info.title, process=info.process,
                                element=elem.name if elem else "")
            return f"Not done: the person declined — {description}."

        series.wait_turn()
        if series.stopped:
            return f"Stopped: {series.stop_reason}. Nothing was done."
        series.note_action()

        control = None
        if elem is not None:
            control = uia.resolve(info, elem)
            if control is None:
                return ("Error: that control is no longer where it was (the window changed). "
                        "Observe the window again.")
        if target is not None and uia.resolve(info, target) is None:
            return ("Error: the drop target is no longer where it was (the window changed). "
                    "Observe the window again.")

        # keys and clicks by coordinates go to whatever is in front: make sure it is the window we mean
        if not uia.bring_to_front(info):
            return "Error: could not bring the window to the front."
        time.sleep(0.15)
        front = uia.foreground_window()
        if front is None or front.hwnd != info.hwnd:
            return (f"Refused: another window ('{front.title if front else '?'}') is in front, so input would "
                    f"go there. Nothing was done.")

        if series.stopped:
            return f"Stopped: {series.stop_reason}. Nothing was done."

        detail = ""
        if action == "focus":
            detail = "window in front"
        elif action == "hotkey":
            uia.send_keys(keys)
            detail = f"keys {keys}"
        elif action in ("click", "double_click", "right_click"):
            if control is not None and action == "click":
                detail = f"pressed via {uia.invoke(control)}"
            else:
                cx, cy = elem.center if elem is not None else point
                uia.click_at(cx, cy, button="right" if action == "right_click" else "left",
                             double=action == "double_click")
                detail = f"clicked at {cx},{cy}"
        elif action == "scroll":
            wx, wy, ww, wh = info.rect
            where = elem.center if elem is not None else (point or (wx + ww // 2, wy + wh // 2))
            detail = f"scrolled {direction} ×{amount} via {uia.scroll(control, where, direction, amount)}"
        elif action == "drag":
            sx, sy = elem.center if elem is not None else point
            tx, ty = target.center if target is not None else drop
            uia.drag(sx, sy, tx, ty)
            detail = f"dragged {sx},{sy} -> {tx},{ty}"
        elif action == "set_value":
            if not uia.set_value(control, text):
                return "Error: this control cannot take a value directly; use action='type'."
            detail = f"value set ({len(text)} characters)"
        elif action == "type":
            try:
                uia.focus(control)
            except Exception:
                pass
            time.sleep(0.05)
            uia.send_text(text)
            detail = f"typed {len(text)} characters"

        desktop_guard.audit("done", action=action, window=info.title, process=info.process,
                            element=elem.name if elem else "", text=text or None, keys=keys or None, detail=detail)
        time.sleep(0.3)
        fresh = uia.observe(info)
        _store(fresh)
        return f"Done: {description} ({detail}).\n" + uia.render(fresh, limit=60)
    except uia.UiaUnavailable as e:
        return f"Error: {e}"
    except Exception as e:
        log.warning("desktop_act failed: %s", e)
        desktop_guard.audit("error", action=action, window=window, error=str(e)[:200])
        return f"Error: {e}"
