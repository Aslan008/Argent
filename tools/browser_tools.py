import re
import threading

from logger import get_logger

log = get_logger("tools")

from browser_engine import browser_engine


_SESSION_POLL_TRACKER: dict[str, dict] = {}
_SESSION_EVAL_TRACKER: dict[str, dict] = {}
_SESSION_EVAL_LOCK = threading.Lock()


def _reset_poll_tracker(session: str = "default") -> None:
    if session in _SESSION_POLL_TRACKER:
        _SESSION_POLL_TRACKER[session] = {"count": 0, "last_query": None}
    with _SESSION_EVAL_LOCK:
        if session in _SESSION_EVAL_TRACKER:
            _SESSION_EVAL_TRACKER[session] = {
                "count": 0,
                "last_expr": None,
                "last_result": None,
                "consecutive_void_probes": 0,
            }


def _format_error(e: Exception) -> str:
    if isinstance(e, KeyError):
        return e.args[0] if e.args else str(e)
    return str(e)


def run_browser_task(task: str, url: str = None) -> str:
    """Delegate a browser automation task to a specialized headless sub-agent."""
    from agent import ArgentSubAgent
    
    # We provide the sub-agent with manual browser tools so IT can do the navigation,
    # while the main agent just waits for the result.
    allowed_tools = [
        "browser_open", "browser_state", "browser_click", 
        "browser_input", "browser_select", "browser_get_content", "browser_scroll",
        "browser_wait_for", "browser_press_key", "browser_fill_form", "browser_upload_file",
        "browser_back", "browser_reload", "browser_eval_js", "browser_screenshot",
        "browser_close", "browser_switch_tab",
        "browser_act", "browser_extract", "browser_observe",
    ]
    
    prompt = f"You are an autonomous browser agent. Your task is to navigate the web and retrieve the requested information or perform the requested action.\n\nTask: {task}"
    if url:
        prompt += f"\n\nStart by opening this URL: {url}"
        
    try:
        agent = ArgentSubAgent(role="BrowserWorker", task=prompt, tools_override=allowed_tools)
        result = agent.execute()
        return f"Browser Sub-Agent completed the task.\n\nResult:\n{result}"
    except Exception as e:
        log.error("run_browser_task error: %s", e)
        return f"Error running browser task: {e}"


def browser_open(url: str, session: str = "default", headed: bool = None, return_state: bool = True, mode: str = "auto") -> str:
    """Open a URL in the browser. Creates a new session if needed.

    Args:
        url: The URL to navigate to (e.g. 'https://example.com').
        session: Name of the browser session. Default: 'default'.
        headed: True shows a browser window on the user's desktop, False runs in the background, None (not
            given) keeps the browser as it is - a later call that did not mention it used to close the
            window the user was watching and restart the browser headless.
        return_state: If true (default), automatically returns updated page state.
        mode: Browser environment to use:
            - 'user': The user's real installed browser (Yandex/Chrome/Edge). Use this when the user asks "в моём браузере" / "через мой браузер" or when the task requires the user's existing logins/cookies.
            - 'isolated': Clean, disposable Chromium for anonymous scraping, testing, or background tasks.
            - 'auto' (default): Automatically chooses based on system config or task context.
    """
    _reset_poll_tracker(session)
    if mode is not None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in ("auto", "user", "isolated"):
            return f"Error: Invalid browser mode '{mode}'. Allowed modes: 'auto', 'user', 'isolated'."
        mode = normalized_mode
    else:
        mode = "auto"

    try:
        return browser_engine.run(browser_engine.open_page(url, session, headed, return_state=return_state, mode=mode))
    except Exception as e:
        log.error("browser_open error: %s", e)
        return f"Error opening browser: {_format_error(e)}"



def browser_state(session: str = "default", query: str = None, scroll_depth: int = 0, mode: str = None) -> str:
    """Get the current page state: numbered list of interactive elements, optionally filtered by query.

    By default uses the accessibility tree (semantic roles, hierarchy, structural
    context) for richer page understanding. Set mode='dom' for a compact flat
    list of interactive elements only — useful for quick checks after an action.
    """
    tracker = _SESSION_POLL_TRACKER.setdefault(session, {"count": 0, "last_query": None})
    if tracker["last_query"] == query:
        tracker["count"] += 1
    else:
        tracker["last_query"] = query
        tracker["count"] = 1

    try:
        res = browser_engine.run(browser_engine.get_state(session, query, scroll_depth, mode))
        if tracker["count"] >= 3:
            hint = (
                "[POLLING HINT]: You have called browser_inspect multiple times with the same query. "
                "Stop blind polling! Use browser_action(action=\"wait_for\", text=\"...\", amount=15) to wait for the element to appear/render, "
                "or browser_inspect(mode=\"text_content\") to read the current section directly.\n\n"
            )
            res = hint + str(res)
        return res
    except Exception as e:
        log.error("browser_state error: %s", e)
        return f"Error getting browser state: {_format_error(e)}"


def browser_click(index: int = None, selector: str = None, text: str = None,
                  session: str = "default", force: bool = False, return_state: bool = True) -> str:
    """Click an element by its index, CSS selector, or visible text."""
    _reset_poll_tracker(session)
    try:
        return browser_engine.run(browser_engine.click(index, selector, text, session, force=force, return_state=return_state))
    except Exception as e:
        log.error("browser_click error: %s", e)
        return f"Error clicking element: {_format_error(e)}"


def browser_input(index: int = None, text: str = "", selector: str = None,
                  session: str = "default", return_state: bool = True,
                  clear_first: bool = False, **kwargs) -> str:
    """Type text into an input element by its index or CSS selector."""
    _reset_poll_tracker(session)
    try:
        return browser_engine.run(browser_engine.fill_input(index, text, selector, session, return_state=return_state))
    except Exception as e:
        log.error("browser_input error: %s", e)
        return f"Error filling input: {_format_error(e)}"


def browser_select(index: int = None, selector: str = None, value: str = None,
                   label: str = None, session: str = "default", return_state: bool = True) -> str:
    """Select an option in a <select> dropdown by index, selector, value, or visible label."""
    _reset_poll_tracker(session)
    try:
        return browser_engine.run(browser_engine.select_option(index, selector, value, label, session, return_state=return_state))
    except Exception as e:
        log.error("browser_select error: %s", e)
        return f"Error selecting option: {_format_error(e)}"


def browser_screenshot(path: str = None, full_page: bool = False, session: str = "default") -> str:
    """Take a screenshot of the current page and attach to conversation for vision models."""
    try:
        res = browser_engine.run(browser_engine.screenshot(path, full_page, session))
        if "Screenshot saved to:" in str(res):
            saved_path = str(res).split("Screenshot saved to:")[-1].strip()
            from config import get_current_model, get_provider
            from vision import encode_image, model_supports_vision, queue_image
            supports = model_supports_vision(get_current_model(), get_provider())
            if supports is not False:
                payload, media_type, note = encode_image(saved_path)
                if payload:
                    queue_image(payload, media_type, saved_path)
                    res += (
                        "\n\n[VISION ATTACHED]: The browser screenshot has been attached to the conversation. "
                        "You will SEE it in your very next message — continue and analyze the page."
                    )
        return res
    except Exception as e:
        log.error("browser_screenshot error: %s", e)
        return f"Error taking screenshot: {_format_error(e)}"


def browser_scroll(direction: str = "down", amount: int = 500, index: int = None,
                   selector: str = None, session: str = "default", return_state: bool = True) -> str:
    """Scroll the page or scroll a specific element into view."""
    _reset_poll_tracker(session)
    try:
        return browser_engine.run(browser_engine.scroll(direction, amount, index, selector=selector, session=session, return_state=return_state))
    except Exception as e:
        log.error("browser_scroll error: %s", e)
        return f"Error scrolling: {_format_error(e)}"


def browser_wait_for(text: str = None, selector: str = None, expression: str = None,
                     state: str = "visible", timeout: float = 15.0,
                     session: str = "default", return_state: bool = True) -> str:
    """Wait for an element or JavaScript condition to satisfy a state.

    Supports visible text, CSS selector, or an arbitrary JavaScript expression (evaluated until truthy).
    Valid states: 'visible', 'attached', 'detached', 'hidden', 'enabled', 'disabled'.
    Automatically returns the fresh page state once ready, eliminating blind polling loops.
    """
    _reset_poll_tracker(session)
    try:
        return browser_engine.run(browser_engine.wait_for(
            text=text, selector=selector, expression=expression, state=state,
            timeout=timeout, session=session, return_state=return_state
        ))
    except Exception as e:
        log.error("browser_wait_for error: %s", e)
        return f"Error waiting for condition: {_format_error(e)}"


def browser_get_content(content_type: str = "text", index: int = None, selector: str = None,
                        text: str = None, session: str = "default") -> str:
    """Extract content from the page. content_type: 'text', 'markdown', or 'html'. Use text, selector, or index for targeting."""
    try:
        if content_type == "markdown":
            return browser_engine.run(browser_engine.get_markdown(selector=selector, index=index, text=text, session=session))
        elif content_type == "html":
            return browser_engine.run(browser_engine.get_html(selector=selector, index=index, text=text, session=session))
        else:
            return browser_engine.run(browser_engine.get_text(index=index, selector=selector, text=text, session=session))
    except Exception as e:
        log.error("browser_get_content error: %s", e)
        return f"Error getting content: {_format_error(e)}"


def browser_close(session: str = "default", index: int = None, text: str = None) -> str:
    """Close a browser session, or close a specific tab if index or text is provided."""
    if index is not None or text is not None:
        return browser_close_tab(index=index, text=text, session=session)
    _reset_poll_tracker(session)
    try:
        return browser_engine.run(browser_engine.close_session(session))
    except Exception as e:
        log.error("browser_close error: %s", e)
        return f"Error closing browser: {_format_error(e)}"


def browser_close_tab(index: int = None, text: str = None, session: str = "default") -> str:
    """Close an open browser tab by its 1-based index, URL/title match, or the currently active tab."""
    try:
        return browser_engine.run(browser_engine.close_tab(index=index, text=text, session=session))
    except Exception as e:
        log.error("browser_close_tab error: %s", e)
        return f"Error closing tab: {_format_error(e)}"


def browser_list_tabs(session: str = "default") -> str:
    """List all open tabs in the browser session with indices, URLs, and titles."""
    try:
        return browser_engine.run(browser_engine.list_tabs(session=session))
    except Exception as e:
        log.error("browser_list_tabs error: %s", e)
        return f"Error listing tabs: {_format_error(e)}"


def browser_switch_tab(index: int, session: str = "default") -> str:
    """Switch active tab in the browser context by index."""
    try:
        return browser_engine.run(browser_engine.switch_tab(index, session))
    except Exception as e:
        log.error("browser_switch_tab error: %s", e)
        return f"Error switching tab: {_format_error(e)}"


def browser_press_key(key: str, session: str = "default", return_state: bool = True) -> str:
    """Press a keyboard key (Enter, Escape, Tab, ArrowDown, Backspace, etc.)."""
    _reset_poll_tracker(session)
    try:
        return browser_engine.run(browser_engine.press_key(key, session, return_state=return_state))
    except Exception as e:
        log.error("browser_press_key error: %s", e)
        return f"Error pressing key: {_format_error(e)}"


def browser_fill_form(fields: list | str, submit_index: int = None, submit_selector: str = None,
                      session: str = "default", return_state: bool = True) -> str:
    """Batch-fill multiple form inputs/selects and optionally click submit in one turn."""
    _reset_poll_tracker(session)
    try:
        return browser_engine.run(browser_engine.fill_form(fields, submit_index, submit_selector, session, return_state=return_state))
    except Exception as e:
        log.error("browser_fill_form error: %s", e)
        return f"Error filling form: {_format_error(e)}"


def browser_upload_file(file_path: str | list, index: int = None, selector: str = None,
                        session: str = "default", return_state: bool = True) -> str:
    """Upload one or more local files to a file input or trigger a file chooser dialog.
    Do NOT call browser_click beforehand — call browser_upload_file directly with the target element.
    """
    _reset_poll_tracker(session)
    try:
        return browser_engine.run(browser_engine.upload_file(file_path, index, selector, session, return_state=return_state))
    except Exception as e:
        log.error("browser_upload_file error: %s", e)
        return f"Error uploading file: {_format_error(e)}"


def browser_back(session: str = "default", return_state: bool = True) -> str:
    """Navigate back to the previous page in history."""
    _reset_poll_tracker(session)
    try:
        return browser_engine.run(browser_engine.go_back(session, return_state=return_state))
    except Exception as e:
        log.error("browser_back error: %s", e)
        return f"Error navigating back: {_format_error(e)}"


def browser_reload(session: str = "default", return_state: bool = True) -> str:
    """Reload the current page."""
    _reset_poll_tracker(session)
    try:
        return browser_engine.run(browser_engine.reload(session, return_state=return_state))
    except Exception as e:
        log.error("browser_reload error: %s", e)
        return f"Error reloading page: {_format_error(e)}"


_MUTATIVE_JS_PATTERN = re.compile(
    r"\.(scrollBy|scrollTo|scrollIntoView|click|focus|blur|remove|submit|dispatchEvent)\s*\(|"
    r"(?:^|[^\w=<>!])[\w$.\[\]\'\"]+\s*=(?!=)",
    re.IGNORECASE
)


def _is_unproductive_probe(res: str) -> bool:
    """Classify whether a browser_eval_js outcome represents an unproductive probe."""
    from src.agent.semantic_loop_guard import SemanticLoopGuard
    return SemanticLoopGuard._is_unproductive_eval_result(res)


def browser_eval_js(expression: str, session: str = "default",
                    return_state: bool = False, timeout: float = 15.0) -> str:
    """Evaluate a JavaScript expression in the page context and return the result."""
    with _SESSION_EVAL_LOCK:
        tracker = _SESSION_EVAL_TRACKER.setdefault(session, {
            "count": 0,
            "last_expr": None,
            "last_result": None,
            "consecutive_void_probes": 0,
        })

    try:
        res = browser_engine.run(browser_engine.eval_js(
            expression, session=session, timeout=timeout, return_state=return_state
        ))
    except Exception as e:
        log.error("browser_eval_js error: %s", e)
        return f"Error evaluating JS: {_format_error(e)}"

    norm_res = str(res).strip()
    norm_expr = (expression or "").strip()

    is_mutative = bool(_MUTATIVE_JS_PATTERN.search(norm_expr))
    is_void = norm_res in ("JS result: null", "JS result: undefined", 'JS result: "__ARGENT_UNDEFINED__"')

    # Mutative actions (scrolling, clicking, removals) that return void/null should not be treated as blind read polling
    if is_mutative and is_void:
        with _SESSION_EVAL_LOCK:
            tracker["consecutive_void_probes"] = 0
            tracker["last_expr"] = norm_expr
            tracker["last_result"] = norm_res
            tracker["count"] = 1
        return res

    is_unproductive = _is_unproductive_probe(norm_res)

    # Track repetition and void probing under lock
    with _SESSION_EVAL_LOCK:
        if tracker.get("last_expr") == norm_expr and tracker.get("last_result") == norm_res:
            tracker["count"] += 1
        else:
            tracker["last_expr"] = norm_expr
            tracker["last_result"] = norm_res
            tracker["count"] = 1
        count = tracker["count"]

        if is_unproductive and not is_mutative:
            tracker["consecutive_void_probes"] = tracker.get("consecutive_void_probes", 0) + 1
        else:
            tracker["consecutive_void_probes"] = 0
        void_probes = tracker["consecutive_void_probes"]

    # If the expression resulted in an error, check repeated errors first
    is_error = norm_res.startswith("Error evaluating JS:") or norm_res.startswith("Error:")
    if is_error:
        if count >= 3:
            return (
                f"[REPEATED JS ERROR]: This JavaScript expression has failed {count} times with the identical error.\n"
                f"{norm_res}\n\n"
                "Fix your script syntax, inspect element attributes via browser_inspect(mode='dom_interactive'), or verify DOM selectors instead of repeating the same failing code."
            )
        if void_probes >= 6:
            return (
                f"[PROBING HALTED]: You have executed {void_probes} consecutive exploratory JavaScript evaluations "
                f"that failed or returned unproductive probe results.\n"
                f"Latest error: {norm_res[:150]}\n\n"
                "Do NOT attempt to reverse-engineer private React/Redux internals, closures, or minified SPA stores via JavaScript.\n"
                "Switch immediately to user-level UI interaction: inspect visible elements using browser_inspect(mode=\"dom_interactive\"), "
                "or click / fill inputs directly using browser_action(action=\"click\") and browser_action(action=\"type\")."
            )
        if void_probes >= 4:
            hint = (
                f"[PROBING HINT]: {void_probes} consecutive JavaScript evaluations returned null/errors/ping scalars. "
                "Stop probing internal JavaScript variables or hidden stores! "
                "Interact with the web page through standard UI elements (browser_inspect, browser_action click / type).\n\n"
            )
            return hint + str(res)
        return res

    # Exact expression repetition halts
    if count >= 5:
        return (
            f"[POLLING HALTED]: This exact JavaScript expression has been evaluated {count} times "
            f"with the IDENTICAL result: {norm_res[:150]}.\n"
            "Blind polling has been halted to prevent infinite loops and token waste. The page is not changing automatically.\n"
            "You MUST change your approach: verify required form fields or validation errors via browser_inspect(mode=\"dom_interactive\"), "
            "click the necessary prerequisite buttons/tabs, or use browser_action(action=\"wait_for\", text=\"...\", amount=15) "
            "to wait for the page asynchronously."
        )
    if count >= 3:
        hint = (
            f"[POLLING HINT]: You have evaluated this JavaScript expression {count} times with the IDENTICAL result. "
            "Stop blind polling in an LLM loop! The page state is static. "
            "If waiting for asynchronous rendering or button activation, use browser_action(action=\"wait_for\", text=\"...\", amount=15) "
            "or check if earlier user actions (filling required inputs, saving, or checking agreements) are missing.\n\n"
        )
        return hint + str(res)

    # Consecutive void/echo probing across varying expressions
    if void_probes >= 6:
        return (
            f"[PROBING HALTED]: You have executed {void_probes} consecutive exploratory JavaScript evaluations "
            f"that returned null, undefined, errors, or dummy ping results without interacting with the user interface.\n"
            "Do NOT attempt to reverse-engineer private React/Redux internals, closures, or minified SPA stores via JavaScript. "
            "The application state is protected and not accessible this way.\n"
            "Switch immediately to user-level UI interaction: inspect visible elements using browser_inspect(mode=\"dom_interactive\"), "
            "or click / fill inputs directly using browser_action(action=\"click\") and browser_action(action=\"type\")."
        )
    if void_probes >= 4:
        hint = (
            f"[PROBING HINT]: {void_probes} consecutive JavaScript evaluations returned null/errors/ping scalars. "
            "Stop probing internal JavaScript variables or hidden stores! "
            "Interact with the web page through standard UI elements (browser_inspect, browser_action click / type).\n\n"
        )
        return hint + str(res)

    return res


# ---------------------------------------------------------------------------
# Consolidated Browser Action & Inspect Facades
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stagehand v4 Semantic AI Browser Automation (Side-by-side with Playwright)
# ---------------------------------------------------------------------------
from tools.stagehand_tools import (
    browser_act,
    browser_extract,
    browser_observe,
    reset_stagehand,
)


def browser_action(
    action: str,
    selector: str = None,
    text: str = None,
    value: str = None,
    key: str = None,
    direction: str = "down",
    amount: int = 500,
    index: int = None,
    instruction: str = None,
    session: str = "default",
    return_state: bool = True,
) -> str:
    """Execute an interactive action on the active browser page.
    
    Actions:
    - 'click': Click an element identified by selector or text.
    - 'type' / 'input': Type text into an input field or element.
    - 'select': Select an option from a dropdown.
    - 'scroll': Scroll the page in a direction ('down', 'up') or by amount.
    - 'press_key' / 'press': Press a keyboard key (Enter, Escape, Tab, etc.).
    - 'upload_file': Upload a local file to a file input.
    - 'wait_for' / 'wait': Wait for an element or text to appear.
    - 'back': Navigate back in history.
    - 'reload': Refresh the current page.
    - 'switch_tab': Switch active tab by index.
    - 'eval_js': Execute JavaScript in the page context.
    - 'act': Semantic action via natural language instruction (Stagehand).
    """
    act = (action or "").strip().lower()
    
    try:
        active_sessions = getattr(browser_engine, "_sessions", {})
        if not active_sessions and act not in ("open",):
            return "Error: No active browser session. Call browser_open(url) first to open a page."
    except Exception:
        pass

    target_text = text if text is not None else (value if value is not None else instruction)
    
    if act == "click":
        return browser_click(index=index, selector=selector, text=target_text, session=session, return_state=return_state)
    elif act in ("type", "input"):
        return browser_input(text=target_text or "", index=index, selector=selector, session=session, return_state=return_state)
    elif act == "select":
        return browser_select(value=target_text or "", index=index, selector=selector, session=session, return_state=return_state)
    elif act == "scroll":
        scroll_dir = direction or target_text or "down"
        return browser_scroll(direction=scroll_dir, amount=amount or 500, index=index, selector=selector, session=session, return_state=return_state)
    elif act in ("press_key", "press"):
        key_name = key or target_text or "Enter"
        return browser_press_key(key=key_name, session=session, return_state=return_state)
    elif act in ("upload_file", "upload"):
        return browser_upload_file(file_path=target_text or "", index=index, selector=selector, session=session, return_state=return_state)
    elif act in ("wait_for", "wait"):
        return browser_wait_for(text=target_text, selector=selector, timeout=amount if isinstance(amount, (int, float)) else 5.0, session=session, return_state=return_state)
    elif act == "back":
        return browser_back(session=session)
    elif act == "reload":
        return browser_reload(session=session)
    elif act in ("switch_tab", "switch"):
        try:
            tab_idx = int(target_text or index or 1)
        except (ValueError, TypeError):
            tab_idx = 1
        return browser_switch_tab(index=tab_idx, session=session)
    elif act in ("close_tab", "close"):
        target_idx = index
        if target_idx is None and target_text and str(target_text).strip().isdigit():
            try:
                target_idx = int(str(target_text).strip())
            except (ValueError, TypeError):
                pass
        filter_text = None if target_idx is not None else target_text
        return browser_close_tab(index=target_idx, text=filter_text, session=session)
    elif act in ("eval_js", "eval"):
        return browser_eval_js(expression=target_text or "", session=session)
    elif act in ("act", "stagehand_act", "ai_act"):
        instr = instruction or target_text or ""
        return browser_act(instruction=instr, session=session)
    else:
        return f"Error: Unknown browser action '{action}'. Valid actions: click, type, select, scroll, press_key, upload_file, wait_for, back, reload, switch_tab, close_tab, eval_js, act."


def browser_inspect(
    mode: str = "dom_interactive",
    selector: str = None,
    index: int = None,
    path: str = None,
    instruction: str = None,
    query: str = None,
    session: str = "default",
) -> str:
    """Inspect the current browser page state, console logs, or take a screenshot.
    
    Modes:
    - 'dom_interactive' / 'dom': Get compact accessibility tree of interactive elements (buttons, inputs, links).
    - 'tabs' / 'list_tabs': List all open browser tabs with their indices, titles, and URLs.
    - 'screenshot': Capture visual screenshot of the page and save to disk / attach vision.
    - 'console_errors' / 'console': Get JavaScript errors and warnings from the browser DevTools console.
    - 'text_content' / 'text': Extract readable text content from the page or selector.
    - 'html': Extract HTML markup.
    - 'observe': Semantic element discovery via Stagehand.
    - 'extract': Semantic data extraction via Stagehand.
    """
    m = (mode or "dom_interactive").strip().lower()
    
    try:
        active_sessions = getattr(browser_engine, "_sessions", {})
        if not active_sessions:
            return "Error: No active browser session. Call browser_open(url) first to open a page."
    except Exception:
        pass

    if m in ("screenshot", "image"):
        return browser_screenshot(path=path, session=session)
    elif m in ("tabs", "list_tabs", "open_tabs"):
        return browser_list_tabs(session=session)
    elif m in ("dom_interactive", "dom", "interactive", "state"):
        filter_query = query if query is not None else instruction
        state_str = browser_state(session=session, query=filter_query)
        if len(state_str) > 15000:
            return state_str[:15000] + "\n\n... [State truncated to 15KB to preserve context. Use selector/index to inspect specific elements]"
        return state_str
    elif m in ("console_errors", "console", "logs"):
        state_str = browser_state(session=session)
        lines = state_str.splitlines()
        console_lines = [l for l in lines if "[Console" in l or "Error:" in l or "Warning:" in l]
        if console_lines:
            return "Console Messages / Errors:\n" + "\n".join(console_lines)
        return "No console errors detected on the page."
    elif m in ("text_content", "text"):
        return browser_get_content(content_type="text", selector=selector, index=index, session=session)
    elif m in ("html",):
        return browser_get_content(content_type="html", selector=selector, index=index, session=session)
    elif m in ("markdown",):
        return browser_get_content(content_type="markdown", selector=selector, index=index, session=session)
    elif m in ("observe", "stagehand_observe", "ai_observe"):
        instr = instruction or query or selector or ""
        return browser_observe(instruction=instr, session=session)
    elif m in ("extract", "stagehand_extract", "ai_extract"):
        instr = instruction or query or selector or ""
        return browser_extract(instruction=instr, session=session)
    else:
        return f"Error: Unknown inspect mode '{mode}'. Valid modes: dom_interactive, tabs, screenshot, console_errors, text_content, html, observe, extract."


