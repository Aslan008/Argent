from logger import get_logger

log = get_logger("tools")

from browser_engine import browser_engine

def browser_open(url: str, session: str = "default", headed: bool = False) -> str:
    """Open a URL in the browser. Creates a new session if needed."""
    try:
        return browser_engine.run(browser_engine.open_page(url, session, headed))
    except Exception as e:
        log.error("browser_open error: %s", e)
        return f"Error opening browser: {e}"

def browser_state(session: str = "default", query: str = None, scroll_depth: int = 0) -> str:
    """Get the current page state: numbered list of interactive elements, optionally filtered by query."""
    try:
        return browser_engine.run(browser_engine.get_state(session, query, scroll_depth))
    except KeyError as e:
        return str(e)
    except Exception as e:
        log.error("browser_state error: %s", e)
        return f"Error getting browser state: {e}"

def browser_click(index: int = None, selector: str = None, text: str = None, session: str = "default") -> str:
    """Click an element by its index, CSS selector, or visible text."""
    try:
        return browser_engine.run(browser_engine.click(index, selector, text, session))
    except KeyError as e:
        return str(e)
    except Exception as e:
        log.error("browser_click error: %s", e)
        return f"Error clicking element: {e}"

def browser_input(index: int = None, text: str = "", selector: str = None, session: str = "default") -> str:
    """Type text into an input element by its index or CSS selector."""
    try:
        return browser_engine.run(browser_engine.fill_input(index, text, selector, session))
    except KeyError as e:
        return str(e)
    except Exception as e:
        log.error("browser_input error: %s", e)
        return f"Error filling input: {e}"

def browser_screenshot(path: str = None, full_page: bool = False, session: str = "default") -> str:
    """Take a screenshot of the current page."""
    try:
        return browser_engine.run(browser_engine.screenshot(path, full_page, session))
    except KeyError as e:
        return str(e)
    except Exception as e:
        log.error("browser_screenshot error: %s", e)
        return f"Error taking screenshot: {e}"

def browser_scroll(direction: str = "down", amount: int = 500, index: int = None, session: str = "default") -> str:
    """Scroll the page or scroll a specific element into view."""
    try:
        return browser_engine.run(browser_engine.scroll(direction, amount, index, session))
    except KeyError as e:
        return str(e)
    except Exception as e:
        log.error("browser_scroll error: %s", e)
        return f"Error scrolling: {e}"

def browser_get_content(content_type: str = "text", index: int = None, selector: str = None, session: str = "default") -> str:
    """Extract content from the page. content_type: 'text', 'markdown', or 'html'. Use selector for CSS targeting."""
    try:
        if content_type == "markdown":
            return browser_engine.run(browser_engine.get_markdown(session))
        elif content_type == "html":
            return browser_engine.run(browser_engine.get_html(session))
        else:
            return browser_engine.run(browser_engine.get_text(index, selector, session))
    except KeyError as e:
        return str(e)
    except Exception as e:
        log.error("browser_get_content error: %s", e)
        return f"Error getting content: {e}"

def browser_close(session: str = "default") -> str:
    """Close a browser session."""
    try:
        return browser_engine.run(browser_engine.close_session(session))
    except Exception as e:
        log.error("browser_close error: %s", e)
        return f"Error closing browser: {e}"

def browser_switch_tab(index: int, session: str = "default") -> str:
    """Switch active tab in the browser context by index."""
    try:
        return browser_engine.run(browser_engine.switch_tab(index, session))
    except KeyError as e:
        return str(e)
    except Exception as e:
        log.error("browser_switch_tab error: %s", e)
        return f"Error switching tab: {e}"
