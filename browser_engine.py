"""
browser_engine.py — Playwright-based browser automation engine for Argent.

Architecture inspired by BrowserAct's Observe→Think→Act pattern:
  1. get_state() returns numbered interactive elements for LLM consumption
  2. LLM decides which element to interact with
  3. Action functions use element indices from get_state()
  4. LLM verifies result via get_state() again

The engine is a lazy-init singleton: browser launches on first use and
persists across tool calls within a session.
"""

import asyncio
import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from logger import get_logger

log = get_logger("browser")

# ---------------------------------------------------------------------------
# Minimal HTML-to-Markdown converter (no extra dependency)
# ---------------------------------------------------------------------------

def _html_to_markdown(html: str) -> str:
    """Convert HTML to readable Markdown using BeautifulSoup.
    Strips scripts/styles and converts common tags to Markdown equivalents."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "noscript", "svg", "path"]):
        tag.decompose()

    # Convert headers
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            h.replace_with(f"\n{'#' * level} {h.get_text(strip=True)}\n")

    # Convert links
    for a in soup.find_all("a"):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if href and text:
            a.replace_with(f"[{text}]({href})")
        elif text:
            a.replace_with(text)

    # Convert bold/strong
    for tag in soup.find_all(["strong", "b"]):
        tag.replace_with(f"**{tag.get_text(strip=True)}**")

    # Convert italic/em
    for tag in soup.find_all(["em", "i"]):
        tag.replace_with(f"*{tag.get_text(strip=True)}*")

    # Convert code blocks
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        lang = ""
        if code and code.get("class"):
            classes = code.get("class", [])
            for cls in classes:
                if cls.startswith("language-"):
                    lang = cls[9:]
                    break
        text = pre.get_text()
        pre.replace_with(f"\n```{lang}\n{text}\n```\n")

    # Convert inline code
    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text()}`")

    # Convert images
    for img in soup.find_all("img"):
        alt = img.get("alt", "image")
        src = img.get("src", "")
        img.replace_with(f"![{alt}]({src})")

    # Convert list items
    for li in soup.find_all("li"):
        li.replace_with(f"\n- {li.get_text(strip=True)}")

    # Convert paragraphs
    for p in soup.find_all("p"):
        p.replace_with(f"\n{p.get_text(strip=True)}\n")

    # Convert line breaks
    for br in soup.find_all("br"):
        br.replace_with("\n")

    text = soup.get_text()
    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Interactive Elements State Extractor (runs in browser via JS)
# ---------------------------------------------------------------------------

# This JavaScript runs inside the browser page to extract all interactive
# elements and assign them index numbers. It returns a JSON array that
# get_state() formats into the numbered list the LLM consumes.
_STATE_EXTRACTION_JS = """
() => {
    // ---------------------------------------------------------------
    // Shadow DOM-aware interactive element extractor for Argent.
    // Modern sites (YouTube, GitHub, etc.) use Web Components with
    // Shadow DOM. Standard TreeWalker does NOT cross shadow boundaries,
    // so we recursively walk both light DOM and all shadow roots.
    // ---------------------------------------------------------------

    // Clean up previous markers (including inside shadow roots)
    function removeMarkers(root) {
        try {
            root.querySelectorAll('[data-argent-idx]').forEach(el => {
                el.removeAttribute('data-argent-idx');
            });
            root.querySelectorAll('*').forEach(el => {
                if (el.shadowRoot) removeMarkers(el.shadowRoot);
            });
        } catch(e) {}
    }
    removeMarkers(document);

    const interactiveTags = new Set([
        'A', 'BUTTON', 'INPUT', 'TEXTAREA', 'SELECT', 'DETAILS', 'SUMMARY'
    ]);

    const interactiveRoles = new Set([
        'button', 'link', 'textbox', 'checkbox', 'radio', 'combobox',
        'listbox', 'menuitem', 'option', 'searchbox', 'switch', 'tab',
        'slider', 'spinbutton', 'menuitemcheckbox', 'menuitemradio'
    ]);

    const elements = [];
    let idx = 1;
    const MAX_ELEMENTS = 600;

    function isVisible(node) {
        try {
            const style = window.getComputedStyle(node);
            if (style.display === 'none' || style.visibility === 'hidden' ||
                style.opacity === '0') {
                return false;
            }
            // offsetParent is null for hidden elements, but also for
            // position:fixed and elements inside shadow DOM, so we
            // only use it as a hint combined with dimensions
            const rect = node.getBoundingClientRect();
            if (rect.width === 0 && rect.height === 0) {
                return false;
            }
            return true;
        } catch(e) {
            return false;
        }
    }

    function processNode(node) {
        if (idx > MAX_ELEMENTS) return;
        if (!node.tagName) return;
        if (!isVisible(node)) return;

        const tag = node.tagName;
        const role = node.getAttribute && node.getAttribute('role');
        const isClickable = node.onclick != null ||
                           (node.getAttribute && node.getAttribute('onclick'));
        const tabindex = node.getAttribute && node.getAttribute('tabindex');
        const contentEditable = node.isContentEditable &&
                               node.getAttribute('contenteditable') !== 'false';
        const isInteractive = interactiveTags.has(tag) ||
                              interactiveRoles.has(role) ||
                              isClickable ||
                              contentEditable ||
                              (tabindex !== null && tabindex !== '-1');

        if (isInteractive) {
            let descriptor = tag.toLowerCase();
            const type = node.getAttribute('type');
            if (type) descriptor += `[type=${type}]`;

            // Gather label text
            let label = '';
            const ariaLabel = node.getAttribute('aria-label');
            if (ariaLabel) {
                label = ariaLabel;
            } else if (node.placeholder) {
                label = node.placeholder;
            } else if (node.title) {
                label = node.title;
            } else {
                const text = (node.innerText || node.textContent || '').trim();
                label = text.substring(0, 80);
            }

            // Current value for inputs
            let value = '';
            if (tag === 'INPUT' || tag === 'TEXTAREA') {
                value = (node.value || '').substring(0, 50);
            } else if (tag === 'SELECT') {
                const selected = node.options && node.options[node.selectedIndex];
                value = selected ? selected.text : '';
            }

            // Checked state
            let checked = null;
            if (type === 'checkbox' || type === 'radio') {
                checked = node.checked;
            }

            // Mark element for later retrieval
            node.setAttribute('data-argent-idx', idx);

            elements.push({
                idx: idx,
                tag: descriptor,
                label: label,
                value: value,
                checked: checked,
                href: tag === 'A' ? (node.href || '') : '',
            });

            idx++;
        }
    }

    // Recursive DOM walker that crosses Shadow DOM boundaries
    function walkDOM(root) {
        if (idx > MAX_ELEMENTS) return;
        
        // Get all elements in this root (light DOM or shadow root)
        let allElements;
        try {
            allElements = root.querySelectorAll('*');
        } catch(e) {
            return;
        }

        for (const el of allElements) {
            if (idx > MAX_ELEMENTS) break;
            processNode(el);

            // If this element has a shadow root, recurse into it
            if (el.shadowRoot) {
                walkDOM(el.shadowRoot);
            }
        }
    }

    walkDOM(document);

    return {
        url: window.location.href,
        title: document.title,
        elements: elements
    };
}
"""


# ---------------------------------------------------------------------------
# Browser Engine
# ---------------------------------------------------------------------------

class _SessionContext:
    """Holds a browser context + the active page for one named session."""
    __slots__ = ("context", "page")

    def __init__(self, context, page):
        self.context = context
        self.page = page


class BrowserEngine:
    """
    Singleton Playwright browser manager.
    
    Provides high-level methods that Argent's tool functions call.
    All public methods are async — callers use engine.run() to bridge sync/async.
    
    Usage pattern in tools:
        result = browser_engine.run(browser_engine.navigate("https://example.com"))
    """

    def __init__(self):
        self._playwright = None       # Playwright instance
        self._browser = None          # Shared Chromium browser instance
        self._sessions: Dict[str, _SessionContext] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._headed = False          # Current headed mode state
        self._stealth_applied = False

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def _ensure_browser(self, headed: bool = False) -> None:
        """Lazy-initialize Playwright and launch browser if not running."""
        # If the browser exists but headed mode changed, restart
        if self._browser and self._headed != headed:
            await self.shutdown()

        if self._browser:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            )

        self._playwright = await async_playwright().start()

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
        ]

        self._browser = await self._playwright.chromium.launch(
            headless=not headed,
            args=launch_args,
        )
        self._headed = headed
        log.info("Browser launched (headed=%s)", headed)

    async def shutdown(self) -> None:
        """Close all sessions and the browser."""
        for name in list(self._sessions.keys()):
            try:
                ctx = self._sessions.pop(name)
                await ctx.context.close()
            except Exception:
                pass

        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

        self._stealth_applied = False
        log.info("Browser engine shut down")

    def run(self, coro, timeout: int = 120):
        """Run an async coroutine from synchronous code.
        Uses a persistent background event loop so that Playwright connections
        (which are bound to a specific loop) survive between calls.

        Args:
            coro: The async coroutine to execute.
            timeout: Max seconds to wait for the result (default 120).
        """
        # Ensure a persistent loop exists in a background daemon thread.
        # This is critical: asyncio.run() would create+destroy a loop each time,
        # killing Playwright's internal connection channel between calls.
        if self._loop is None or self._loop.is_closed():
            import threading
            self._loop = asyncio.new_event_loop()
            t = threading.Thread(target=self._loop.run_forever, daemon=True)
            t.start()

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"Browser operation timed out after {timeout}s"
            )

    # -----------------------------------------------------------------------
    # Session Management
    # -----------------------------------------------------------------------

    async def _get_session(self, session: str = "default") -> _SessionContext:
        """Get or create a named session (browser context + page)."""
        if session in self._sessions:
            sc = self._sessions[session]
            if sc.page.is_closed():
                try:
                    sc.page = await sc.context.new_page()
                    await self._apply_stealth(sc.page)
                except Exception:
                    log.warning("Session '%s' context is dead, recreating.", session)
                    del self._sessions[session]
                    return await self._create_session(session)
            return sc

        raise KeyError(f"Session '{session}' does not exist. Use browser_open first.")

    async def _create_session(self, session: str, headed: bool = False) -> _SessionContext:
        """Create a new named browser session."""
        await self._ensure_browser(headed)

        # Create isolated browser context (separate cookies, cache, etc.)
        context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
            # Disable webdriver detection flag
            java_script_enabled=True,
        )

        page = await context.new_page()
        await self._apply_stealth(page)

        sc = _SessionContext(context=context, page=page)
        self._sessions[session] = sc
        log.info("Session '%s' created", session)
        return sc

    async def _apply_stealth(self, page) -> None:
        """Apply anti-detection patches to a page.
        Uses playwright-stealth if available, otherwise applies manual patches."""
        # Try playwright-stealth library first
        try:
            from playwright_stealth import stealth_async
            await stealth_async(page)
            return
        except ImportError:
            pass

        # Manual stealth patches (fallback)
        await page.add_init_script("""
            // Override navigator.webdriver
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            
            // Override chrome runtime  
            window.chrome = { runtime: {} };
            
            // Override permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
            );
            
            // Override plugins length
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            // Override languages  
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
        """)

    # -----------------------------------------------------------------------
    # Public API: Navigation
    # -----------------------------------------------------------------------

    async def open_page(self, url: str, session: str = "default",
                        headed: bool = False) -> str:
        """Open a URL in a session. Creates the session if it doesn't exist."""
        if session in self._sessions:
            sc = self._sessions[session]
            # Re-check if headed mode mismatch requires restart
            if headed != self._headed:
                await self.shutdown()
                sc = await self._create_session(session, headed)
        else:
            sc = await self._create_session(session, headed)

        try:
            await sc.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Wait a bit for JS to render
            await sc.page.wait_for_timeout(1000)
        except Exception as e:
            return f"Error navigating to {url}: {e}"

        title = await sc.page.title()
        current_url = sc.page.url
        return f"Opened: {current_url}\nTitle: {title}"

    async def navigate(self, url: str, session: str = "default") -> str:
        """Navigate the current session to a new URL."""
        sc = await self._get_session(session)
        try:
            await sc.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await sc.page.wait_for_timeout(1000)
        except Exception as e:
            return f"Error navigating to {url}: {e}"

        title = await sc.page.title()
        return f"Navigated to: {sc.page.url}\nTitle: {title}"

    async def go_back(self, session: str = "default") -> str:
        """Go back in browser history."""
        sc = await self._get_session(session)
        await sc.page.go_back(wait_until="domcontentloaded", timeout=15000)
        title = await sc.page.title()
        return f"Went back to: {sc.page.url}\nTitle: {title}"

    async def reload(self, session: str = "default") -> str:
        """Reload the current page."""
        sc = await self._get_session(session)
        await sc.page.reload(wait_until="domcontentloaded", timeout=15000)
        title = await sc.page.title()
        return f"Reloaded: {sc.page.url}\nTitle: {title}"

    # -----------------------------------------------------------------------
    # Public API: State Extraction (the key BrowserAct-like feature)
    # -----------------------------------------------------------------------

    async def get_state(self, session: str = "default") -> str:
        """
        Extract interactive elements from the page with numbered indices.
        
        Returns a text representation optimized for LLM consumption:
            Page: https://example.com | Title: Example
            ---
            [1] input[type=email] placeholder="Email"
            [2] input[type=password] placeholder="Password"
            [3] button "Sign In"
        """
        sc = await self._get_session(session)

        try:
            result = await sc.page.evaluate(_STATE_EXTRACTION_JS)
        except Exception as e:
            return f"Error extracting state: {e}"

        url = result.get("url", "?")
        title = result.get("title", "?")
        elements = result.get("elements", [])

        lines = [f"Page: {url} | Title: {title}", "---"]

        if not elements:
            lines.append("(No interactive elements found on this page)")
        else:
            for el in elements:
                parts = [f"[{el['idx']}]", el["tag"]]

                if el.get("label"):
                    # Truncate long labels
                    label = el["label"][:60]
                    parts.append(f'"{label}"')

                if el.get("value"):
                    parts.append(f'value="{el["value"]}"')

                if el.get("checked") is not None:
                    parts.append(f'checked={el["checked"]}')

                if el.get("href"):
                    # Show shortened href
                    href = el["href"]
                    if len(href) > 60:
                        href = href[:57] + "..."
                    parts.append(f"-> {href}")

                lines.append(" ".join(parts))

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Public API: Interaction
    # -----------------------------------------------------------------------

    async def click(self, index: int, session: str = "default") -> str:
        """Click element by its state index."""
        sc = await self._get_session(session)
        try:
            el = sc.page.locator(f'[data-argent-idx="{index}"]')
            count = await el.count()
            if count == 0:
                return (
                    f"Error: Element [{index}] not found. "
                    "The page may have changed — call browser_state to get fresh indices."
                )
            await el.first.click(timeout=10000)
            await sc.page.wait_for_timeout(800)
            return f"Clicked element [{index}]. Call browser_state to see the updated page."
        except Exception as e:
            return f"Error clicking element [{index}]: {e}"

    async def fill_input(self, index: int, text: str,
                         session: str = "default") -> str:
        """Clear and fill text into an input element by its state index."""
        sc = await self._get_session(session)
        try:
            el = sc.page.locator(f'[data-argent-idx="{index}"]')
            count = await el.count()
            if count == 0:
                return (
                    f"Error: Element [{index}] not found. "
                    "Call browser_state to get fresh indices."
                )
            await el.first.click(timeout=5000)
            await el.first.fill(text, timeout=5000)
            return f"Filled element [{index}] with text."
        except Exception as e:
            return f"Error filling element [{index}]: {e}"

    async def scroll(self, direction: str = "down", amount: int = 500,
                     session: str = "default") -> str:
        """Scroll the page up or down using JavaScript.
        Uses window.scrollBy which reliably triggers lazy-loading on SPAs
        like YouTube, unlike mouse.wheel which can miss scroll listeners."""
        sc = await self._get_session(session)
        delta = amount if direction.lower() == "down" else -amount
        # JS scroll triggers IntersectionObserver and scroll event listeners
        await sc.page.evaluate(f"window.scrollBy(0, {delta})")
        # Wait for lazy-loaded content (SPA sites need this)
        await sc.page.wait_for_timeout(1500)
        # Report actual scroll position for context
        scroll_y = await sc.page.evaluate("window.scrollY")
        page_height = await sc.page.evaluate(
            "Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)"
        )
        return f"Scrolled {direction} by {amount}px. Position: {scroll_y}/{page_height}px."

    async def press_key(self, key: str, session: str = "default") -> str:
        """Press a keyboard key (Enter, Escape, Tab, etc.)."""
        sc = await self._get_session(session)
        await sc.page.keyboard.press(key)
        await sc.page.wait_for_timeout(500)
        return f"Pressed key: {key}"

    # -----------------------------------------------------------------------
    # Public API: Data Extraction
    # -----------------------------------------------------------------------

    async def screenshot(self, path: str = None, full_page: bool = False,
                         session: str = "default") -> str:
        """Take a screenshot and save to disk."""
        sc = await self._get_session(session)

        if not path:
            from config import get_visuals_dir
            visuals_dir = Path(get_visuals_dir()).expanduser().resolve()
            visuals_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(visuals_dir / f"browser_{timestamp}.png")

        try:
            await sc.page.screenshot(path=path, full_page=full_page)
            return f"Screenshot saved to: {path}"
        except Exception as e:
            return f"Error taking screenshot: {e}"

    async def get_text(self, index: int = None, selector: str = None,
                       session: str = "default") -> str:
        """Get text content of a specific element, CSS selector, or the full page.
        
        Priority: index > selector > full page.
        selector allows targeting specific page sections, e.g.:
          - '#comments' for YouTube comments
          - '.job-list' for vacancy listings
          - 'main' for main content area
        """
        sc = await self._get_session(session)

        if index is not None:
            try:
                el = sc.page.locator(f'[data-argent-idx="{index}"]')
                count = await el.count()
                if count == 0:
                    return f"Error: Element [{index}] not found."
                text = await el.first.inner_text()
                return f"Text of element [{index}]:\n{text}"
            except Exception as e:
                return f"Error getting text of element [{index}]: {e}"
        elif selector is not None:
            try:
                el = sc.page.locator(selector)
                count = await el.count()
                if count == 0:
                    return f"Error: No elements found for selector '{selector}'. Try browser_get_content with content_type='text' to see the full page."
                text = await el.first.inner_text()
                if len(text) > 15000:
                    text = text[:15000] + "\n... [Content Truncated]"
                return f"Content of '{selector}':\n{text}"
            except Exception as e:
                return f"Error getting text for selector '{selector}': {e}"
        else:
            try:
                text = await sc.page.inner_text("body")
                # Truncate if too long
                if len(text) > 15000:
                    text = text[:15000] + "\n... [Content Truncated]"
                return text
            except Exception as e:
                return f"Error getting page text: {e}"

    async def get_markdown(self, session: str = "default") -> str:
        """Get page content converted to Markdown."""
        sc = await self._get_session(session)
        try:
            html = await sc.page.content()
            md = _html_to_markdown(html)
            if len(md) > 15000:
                md = md[:15000] + "\n... [Content Truncated]"
            return md
        except Exception as e:
            return f"Error converting page to markdown: {e}"

    async def get_html(self, session: str = "default") -> str:
        """Get raw HTML of the page."""
        sc = await self._get_session(session)
        try:
            html = await sc.page.content()
            if len(html) > 20000:
                html = html[:20000] + "\n... [Content Truncated]"
            return html
        except Exception as e:
            return f"Error getting page HTML: {e}"

    async def eval_js(self, expression: str,
                      session: str = "default") -> str:
        """Evaluate a JavaScript expression in the page context."""
        sc = await self._get_session(session)
        try:
            result = await sc.page.evaluate(expression)
            return f"JS result: {json.dumps(result, ensure_ascii=False, default=str)}"
        except Exception as e:
            return f"Error evaluating JS: {e}"

    # -----------------------------------------------------------------------
    # Public API: Wait
    # -----------------------------------------------------------------------

    async def wait_stable(self, timeout: int = 30000,
                          session: str = "default") -> str:
        """Wait for the page to be fully loaded and network idle."""
        sc = await self._get_session(session)
        try:
            await sc.page.wait_for_load_state("networkidle", timeout=timeout)
            return "Page is stable (network idle)."
        except Exception:
            return "Timeout waiting for page stability. The page may still be loading."

    # -----------------------------------------------------------------------
    # Public API: Session Cleanup
    # -----------------------------------------------------------------------

    async def close_session(self, session: str = "default") -> str:
        """Close a named browser session."""
        if session not in self._sessions:
            return f"Session '{session}' is not active."

        sc = self._sessions.pop(session)
        try:
            await sc.context.close()
        except Exception:
            pass

        # If no sessions left, shut down the browser entirely
        if not self._sessions:
            await self.shutdown()

        return f"Session '{session}' closed."

    async def list_sessions(self) -> str:
        """List active browser sessions."""
        if not self._sessions:
            return "No active browser sessions."
        lines = ["Active browser sessions:"]
        for name, sc in self._sessions.items():
            try:
                url = sc.page.url
                title = await sc.page.title()
            except Exception:
                url = "?"
                title = "?"
            lines.append(f"  - {name}: {url} ({title})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Global singleton instance
# ---------------------------------------------------------------------------

browser_engine = BrowserEngine()
