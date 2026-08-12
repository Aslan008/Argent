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

# Browser navigation is scheme-restricted. A page fetched through the browser
# is rendered AND its text is handed to the model, so `file:///C:/Users/...`
# would turn the browser into a file-exfiltration channel that bypasses every
# filesystem guard in tools/. Same for data:/javascript:/view-source: (script
# execution, embedded payloads) and chrome:// (browser internals).
#
# Note this is deliberately NOT the SSRF guard used by src/research/fetch.py:
# there, a URL comes from arbitrary web content, so private addresses are
# blocked. Here the target is user/task driven and http://localhost is a
# first-class use case — the dev server of the project being built.
_ALLOWED_URL_SCHEMES = {"http", "https"}


def _blocked_url_reason(url: str) -> str | None:
    """Error string when this URL must not be opened, else None."""
    import urllib.parse
    if not isinstance(url, str):
        return f"Error: URL must be a string, got {type(url).__name__}"
    if not url:
        return "Error: no URL provided."
    raw = str(url).strip()
    if raw.lower() in ("about:blank", "about:"):
        return None
    scheme = urllib.parse.urlparse(raw).scheme.lower()
    if not scheme:
        return None                       # bare host: Playwright prepends http
    if scheme in _ALLOWED_URL_SCHEMES:
        return None
    return (f"Error: refusing to open a '{scheme}:' URL in the browser — only http/https "
            f"are allowed. To read a local file use read_file, not the browser.")


def _html_to_markdown(html: str) -> str:
    """Convert HTML to readable Markdown using BeautifulSoup.
    Strips scripts/styles and converts common tags to Markdown equivalents."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        import re
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.IGNORECASE | re.DOTALL)
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
(args) => {
    const startIdx = args.startIdx || 1;
    const query = args.query || null;
    const MAX_ELEMENTS = 600;

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

    function getParentNode(node) {
        if (node.parentNode) {
            return node.parentNode;
        }
        if (node.parentNode === null && node.host) {
            return node.host;
        }
        const root = node.getRootNode ? node.getRootNode() : null;
        if (root && root.host) {
            return root.host;
        }
        return null;
    }

    function isVisible(node, style) {
        if (!style) return false;
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
            return false;
        }
        const rect = node.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) {
            return false;
        }
        return true;
    }

    function hasInteractiveParent(node, candidatesSet) {
        let parent = getParentNode(node);
        while (parent) {
            if (candidatesSet.has(parent)) {
                const tag = node.tagName;
                if (interactiveTags.has(tag) || (node.getAttribute && interactiveRoles.has(node.getAttribute('role')))) {
                    return false;
                }
                return true;
            }
            parent = getParentNode(parent);
        }
        return false;
    }

    const genericLabels = new Set([
        'слушать', 'play', 'воспроизвести', 'открыть', 'open', 'купить', 'buy', 
        'подробнее', 'more', 'details', 'нажать', 'click', 'show', 'показать',
        'кнопка', 'button', 'delete', 'удалить', 'edit', 'изменить', 'запустить',
        'начать прослушивание', 'воспроизведение', 'play music', 'слушать музыку'
    ]);

    function contextualizeLabel(node, label) {
        const norm = (label || '').toLowerCase().trim();
        
        // Если лейбл пустой или совпадает с общими словами, попробуем поискать контекст
        if (!norm || genericLabels.has(norm)) {
            let parent = getParentNode(node);
            // 1. Ищем первый родительский блок, содержащий заголовок или имя
            for (let i = 0; i < 7 && parent; i++) {
                const heading = parent.querySelector('h1, h2, h3, h4, h5, h6, [class*="title" i], [class*="name" i]');
                if (heading) {
                    const headingText = (heading.innerText || heading.textContent || '').trim();
                    if (headingText && headingText.toLowerCase() !== norm) {
                        return label ? `${label} (${headingText.substring(0, 45)})` : headingText.substring(0, 45);
                    }
                }
                parent = getParentNode(parent);
            }
            
            // 2. Фолбек: ищем первый родительский блок, содержащий сильный текст или обычные строки
            parent = getParentNode(node);
            for (let i = 0; i < 4 && parent; i++) {
                const bold = parent.querySelector('strong, b');
                if (bold) {
                    const boldText = (bold.innerText || bold.textContent || '').trim();
                    if (boldText && boldText.toLowerCase() !== norm) {
                        return label ? `${label} (${boldText.substring(0, 45)})` : boldText.substring(0, 45);
                    }
                }
                const parentText = (parent.innerText || parent.textContent || '').trim();
                const lines = parentText.split('\\n').map(l => l.trim()).filter(l => l.length > 0 && l.toLowerCase() !== norm);
                if (lines.length > 0) {
                    return label ? `${label} (${lines[0].substring(0, 45)})` : lines[0].substring(0, 45);
                }
                parent = getParentNode(parent);
            }
        }
        return label;
    }

    function extractLabel(node) {
        // 1. Атрибут aria-labelledby
        const ariaLabelledBy = node.getAttribute && node.getAttribute('aria-labelledby');
        if (ariaLabelledBy) {
            const labelEl = document.getElementById(ariaLabelledBy.trim());
            if (labelEl) {
                const lblText = labelEl.innerText || labelEl.textContent;
                if (lblText && lblText.trim()) return lblText.trim();
            }
        }

        // 2. Атрибуты самого элемента
        const ariaLabel = node.getAttribute && node.getAttribute('aria-label');
        if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim();
        
        if (node.placeholder && node.placeholder.trim()) return node.placeholder.trim();
        if (node.title && node.title.trim()) return node.title.trim();
        
        // 3. Для полей ввода — тег <label for="id">
        if (node.id) {
            const labelFor = document.querySelector(`label[for="${node.id}"]`);
            if (labelFor) {
                const lblText = labelFor.innerText || labelFor.textContent;
                if (lblText && lblText.trim()) return lblText.trim();
            }
        }

        // 4. Текстовое содержимое самого элемента
        let text = (node.innerText || node.textContent || '').trim();
        if (text) return text.substring(0, 80);

        // 5. Вложенные изображения и SVG
        try {
            // Вложенные img
            const imgs = node.querySelectorAll('img');
            for (const img of imgs) {
                const alt = img.getAttribute('alt');
                if (alt && alt.trim()) return alt.trim();
                const title = img.getAttribute('title');
                if (title && title.trim()) return title.trim();
            }

            // Вложенные SVG
            const svgs = node.querySelectorAll('svg');
            for (const svg of svgs) {
                const svgAria = svg.getAttribute('aria-label');
                if (svgAria && svgAria.trim()) return svgAria.trim();
                
                const titleTag = svg.querySelector('title');
                if (titleTag && titleTag.textContent && titleTag.textContent.trim()) {
                    return titleTag.textContent.trim();
                }
            }
        } catch (e) {}

        // 6. Поиск по родителям (до 3 уровней вверх) для поиска aria-label
        let parent = getParentNode(node);
        for (let i = 0; i < 3 && parent; i++) {
            if (parent.getAttribute) {
                const pAria = parent.getAttribute('aria-label');
                if (pAria && pAria.trim()) return pAria.trim() + " (parent)";
                
                const pTitle = parent.getAttribute('title');
                if (pTitle && pTitle.trim()) return pTitle.trim() + " (parent)";
            }
            parent = getParentNode(parent);
        }

        // 7. Поиск по URL для пустых ссылок
        if (node.tagName === 'A' && node.href) {
            try {
                const url = new URL(node.href);
                if (url.pathname && url.pathname !== '/') {
                    return url.pathname + (url.search ? url.search : '');
                }
            } catch(e) {}
        }

        return '';
    }

    function isInViewport(node) {
        try {
            const rect = node.getBoundingClientRect();
            const windowHeight = (window.innerHeight || document.documentElement.clientHeight);
            const windowWidth = (window.innerWidth || document.documentElement.clientWidth);
            return (
                rect.top < windowHeight &&
                rect.bottom > 0 &&
                rect.left < windowWidth &&
                rect.right > 0
            );
        } catch (e) {
            return false;
        }
    }

    const candidates = [];
    const candidatesSet = new Set();

    function processNode(node) {
        if (!node.tagName) return;
        
        let style = null;
        try {
            style = window.getComputedStyle(node);
        } catch (e) {
            return;
        }

        if (!isVisible(node, style)) return;

        const tag = node.tagName;
        const role = node.getAttribute && node.getAttribute('role');
        const isClickable = node.onclick != null ||
                           (node.getAttribute && node.getAttribute('onclick'));
        const tabindex = node.getAttribute && node.getAttribute('tabindex');
        const contentEditable = node.isContentEditable &&
                               node.getAttribute('contenteditable') !== 'false';
        
        // CSS pointer cursor heuristic
        const isPointerCursor = style && style.cursor === 'pointer';

        const isInteractive = interactiveTags.has(tag) ||
                              interactiveRoles.has(role) ||
                              isClickable ||
                              contentEditable ||
                              isPointerCursor ||
                              (tabindex !== null && tabindex !== '-1');

        if (isInteractive) {
            if (hasInteractiveParent(node, candidatesSet)) {
                return; // Skip duplicates
            }

            candidatesSet.add(node);

            let descriptor = tag.toLowerCase();
            const type = node.getAttribute('type');
            if (type) descriptor += `[type=${type}]`;

            let label = extractLabel(node);
            label = contextualizeLabel(node, label);

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

            candidates.push({
                node: node,
                tag: descriptor,
                label: label,
                value: value,
                checked: checked,
                href: tag === 'A' ? (node.href || '') : '',
                inViewport: isInViewport(node)
            });
        }
    }

    // Recursive DOM walker that crosses Shadow DOM boundaries
    function walkDOM(root) {
        let allElements;
        try {
            allElements = root.querySelectorAll('*');
        } catch(e) {
            return;
        }

        for (const el of allElements) {
            processNode(el);
            if (el.shadowRoot) {
                walkDOM(el.shadowRoot);
            }
        }
    }

    walkDOM(document);

    // Filter by query if present
    let filteredCandidates = candidates;
    let totalFiltered = 0;
    if (query) {
        const keywords = query.split(',').map(s => s.trim().toLowerCase()).filter(s => s.length > 0);
        if (keywords.length > 0) {
            filteredCandidates = candidates.filter(c => {
                const matches = keywords.some(q => {
                    return (c.label || '').toLowerCase().includes(q) ||
                           c.tag.toLowerCase().includes(q) ||
                           (c.value || '').toLowerCase().includes(q) ||
                           (c.href || '').toLowerCase().includes(q);
                });
                if (!matches) totalFiltered++;
                return matches;
            });
        }
    }

    // Prioritize elements in viewport
    const inViewport = [];
    const outOfViewport = [];
    filteredCandidates.forEach(c => {
        if (c.inViewport) {
            inViewport.push(c);
        } else {
            outOfViewport.push(c);
        }
    });

    const finalElements = inViewport.concat(outOfViewport).slice(0, MAX_ELEMENTS);

    // Apply indexes and write data-argent-idx attributes
    const elementsResult = [];
    finalElements.forEach((el, index) => {
        const idxVal = startIdx + index;
        try {
            el.node.setAttribute('data-argent-idx', idxVal);
        } catch (e) {}
        
        elementsResult.push({
            idx: idxVal,
            tag: el.tag,
            label: el.label,
            value: el.value,
            checked: el.checked,
            href: el.href
        });
    });

    return {
        url: window.location.href,
        title: document.title,
        elements: elementsResult,
        nextIdx: startIdx + finalElements.length,
        totalFiltered: totalFiltered
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
        self._cdp_mode = False        # True when connected via CDP to user's browser
        self._cdp_process = None      # subprocess.Popen if we launched the browser ourselves
        self._element_frames = {}     # Maps element index -> Playwright Frame object

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def _ensure_browser(self, headed: bool = False) -> None:
        """Lazy-initialize browser. Routes to isolated or CDP mode based on config."""
        # If the browser exists but headed mode changed, or it is disconnected, restart
        if self._browser and (self._headed != headed or not self._browser.is_connected):
            await self.shutdown()

        if self._browser:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            )

        # Start Playwright ONCE — both modes reuse this instance.
        if not self._playwright:
            self._playwright = await async_playwright().start()

        from config import get_browser_mode
        mode = get_browser_mode()

        if mode == "user":
            await self._connect_user_browser(headed)
        else:
            await self._launch_isolated(headed)

    async def _launch_isolated(self, headed: bool = False) -> None:
        """Launch Playwright's bundled Chromium (isolated, no user data).
        Assumes self._playwright is already started.
        """
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
        self._cdp_mode = False
        log.info("Isolated browser launched (headed=%s)", headed)

    @staticmethod
    def _is_browser_running(exe_name: str) -> bool:
        """Check if a browser process is already running."""
        import platform
        from src.agent.shell import run_text
        if platform.system() != "Windows":
            try:
                result = run_text(["pgrep", "-f", exe_name], capture_output=True)
                return result.returncode == 0
            except Exception:
                return False
        
        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and proc.info['name'].lower() == exe_name.lower():
                    return True
            return False
        except ImportError:
            try:
                result = run_text(
                    ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/NH"],
                    capture_output=True, timeout=5,
                )
                return exe_name.lower() in result.stdout.lower()
            except Exception:
                return False

    async def _connect_user_browser(self, headed: bool = False) -> None:
        """Connect to the user's real browser via Chrome DevTools Protocol.
        Assumes self._playwright is already started.

        Strategy:
        1. Try connecting to an already-running CDP endpoint (localhost:9222)
        2. If not available, check if browser is running (profile lock risk)
        3. If browser is NOT running, launch it with CDP
        4. If nothing works, fall back to isolated mode
        """
        from config import get_browser_name
        from browser_detect import find_browser, get_default_browser
        import os

        # 1. Try connecting to an already-running CDP endpoint
        try:
            self._browser = await self._playwright.chromium.connect_over_cdp(
                "http://localhost:9222", timeout=3000
            )
            self._cdp_mode = True
            self._headed = True
            log.info("Connected to existing browser via CDP on port 9222")
            return
        except Exception:
            log.info("No existing CDP endpoint found, will launch browser")

        # 2. Find the user's browser
        browser_name = get_browser_name()
        if browser_name == "auto":
            info = get_default_browser()
        else:
            info = find_browser(browser_name)

        if not info:
            log.warning("No user browser found, falling back to isolated mode")
            await self._launch_isolated(headed)
            return

        # 3. Check if this browser is already running (profile lock risk)
        exe_name = os.path.basename(info.exe_path)
        if self._is_browser_running(exe_name):
            log.warning(
                "%s is already running without CDP. "
                "Cannot use the same profile. "
                "Close %s and retry, or switch to isolated mode. "
                "Falling back to isolated mode.",
                info.name, info.name,
            )
            await self._launch_isolated(headed)
            return

        # 4. Launch the browser with CDP enabled
        import subprocess
        cdp_args = [
            info.exe_path,
            "--remote-debugging-port=9222",
            f"--user-data-dir={info.user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        log.info("Launching %s with CDP: %s", info.name, info.exe_path)

        self._cdp_process = subprocess.Popen(
            cdp_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 5. Wait for CDP endpoint to become available
        from urllib.request import urlopen
        from urllib.error import URLError
        connected = False
        for attempt in range(30):  # 30 * 0.5s = 15s max
            try:
                resp = urlopen("http://localhost:9222/json/version", timeout=2)
                if resp.status == 200:
                    connected = True
                    break
            except (URLError, OSError):
                pass
            await asyncio.sleep(0.5)

        if not connected:
            log.error("Failed to connect to %s CDP after 15s", info.name)
            if self._cdp_process:
                self._cdp_process.terminate()
                self._cdp_process = None
            log.warning("Falling back to isolated mode")
            await self._launch_isolated(headed)
            return

        # 6. Connect Playwright to the browser via CDP
        self._browser = await self._playwright.chromium.connect_over_cdp(
            "http://localhost:9222"
        )
        self._cdp_mode = True
        self._headed = True
        log.info("Connected to %s via CDP", info.name)

    async def shutdown(self) -> None:
        """Close all sessions and disconnect from the browser.
        In CDP mode, we do NOT kill the user's browser — only disconnect.
        """
        for name in list(self._sessions.keys()):
            try:
                ctx = self._sessions.pop(name)
                # In CDP mode, don't close the context (it belongs to the user)
                if not self._cdp_mode:
                    # Save storage state before closing context
                    try:
                        state_dir = Path.home() / ".argent" / "browser_sessions"
                        state_dir.mkdir(parents=True, exist_ok=True)
                        state_file = state_dir / f"{name}_storage.json"
                        await ctx.context.storage_state(path=str(state_file))
                        log.info("Saved storage state to %s", state_file)
                    except Exception as e:
                        log.warning("Failed to save storage state in shutdown: %s", e)
                    await ctx.context.close()
            except Exception:
                pass

        if self._browser:
            try:
                if self._cdp_mode:
                    # Disconnect only — do NOT close the user's browser
                    await self._browser.close()
                else:
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

        # Do NOT terminate CDP process — it's the user's real browser
        self._cdp_process = None
        self._cdp_mode = False
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

        # Auto-attach fallback for CDP mode
        from config import get_browser_mode
        if get_browser_mode() == "user":
            try:
                await self._ensure_browser()
                if self._cdp_mode and self._browser:
                    contexts = self._browser.contexts
                    if contexts:
                        context = contexts[0]
                        pages = context.pages
                        if pages:
                            page = pages[-1]
                            sc = _SessionContext(context=context, page=page)
                            self._sessions[session] = sc
                            log.info("Auto-attached to existing tab for session '%s'", session)
                            return sc
            except Exception as e:
                log.warning("Failed to auto-attach to existing CDP page: %s", e)

        raise KeyError(f"Session '{session}' does not exist. Use browser_open first.")

    async def _create_session(self, session: str, headed: bool = False) -> _SessionContext:
        """Create a new named browser session.

        If the Playwright driver has died (stale singleton), a connection error
        will be raised on new_context(). We catch it, force a full shutdown, and
        retry once with a fresh Playwright instance.
        """
        try:
            return await self._create_session_inner(session, headed)
        except Exception as e:
            if "Connection closed" in str(e) or "Target closed" in str(e):
                log.warning("Playwright driver appears dead (%s), restarting...", e)
                await self.shutdown()
                return await self._create_session_inner(session, headed)
            raise

    async def _create_session_inner(self, session: str, headed: bool = False) -> _SessionContext:
        """Inner session creation logic (called by _create_session, may retry)."""
        await self._ensure_browser(headed)

        if self._cdp_mode:
            # CDP mode: use the user's existing browser context.
            # No stealth patches — this is their real browser with real fingerprint.
            contexts = self._browser.contexts
            if contexts:
                context = contexts[0]
            else:
                context = await self._browser.new_context()
            page = await context.new_page()
        else:
            # Isolated mode: check if storage state exists for this session
            state_dir = Path.home() / ".argent" / "browser_sessions"
            state_file = state_dir / f"{session}_storage.json"
            
            kwargs = {
                "viewport": {"width": 1280, "height": 720},
                "user_agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "locale": "en-US",
                "timezone_id": "America/New_York",
                "java_script_enabled": True,
                "accept_downloads": True,
            }
            if state_file.exists():
                kwargs["storage_state"] = str(state_file)
                log.info("Loading storage state from %s for session %s", state_file, session)
                
            context = await self._browser.new_context(**kwargs)
            page = await context.new_page()
            await self._apply_stealth(page)

        sc = _SessionContext(context=context, page=page)
        self._sessions[session] = sc
        log.info("Session '%s' created (cdp=%s)", session, self._cdp_mode)

        # Download handler
        async def handle_download(download):
            try:
                downloads_dir = Path.home() / "Downloads"
                downloads_dir.mkdir(exist_ok=True)
                final_path = downloads_dir / (download.suggested_filename or "downloaded_file")
                log.info("Downloading to %s...", final_path)
                await download.save_as(str(final_path))
                log.info("Successfully downloaded: %s", final_path)
            except Exception as e:
                log.error("Download failed: %s", e)

        # Attach to the initial page
        page.on("download", handle_download)

        # Set up auto-tab-switching listener
        def make_page_handler(session_ctx):
            def handle_page(new_page):
                session_ctx.page = new_page
                new_page.on("download", handle_download)
                log.info("Auto-switched session to new tab: %s", new_page.url)
            return handle_page
        
        context.on("page", make_page_handler(sc))

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

    async def _wait_for_page_settle(self, page, timeout_ms: int = 1500) -> None:
        """Wait for page DOM content to load and network to settle down."""
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:
            pass
        await page.wait_for_timeout(500)

    # -----------------------------------------------------------------------
    # Public API: Navigation
    # -----------------------------------------------------------------------

    async def open_page(self, url: str, session: str = "default",
                        headed: bool = False) -> str:
        """Open a URL in a session. Creates the session if it doesn't exist."""
        blocked = _blocked_url_reason(url)
        if blocked:
            return blocked
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
        mode_label = "CDP (user browser)" if self._cdp_mode else "isolated (Playwright)"
        return f"Opened: {current_url}\nTitle: {title}\nMode: {mode_label}"

    async def navigate(self, url: str, session: str = "default") -> str:
        """Navigate the current session to a new URL."""
        blocked = _blocked_url_reason(url)
        if blocked:
            return blocked
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

    async def get_state(self, session: str = "default", query: str = None,
                        scroll_depth: int = 0, mode: str = None) -> str:
        """
        Dispatcher: extract page state as numbered interactive elements.

        mode=None (default) resolves via get_effective_browser_state_mode():
            'a11y' → accessibility tree (semantic roles, hierarchy, structural context)
            'dom'  → flat JS-extraction of interactive elements only

        Both modes produce [idx]-prefixed lines compatible with browser_click,
        browser_input, and browser_scroll. Falls back to _get_dom_state() on
        CDP failure (when mode='a11y').
        """
        if mode is None:
            from config import get_effective_browser_state_mode
            mode = get_effective_browser_state_mode()
        if mode == "a11y":
            return await self.get_accessibility_tree(session, query, scroll_depth)
        return await self._get_dom_state(session, query, scroll_depth)

    async def _get_dom_state(self, session: str = "default", query: str = None,
                             scroll_depth: int = 0) -> str:
        """
        Extract interactive elements from the DOM via JS injection.

        Returns a text representation optimized for LLM consumption:
            Page: https://example.com | Title: Example
            ---
            [1] input[type=email] placeholder="Email"
            [2] input[type=password] placeholder="Password"
            [3] button "Sign In"
        """
        sc = await self._get_session(session)
        
        # Pre-scrolling for infinite scroll / lazy-loaded pages
        if scroll_depth > 0:
            log.info("Pre-scrolling page for session %s with depth %d", session, scroll_depth)
            for _ in range(scroll_depth):
                try:
                    await sc.page.evaluate("window.scrollBy(0, window.innerHeight);")
                    await sc.page.wait_for_timeout(600)
                except Exception:
                    break
                    
        self._element_frames.clear()

        combined_elements = []
        page_url = "?"
        page_title = "?"
        idx = 1

        total_filtered = 0

        try:
            # Get all frames in the page
            frames = sc.page.frames
            for frame in frames:
                try:
                    result = await frame.evaluate(_STATE_EXTRACTION_JS, {"startIdx": idx, "query": query})
                    if result:
                        frame_elements = result.get("elements", [])
                        for el in frame_elements:
                            self._element_frames[el["idx"]] = frame
                            combined_elements.append(el)
                        idx = result.get("nextIdx", idx)
                        total_filtered += result.get("totalFiltered", 0)

                        # Set page url and title based on the main frame
                        if frame == sc.page.main_frame:
                            page_url = result.get("url", "?")
                            page_title = result.get("title", "?")
                except Exception as e:
                    # Ignore frames that can't be evaluated (e.g. detached, or security restrictions)
                    log.debug("Frame evaluation skipped for frame: %s", e)
        except Exception as e:
            return f"Error extracting state: {e}"

        # Build Open Tabs list
        tabs_list = []
        try:
            pages = sc.context.pages
            for i, p in enumerate(pages):
                active_mark = " (active)" if p == sc.page else ""
                try:
                    t_title = await p.title()
                except Exception:
                    t_title = "Untitled"
                tabs_list.append(f"[{i+1}] {p.url} ({t_title}){active_mark}")
        except Exception:
            pass

        lines = [f"Page: {page_url} | Title: {page_title}"]
        if tabs_list:
            lines.append("Open Tabs: " + ", ".join(tabs_list))
        lines.append("---")

        if not combined_elements:
            lines.append("(No interactive elements found on this page)")
        else:
            for el in combined_elements:
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

        if query and total_filtered > 0:
            lines.append(f"\n(Note: {total_filtered} interactive elements were filtered out by query '{query}'. If you cannot find the element you need, call browser_state without a query or with a different keyword.)")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Public API: Accessibility Tree
    # -----------------------------------------------------------------------

    # Roles that are always excluded from the a11y tree output — they are
    # DOM-internal noise (div wrappers, text leaf nodes, etc.).
    _NOISE_A11Y_ROLES = frozenset({
        "generic", "InlineTextBox", "line break", "none", "LineBreak",
    })

    # Roles that are excluded only when they have no accessible name and no
    # interesting properties — they add visual noise without semantic value.
    _NOISE_IF_UNNAMED_A11Y_ROLES = frozenset({
        "list", "listitem", "paragraph", "strong", "LabelText", "Legend",
        "div", "span", "group",
    })

    # Roles that are always included regardless of name/properties.
    _INTERACTIVE_A11Y_ROLES = frozenset({
        "link", "button", "textbox", "checkbox", "radio", "combobox",
        "listbox", "menuitem", "menuitemcheckbox", "menuitemradio",
        "option", "switch", "tab", "slider", "spinbutton", "searchbox",
        "menu", "menubar", "treeitem", "tree", "treegrid",
    })

    # Properties worth showing in the text output.
    _INTERESTING_A11Y_PROPS = frozenset({
        "checked", "selected", "expanded", "level", "disabled", "pressed",
        "readonly", "required", "invalid",
    })

    # JS: wait for two animation frames so the browser finishes layout/paint
    # after a scroll before we snapshot the AX tree.
    _WAIT_RAF_JS = """
new Promise(resolve => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
})
"""

    # JS: clear all data-argent-idx markers in the document (and shadow roots).
    _CLEAR_MARKERS_JS = """
() => {
    function clear(root) {
        try {
            root.querySelectorAll('[data-argent-idx]').forEach(el => {
                el.removeAttribute('data-argent-idx');
            });
            root.querySelectorAll('*').forEach(el => {
                if (el.shadowRoot) clear(el.shadowRoot);
            });
        } catch(e) {}
    }
    clear(document);
}
"""

    # JS: collect all data-argent-idx values present in this frame's document.
    _COLLECT_IDX_JS = """
() => {
    return Array.from(document.querySelectorAll('[data-argent-idx]'))
        .map(el => parseInt(el.getAttribute('data-argent-idx'), 10))
        .filter(n => !isNaN(n));
}
"""

    @staticmethod
    def _a11y_node_name(node: dict) -> str:
        """Extract the accessible name from an AX node."""
        name = node.get("name")
        if name and isinstance(name, dict):
            return name.get("value", "")
        return ""

    @staticmethod
    def _a11y_node_role(node: dict) -> str:
        """Extract the role from an AX node."""
        role = node.get("role")
        if role and isinstance(role, dict):
            return role.get("value", "")
        return ""

    @staticmethod
    def _a11y_node_value(node: dict) -> str:
        """Extract the value from an AX node."""
        val = node.get("value")
        if val and isinstance(val, dict):
            v = val.get("value")
            return "" if v is None else str(v)
        return ""

    @staticmethod
    def _a11y_interesting_props(node: dict) -> list:
        """Extract interesting properties (checked, level, etc.) from an AX node."""
        props = []
        for prop in node.get("properties", []):
            pname = prop.get("name", "")
            if pname in BrowserEngine._INTERESTING_A11Y_PROPS:
                pval = prop.get("value", {})
                pval_v = pval.get("value", "") if isinstance(pval, dict) else ""
                props.append((pname, pval_v))
        return props

    def _a11y_is_interesting(self, role: str, name: str, props: list) -> bool:
        """Determine whether an AX node should appear in the output tree."""
        if role in self._NOISE_A11Y_ROLES:
            return False
        if role in self._INTERACTIVE_A11Y_ROLES:
            return True
        if role in ("heading", "img", "image"):
            return True
        if role in self._NOISE_IF_UNNAMED_A11Y_ROLES:
            return bool(name) or bool(props)
        # Everything else: include if it has a name or interesting properties
        return bool(name) or bool(props)

    async def get_accessibility_tree(self, session: str = "default",
                                     query: str = None,
                                     scroll_depth: int = 0) -> str:
        """
        Extract the browser's accessibility tree and return it as indented text
        with numbered indices for interactive elements.

        Indices are compatible with browser_click(index), browser_input(index),
        and browser_scroll(index) — DOM elements are tagged with data-argent-idx
        via CDP, and the existing click mechanism works unchanged.

        Falls back to get_state() if CDP or the Accessibility domain is
        unavailable.
        """
        sc = await self._get_session(session)

        # --- Pre-scroll with RAF-based stability wait ---
        if scroll_depth > 0:
            log.info("Pre-scrolling page for a11y tree, depth %d", scroll_depth)
            for _ in range(scroll_depth):
                try:
                    await sc.page.evaluate("window.scrollBy(0, window.innerHeight);")
                    await sc.page.wait_for_timeout(400)
                except Exception:
                    break
            # Wait for two animation frames so layout/paint settles
            try:
                await sc.page.evaluate(self._WAIT_RAF_JS)
            except Exception:
                pass

        # --- Clear previous data-argent-idx markers ---
        try:
            await sc.page.evaluate(self._CLEAR_MARKERS_JS)
        except Exception:
            pass
        self._element_frames.clear()

        # --- Get the AX tree via CDP ---
        client = None
        try:
            client = await sc.context.new_cdp_session(sc.page)
            ax_result = await client.send("Accessibility.getFullAXTree")
        except Exception as e:
            log.warning("CDP Accessibility.getFullAXTree failed: %s — falling back to DOM extraction", e)
            if client:
                try:
                    await client.detach()
                except Exception:
                    pass
            return await self._get_dom_state(session, query, scroll_depth)

        nodes = ax_result.get("nodes", [])
        if not nodes:
            if client:
                try:
                    await client.detach()
                except Exception:
                    pass
            return await self._get_dom_state(session, query, scroll_depth)

        # Build node-id → node map
        node_map = {}
        for n in nodes:
            node_map[n["nodeId"]] = n

        # Find root: the node whose parentId is not in the map (or first node)
        root_id = nodes[0]["nodeId"]
        for n in nodes:
            pid = n.get("parentId")
            if pid and pid not in node_map:
                root_id = n["nodeId"]
                break

        # --- Recursive walk: filter + assign candidate indices ---
        candidates = []  # list of dicts: {idx, role, name, value, props, depth, backend_dom_node_id}
        idx_counter = [1]  # mutable counter for closure
        MAX_NODES = 600

        def walk(node_id: str, depth: int):
            if len(candidates) >= MAX_NODES:
                return
            node = node_map.get(node_id)
            if not node:
                return

            role = self._a11y_node_role(node)
            name = self._a11y_node_name(node)
            value = self._a11y_node_value(node)
            props = self._a11y_interesting_props(node)
            backend_id = node.get("backendDOMNodeId")

            is_interesting = self._a11y_is_interesting(role, name, props)

            if is_interesting:
                candidates.append({
                    "idx": idx_counter[0],
                    "role": role,
                    "name": name,
                    "value": value,
                    "props": props,
                    "depth": depth,
                    "backend_dom_node_id": backend_id,
                })
                idx_counter[0] += 1

            for child_id in node.get("childIds", []):
                walk(child_id, depth + 1 if is_interesting else depth)

        walk(root_id, 0)

        if not candidates:
            if client:
                try:
                    await client.detach()
                except Exception:
                    pass
            return await self._get_dom_state(session, query, scroll_depth)

        # --- Tag DOM elements with data-argent-idx via CDP ---
        # 1. Collect backendDOMNodeIds (only for nodes that have one)
        taggable = [c for c in candidates if c["backend_dom_node_id"]]

        tagged_indices = set()  # indices that were successfully tagged

        if taggable:
            try:
                # Enable DOM + Runtime domains — required for resolveNode + callFunctionOn
                await client.send("DOM.enable")
                await client.send("Runtime.enable")

                # 2. For each candidate: resolveNode(backendNodeId) → objectId,
                #    then callFunctionOn to set data-argent-idx attribute.
                #    This is more reliable than pushNodesByBackendIdsToFrontend,
                #    which can return 0 nodes in certain CDP session contexts.
                BATCH_SIZE = 50

                async def tag_one(cand):
                    try:
                        resolve_result = await client.send(
                            "DOM.resolveNode",
                            {"backendNodeId": cand["backend_dom_node_id"]},
                        )
                        object_id = resolve_result.get("object", {}).get("objectId")
                        if not object_id:
                            return None
                        await client.send("Runtime.callFunctionOn", {
                            "objectId": object_id,
                            "functionDeclaration": (
                                "function(idx) { "
                                "this.setAttribute('data-argent-idx', String(idx)); "
                                "return this.getAttribute('data-argent-idx'); "
                                "}"
                            ),
                            "arguments": [{"value": cand["idx"]}],
                            "returnByValue": True,
                        })
                        return cand["idx"]
                    except Exception:
                        return None

                for i in range(0, len(taggable), BATCH_SIZE):
                    batch = taggable[i:i + BATCH_SIZE]
                    results = await asyncio.gather(*[tag_one(c) for c in batch])
                    for r in results:
                        if r is not None:
                            tagged_indices.add(r)

            except Exception as e:
                log.warning("CDP DOM tagging failed: %s — some elements may not be clickable", e)

        # --- Map indices to frames ---
        try:
            for frame in sc.page.frames:
                try:
                    found_indices = await frame.evaluate(self._COLLECT_IDX_JS)
                    for fi in found_indices:
                        if fi not in self._element_frames:
                            self._element_frames[fi] = frame
                except Exception:
                    pass
        except Exception:
            pass

        # --- Build text output ---
        # IMPORTANT: nodes stay in the tree for structural context even if they
        # weren't tagged. Only the [idx] prefix is omitted for untagged nodes.
        # Apply query filter on the text representation.
        query_keywords = []
        if query:
            query_keywords = [q.strip().lower() for q in query.split(",") if q.strip()]

        # Build open tabs list
        tabs_list = []
        try:
            pages = sc.context.pages
            for i, p in enumerate(pages):
                active_mark = " (active)" if p == sc.page else ""
                try:
                    t_title = await p.title()
                except Exception:
                    t_title = "Untitled"
                tabs_list.append(f"[{i+1}] {p.url} ({t_title}){active_mark}")
        except Exception:
            pass

        lines = [f"Page: {sc.page.url} | Title: {await sc.page.title() if sc.page else '?'}"]
        if tabs_list:
            lines.append("Open Tabs: " + ", ".join(tabs_list))
        lines.append("---")

        total_filtered_by_query = 0

        for c in candidates:
            # Query filtering
            if query_keywords:
                haystack = " ".join([
                    c["role"].lower(),
                    c["name"].lower(),
                    c["value"].lower(),
                ])
                if not any(kw in haystack for kw in query_keywords):
                    total_filtered_by_query += 1
                    continue

            indent = "  " * c["depth"]
            is_tagged = c["idx"] in tagged_indices or c["idx"] in self._element_frames

            # Build the line: [idx] role "name" props...
            # If the node wasn't tagged, omit the [idx] — it's not clickable
            idx_prefix = f"[{c['idx']}]" if is_tagged else "   "
            parts = [f"{indent}{idx_prefix}", c["role"]]

            if c["name"]:
                parts.append(f'"{c["name"][:60]}"')

            if c["value"]:
                parts.append(f'value="{c["value"][:50]}"')

            for pname, pval in c["props"]:
                parts.append(f"{pname}={pval}")

            lines.append(" ".join(parts))

        if query and total_filtered_by_query > 0:
            lines.append(
                f"\n(Note: {total_filtered_by_query} nodes were filtered out by query '{query}'. "
                "Call without a query to see the full tree.)"
            )

        # Detach CDP session
        if client:
            try:
                await client.detach()
            except Exception:
                pass

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Public API: Interaction
    # -----------------------------------------------------------------------

    async def click(self, index: int = None, selector: str = None, text: str = None, session: str = "default") -> str:
        """Click element by its state index, CSS selector, or visible text."""
        sc = await self._get_session(session)
        
        # Priority: index > selector > text
        if index is not None:
            try:
                frame = self._element_frames.get(index, sc.page)
                el = frame.locator(f'[data-argent-idx="{index}"]')
                count = await el.count()
                if count == 0:
                    return (
                        f"Error: Element [{index}] not found. "
                        "The page may have changed — call browser_state to get fresh indices."
                    )
                await el.first.click(timeout=10000)
                await self._wait_for_page_settle(sc.page)
                return f"Clicked element [{index}]. Call browser_state to see the updated page."
            except Exception as e:
                return f"Error clicking element [{index}]: {e}"
        elif selector is not None:
            try:
                target_locator = None
                for frame in sc.page.frames:
                    try:
                        el = frame.locator(selector)
                        if await el.count() > 0:
                            target_locator = el.first
                            break
                    except Exception:
                        pass
                
                if not target_locator:
                    target_locator = sc.page.locator(selector).first
                    
                await target_locator.click(timeout=10000)
                await self._wait_for_page_settle(sc.page)
                return f"Clicked element matching selector '{selector}'."
            except Exception as e:
                return f"Error clicking selector '{selector}': {e}"
        elif text is not None:
            try:
                target_locator = None
                for frame in sc.page.frames:
                    try:
                        el = frame.get_by_text(text, exact=False)
                        if await el.count() > 0:
                            target_locator = el.first
                            break
                    except Exception:
                        pass
                
                if not target_locator:
                    target_locator = sc.page.get_by_text(text, exact=False).first
                    
                await target_locator.click(timeout=10000)
                await self._wait_for_page_settle(sc.page)
                return f"Clicked element matching text '{text}'."
            except Exception as e:
                return f"Error clicking text '{text}': {e}"
        else:
            return "Error: You must specify index, selector, or text to click."

    async def switch_tab(self, index: int, session: str = "default") -> str:
        """Switch the active tab/page of a named session by its 1-based index."""
        sc = await self._get_session(session)
        pages = sc.context.pages
        if not pages:
            return "Error: No open tabs in browser context."
        
        if not (1 <= index <= len(pages)):
            return f"Error: Invalid tab index {index}. Available indices: 1-{len(pages)}."
        
        target_page = pages[index - 1]
        await target_page.bring_to_front()
        sc.page = target_page
        
        title = "?"
        try:
            title = await target_page.title()
        except Exception:
            pass
        return f"Switched to tab [{index}]: {target_page.url} ({title})"

    async def fill_input(self, index: int = None, text: str = "", selector: str = None,
                         session: str = "default") -> str:
        """Clear and fill text into an input element by its state index or CSS selector."""
        sc = await self._get_session(session)
        
        if index is not None:
            try:
                frame = self._element_frames.get(index, sc.page)
                el = frame.locator(f'[data-argent-idx="{index}"]')
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
        elif selector is not None:
            try:
                target_locator = None
                for frame in sc.page.frames:
                    try:
                        el = frame.locator(selector)
                        if await el.count() > 0:
                            target_locator = el.first
                            break
                    except Exception:
                        pass
                
                if not target_locator:
                    target_locator = sc.page.locator(selector).first
                    
                await target_locator.click(timeout=5000)
                await target_locator.fill(text, timeout=5000)
                return f"Filled element matching selector '{selector}' with text."
            except Exception as e:
                return f"Error filling selector '{selector}': {e}"
        else:
            return "Error: You must specify either index or selector to fill input."

    async def scroll(self, direction: str = "down", amount: int = 500, index: int = None,
                     session: str = "default") -> str:
        """Scroll the page or scroll a specific element into view."""
        sc = await self._get_session(session)
        
        if index is not None:
            try:
                frame = self._element_frames.get(index, sc.page)
                el = frame.locator(f'[data-argent-idx="{index}"]')
                count = await el.count()
                if count == 0:
                    return f"Error: Element [{index}] not found to scroll to."
                await el.first.scroll_into_view_if_needed(timeout=5000)
                await sc.page.wait_for_timeout(800)
                return f"Scrolled element [{index}] into view."
            except Exception as e:
                return f"Error scrolling to element [{index}]: {e}"
        else:
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
            # BeautifulSoup parsing of a large DOM is CPU-bound and synchronous;
            # run it off the event loop so it can't stall the browser loop.
            import asyncio
            md = await asyncio.to_thread(_html_to_markdown, html)
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
        
        # Save storage state for isolated mode sessions before closing context
        if not self._cdp_mode:
            try:
                state_dir = Path.home() / ".argent" / "browser_sessions"
                state_dir.mkdir(parents=True, exist_ok=True)
                state_file = state_dir / f"{session}_storage.json"
                await sc.context.storage_state(path=str(state_file))
                log.info("Saved storage state to %s", state_file)
            except Exception as e:
                log.warning("Failed to save storage state for session %s: %s", session, e)
                
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
