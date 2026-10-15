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
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    if not url or not url.strip():
        return "Error: no URL provided."
    raw = url.strip()
    if raw.startswith(("\\\\", "//")) and not raw.lower().startswith(("http://", "https://")):
        return "Error: refusing to open UNC or protocol-relative network paths in the browser."
    if raw.lower() in ("about:blank", "about:"):
        return None
    scheme = urllib.parse.urlparse(raw).scheme.lower()
    if not scheme:
        return None                       # bare host: will be normalized to http:// by _normalize_url
    if scheme in _ALLOWED_URL_SCHEMES:
        return None
    return (f"Error: refusing to open a '{scheme}:' URL in the browser — only http/https "
            f"are allowed. To read a local file use read_file, not the browser.")


def _normalize_url(url: str) -> str:
    """Normalize user/model input URL for browser navigation.

    Strips whitespace, handles 'about:blank', and prepends 'http://' to bare
    domains or host:port pairs (e.g. 'localhost:3000', 'google.com') because
    Playwright requires an explicit scheme.
    """
    raw = str(url).strip()
    if raw.lower() in ("about:blank", "about:"):
        return "about:blank"
    # If no URI scheme prefix (like http://, https://) is present
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', raw):
        return f"http://{raw}"
    return raw


def _html_to_markdown(html: str) -> str:
    """Convert HTML to readable Markdown using BeautifulSoup.
    Strips scripts/styles and converts common tags to Markdown equivalents."""
    if not html:
        return ""
    import re  # needed for whitespace cleanup at the end regardless of bs4 availability

    try:
        from bs4 import BeautifulSoup
    except ImportError:
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

        // 3b. Поле обёрнуто в <label>…</label>: подпись — текст обёртки
        try {
            const wrap = node.closest ? node.closest('label') : null;
            if (wrap) {
                const wrapText = (wrap.innerText || wrap.textContent || '').trim();
                if (wrapText) return wrapText.substring(0, 80);
            }
        } catch (e) {}

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

            // Disabled state
            let disabled = Boolean(node.disabled || (node.getAttribute && node.getAttribute('aria-disabled') === 'true'));

            candidates.push({
                node: node,
                tag: descriptor,
                label: label,
                value: value,
                checked: checked,
                disabled: disabled,
                href: tag === 'A' ? (node.href || '') : '',
                text: ((node.innerText || node.textContent || '') + '').trim().substring(0, 80),
                name: (type === 'radio' && node.getAttribute('name')) || '',
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

    // Prioritize elements in viewport
    const inViewport = [];
    const outOfViewport = [];
    candidates.forEach(c => {
        if (c.inViewport) {
            inViewport.push(c);
        } else {
            outOfViewport.push(c);
        }
    });

    const finalElements = inViewport.concat(outOfViewport).slice(0, MAX_ELEMENTS);

    // Apply indexes and write data-argent-idx attributes to all finalElements
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
            disabled: el.disabled,
            href: el.href,
            text: el.text,
            name: el.name
        });
    });

    // Filter elementsResult by query AFTER indexes have been assigned in DOM
    let filteredElements = elementsResult;
    let totalFiltered = 0;
    if (query) {
        const keywords = query.split(',').map(s => s.trim().toLowerCase()).filter(s => s.length > 0);
        if (keywords.length > 0) {
            filteredElements = elementsResult.filter(c => {
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

    return {
        url: window.location.href,
        title: document.title,
        elements: filteredElements,
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


def _get_stagehand_extension_dir() -> Optional[str]:
    """Dynamically locate Stagehand's built-in extension if installed."""
    try:
        clean_dir = Path.home() / ".argent" / "stagehand_extension"
        if clean_dir.is_dir() and (clean_dir / "manifest.json").is_file():
            return str(clean_dir)
        import stagehand.extension_assets
        ext_dir = stagehand.extension_assets.extension_directory()
        if ext_dir and (ext_dir / "manifest.json").is_file():
            return str(ext_dir)
    except Exception:
        pass
    return None


class BrowserEngine:
    """
    Singleton Playwright browser manager.
    
    Provides high-level methods that Argent's tool functions call.
    All public methods are async — callers use engine.run() to bridge sync/async.
    
    Usage pattern in tools:
        result = browser_engine.run(browser_engine.navigate("https://example.com"))
    """

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._playwright = None       # Playwright instance
        self._browser = None          # Shared Chromium browser instance
        self._sessions: Dict[str, _SessionContext] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._headed = False          # Current headed mode state
        self._stealth_applied = False
        self._cdp_mode = False        # True when connected via CDP to user's browser
        self._cdp_process = None      # subprocess.Popen if we launched the browser ourselves
        self._cdp_port: Optional[int] = None  # Active CDP port (e.g. 9222)
        self._element_frames = {}     # Maps element index -> Playwright Frame object
        self._element_metadata = {}   # Maps element index -> dict of node metadata for resilient recovery
        self._last_fallback_notice: Optional[str] = None  # Reason when user mode falls back to isolated

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    @staticmethod
    def _find_free_port(preferred: int = 9222) -> int:
        """Find an available TCP port for CDP, preferring 9222."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", preferred))
                return preferred
        except OSError:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                return s.getsockname()[1]

    async def _ensure_browser(
        self,
        headed: bool = False,
        initial_url: Optional[str] = None,
        mode_override: Optional[str] = None,
    ) -> None:
        """Lazy-initialize browser. Routes to isolated or CDP mode based on config or mode_override."""
        from config import get_browser_mode
        if mode_override is not None:
            normalized_override = str(mode_override).strip().lower()
            if normalized_override not in ("auto", "user", "isolated"):
                raise ValueError(
                    f"Invalid browser mode '{mode_override}'. Allowed modes: 'auto', 'user', 'isolated'."
                )
            mode = normalized_override if normalized_override != "auto" else get_browser_mode()
        else:
            mode = get_browser_mode()
        desired_cdp = (mode == "user")

        # If the browser exists but headed mode or CDP mode changed, or it is disconnected, restart
        if self._browser and (self._headed != headed or self._cdp_mode != desired_cdp or not self._browser.is_connected):
            await self.shutdown()

        if self._browser:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            )

        # Suppress noisy Node.js deprecation warnings (e.g. url.parse() DEP0169 in Playwright driver)
        import os
        os.environ.setdefault("NODE_OPTIONS", "--no-deprecation")

        # Start Playwright ONCE — both modes reuse this instance.
        if not self._playwright:
            self._playwright = await async_playwright().start()

        if mode == "user":
            await self._connect_user_browser(headed, initial_url=initial_url)
        else:
            await self._launch_isolated(headed)

    async def _launch_isolated(self, headed: bool = False) -> None:
        """Launch Playwright's bundled Chromium (isolated, no user data).
        Assumes self._playwright is already started.
        """
        port = self._find_free_port(9222)
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
            f"--remote-debugging-port={port}",
            "--remote-allow-origins=*",
            "--enable-unsafe-extension-debugging",
        ]
        ext_dir = _get_stagehand_extension_dir()
        if ext_dir:
            launch_args.append(f"--load-extension={ext_dir}")

        self._browser = await self._playwright.chromium.launch(
            headless=not headed,
            args=launch_args,
        )
        self._cdp_port = port
        self._headed = headed
        self._cdp_mode = False
        log.info("Isolated browser launched (headed=%s, cdp_port=%s)", headed, port)

    @staticmethod
    def _is_browser_running(exe_name: str) -> bool:
        """Check if a browser process is already running."""
        import host_os
        from src.agent.shell import run_text
        if host_os.system() != "Windows":
            try:
                result = run_text(["pgrep", "-f", exe_name], capture_output=True)
                return result.returncode == 0
            except Exception:
                return False
        
        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                try:
                    pname = proc.info.get('name') if proc.info else None
                    if pname and pname.lower() == exe_name.lower():
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
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

    async def _connect_user_browser(self, headed: bool = False, initial_url: Optional[str] = None) -> None:
        """Connect to the user's real browser via Chrome DevTools Protocol.
        Assumes self._playwright is already started.

        Strategy:
        0. If headless was requested, skip CDP entirely (CDP is always visible)
        1. Try connecting to an already-running CDP endpoint (127.0.0.1:9222)
        2. If not available, check if browser is running (profile lock risk)
        3. If browser is NOT running, launch it with CDP and target URL
        4. If nothing works, fall back to isolated mode
        """
        from config import get_browser_name
        from browser_detect import find_browser, get_default_browser

        # 0. CDP mode always shows a visible browser window. If the caller
        #    requested headless (background) operation, skip CDP entirely
        #    and use isolated headless mode instead — otherwise we'd silently
        #    ignore headed=False and open tabs in the user's visible browser.
        if not headed:
            log.info(
                "Headless requested in user mode — CDP requires a visible "
                "browser window, falling back to isolated headless"
            )
            self._last_fallback_notice = (
                "Notice: Headless mode requested in 'user' browser mode. The user's real browser "
                "cannot run headlessly without disrupting open sessions, so Argent automatically "
                "switched to isolated headless Chromium."
            )
            await self._launch_isolated(headed)
            return

        # 1. Try connecting to an already-running CDP endpoint (fast probe via 127.0.0.1)
        try:
            self._browser = await self._playwright.chromium.connect_over_cdp(
                "http://127.0.0.1:9222", timeout=2000
            )
            self._cdp_port = 9222
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
            self._last_fallback_notice = (
                "Notice: No supported Chromium-based user browser (Chrome, Yandex, Edge, Brave) "
                "was found installed. Switched to isolated Chromium."
            )
            await self._launch_isolated(headed)
            return

        # Dedicated profile directory for Argent user-browser automation
        # Prevents Chromium from blocking --remote-debugging-port due to default profile restrictions,
        # and avoids SingletonLock collisions with the user's open personal browser.
        argent_user_data = Path.home() / ".argent" / "browser_profiles" / info.key
        argent_user_data.mkdir(parents=True, exist_ok=True)

        # Pre-configure Developer Mode, extension whitelists, and permissions in automated profile Preferences
        try:
            pref_file = argent_user_data / "Default" / "Preferences"
            pref_file.parent.mkdir(parents=True, exist_ok=True)
            import json
            prefs = {}
            if pref_file.is_file():
                prefs = json.loads(pref_file.read_text(encoding="utf-8", errors="ignore"))
            ext_prefs = prefs.setdefault("extensions", {})
            ext_prefs.setdefault("ui", {})["developer_mode"] = True
            allowed = ext_prefs.setdefault("allowed_install_sites", [])
            for site in [
                "https://chromewebstore.google.com/*",
                "https://chrome.google.com/*",
                "https://addons.opera.com/*",
                "<all_urls>",
            ]:
                if site not in allowed:
                    allowed.append(site)
            prefs.setdefault("safebrowsing", {})["enabled"] = False
            pref_file.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
        except Exception as e:
            log.debug("Could not pre-configure profile Preferences: %s", e)

        # 3. Launch the browser with CDP enabled and target URL attached (Variant 3)
        import subprocess
        cdp_args = [
            info.exe_path,
            "--remote-debugging-port=9222",
            "--remote-allow-origins=*",
            f"--user-data-dir={argent_user_data}",
            "--no-first-run",
            "--no-default-browser-check",
            "--test-type",
            "--enable-unsafe-extension-debugging",
            "--whitelisted-extension-id=kmgeemlmoiohlehomoligkpelnhjbahl,kbllhomkfighlohagocmnjkhfehkombc,knkbjekpciapkagiphkcfdmhgaggjilb,hjkgacheefjckdpkhppjbhldfpdmbbmh",
            "--disable-features=SafeBrowsing,YandexProtect",
        ]
        ext_dir = _get_stagehand_extension_dir()
        if ext_dir:
            cdp_args.append(f"--load-extension={ext_dir}")
        if initial_url:
            cdp_args.append(initial_url)
        log.info("Launching %s with CDP (Variant 3): %s (url=%s)", info.name, info.exe_path, initial_url)

        try:
            self._cdp_process = subprocess.Popen(
                cdp_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            # The executable was found once (a detected install, a saved name) and
            # is gone or unusable now: uninstalled, moved, blocked. Raising here
            # surfaced as a failed browser_open with nothing the model could do
            # about it, while the isolated browser works.
            log.warning("Could not start %s (%s): %s", info.name, info.exe_path, e)
            self._cdp_process = None
            self._last_fallback_notice = (
                f"Notice: Could not start {info.name} ({e}). Switched to isolated Chromium."
            )
            await self._launch_isolated(headed)
            return

        # 4. Wait for CDP endpoint to become available (fast TCP probe, max ~12s for Windows cold-start)
        connected = False
        for _ in range(48):  # 48 * 0.25s = ~12s max
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", 9222),
                    timeout=0.25,
                )
                writer.close()
                await writer.wait_closed()
                connected = True
                break
            except Exception:
                pass
            await asyncio.sleep(0.25)

        if not connected:
            log.warning("Failed to connect to %s CDP on 127.0.0.1:9222 after 12s", info.name)
            # Do not violently kill the user's browser; leave it running if opened
            self._cdp_process = None
            log.warning("Falling back to isolated mode")
            self._last_fallback_notice = (
                f"Notice: Could not connect to {info.name} via CDP on 127.0.0.1:9222 after 12s. "
                f"Switched to isolated Chromium."
            )
            await self._launch_isolated(headed)
            return

        # 5. Connect Playwright to the browser via CDP with safety timeout
        try:
            self._browser = await self._playwright.chromium.connect_over_cdp(
                "http://127.0.0.1:9222", timeout=5000
            )
            self._cdp_port = 9222
            self._cdp_mode = True
            self._headed = True
            log.info("Connected to %s via CDP", info.name)
        except Exception as e:
            log.error("Failed to connect_over_cdp to %s: %s, falling back to isolated", info.name, e)
            self._last_fallback_notice = (
                f"Notice: Failed to connect to {info.name} over CDP ({e}). Switched to isolated Chromium."
            )
            await self._launch_isolated(headed)
            return

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

        # Invalidate Stagehand instance if active
        try:
            from tools.stagehand_tools import reset_stagehand
            reset_stagehand()
        except ImportError:
            pass
        except Exception as e:
            log.warning("Failed to reset Stagehand in shutdown: %s", e)

        # Do NOT terminate CDP process — it's the user's real browser
        self._cdp_process = None
        self._element_frames.clear()
        self._cdp_mode = False
        self._cdp_port = None
        self._stealth_applied = False
        self._last_fallback_notice = None
        log.info("Browser engine shut down")

    def get_cdp_url(self) -> Optional[str]:
        """Return the CDP URL of the running browser if available."""
        if self._cdp_port and self._browser:
            return f"http://127.0.0.1:{self._cdp_port}"
        return None

    async def get_page(self, session: str = "default"):
        """Get the active Playwright Page object for a session."""
        sc = await self._get_session(session)
        return sc.page

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
        with self._lock:
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
                alive = [p for p in sc.context.pages if not p.is_closed()]
                content_pages = [p for p in alive if p.url not in ("about:blank", "chrome://newtab/", "chrome://new-tab-page/")]
                if content_pages:
                    sc.page = content_pages[-1]
                    log.info("Recovered active content page from context for session '%s': %s", session, sc.page.url)
                elif alive:
                    sc.page = alive[-1]
                    log.info("Recovered alive page from context for session '%s': %s", session, sc.page.url)
                else:
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

    async def _create_session(
        self,
        session: str,
        headed: bool = False,
        initial_url: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> _SessionContext:
        """Create a new named browser session.

        If the Playwright driver has died (stale singleton), a connection error
        will be raised on new_context(). We catch it, force a full shutdown, and
        retry once with a fresh Playwright instance.
        """
        try:
            return await self._create_session_inner(session, headed, initial_url=initial_url, mode=mode)
        except Exception as e:
            if "Connection closed" in str(e) or "Target closed" in str(e):
                log.warning("Playwright driver appears dead (%s), restarting...", e)
                await self.shutdown()
                return await self._create_session_inner(session, headed, initial_url=initial_url, mode=mode)
            raise

    async def _create_session_inner(
        self,
        session: str,
        headed: bool = False,
        initial_url: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> _SessionContext:
        """Inner session creation logic (called by _create_session, may retry)."""
        await self._ensure_browser(headed, initial_url=initial_url, mode_override=mode)

        if self._cdp_mode:
            # CDP mode: use the user's existing browser context.
            # No stealth patches — this is their real browser with real fingerprint.
            contexts = self._browser.contexts
            if contexts:
                context = contexts[0]
            else:
                context = await self._browser.new_context()

            # Reuse active blank page if available instead of opening duplicate tabs
            page = None
            if context.pages:
                active = context.pages[-1]
                try:
                    if active.url in ("about:blank", "chrome://newtab/", "chrome://new-tab-page/"):
                        page = active
                except Exception:
                    pass
            if page is None:
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

        def on_initial_close():
            if sc.page == page or sc.page.is_closed():
                alive = [p for p in sc.context.pages if not p.is_closed() and p != page]
                content = [p for p in alive if p.url not in ("about:blank", "chrome://newtab/", "chrome://new-tab-page/")]
                if content:
                    sc.page = content[-1]
                elif alive:
                    sc.page = alive[-1]
                log.info("Initial page closed; session '%s' active tab is now: %s", session, getattr(sc.page, 'url', 'none'))
        page.on("close", on_initial_close)

        # Set up auto-tab-switching listener
        def make_page_handler(session_ctx):
            def handle_page(new_page):
                prev_page = session_ctx.page
                session_ctx.page = new_page
                new_page.on("download", handle_download)
                log.info("Auto-switched session to new tab: %s", new_page.url)

                def on_tab_close():
                    if session_ctx.page == new_page or session_ctx.page.is_closed():
                        alive = [p for p in session_ctx.context.pages if not p.is_closed() and p != new_page]
                        content = [p for p in alive if p.url not in ("about:blank", "chrome://newtab/", "chrome://new-tab-page/")]
                        if content:
                            session_ctx.page = content[-1]
                        elif alive:
                            session_ctx.page = alive[-1]
                        elif prev_page and not prev_page.is_closed():
                            session_ctx.page = prev_page
                        log.info("Active tab closed; session reverted to: %s", getattr(session_ctx.page, 'url', 'none'))
                new_page.on("close", on_tab_close)
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
        try:
            await page.wait_for_timeout(500)
        except Exception:
            pass

    async def _format_action_result(self, action_msg: str, session: str,
                                    return_state: bool = True, max_state_lines: int = 50) -> str:
        """Format an action result, optionally appending current compact page state."""
        if not return_state:
            return action_msg
        try:
            state_text = await self.get_state(session=session)
            lines = state_text.splitlines()
            header_idx = -1
            for i, l in enumerate(lines):
                if l.strip() == "---":
                    header_idx = i
                    break
            if header_idx != -1 and len(lines) - (header_idx + 1) > max_state_lines:
                header = lines[:header_idx + 1]
                elements = lines[header_idx + 1 : header_idx + 1 + max_state_lines]
                remaining = len(lines) - (header_idx + 1 + max_state_lines)
                elements.append(f"... ({remaining} more elements off-screen. Call browser_inspect(mode='dom_interactive') for the full list)")
                state_text = "\n".join(header + elements)
            return f"{action_msg}\n\n[Current Page State]:\n{state_text}"
        except Exception as e:
            log.warning("Failed to get auto-state after action: %s", e)
            return f"{action_msg}\n(Call browser_inspect(mode='dom_interactive') to inspect the page)"

    # -----------------------------------------------------------------------
    # Public API: Navigation
    # -----------------------------------------------------------------------

    async def open_page(self, url: str, session: str = "default",
                        headed: bool = False, return_state: bool = True,
                        mode: Optional[str] = None) -> str:
        """Open a URL in a session. Creates the session if it doesn't exist."""
        blocked = _blocked_url_reason(url)
        if blocked:
            return blocked
        target_url = _normalize_url(url)
        desired_user = (mode == "user") if (mode and mode != "auto") else None
        if session in self._sessions:
            sc = self._sessions[session]
            mode_mismatch = (desired_user is not None and desired_user != self._cdp_mode)
            if sc.page.is_closed():
                try:
                    sc.page = await sc.context.new_page()
                    await self._apply_stealth(sc.page)
                except Exception:
                    sc = await self._create_session(session, headed, initial_url=target_url, mode=mode)
            elif headed != self._headed or mode_mismatch:
                await self.shutdown()
                sc = await self._create_session(session, headed, initial_url=target_url, mode=mode)
        else:
            sc = await self._create_session(session, headed, initial_url=target_url, mode=mode)

        try:
            # If already on the target URL (e.g. launched with it via CLI), don't reload unnecessarily
            if sc.page.url.rstrip("/") != target_url.rstrip("/"):
                await sc.page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
                # Wait a bit for JS to render
                await sc.page.wait_for_timeout(1000)
        except Exception as e:
            fallback_part = f"\n[{self._last_fallback_notice}]" if self._last_fallback_notice else ""
            self._last_fallback_notice = None
            return f"Error navigating to {target_url}: {e}{fallback_part}"

        title = await sc.page.title()
        current_url = sc.page.url
        mode_label = "CDP (user browser)" if self._cdp_mode else "isolated (Playwright)"
        window_status = (
            "VISIBLE (window is open on user's desktop)"
            if self._headed
            else "HEADLESS (background process — no visible window on screen)"
        )
        msg_parts = [
            f"Opened: {current_url}",
            f"Title: {title}",
            f"Mode: {mode_label}",
            f"Window: {window_status}",
        ]
        if self._last_fallback_notice:
            msg_parts.append(f"[{self._last_fallback_notice}]")
            self._last_fallback_notice = None

        msg = "\n".join(msg_parts)
        return await self._format_action_result(msg, session, return_state=return_state)

    async def navigate(self, url: str, session: str = "default", return_state: bool = True) -> str:
        """Navigate the current session to a new URL."""
        blocked = _blocked_url_reason(url)
        if blocked:
            return blocked
        target_url = _normalize_url(url)
        sc = await self._get_session(session)
        try:
            await sc.page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
            await sc.page.wait_for_timeout(1000)
        except Exception as e:
            return f"Error navigating to {target_url}: {e}"

        title = await sc.page.title()
        window_status = (
            "VISIBLE (window is open on user's desktop)"
            if self._headed
            else "HEADLESS (background process — no visible window on screen)"
        )
        msg = f"Navigated to: {sc.page.url}\nTitle: {title}\nWindow: {window_status}"
        return await self._format_action_result(msg, session, return_state=return_state)

    async def go_back(self, session: str = "default", return_state: bool = True) -> str:
        """Go back in browser history."""
        sc = await self._get_session(session)
        await sc.page.go_back(wait_until="domcontentloaded", timeout=15000)
        title = await sc.page.title()
        msg = f"Went back to: {sc.page.url}\nTitle: {title}"
        return await self._format_action_result(msg, session, return_state=return_state)

    async def reload(self, session: str = "default", return_state: bool = True) -> str:
        """Reload the current page."""
        sc = await self._get_session(session)
        if sc.page.url in ("about:blank", "chrome://newtab/", "chrome://new-tab-page/"):
            alive_content = [p for p in sc.context.pages if not p.is_closed() and p.url not in ("about:blank", "chrome://newtab/", "chrome://new-tab-page/")]
            if alive_content:
                sc.page = alive_content[-1]
                log.info("Reload: switched from blank tab to active content page: %s", sc.page.url)
        await sc.page.reload(wait_until="domcontentloaded", timeout=15000)
        title = await sc.page.title()
        msg = f"Reloaded: {sc.page.url}\nTitle: {title}"
        if sc.page.url == "about:blank":
            msg += "\n[Warning]: Current tab is about:blank. Use browser_open or browser_action(action='switch_tab') to target your web page."
        return await self._format_action_result(msg, session, return_state=return_state)

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
        
        try:
            scroll_depth = int(scroll_depth or 0)
        except (TypeError, ValueError):
            scroll_depth = 0

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
        self._element_metadata.clear()

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
                            self._element_metadata[el["idx"]] = {
                                "tag": el.get("tag", ""),
                                "role": el.get("tag", ""),
                                "name": el.get("label", ""),
                                "value": el.get("value", ""),
                                "disabled": el.get("disabled", False),
                                "frame": frame,
                            }
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

                if el.get("disabled"):
                    parts.append("[disabled]")

                if el.get("href"):
                    # Show shortened href
                    href = el["href"]
                    if len(href) > 60:
                        href = href[:57] + "..."
                    parts.append(f"-> {href}")

                lines.append(" ".join(parts))

        if query and total_filtered > 0:
            lines.append(f"\n(Note: {total_filtered} interactive elements were filtered out by query '{query}'. If you cannot find the element you need, call browser_inspect(mode='dom_interactive') without a query or with a different keyword.)")

        return "\n".join(lines)

    async def get_interactive_elements_raw(self, session: str = "default") -> List[Dict]:
        """Returns raw interactive elements for fast programmatic ranking and routing."""
        sc = await self._get_session(session)
        self._element_frames.clear()
        self._element_metadata.clear()
        combined_elements = []
        idx = 1
        try:
            for frame in sc.page.frames:
                try:
                    result = await frame.evaluate(_STATE_EXTRACTION_JS, {"startIdx": idx, "query": None})
                    if result:
                        frame_elements = result.get("elements", [])
                        for el in frame_elements:
                            self._element_frames[el["idx"]] = frame
                            self._element_metadata[el["idx"]] = {
                                "tag": el.get("tag", ""),
                                "role": el.get("tag", ""),
                                "name": el.get("label", ""),
                                "value": el.get("value", ""),
                                "disabled": el.get("disabled", False),
                                "frame": frame,
                            }
                            combined_elements.append(el)
                        idx = result.get("nextIdx", idx)
                except Exception:
                    pass
        except Exception as e:
            log.warning("get_interactive_elements_raw error: %s", e)
        return combined_elements

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

        try:
            scroll_depth = int(scroll_depth or 0)
        except (TypeError, ValueError):
            scroll_depth = 0

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
        self._element_metadata.clear()

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
                c_idx = idx_counter[0]
                candidates.append({
                    "idx": c_idx,
                    "role": role,
                    "name": name,
                    "value": value,
                    "props": props,
                    "depth": depth,
                    "backend_dom_node_id": backend_id,
                })
                self._element_metadata[c_idx] = {
                    "role": role,
                    "name": name,
                    "value": value,
                    "tag": "",
                }
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
        if len(taggable) > 250:
            taggable = taggable[:250]

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
                        resolve_result = await asyncio.wait_for(
                            client.send(
                                "DOM.resolveNode",
                                {"backendNodeId": cand["backend_dom_node_id"]},
                            ),
                            timeout=1.0,
                        )
                        object_id = resolve_result.get("object", {}).get("objectId")
                        if not object_id:
                            return None
                        await asyncio.wait_for(
                            client.send("Runtime.callFunctionOn", {
                                "objectId": object_id,
                                "functionDeclaration": (
                                    "function(idx) { "
                                    "this.setAttribute('data-argent-idx', String(idx)); "
                                    "return this.getAttribute('data-argent-idx'); "
                                    "}"
                                ),
                                "arguments": [{"value": cand["idx"]}],
                                "returnByValue": True,
                            }),
                            timeout=1.0,
                        )
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
                        if fi in self._element_metadata:
                            self._element_metadata[fi]["frame"] = frame
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

    async def _click_locator(self, locator, sc, force: bool = False) -> str:
        """Click a locator with cascading resilient fallbacks for overlays and pointer interception."""
        tag = ""
        try:
            tag = (await locator.evaluate("el => (el.tagName || '').toLowerCase()")) or ""
        except Exception:
            pass

        if tag == "option":
            res = await locator.evaluate("""el => {
                const select = el.closest('select');
                if (select) {
                    select.value = el.value;
                    select.dispatchEvent(new Event('input', { bubbles: true }));
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                    return el.text || el.value;
                }
                return null;
            }""")
            await self._wait_for_page_settle(sc.page)
            if res is not None:
                return f"Selected option '{res}' in dropdown."
            return "Error: Could not find enclosing <select> for option."

        # Level 1: Standard actionable click (or direct force click if force=True requested)
        err_l1 = None
        if force:
            try:
                await locator.click(timeout=3000, force=True)
                await self._wait_for_page_settle(sc.page)
                return ""
            except Exception as e:
                err_l1 = e
        else:
            try:
                await locator.click(timeout=2500, force=False)
                await self._wait_for_page_settle(sc.page)
                return ""
            except Exception as e:
                err_l1 = e
                log.debug("Standard click failed (%s), escalating to force click", e)

        # Level 2: Force click fallback (bypasses pointer interception checks)
        err_l2 = None
        try:
            await locator.click(timeout=2000, force=True)
            await self._wait_for_page_settle(sc.page)
            return ""
        except Exception as e:
            err_l2 = e
            log.debug("Force click failed (%s), escalating to DOM JS dispatch click", e)

        # Level 3: DOM synthetic MouseEvent dispatch & native el.click()
        try:
            js_res = await locator.evaluate("""el => {
                if (!el) return false;
                const opts = { bubbles: true, cancelable: true, view: window, buttons: 1 };
                el.dispatchEvent(new MouseEvent('mousedown', opts));
                el.dispatchEvent(new MouseEvent('mouseup', opts));
                el.dispatchEvent(new MouseEvent('click', opts));
                if (typeof el.click === 'function') {
                    el.click();
                }
                return true;
            }""")
            if js_res:
                await self._wait_for_page_settle(sc.page)
                return ""
        except Exception as e:
            log.debug("DOM JS click failed: %s", e)

        # Level 4: Container / child delegation click
        try:
            delegated = await locator.evaluate("""el => {
                if (!el) return false;
                const clickable = el.closest('[role="button"], [role="link"], li, tr, [class*="item"], [class*="row"]') || el.firstElementChild;
                if (clickable && clickable !== el) {
                    const opts = { bubbles: true, cancelable: true, view: window, buttons: 1 };
                    clickable.dispatchEvent(new MouseEvent('mousedown', opts));
                    clickable.dispatchEvent(new MouseEvent('mouseup', opts));
                    clickable.dispatchEvent(new MouseEvent('click', opts));
                    if (typeof clickable.click === 'function') {
                        clickable.click();
                    }
                    return true;
                }
                return false;
            }""")
            if delegated:
                await self._wait_for_page_settle(sc.page)
                return ""
        except Exception as e:
            log.debug("Delegated click failed: %s", e)

        # If all resilient levels failed, raise the most informative error
        raise err_l2 or err_l1 or RuntimeError("Failed to click element across all resilient strategies.")

    async def _fill_locator(self, locator, text: str, sc) -> str:
        """Fill an input locator or select option if target is a <select> element."""
        tag = ""
        try:
            tag = (await locator.evaluate("el => (el.tagName || '').toLowerCase()")) or ""
        except Exception:
            pass

        if tag == "select":
            res = await locator.evaluate("""(select, val) => {
                let matched = null;
                for (let opt of select.options) {
                    if (opt.value === val || opt.text.trim() === val.trim() || opt.value.toLowerCase() === val.toLowerCase() || opt.text.trim().toLowerCase() === val.trim().toLowerCase()) {
                        select.value = opt.value;
                        matched = opt.text || opt.value;
                        break;
                    }
                }
                if (matched !== null) {
                    select.dispatchEvent(new Event('input', { bubbles: true }));
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                    return matched;
                }
                return null;
            }""", text)
            await self._wait_for_page_settle(sc.page)
            if res is not None:
                return f"Selected option '{res}' in <select>."
            else:
                return f"Warning: Option '{text}' not found in <select> dropdown."

        # Safe focus / activation prior to fill (resilient to floating label overlays)
        try:
            await locator.click(timeout=2000, force=False)
        except Exception:
            try:
                await locator.click(timeout=1000, force=True)
            except Exception:
                try:
                    await locator.focus(timeout=1000)
                except Exception:
                    pass

        try:
            await locator.fill(str(text), timeout=3000)
        except Exception:
            # Fallback to direct DOM value assignment with React/Vue input event dispatch
            await locator.evaluate("""(el, val) => {
                if (typeof el.focus === 'function') el.focus();
                el.value = val;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }""", str(text))

        await self._wait_for_page_settle(sc.page)
        return ""

    async def _resolve_target_locator(self, sc, index: int = None, selector: str = None,
                                      text: str = None) -> tuple[Optional[Any], Optional[Any], str, str]:
        """Resolve a target element across index, selector, or text with resilient recovery.

        Returns:
            (locator, frame, display_name, recovery_note)
            If locator is None, recovery_note contains an error message.
        """
        if index is not None:
            try:
                index = int(index)
            except (TypeError, ValueError):
                return None, sc.page, f"[{index}]", f"Error: Invalid index {index!r}."

            frame = self._element_frames.get(index, sc.page)
            el = frame.locator(f'[data-argent-idx="{index}"]')
            try:
                if await el.count() > 0:
                    return el.first, frame, f"element [{index}]", ""
            except Exception:
                pass

            # Smart recovery: index missing (DOM re-rendered or indices shifted)
            meta = self._element_metadata.get(index)
            if meta:
                name = (meta.get("name") or "").strip()
                role = (meta.get("role") or "").strip()
                # 1. Try finding by role and name
                if role and name and role.lower() in ("button", "link", "checkbox", "radio", "textbox", "combobox", "tab", "menuitem", "heading"):
                    try:
                        by_role = frame.get_by_role(role.lower(), name=name)
                        if await by_role.count() > 0:
                            log.info("Recovered element [%d] via role '%s' and name '%s'", index, role, name)
                            return by_role.first, frame, f"element [{index}]", f" (recovered by role '{role}' and name '{name}')"
                    except Exception:
                        pass
                # 2. Try finding by text
                if name:
                    try:
                        by_text = frame.get_by_text(name, exact=False)
                        if await by_text.count() > 0:
                            log.info("Recovered element [%d] via text '%s'", index, name)
                            return by_text.first, frame, f"element [{index}]", f" (recovered by text '{name}')"
                    except Exception:
                        pass
                return None, frame, f"element [{index}]", (
                    f"Error: Element [{index}] ('{name}') not found. "
                    "The page appears to have re-rendered. Call browser_inspect(mode='dom_interactive') for fresh indices, "
                    f"or target by text='{name}'."
                )

            return None, frame, f"element [{index}]", (
                f"Error: Element [{index}] not found. "
                "The page may have changed — call browser_inspect(mode='dom_interactive') to get fresh indices."
            )

        elif selector is not None:
            target_locator = None
            target_frame = sc.page
            for frame in sc.page.frames:
                try:
                    el = frame.locator(selector)
                    if await el.count() > 0:
                        target_locator = el.first
                        target_frame = frame
                        break
                except Exception:
                    pass
            if not target_locator:
                target_locator = sc.page.locator(selector).first
                target_frame = sc.page
                try:
                    if await target_locator.count() == 0:
                        return None, target_frame, f"selector '{selector}'", f"Error: Element matching selector '{selector}' not found."
                except Exception:
                    pass
            return target_locator, target_frame, f"element matching selector '{selector}'", ""

        elif text is not None:
            target_locator = None
            target_frame = sc.page
            for frame in sc.page.frames:
                try:
                    el = frame.get_by_text(text, exact=False)
                    if await el.count() > 0:
                        target_locator = el.first
                        target_frame = frame
                        break
                except Exception:
                    pass
            if not target_locator:
                target_locator = sc.page.get_by_text(text, exact=False).first
                target_frame = sc.page
                try:
                    if await target_locator.count() == 0:
                        return None, target_frame, f"text '{text}'", f"Error: Element matching text '{text}' not found."
                except Exception:
                    pass
            return target_locator, target_frame, f"element matching text '{text}'", ""

        else:
            return None, sc.page, "none", "Error: You must specify index, selector, or text."

    async def click(self, index: int = None, selector: str = None, text: str = None,
                    session: str = "default", force: bool = False, return_state: bool = True) -> str:
        """Click element by its state index, CSS selector, or visible text."""
        sc = await self._get_session(session)
        locator, frame, display_name, note = await self._resolve_target_locator(sc, index=index, selector=selector, text=text)
        if not locator:
            return note
        try:
            msg = await self._click_locator(locator, sc, force=force)
            final_msg = msg if msg else f"Clicked {display_name}{note}."
            return await self._format_action_result(final_msg, session, return_state=return_state)
        except Exception as e:
            return f"Error clicking {display_name}: {e}"

    async def switch_tab(self, index: int, session: str = "default") -> str:
        """Switch the active tab/page of a named session by its 1-based index."""
        sc = await self._get_session(session)
        pages = sc.context.pages
        if not pages:
            return "Error: No open tabs in browser context."
        
        try:
            index = int(index)
        except (TypeError, ValueError):
            return f"Error: Invalid tab index {index!r}. Must be an integer."
        
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

    async def list_tabs(self, session: str = "default") -> str:
        """List all open tabs in the browser context with their 1-based index, title, and URL."""
        sc = await self._get_session(session)
        pages = [p for p in sc.context.pages if not p.is_closed()]
        if not pages:
            return "No open tabs in browser session."
        
        lines = [f"Open browser tabs ({len(pages)}):"]
        for idx, page in enumerate(pages, 1):
            is_active = " (Active)" if page == sc.page else ""
            title = "?"
            try:
                title = await page.title()
            except Exception:
                pass
            lines.append(f"[{idx}]{is_active} {page.url} (\"{title}\")")
        return "\n".join(lines)

    async def close_tab(self, index: Optional[int] = None, text: Optional[str] = None, session: str = "default") -> str:
        """Close a specific browser tab by 1-based index, URL/title match, or the currently active tab."""
        sc = await self._get_session(session)
        pages = [p for p in sc.context.pages if not p.is_closed()]
        if not pages:
            return "Error: No open tabs to close."

        target_page = None
        target_desc = ""

        # 1. Match by 1-based integer index
        if index is not None:
            try:
                idx = int(index)
                if 1 <= idx <= len(pages):
                    target_page = pages[idx - 1]
                    target_desc = f"tab [{idx}]"
                else:
                    return f"Error: Invalid tab index {idx}. Available indices: 1-{len(pages)}."
            except (TypeError, ValueError):
                return f"Error: Invalid tab index {index!r}. Must be an integer."

        # 2. Match by URL or title substring
        elif text:
            query = text.strip().lower()
            # Try to parse numeric string like "2" as index first
            if query.isdigit():
                try:
                    idx = int(query)
                    if 1 <= idx <= len(pages):
                        target_page = pages[idx - 1]
                        target_desc = f"tab [{idx}]"
                except (TypeError, ValueError):
                    pass
            if not target_page:
                for idx, page in enumerate(pages, 1):
                    p_title = ""
                    try:
                        p_title = await page.title()
                    except Exception:
                        pass
                    if query in page.url.lower() or query in p_title.lower():
                        target_page = page
                        target_desc = f"tab [{idx}] matching '{text}'"
                        break
            if not target_page:
                return f"Error: No tab found matching '{text}'. Call browser_inspect(mode='tabs') to view open tabs."

        # 3. Default: close active page
        else:
            target_page = sc.page if not sc.page.is_closed() else pages[-1]
            idx = pages.index(target_page) + 1 if target_page in pages else 1
            target_desc = f"active tab [{idx}]"

        closed_url = target_page.url
        try:
            await target_page.close()
        except Exception as e:
            return f"Error closing {target_desc}: {e}"

        # Reassign active page if the closed page was the active one
        remaining = [p for p in sc.context.pages if not p.is_closed()]
        if remaining:
            if target_page == sc.page:
                sc.page = remaining[-1]
                try:
                    await sc.page.bring_to_front()
                except Exception:
                    pass
            new_title = "?"
            try:
                new_title = await sc.page.title()
            except Exception:
                pass
            return f"Closed {target_desc} ({closed_url}). Active tab is now: {sc.page.url} (\"{new_title}\"). {len(remaining)} tab(s) remaining."
        else:
            # If all tabs closed, create a blank page so session stays alive
            sc.page = await sc.context.new_page()
            return f"Closed {target_desc} ({closed_url}). All tabs were closed; opened new blank tab."


    async def fill_input(self, index: int = None, text: str = "", selector: str = None,
                         session: str = "default", return_state: bool = True) -> str:
        """Clear and fill text into an input element by its state index or CSS selector."""
        sc = await self._get_session(session)
        text_str = "" if text is None else str(text)
        locator, frame, display_name, note = await self._resolve_target_locator(sc, index=index, selector=selector)
        if not locator:
            return note
        try:
            msg = await self._fill_locator(locator, text_str, sc)
            final_msg = f"{msg} ({display_name}{note})" if msg else f"Filled {display_name}{note} with text."
            return await self._format_action_result(final_msg, session, return_state=return_state)
        except Exception as e:
            return f"Error filling {display_name}: {e}"

    async def select_option(self, index: int = None, selector: str = None,
                            value: str = None, label: str = None,
                            session: str = "default", return_state: bool = True) -> str:
        """Select an option in a <select> dropdown by index, selector, value, or visible label."""
        sc = await self._get_session(session)
        locator, frame, display_name, note = await self._resolve_target_locator(sc, index=index, selector=selector)
        if not locator:
            return note
        target_locator = locator

        try:
            tag = (await target_locator.evaluate("el => (el.tagName || '').toLowerCase()")) or ""
            if tag == "option":
                res = await target_locator.evaluate("""el => {
                    const select = el.closest('select');
                    if (select) {
                        select.value = el.value;
                        select.dispatchEvent(new Event('input', { bubbles: true }));
                        select.dispatchEvent(new Event('change', { bubbles: true }));
                        return el.text || el.value;
                    }
                    return null;
                }""")
                await self._wait_for_page_settle(sc.page)
                if res is not None:
                    final_msg = f"Selected option '{res}'."
                    return await self._format_action_result(final_msg, session, return_state=return_state)
                return "Error: Could not find enclosing <select> for option."
            elif tag == "select":
                res = await target_locator.evaluate("""(select, [val, lbl]) => {
                    let matched = null;
                    for (let opt of select.options) {
                        const valMatch = val && (opt.value === val || opt.value.toLowerCase() === val.toLowerCase());
                        const lblMatch = lbl && (opt.text.trim() === lbl.trim() || opt.text.trim().toLowerCase() === lbl.trim().toLowerCase());
                        if (valMatch || lblMatch || (!val && !lbl)) {
                            select.value = opt.value;
                            matched = opt.text || opt.value;
                            break;
                        }
                    }
                    if (matched !== null) {
                        select.dispatchEvent(new Event('input', { bubbles: true }));
                        select.dispatchEvent(new Event('change', { bubbles: true }));
                        return matched;
                    }
                    return null;
                }""", [value, label])
                await self._wait_for_page_settle(sc.page)
                if res is not None:
                    final_msg = f"Selected option '{res}' in <select>."
                    return await self._format_action_result(final_msg, session, return_state=return_state)
                return f"Error: No matching option found in <select> for value='{value}', label='{label}'."
            else:
                kwargs = {}
                if value is not None:
                    kwargs["value"] = value
                if label is not None:
                    kwargs["label"] = label
                await target_locator.select_option(**kwargs)
                await self._wait_for_page_settle(sc.page)
                final_msg = f"Selected option in '{selector or index}'."
                return await self._format_action_result(final_msg, session, return_state=return_state)
        except Exception as e:
            return f"Error selecting option: {e}"

    async def get_select_options(self, indices: list, session: str = "default") -> Dict[int, List[str]]:
        """Visible option texts of the <select> elements with the given state indices ({index: [text, ...]}); the indices must come from
        the last ``get_interactive_elements_raw`` / ``get_state`` snapshot. Elements that are not a <select> are left out."""
        sc = await self._get_session(session)
        out: Dict[int, List[str]] = {}
        for idx in indices:
            try:
                locator, _frame, _name, _note = await self._resolve_target_locator(sc, index=idx)
                if not locator:
                    continue
                opts = await locator.evaluate(
                    "el => (el.tagName || '').toLowerCase() === 'select' "
                    "? Array.from(el.options).slice(0, 300).map(o => (o.text || '').trim()) : null")
            except Exception as e:  # one broken element must not lose the others
                log.debug("get_select_options [%s]: %s", idx, e)
                continue
            if opts:
                out[int(idx)] = [str(o) for o in opts]
        return out

    # The repeated containers of a page (posts, cards, rows, comments) are found the same way for reading them (get_text_blocks)
    # and for telling identical buttons apart (get_block_contexts), so a post's context key is the same at scan and at click time.
    # Order of evidence: real <article>s; the largest run of same-looking siblings (custom elements and anonymous divs included);
    # list items; table rows; class-name guesses; at last plain paragraphs.
    _BLOCKS_LIB_JS = """
        const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
        const visible = el => {
            const r = el.getBoundingClientRect();
            if (r.width < 2 || r.height < 2) return false;
            const st = getComputedStyle(el);
            return st.visibility !== 'hidden' && st.display !== 'none';
        };
        const chrome = el => !!el.closest('nav, footer, [role="navigation"], [role="banner"], [role="contentinfo"]');
        function argentBlocks(MIN) {
            const usable = el => visible(el) && !chrome(el) && norm(el.innerText).length >= MIN;
            // a wrapper that holds two or more other candidates is a container of posts, not a post
            const leaves = els => {
                els = els.slice(0, 2000);
                const set = new Set(els);
                return els.filter(el => {
                    let inner = 0;
                    for (const o of set) { if (o !== el && el.contains(o)) { inner++; if (inner >= 2) return false; } }
                    return true;
                });
            };
            const bySelector = sel => leaves(Array.from(document.querySelectorAll(sel)).filter(usable));
            const runs = () => {
                const found = [];  // every run of 3+ same-looking siblings that carries text
                const parents = [];
                document.querySelectorAll('body *').forEach(el => { if (el.children.length >= 3) parents.push(el); });
                const cls = x => Array.from(x.classList).filter(n => !/\\d/.test(n));  // hashed / numbered class names say nothing
                for (const p of parents) {
                    const byTag = new Map();
                    for (const c of p.children) {
                        if (!c.tagName || /^(SCRIPT|STYLE|LINK|META|BR|HR|NOSCRIPT)$/.test(c.tagName)) continue;
                        if (!byTag.has(c.tagName)) byTag.set(c.tagName, []);
                        byTag.get(c.tagName).push(c);
                    }
                    const groups = [];
                    for (const g of byTag.values()) {
                        if (g.length < 3) continue;
                        // same-looking = shares a class that most of the siblings carry (flags such as "minor" / "new" vary per row)
                        const freq = new Map();
                        for (const c of g) for (const n of new Set(cls(c))) freq.set(n, (freq.get(n) || 0) + 1);
                        const common = new Set(Array.from(freq).filter(kv => kv[1] >= g.length * 0.6).map(kv => kv[0]));
                        groups.push(common.size ? g.filter(c => cls(c).some(n => common.has(n))) : g.filter(c => cls(c).length === 0));
                        // ... and the strict reading: exactly the same class list (rows of a table that alternate between kinds)
                        const exact = new Map();
                        for (const c of g) {
                            const k = cls(c).sort().join('.');
                            if (!exact.has(k)) exact.set(k, []);
                            exact.get(k).push(c);
                        }
                        for (const e of exact.values()) { if (e.length >= 3) groups.push(e); }
                    }
                    for (const g of groups) {
                        if (g.length < 3) continue;
                        const ok = g.filter(usable);
                        if (ok.length < 3 || ok.length < g.length * 0.6) continue;
                        found.push({els: ok, score: ok.reduce((a, c) => a + Math.min(norm(c.innerText).length, 600), 0)});
                    }
                }
                found.sort((a, b) => b.score - a.score);
                found.length = Math.min(found.length, 60);
                // a run whose members each hold a finer run (day sections of a change list, thread blocks of comments) is a
                // container of the real items, not the items: the finest granularity wins, then the most text
                const holdsFiner = (r) => {
                    const mine = new Set(r.els);
                    return found.some(s => {
                        if (s === r || s.els.some(e => mine.has(e))) return false;
                        let hit = 0;
                        for (const m of r.els) { if (s.els.filter(e => m.contains(e)).length >= 2) hit++; }
                        return hit * 2 >= r.els.length;
                    });
                };
                let best = [], bestScore = 0;
                for (const r of found) {
                    if (holdsFiner(r)) continue;
                    if (r.score > bestScore) { bestScore = r.score; best = r.els; }
                }
                return best;
            };
            const tries = [
                () => bySelector('article'),
                () => bySelector('[role="article"]'),
                runs,
                () => bySelector('li'),
                () => bySelector('tr'),
                () => bySelector('[class*="post" i], [class*="card" i], [class*="item" i], [class*="comment" i], [class*="entry" i], [class*="tweet" i], [class*="message" i]'),
                () => leaves(Array.from(document.querySelectorAll('p, blockquote')).filter(usable)),
            ];
            for (const t of tries) {
                const b = t();
                if (b.length >= 2) return b;
            }
            return [];
        }
    """

    _TEXT_BLOCKS_JS = """
    (opts) => {
        __LIB__
        const MAXC = opts.maxChars, LIMIT = opts.limit;
        const out = [];
        for (const el of argentBlocks(opts.minChars)) {
            if (out.length >= LIMIT) break;
            const text = norm(el.innerText);
            let a = el.querySelector('h1 a[href], h2 a[href], h3 a[href], h4 a[href]');
            if (!a) {
                // the post's own link is the one that carries its title: the longest caption, not the first link (an upvote arrow)
                let bestLen = 5;
                for (const x of el.querySelectorAll('a[href]')) {
                    const h = x.getAttribute('href') || '';
                    if (!h || h[0] === '#' || /^javascript:/i.test(h)) continue;
                    const n = norm(x.innerText).length;
                    if (n > bestLen) { bestLen = n; a = x; }
                }
                if (!a) {
                    a = Array.from(el.querySelectorAll('a[href]')).find(x => {
                        const h = x.getAttribute('href') || '';
                        return h && h[0] !== '#' && !/^javascript:/i.test(h);
                    }) || null;
                }
            }
            const idxs = [];
            if (el.hasAttribute && el.hasAttribute('data-argent-idx')) idxs.push(parseInt(el.getAttribute('data-argent-idx'), 10));
            el.querySelectorAll('[data-argent-idx]').forEach(n => {
                if (idxs.length < 16) idxs.push(parseInt(n.getAttribute('data-argent-idx'), 10));
            });
            out.push({text: text.slice(0, MAXC), href: a ? a.href : '', idxs: idxs.filter(n => !isNaN(n)),
                      y: Math.round(el.getBoundingClientRect().top + window.scrollY)});
        }
        return out;
    }
    """

    _BLOCK_CONTEXT_JS = """
    (opts) => {
        __LIB__
        const out = {};
        for (const b of argentBlocks(opts.minChars)) {
            const key = norm(b.innerText).slice(0, 80);
            const own = (b.hasAttribute && b.hasAttribute('data-argent-idx')) ? [b] : [];
            for (const n of own.concat(Array.from(b.querySelectorAll('[data-argent-idx]')))) {
                out[n.getAttribute('data-argent-idx')] = key;
            }
        }
        return out;
    }
    """

    async def get_text_blocks(self, session: str = "default", max_blocks: int = 200, min_chars: int = 25, max_chars: int = 600) -> List[Dict]:
        """Repeated text containers of the page (posts, cards, rows, comments): [{text, href, idxs, y}].

        ``idxs`` are the state indices of the interactive elements INSIDE the block: call ``get_interactive_elements_raw`` first, it
        tags the DOM with them. Blocks of all frames are returned, in page order per frame."""
        sc = await self._get_session(session)
        js = self._TEXT_BLOCKS_JS.replace("__LIB__", self._BLOCKS_LIB_JS)
        out: List[Dict] = []
        for frame in sc.page.frames:
            try:
                rows = await frame.evaluate(js, {"minChars": int(min_chars), "maxChars": int(max_chars), "limit": int(max_blocks)})
            except Exception as e:  # a detached / cross-origin frame must not lose the others
                log.debug("get_text_blocks frame: %s", e)
                continue
            out.extend(rows or [])
            if len(out) >= max_blocks:
                break
        return out[: int(max_blocks)]

    async def get_block_contexts(self, session: str = "default", min_chars: int = 25) -> Dict[int, str]:
        """{state index: first 80 characters of the post / row that contains the element}, for the elements tagged by the last
        ``get_interactive_elements_raw``. Tells apart identical buttons ("Delete", "Like") of different posts."""
        sc = await self._get_session(session)
        js = self._BLOCK_CONTEXT_JS.replace("__LIB__", self._BLOCKS_LIB_JS)
        out: Dict[int, str] = {}
        for frame in sc.page.frames:
            try:
                rows = await frame.evaluate(js, {"minChars": int(min_chars)})
            except Exception as e:
                log.debug("get_block_contexts frame: %s", e)
                continue
            for k, v in (rows or {}).items():
                try:
                    out[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
        return out

    async def scroll(self, direction: str = "down", amount: int = 500, index: int = None,
                     selector: str = None, session: str = "default", return_state: bool = True) -> str:
        """Scroll the page or scroll a specific element into view."""
        sc = await self._get_session(session)
        
        if index is not None or selector is not None:
            locator, frame, display_name, note = await self._resolve_target_locator(sc, index=index, selector=selector)
            if not locator:
                return note
            try:
                await locator.scroll_into_view_if_needed(timeout=5000)
                await self._wait_for_page_settle(sc.page)
                final_msg = f"Scrolled {display_name}{note} into view."
                return await self._format_action_result(final_msg, session, return_state=return_state)
            except Exception as e:
                return f"Error scrolling {display_name}: {e}"
        else:
            try:
                amount = abs(int(amount))
            except (TypeError, ValueError):
                amount = 500

            dir_lower = (direction or "down").lower()
            if dir_lower == "up":
                dx, dy = 0, -amount
            elif dir_lower == "left":
                dx, dy = -amount, 0
            elif dir_lower == "right":
                dx, dy = amount, 0
            else:
                dx, dy = 0, amount

            # JS scroll triggers IntersectionObserver and scroll event listeners
            await sc.page.evaluate(f"window.scrollBy({dx}, {dy})")
            await self._wait_for_page_settle(sc.page)
            # Report actual scroll position for context
            scroll_x = await sc.page.evaluate("window.scrollX")
            scroll_y = await sc.page.evaluate("window.scrollY")
            page_height = await sc.page.evaluate(
                "Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)"
            )
            final_msg = f"Scrolled {dir_lower} by {amount}px. Position: ({scroll_x}, {scroll_y}) / height: {page_height}px."
            return await self._format_action_result(final_msg, session, return_state=return_state)

    async def press_key(self, key: str, session: str = "default", return_state: bool = True) -> str:
        """Press a keyboard key (Enter, Escape, Tab, etc.)."""
        sc = await self._get_session(session)
        try:
            await sc.page.keyboard.press(key)
            await self._wait_for_page_settle(sc.page)
            final_msg = f"Pressed key: {key}"
            return await self._format_action_result(final_msg, session, return_state=return_state)
        except Exception as e:
            return f"Error pressing key '{key}': {e}"

    async def fill_form(self, fields: list[dict] | str, submit_index: int = None,
                        submit_selector: str = None, session: str = "default",
                        return_state: bool = True) -> str:
        """Batch-fill multiple form fields and optionally submit the form in one step."""
        sc = await self._get_session(session)
        if isinstance(fields, str):
            try:
                fields = json.loads(fields)
            except Exception as e:
                return f"Error: Invalid JSON for fields: {e}"
        if not isinstance(fields, list):
            return "Error: fields must be a list of field specifications."

        filled_summary = []
        for item in fields:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            sel = item.get("selector")
            val = item.get("value", "")
            ftype = str(item.get("type", "input")).lower()

            if idx is None and sel is None:
                continue

            if ftype == "select":
                res = await self.select_option(index=idx, selector=sel, value=val, session=session, return_state=False)
                filled_summary.append(f"Select [{idx if idx is not None else sel}] -> {res}")
            else:
                res = await self.fill_input(text=val, index=idx, selector=sel, session=session, return_state=False)
                filled_summary.append(f"Field [{idx if idx is not None else sel}] -> {res}")

        if submit_index is not None or submit_selector is not None:
            click_res = await self.click(index=submit_index, selector=submit_selector, session=session, return_state=False)
            filled_summary.append(f"Submit [{submit_index if submit_index is not None else submit_selector}] -> {click_res}")

        final_msg = "Form filled:\n" + "\n".join(filled_summary)
        return await self._format_action_result(final_msg, session, return_state=return_state)

    async def upload_file(self, file_path: str | list, index: int = None, selector: str = None,
                          session: str = "default", return_state: bool = True) -> str:
        """Upload one or more local files to a file input or trigger a file chooser dialog."""
        raw_paths = []
        if isinstance(file_path, (str, Path)):
            s = str(file_path).strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    import json
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        raw_paths = parsed
                except Exception:
                    try:
                        import ast
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, list):
                            raw_paths = parsed
                    except Exception:
                        raw_paths = [file_path]
            else:
                raw_paths = [file_path]
        else:
            raw_paths = list(file_path or [])

        resolved_paths = []
        for p in raw_paths:
            rp = Path(p).expanduser().resolve()
            if not rp.exists():
                return f"Error: Local file '{p}' does not exist."
            resolved_paths.append(rp)

        if not resolved_paths:
            return "Error: No valid file paths provided for upload."

        sc = await self._get_session(session)

        file_payload = [str(rp) for rp in resolved_paths]
        names_str = ", ".join(f"'{rp.name}'" for rp in resolved_paths)

        target_locator = None
        display_name = "input[type=file]"
        note = ""
        if index is not None or selector is not None:
            locator, frame, display_name, note = await self._resolve_target_locator(sc, index=index, selector=selector)
            if not locator:
                return note
            target_locator = locator
        else:
            file_input = sc.page.locator('input[type="file"]')
            if await file_input.count() > 0:
                target_locator = file_input.first
            else:
                return "Error: You must specify either index or selector for file upload (no <input type='file'> found on page)."

        # Strategy 1: Direct set_input_files on standard file input
        direct_err = None
        try:
            await target_locator.set_input_files(file_payload, timeout=3000)
            await self._wait_for_page_settle(sc.page)
            final_msg = f"Uploaded {names_str} to {display_name}{note}."
            return await self._format_action_result(final_msg, session, return_state=return_state)
        except Exception as e:
            direct_err = e
            log.debug("Direct set_input_files failed on [%s]: %s; trying expect_file_chooser", index or selector, e)

        # Strategy 2: Intercept file chooser dialog upon clicking button/element
        fc_err = None
        try:
            async with sc.page.expect_file_chooser(timeout=5000) as fc_info:
                await target_locator.click(force=True, timeout=5000)
            file_chooser = await fc_info.value
            await file_chooser.set_files(file_payload)
            await self._wait_for_page_settle(sc.page)
            final_msg = f"Uploaded {names_str} via file chooser on {display_name}{note}."
            return await self._format_action_result(final_msg, session, return_state=return_state)
        except Exception as e:
            fc_err = e

        # Strategy 3: Nearby input[type="file"] fallback
        try:
            hidden_input = target_locator.locator('xpath=.//input[@type="file"] | ./following::input[@type="file"][1]').first
            if await hidden_input.count() > 0:
                await hidden_input.set_input_files(file_payload)
                await self._wait_for_page_settle(sc.page)
                final_msg = f"Uploaded {names_str} via nearby file input for {display_name}{note}."
                return await self._format_action_result(final_msg, session, return_state=return_state)
        except Exception:
            pass

        return f"Error uploading file: direct upload failed ({direct_err}), and file chooser click failed ({fc_err})"

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
        else:
            path_obj = Path(path).expanduser().resolve()
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path = str(path_obj)

        try:
            await sc.page.screenshot(path=path, full_page=full_page)
            return f"Screenshot saved to: {path}"
        except Exception as e:
            return f"Error taking screenshot: {e}"

    async def get_text(self, index: int = None, selector: str = None, text: str = None,
                       session: str = "default") -> str:
        """Get text content of a specific element, CSS selector, or the full page."""
        sc = await self._get_session(session)

        if index is not None or selector is not None or text is not None:
            try:
                locator, frame, display_name, note = await self._resolve_target_locator(sc, index=index, selector=selector, text=text)
                if not locator:
                    return note

                # If targeted by text, expand to nearest section/card/form container if the matched node is short
                if text:
                    try:
                        cur_text = await locator.inner_text()
                        if len(cur_text.strip()) < 60:
                            container = locator.locator(
                                "xpath=./ancestor-or-self::*[self::section or self::article or self::fieldset or self::form or @role='region' or @role='group' or contains(@class, 'card') or contains(@class, 'section')][1]"
                            )
                            if await container.count() > 0:
                                locator = container.first
                    except Exception:
                        pass

                extracted_text = await locator.inner_text()
                if len(extracted_text) > 15000:
                    extracted_text = extracted_text[:15000] + "\n... [Content Truncated]"
                return f"Content of {display_name}{note}:\n{extracted_text}"
            except Exception as e:
                target_desc = f"element [{index}]" if index is not None else (f"selector '{selector}'" if selector else f"text '{text}'")
                return f"Error getting text for {target_desc}: {e}"
        else:
            try:
                page_text = await sc.page.inner_text("body")
                # Truncate if too long
                if len(page_text) > 15000:
                    page_text = page_text[:15000] + "\n... [Content Truncated]"
                return page_text
            except Exception as e:
                return f"Error getting page text: {e}"

    async def get_markdown(self, selector: str = None, index: int = None, text: str = None,
                           session: str = "default") -> str:
        """Get page content (or a specific element) converted to Markdown."""
        sc = await self._get_session(session)
        try:
            if index is not None or selector is not None or text is not None:
                locator, frame, display_name, note = await self._resolve_target_locator(sc, index=index, selector=selector, text=text)
                if not locator:
                    return note

                if text:
                    try:
                        cur_text = await locator.inner_text()
                        if len(cur_text.strip()) < 60:
                            container = locator.locator(
                                "xpath=./ancestor-or-self::*[self::section or self::article or self::fieldset or self::form or @role='region' or @role='group' or contains(@class, 'card') or contains(@class, 'section')][1]"
                            )
                            if await container.count() > 0:
                                locator = container.first
                    except Exception:
                        pass

                html = await locator.inner_html()
            else:
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

    async def get_html(self, selector: str = None, index: int = None, text: str = None,
                       session: str = "default") -> str:
        """Get raw HTML of the page or a specific element matched by CSS selector, index, or text."""
        sc = await self._get_session(session)
        try:
            if index is not None or selector is not None or text is not None:
                locator, frame, display_name, note = await self._resolve_target_locator(sc, index=index, selector=selector, text=text)
                if not locator:
                    return note

                if text:
                    try:
                        cur_text = await locator.inner_text()
                        if len(cur_text.strip()) < 60:
                            container = locator.locator(
                                "xpath=./ancestor-or-self::*[self::section or self::article or self::fieldset or self::form or @role='region' or @role='group' or contains(@class, 'card') or contains(@class, 'section')][1]"
                            )
                            if await container.count() > 0:
                                locator = container.first
                    except Exception:
                        pass

                html = await locator.inner_html()
            else:
                html = await sc.page.content()
            if len(html) > 20000:
                html = html[:20000] + "\n... [Content Truncated]"
            return html
        except Exception as e:
            return f"Error getting page HTML: {e}"

    async def eval_js(self, expression: str,
                      session: str = "default",
                      timeout: float = 15.0,
                      return_state: bool = False) -> str:
        """Evaluate a JavaScript expression in the page context with safety bounds and smart serialization."""
        expr = (expression or "").strip()
        if not expr:
            return "Error: 'expression' cannot be empty."

        sc = await self._get_session(session)

        import re
        if re.search(r"\breturn\b", expr):
            eval_target = f"(async () => {{\n{expr}\n}})()"
        else:
            eval_target = expr

        try:
            timeout_sec = max(0.5, float(timeout)) if timeout else 15.0
        except (TypeError, ValueError):
            timeout_sec = 15.0

        try:
            handle = await asyncio.wait_for(
                sc.page.evaluate_handle(eval_target),
                timeout=timeout_sec
            )
        except asyncio.TimeoutError:
            return f"Error evaluating JS: Execution timed out after {timeout_sec}s. Ensure your script does not contain infinite loops or unresolved Promises."
        except Exception as e:
            return f"Error evaluating JS: {e}"

        try:
            element = handle.as_element()
            if element:
                tag = await element.evaluate("el => (el.tagName || '').toLowerCase()")
                el_id = await element.evaluate("el => el.id || ''")
                el_cls = await element.evaluate("el => el.className || ''")
                text = await element.evaluate("el => (el.textContent || '').trim().slice(0, 200)")
                outer = await element.evaluate("el => (el.outerHTML || '').slice(0, 300)")
                info = {
                    "__type": "Element",
                    "tagName": tag,
                    "id": el_id or None,
                    "className": el_cls or None,
                    "text": text,
                    "outerHTML": outer,
                }
                serialized = {k: v for k, v in info.items() if v is not None}
            else:
                is_node_collection = False
                try:
                    is_node_collection = await handle.evaluate("""h => {
                        return Boolean(h && (Array.isArray(h) || h instanceof NodeList || h instanceof HTMLCollection) && h.length > 0 && h[0] instanceof Node);
                    }""")
                except Exception:
                    pass

                if is_node_collection:
                    serialized = await handle.evaluate("""h => {
                        return Array.from(h).slice(0, 20).map(el => {
                            if (el instanceof Element) {
                                return {
                                    __type: "Element",
                                    tagName: (el.tagName || '').toLowerCase(),
                                    id: el.id || undefined,
                                    className: el.className || undefined,
                                    text: (el.textContent || '').trim().slice(0, 100)
                                };
                            }
                            return String(el);
                        });
                    }""")
                else:
                    serialized = await handle.json_value()
        except Exception as e:
            serialized = f"[Unserializable Result: {e}]"
        finally:
            try:
                await handle.dispose()
            except Exception:
                pass

        try:
            json_str = json.dumps(serialized, ensure_ascii=False, default=str)
        except Exception:
            json_str = str(serialized)

        MAX_OUTPUT = 15000
        if len(json_str) > MAX_OUTPUT:
            json_str = (
                f"[Output truncated at {MAX_OUTPUT} of {len(json_str)} chars. Narrow down your JS query or target specific properties]\n"
                f"{json_str[:MAX_OUTPUT]}\n..."
            )

        res_msg = f"JS result: {json_str}"
        if return_state:
            return await self._format_action_result(res_msg, session, return_state=True)
        return res_msg

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

    async def wait_for(self, text: str = None, selector: str = None, expression: str = None,
                       state: str = "visible", timeout: float = 15.0, session: str = "default",
                       return_state: bool = True) -> str:
        """Wait for an element or JavaScript expression to satisfy a state ('visible', 'attached', 'detached', 'hidden', 'enabled', 'disabled')."""
        if not text and not selector and not expression:
            return "Error: You must specify either 'text', 'selector', or 'expression' to wait for."

        valid_states = ("visible", "attached", "detached", "hidden", "enabled", "disabled")
        state_clean = (state or "visible").lower().strip()
        if state_clean not in valid_states:
            return f"Error: Invalid state '{state}'. Valid options: {', '.join(valid_states)}."

        try:
            timeout_ms = max(100, int(float(timeout) * 1000))
        except (TypeError, ValueError):
            timeout_ms = 15000

        sc = await self._get_session(session)

        # 1. JavaScript expression waiting (runs in browser context, eliminates LLM polling)
        if expression:
            try:
                await sc.page.wait_for_function(expression, timeout=timeout_ms)
                await self._wait_for_page_settle(sc.page)
                msg = f"JavaScript condition '{expression}' satisfied."
                return await self._format_action_result(msg, session, return_state=return_state)
            except Exception as e:
                err_str = str(e)
                if "Timeout" in err_str or "timed out" in err_str.lower():
                    return f"Timeout waiting {timeout}s for JavaScript expression '{expression}' to become truthy."
                return f"Error waiting for JavaScript expression '{expression}': {e}"

        if selector:
            target_locator = sc.page.locator(selector).first
            desc = f"selector '{selector}'"
        else:
            target_locator = sc.page.get_by_text(text, exact=False).first
            desc = f"text '{text}'"

        try:
            if state_clean in ("visible", "attached", "detached", "hidden"):
                await target_locator.wait_for(state=state_clean, timeout=timeout_ms)
            elif state_clean == "enabled":
                await target_locator.wait_for(state="visible", timeout=timeout_ms)
                start_t = time.monotonic()
                deadline = start_t + (timeout_ms / 1000.0)
                while time.monotonic() < deadline:
                    if not await target_locator.is_disabled():
                        break
                    await asyncio.sleep(0.15)
                else:
                    if await target_locator.is_disabled():
                        return f"Timeout waiting {timeout}s for element matching {desc} to become enabled."
            elif state_clean == "disabled":
                await target_locator.wait_for(state="visible", timeout=timeout_ms)
                start_t = time.monotonic()
                deadline = start_t + (timeout_ms / 1000.0)
                while time.monotonic() < deadline:
                    if await target_locator.is_disabled():
                        break
                    await asyncio.sleep(0.15)
                else:
                    if not await target_locator.is_disabled():
                        return f"Timeout waiting {timeout}s for element matching {desc} to become disabled."

            await self._wait_for_page_settle(sc.page)
            msg = f"Element matching {desc} reached state '{state_clean}'."
            return await self._format_action_result(msg, session, return_state=return_state)
        except Exception as e:
            err_str = str(e)
            if "Timeout" in err_str or "timed out" in err_str.lower():
                return f"Timeout waiting {timeout}s for element matching {desc} to become {state_clean}."
            return f"Error waiting for element matching {desc}: {e}"

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
        else:
            try:
                await sc.page.close()
            except Exception:
                pass

        # If no sessions left, shut down the browser entirely (only in non-CDP mode)
        if not self._sessions and not self._cdp_mode:
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
