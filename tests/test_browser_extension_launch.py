"""The isolated browser loads Stagehand's extension, keeps a visible window visible, and notices a dead browser.

* `--load-extension` did nothing: Playwright adds `--disable-extensions` to its defaults, so Stagehand found
  no extension and failed with "This Chrome build does not support Extensions.loadUnpacked".
* `browser_open` defaulted to headed=False, so any later call that did not mention it closed the window the
  person was watching and restarted the browser headless.
* `not browser.is_connected` (a method, not called) is always False: a crashed browser was never relaunched.
"""

import asyncio

import browser_engine
from browser_engine import _browser_alive, _extension_launch_options


def test_without_an_extension_nothing_changes():
    assert _extension_launch_options(None, headed=False) == {"args": []}


def test_the_default_disable_extensions_is_dropped():
    opts = _extension_launch_options("C:/ext", headed=True)
    assert opts["ignore_default_args"] == ["--disable-extensions"]
    assert "--load-extension=C:/ext" in opts["args"]
    assert "--disable-extensions-except=C:/ext" in opts["args"]
    assert "channel" not in opts                       # a window runs the full Chromium anyway


def test_headless_uses_the_full_chromium_new_headless_mode():
    assert _extension_launch_options("C:/ext", headed=False)["channel"] == "chromium"


def test_is_connected_is_called_not_just_looked_up():
    class Alive:
        def is_connected(self):
            return True

    class Dead:
        def is_connected(self):
            return False

    class Broken:
        def is_connected(self):
            raise RuntimeError("gone")

    assert _browser_alive(Alive()) is True
    assert _browser_alive(Dead()) is False
    assert _browser_alive(Broken()) is False


class _Page:
    def __init__(self):
        self.url = "about:blank"

    def is_closed(self):
        return False

    async def goto(self, url, **kw):
        self.url = url

    async def wait_for_timeout(self, ms):
        pass

    async def title(self):
        return "T"


class _Session:
    def __init__(self):
        self.page = _Page()


class _Browser:
    def is_connected(self):
        return True


def _engine(monkeypatch, headed_now):
    eng = browser_engine.BrowserEngine()
    eng._browser = _Browser()
    eng._headed = headed_now
    eng._sessions = {"default": _Session()}
    restarts = []

    async def shutdown():
        restarts.append("shutdown")

    async def create(session, headed, initial_url=None, mode=None):
        restarts.append(("create", headed))
        eng._headed = headed
        eng._sessions[session] = _Session()
        return eng._sessions[session]

    async def fmt(msg, session, return_state=True):
        return msg

    monkeypatch.setattr(eng, "shutdown", shutdown)
    monkeypatch.setattr(eng, "_create_session", create)
    monkeypatch.setattr(eng, "_format_action_result", fmt)
    return eng, restarts


def test_not_mentioning_headed_keeps_the_visible_window(monkeypatch):
    eng, restarts = _engine(monkeypatch, headed_now=True)
    out = asyncio.run(eng.open_page("https://example.com", headed=None))
    assert restarts == []
    assert "VISIBLE" in out


def test_asking_for_a_window_still_switches(monkeypatch):
    eng, restarts = _engine(monkeypatch, headed_now=False)
    asyncio.run(eng.open_page("https://example.com", headed=True))
    assert ("create", True) in restarts


def test_the_tool_passes_none_when_headed_is_not_given(monkeypatch):
    from tools import browser_tools
    seen = {}

    async def open_page(url, session, headed, return_state=True, mode=None):
        seen["headed"] = headed
        return "ok"

    monkeypatch.setattr(browser_tools.browser_engine, "open_page", open_page)
    monkeypatch.setattr(browser_tools.browser_engine, "run", lambda coro: asyncio.run(coro))
    browser_tools.browser_open("https://example.com")
    assert seen["headed"] is None
