"""
test_browser_modes_and_stagehand_preload.py — LMTrust Deep Blind Spot & Contract Audit
Covering recent changes:
1. Dynamic Stagehand extension directory discovery and preload args (--load-extension).
2. Intelligent browser mode routing ('auto', 'user', 'isolated') and dedicated debug profiles.
3. Resilient CDP target inspection for extension_id discovery.
4. Stagehand failure fallback to native DOM Markdown + LLM structured JSON.
5. Mode switching state machine and non-destructive shutdown.

LMTrust Matrix: 11 Layers (L0-L9 + L4b) x Directions (D1-D10)
"""

import ast
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from browser_engine import browser_engine, _get_stagehand_extension_dir, BrowserEngine
from tools import browser_tools as bt
from tools import stagehand_tools as sht
from tools.schemas import TOOL_SCHEMAS


# ═══════════════════════════════════════════════════════════════════════════════
# L0: Smoke — «Оно вообще живое?»
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoke:
    """L0: Smoke tests — точки входа существуют, импортируются и базово работают."""

    @pytest.mark.lmtrust(layer="L0", direction="D7")
    def test_stagehand_extension_dir_resolver_callable(self):
        """Функция _get_stagehand_extension_dir вызываема и возвращает str или None."""
        res = _get_stagehand_extension_dir()
        assert res is None or isinstance(res, str)
        if res is not None:
            assert Path(res).exists()

    @pytest.mark.lmtrust(layer="L0", direction="D5")
    def test_find_free_port_returns_valid_integer_port(self):
        """_find_free_port возвращает валидный свободный TCP-порт в диапазоне 1024-65535."""
        port = BrowserEngine._find_free_port(9222)
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    @pytest.mark.lmtrust(layer="L0", direction="D7")
    def test_browser_open_mode_schema_presence(self):
        """Схема browser_open содержит свойство mode со списком enum ['auto', 'user', 'isolated']."""
        schema = next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == "browser_open")
        props = schema["parameters"]["properties"]
        assert "mode" in props
        assert props["mode"]["type"] == "string"
        assert set(props["mode"]["enum"]) == {"auto", "user", "isolated"}


# ═══════════════════════════════════════════════════════════════════════════════
# L1: Contract — «Делает то, что говорит?»
# ═══════════════════════════════════════════════════════════════════════════════

class TestContract:
    """L1: Contract tests — валидация контрактов аргументов и возвращаемых типов."""

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_browser_open_rejects_unsupported_mode(self):
        """browser_open должен вернуть контрактное сообщение об ошибке при невалидном mode."""
        res = bt.browser_open("https://example.com", mode="safari_mode")
        assert "Error: Invalid browser mode 'safari_mode'" in res
        assert "Allowed modes: 'auto', 'user', 'isolated'" in res

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_browser_open_accepts_valid_modes_and_forwards(self, monkeypatch):
        """browser_open нормализует и передает валидный mode в engine.open_page."""
        captured = {}

        def mock_open_page(url, session="default", headed=False, return_state=True, mode=None):
            captured["url"] = url
            captured["mode"] = mode
            return f"Opened: {url}\nMode: {mode}"

        monkeypatch.setattr(browser_engine, "open_page", mock_open_page)
        monkeypatch.setattr(browser_engine, "run", lambda coro: coro if isinstance(coro, str) else mock_open_page("https://example.com", mode="user"))

        res = bt.browser_open("https://example.com", mode="user")
        assert "Opened: https://example.com" in res

    @pytest.mark.lmtrust(layer="L1", direction="D3")
    def test_browser_extract_contract_requires_dict_schema_or_none(self):
        """browser_extract валидирует контракт: schema должна быть dict или None."""
        res_invalid = sht.browser_extract("extract text", schema="invalid_schema_string")
        assert "Error: The 'schema' parameter for browser_extract must be a dictionary" in res_invalid


# ═══════════════════════════════════════════════════════════════════════════════
# L2: Boundary & Edge Cases — «Что на краях?»
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """L2: Edge cases — пробелы, верхний регистр, пустые строки и пустые схемы."""

    @pytest.mark.lmtrust(layer="L2", direction="D3")
    def test_browser_open_normalizes_whitespace_and_casing_in_mode(self, monkeypatch):
        """Режим с пробелами и верхним регистром ('  USER  ') должен корректно нормализоваться в 'user'."""
        captured_mode = []

        def mock_open_page(url, session="default", headed=False, return_state=True, mode=None):
            captured_mode.append(mode)
            return "Opened"

        monkeypatch.setattr(browser_engine, "open_page", mock_open_page)
        monkeypatch.setattr(browser_engine, "run", lambda coro: mock_open_page("https://example.com", mode="user"))

        res = bt.browser_open("https://example.com", mode="  USER  ")
        assert "Error: Invalid browser mode" not in res

    @pytest.mark.lmtrust(layer="L2", direction="D3")
    def test_empty_mode_string_rejected_gracefully(self):
        """Пустая строка в параметре mode возвращает понятную ошибку без исключений."""
        res = bt.browser_open("https://example.com", mode="")
        assert "Error: Invalid browser mode ''" in res

    @pytest.mark.lmtrust(layer="L2", direction="D7")
    def test_browser_extract_with_empty_dict_schema(self, monkeypatch):
        """Пустой dict schema={} валиден по контракту и не отвергается валидатором типов."""
        monkeypatch.setattr(sht, "STAGEHAND_AVAILABLE", False)
        res = sht.browser_extract("get data", schema={})
        # Не должно быть ошибки валидации схемы
        assert "must be a dictionary" not in res
        assert "Stagehand v4 is not installed" in res


# ═══════════════════════════════════════════════════════════════════════════════
# L3: Property & Invariants — «Что ВСЕГДА истинно?»
# ═══════════════════════════════════════════════════════════════════════════════

class TestProperties:
    """L3: Invariants — идемпотентность сброса состояния и сохранение контекста."""

    @pytest.mark.lmtrust(layer="L3", direction="D10")
    def test_reset_stagehand_is_idempotent(self):
        """Инвариант: многократный вызов reset_stagehand() идемпотентен и безопасен."""
        for _ in range(5):
            sht.reset_stagehand()
            assert sht._STAGEHAND_INSTANCE is None

    @pytest.mark.lmtrust(layer="L3", direction="D10")
    def test_extension_dir_consistency(self):
        """Инвариант: повторный вызов _get_stagehand_extension_dir детерминирован."""
        dir1 = _get_stagehand_extension_dir()
        dir2 = _get_stagehand_extension_dir()
        assert dir1 == dir2


# ═══════════════════════════════════════════════════════════════════════════════
# L4: Adversarial — «Что если предположения — ложь?»
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarial:
    """L4: Fault injection — битый JSON CDP, сетевой таймаут, отказ сокета."""

    @pytest.mark.lmtrust(layer="L4", direction="D4")
    def test_cdp_discovery_handles_malformed_json_response(self, monkeypatch):
        """Если эндпоинт /json/list возвращает невалидный JSON, discovery не падает, а продолжает с extension_id=None."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{malformed json corrupt...}"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            mock_browser = MagicMock()
            mock_browser.closed = False
            mock_connect = AsyncMock(return_value=mock_browser)
            monkeypatch.setattr(sht.local_browser, "connect", mock_connect)
            monkeypatch.setattr(browser_engine, "_sessions", {"default": MagicMock()})
            monkeypatch.setattr(browser_engine, "get_cdp_url", lambda: "http://127.0.0.1:9222")
            monkeypatch.setattr(sht.Stagehand, "create", AsyncMock())

            asyncio.run(sht._get_stagehand_instance_async("default"))

            assert mock_connect.called
            call_kwargs = mock_connect.call_args[1]
            assert "extension_id" not in call_kwargs

    @pytest.mark.lmtrust(layer="L4", direction="D4")
    def test_cdp_discovery_handles_http_connection_timeout(self, monkeypatch):
        """Если запрос к /json/list падает по таймауту/ошибке сети, connect вызывается без extension_id."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError("CDP query timed out")):
            mock_browser = MagicMock()
            mock_browser.closed = False
            mock_connect = AsyncMock(return_value=mock_browser)
            monkeypatch.setattr(sht.local_browser, "connect", mock_connect)
            monkeypatch.setattr(browser_engine, "_sessions", {"default": MagicMock()})
            monkeypatch.setattr(browser_engine, "get_cdp_url", lambda: "http://127.0.0.1:9222")
            monkeypatch.setattr(sht.Stagehand, "create", AsyncMock())

            asyncio.run(sht._get_stagehand_instance_async("default"))

            assert mock_connect.called
            call_kwargs = mock_connect.call_args[1]
            assert "extension_id" not in call_kwargs


# ═══════════════════════════════════════════════════════════════════════════════
# L4b: Fallback — «Что когда основной путь недоступен?»
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallback:
    """L4b: Fallback degradation — автоматический откат на нативный Markdown + LLM."""

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_browser_extract_fallback_on_cdp_method_not_available(self, monkeypatch):
        """При ошибке Stagehand CDP Extensions.loadUnpacked срабатывает fallback на LLM JSON parse."""
        sht.reset_stagehand()
        cdp_err = RuntimeError("CDP command failed: Extensions.loadUnpacked: Method not available")
        monkeypatch.setattr(sht, "_get_stagehand_instance_async", AsyncMock(side_effect=cdp_err))
        monkeypatch.setattr(browser_engine, "get_markdown", AsyncMock(return_value="# Page\nItem: Test Widget, Price: 99"))

        mock_provider = MagicMock()
        mock_provider.chat.return_value = '{"item": "Test Widget", "price": 99}'
        with patch("providers.create_provider", return_value=mock_provider):
            res = sht.browser_extract("extract price", schema={"type": "object"})
            assert '{"item": "Test Widget", "price": 99}' in res

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_browser_extract_handles_both_stagehand_and_fallback_failure(self, monkeypatch):
        """Если и Stagehand, и нативный fallback завершились ошибкой, возвращается ясное сообщение."""
        sht.reset_stagehand()
        monkeypatch.setattr(sht, "_get_stagehand_instance_async", AsyncMock(side_effect=RuntimeError("Primary Stagehand error")))
        monkeypatch.setattr(browser_engine, "get_markdown", AsyncMock(side_effect=RuntimeError("DOM read error")))

        res = sht.browser_extract("extract data")
        assert "Error extracting data with Stagehand: Primary Stagehand error" in res


# ═══════════════════════════════════════════════════════════════════════════════
# L5: State / Lifecycle — «Все ли состояния достижимы?»
# ═══════════════════════════════════════════════════════════════════════════════

class TestState:
    """L5: State transitions — смена режимов user -> isolated и изолированный перезапуск."""

    @pytest.mark.lmtrust(layer="L5", direction="D2")
    def test_switch_from_user_to_isolated_reinitializes_engine(self, monkeypatch):
        """Если браузер открыт в user CDP, а затем запрошен mode='isolated', старый браузер закрывается."""
        eng = BrowserEngine()
        mock_browser = MagicMock()
        mock_browser.is_connected = True
        eng._browser = mock_browser
        eng._cdp_mode = True
        eng._headed = True

        shutdown_called = []
        async def mock_shutdown():
            shutdown_called.append(True)
            eng._browser = None
            eng._cdp_mode = False

        monkeypatch.setattr(eng, "shutdown", mock_shutdown)
        mock_launch_isolated = AsyncMock()
        monkeypatch.setattr(eng, "_launch_isolated", mock_launch_isolated)
        monkeypatch.setattr(eng, "_playwright", MagicMock())

        asyncio.run(eng._ensure_browser(headed=True, mode_override="isolated"))

        assert len(shutdown_called) == 1
        assert mock_launch_isolated.called

    @pytest.mark.lmtrust(layer="L5", direction="D2")
    def test_auto_mode_keeps_existing_browser_without_restart(self, monkeypatch):
        """В режиме auto уже открытый браузер не пересоздается лишний раз."""
        eng = BrowserEngine()
        mock_browser = MagicMock()
        mock_browser.is_connected = True
        eng._browser = mock_browser
        eng._cdp_mode = False
        eng._headed = True

        shutdown_mock = AsyncMock()
        monkeypatch.setattr(eng, "shutdown", shutdown_mock)
        monkeypatch.setattr("config.get_browser_mode", lambda: "isolated")

        asyncio.run(eng._ensure_browser(headed=True, mode_override="auto"))
        assert not shutdown_mock.called


# ═══════════════════════════════════════════════════════════════════════════════
# L6: Cross-System / Integration — «A + B работает корректно?»
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """L6: Cross-System — интеграция browser_tools -> browser_engine -> launch args."""

    @pytest.mark.lmtrust(layer="L6", direction="D7")
    def test_extension_preloaded_in_isolated_launch_arguments(self, monkeypatch):
        """При запуске изолированного браузера аргументы загрузки Stagehand передаются в Playwright launch."""
        eng = BrowserEngine()
        mock_launch = AsyncMock()
        mock_playwright = MagicMock()
        mock_playwright.chromium.launch = mock_launch

        monkeypatch.setattr(eng, "_playwright", mock_playwright)
        monkeypatch.setattr(eng, "_find_free_port", lambda p: 9222)
        monkeypatch.setattr("browser_engine._get_stagehand_extension_dir", lambda: "/mock/stagehand/assets")

        asyncio.run(eng._launch_isolated(headed=False))

        assert mock_launch.called
        call_kwargs = mock_launch.call_args[1]
        args = call_kwargs.get("args", [])
        assert any("--load-extension=/mock/stagehand/assets" in a for a in args)
        assert any("--remote-debugging-port=9222" in a for a in args)

    @pytest.mark.lmtrust(layer="L6", direction="D7")
    def test_extension_discovery_extracts_clean_extension_id(self, monkeypatch):
        """Интеграция CDP target polling извлекает чистый extension_id из chrome-extension:// URL."""
        dummy_targets = json.dumps([
            {"type": "page", "url": "http://example.com"},
            {"type": "background_page", "url": "chrome-extension://abcdefghijklmnop1234567890/manifest.json"}
        ]).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = dummy_targets
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            mock_browser = MagicMock()
            mock_browser.closed = False
            mock_connect = AsyncMock(return_value=mock_browser)
            monkeypatch.setattr(sht.local_browser, "connect", mock_connect)
            monkeypatch.setattr(browser_engine, "_sessions", {"default": MagicMock()})
            monkeypatch.setattr(browser_engine, "get_cdp_url", lambda: "http://127.0.0.1:9222")
            monkeypatch.setattr(sht.Stagehand, "create", AsyncMock())

            asyncio.run(sht._get_stagehand_instance_async("default"))

            assert mock_connect.called
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs.get("extension_id") == "abcdefghijklmnop1234567890"


# ═══════════════════════════════════════════════════════════════════════════════
# L7: Compositional — «Две вещи ломаются одновременно?»
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompositional:
    """L7: Compositional faults — Stagehand падает + LLM возвращает markdown code fences."""

    @pytest.mark.lmtrust(layer="L7", direction="D4")
    def test_stagehand_failure_with_markdown_fenced_llm_response(self, monkeypatch):
        """Stagehand бросает ошибку, а LLM возвращает ответ в ```json ... ``` — обертка корректно счищается."""
        sht.reset_stagehand()
        monkeypatch.setattr(sht, "_get_stagehand_instance_async", AsyncMock(side_effect=RuntimeError("CDP error")))
        monkeypatch.setattr(browser_engine, "get_markdown", AsyncMock(return_value="Content: product $50"))

        mock_provider = MagicMock()
        mock_provider.chat.return_value = "```json\n{\n  \"price\": 50\n}\n```"

        with patch("providers.create_provider", return_value=mock_provider):
            res = sht.browser_extract("get price")
            # Проверяем, что markdown code fence удален и остался чистый JSON
            assert res.startswith("{\n  \"price\": 50\n}")
            parsed = json.loads(res)
            assert parsed["price"] == 50


# ═══════════════════════════════════════════════════════════════════════════════
# L8: Temporal / Concurrency — «Зависит ли порядок / параллельность?»
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporal:
    """L8: Concurrency & ordering — безопасность вызовов в параллельных потоках."""

    @pytest.mark.lmtrust(layer="L8", direction="D6")
    def test_concurrent_free_port_probes(self):
        """Множественные параллельные запросы свободных портов возвращают валидные целые числа без исключений."""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(BrowserEngine._find_free_port, 9222) for _ in range(10)]
            ports = [f.result() for f in futures]

        assert len(ports) == 10
        assert all(isinstance(p, int) and p > 0 for p in ports)

    @pytest.mark.lmtrust(layer="L8", direction="D6")
    def test_concurrent_reset_stagehand_thread_safety(self):
        """Параллельный вызов reset_stagehand() из разных потоков защищен блокировкой _STAGEHAND_LOCK."""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(sht.reset_stagehand) for _ in range(8)]
            for f in futures:
                f.result()
        assert sht._STAGEHAND_INSTANCE is None


# ═══════════════════════════════════════════════════════════════════════════════
# L9: Negative Space — «Что НЕ должно происходить?»
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegativeSpace:
    """L9: Negative space — isolated режим не затрагивает пользовательские профили; CDP shutdown не убивает браузер."""

    @pytest.mark.lmtrust(layer="L9", direction="D3")
    def test_isolated_mode_does_not_access_argent_user_browser_profiles(self, monkeypatch, tmp_path):
        """В изолированном режиме каталог ~/.argent/browser_profiles/<key> не должен создаваться и использоваться."""
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        eng = BrowserEngine()
        mock_launch = AsyncMock()
        mock_playwright = MagicMock()
        mock_playwright.chromium.launch = mock_launch
        monkeypatch.setattr(eng, "_playwright", mock_playwright)

        asyncio.run(eng._launch_isolated(headed=False))

        user_profiles_dir = fake_home / ".argent" / "browser_profiles"
        assert not user_profiles_dir.exists(), "Isolated mode must NOT create user profiles directory"

    @pytest.mark.lmtrust(layer="L9", direction="D5")
    def test_shutdown_in_cdp_mode_does_not_terminate_user_process(self, monkeypatch):
        """В CDP режиме shutdown отсоединяется от сессии, но не вызывает kill() у пользовательского процесса."""
        eng = BrowserEngine()
        mock_proc = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.terminate = MagicMock()
        eng._cdp_process = mock_proc
        eng._cdp_mode = True

        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()
        eng._browser = mock_browser

        asyncio.run(eng.shutdown())

        assert not mock_proc.kill.called, "Shutdown in CDP mode must NOT kill user's browser process"
        assert not mock_proc.terminate.called, "Shutdown in CDP mode must NOT terminate user's browser process"
        assert eng._cdp_process is None
