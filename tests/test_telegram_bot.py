import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from config import (
    get_telegram_token,
    set_telegram_token,
    get_telegram_chat_id,
    set_telegram_chat_id,
    get_telegram_enabled,
    set_telegram_enabled,
)
from src.telegram.bot import TelegramClient, TelegramBridge, format_html, format_markdown_for_telegram



class FakeSession:
    def __init__(self):
        self.busy = False
        self.vibe = False
        self._events = []
        self._replies = []
        self._cancelled = False

    def start(self, text: str):
        self.busy = True
        self.last_text = text

    def get_event(self):
        if self._events:
            ev = self._events.pop(0)
            if ev.get("type") == "done":
                self.busy = False
            return ev
        self.busy = False
        return {"type": "done"}

    def reply_approval(self, approval_id: int, decision: str):
        self._replies.append((approval_id, decision))

    def cancel(self):
        self._cancelled = True
        self.busy = False

    def state(self):
        return {
            "model": "test-model",
            "provider": "ollama",
            "tier": "mid",
            "context": {"tokens": 100, "max": 8000, "percent": 1},
            "vibe": self.vibe,
            "checkpoints": [],
        }

    def set_vibe(self, enabled: bool):
        self.vibe = bool(enabled)
        return self.vibe


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Isolate config file so tests never touch ~/.argent_coder_config.json."""
    import config
    test_cfg_file = tmp_path / "test_config.json"
    monkeypatch.setattr(config, "CONFIG_FILE", test_cfg_file)
    monkeypatch.setattr(config, "_CONFIG_CACHE", None)
    yield
    config._CONFIG_CACHE = None


def test_telegram_config():
    """Test get/set telegram configs."""
    set_telegram_token("123456:ABC-DEF")
    assert get_telegram_token() == "123456:ABC-DEF"


    set_telegram_chat_id(987654321)
    assert get_telegram_chat_id() == "987654321"

    set_telegram_enabled(True)
    assert get_telegram_enabled() is True

    set_telegram_enabled(False)
    assert get_telegram_enabled() is False


def test_format_html():
    """Test markdown to HTML conversion for Telegram."""
    raw = (
        "### 📋 Заголовок\n"
        "**Стек**: TypeScript, Vite\n"
        "__Важно__: тест\n"
        "- Пункт 1\n"
        "* Пункт 2\n"
        "[Дока](https://example.com)\n"
        "Hello `code` and <script> & text\n"
        "```python\nprint(1 < 2)\n```"
    )
    formatted = format_html(raw)
    assert "<b>📋 Заголовок</b>" in formatted
    assert "<b>Стек</b>: TypeScript, Vite" in formatted
    assert "<b>Важно</b>: тест" in formatted
    assert "• Пункт 1" in formatted
    assert "• Пункт 2" in formatted
    assert '<a href="https://example.com">Дока</a>' in formatted
    assert "<code>code</code>" in formatted
    assert "&lt;script&gt;" in formatted
    assert "&amp;" in formatted
    assert '<pre><code class="language-python">print(1 &lt; 2)</code></pre>' in formatted


def test_format_markdown_for_telegram():
    """Test paragraph chunking of long markdown text."""
    p1 = "### Section 1\n" + ("text " * 300)
    p2 = "### Section 2\n" + ("more text " * 300)
    full = f"{p1}\n\n{p2}"

    chunks = format_markdown_for_telegram(full, limit=1000)
    assert len(chunks) >= 2
    assert "<b>Section 1</b>" in chunks[0]
    assert any("<b>Section 2</b>" in c for c in chunks[1:])


def test_telegram_client_calls():
    """Test TelegramClient API interaction."""
    client = TelegramClient("dummy_token")

    with patch.object(client.session, "post") as mock_post:
        # Mock getMe
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": {"id": 123, "username": "argent_bot"}}
        mock_post.return_value = mock_resp

        me = client.get_me()
        assert me["username"] == "argent_bot"

        # Mock send_chat_action
        mock_resp.json.return_value = {"ok": True, "result": True}
        assert client.send_chat_action(12345, "typing") is True

        # Mock answer_callback_query
        assert client.answer_callback_query("cb_123", "Done") is True


def test_telegram_client_chunking():
    """Test splitting messages longer than limit."""
    client = TelegramClient("dummy_token")
    long_text = "Line " * 1000  # 5000 chars

    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 1}}
        mock_post.return_value = mock_resp

        client.send_message(12345, long_text)
        assert mock_post.call_count >= 2


def test_telegram_authorization(monkeypatch):
    """Test that unauthorized chat IDs are rejected."""
    set_telegram_chat_id("111222")
    bridge = TelegramBridge()
    bridge.client = MagicMock()

    # Unauthorized message
    bridge._handle_message({
        "chat": {"id": 999999},
        "from": {"id": 999999, "username": "intruder"},
        "text": "rm -rf /"
    })
    bridge.client.send_message.assert_not_called()

    # Authorized message
    bridge._handle_message({
        "chat": {"id": 111222},
        "from": {"id": 111222, "username": "owner"},
        "text": "/help"
    })
    bridge.client.send_message.assert_called()


def test_telegram_slash_commands(monkeypatch):
    """Test handling of slash commands in Telegram bridge."""
    set_telegram_chat_id("111222")
    session = FakeSession()
    bridge = TelegramBridge(session=session)
    bridge.client = MagicMock()

    # /stats
    bridge._handle_command("111222", "/stats")
    bridge.client.send_message.assert_called()
    msg = bridge.client.send_message.call_args[0][1]
    assert "test-model" in msg
    assert "OLLAMA" in msg

    # /vibe on
    bridge._handle_command("111222", "/vibe on")
    assert session.vibe is True

    # /clear
    bridge._handle_command("111222", "/clear")
    assert bridge.session is not session  # fresh session created

    # /stop when busy
    active_session = FakeSession()
    active_session.busy = True
    bridge.session = active_session
    bridge._handle_command("111222", "/stop")
    assert active_session._cancelled is True


def test_telegram_approval_flow():
    """Test sending approval buttons and processing callback queries."""
    set_telegram_chat_id("111222")
    session = FakeSession()
    bridge = TelegramBridge(session=session)
    bridge.client = MagicMock()

    # 1. Simulate approval request event
    event = {
        "type": "approval_request",
        "id": 42,
        "action": "pytest tests/",
        "destructive": False,
        "grant_key": "pytest",
    }
    bridge._send_approval_request("111222", event)
    bridge.client.send_message.assert_called_once()
    call_args = bridge.client.send_message.call_args
    reply_markup = call_args[1]["reply_markup"]
    buttons = reply_markup["inline_keyboard"]
    assert any(b["callback_data"] == "appr:42:once" for row in buttons for b in row)
    assert any(b["callback_data"] == "appr:42:always" for row in buttons for b in row)
    assert any(b["callback_data"] == "appr:42:deny" for row in buttons for b in row)

    # 2. Simulate callback query response from user clicking [✅ Разрешить]
    cb_query = {
        "id": "query_99",
        "from": {"id": 111222},
        "message": {"chat": {"id": 111222}, "message_id": 55, "text": "Запрос разрешения"},
        "data": "appr:42:once",
    }
    bridge._handle_callback_query(cb_query)

    # Verify session received decision
    assert session._replies == [(42, "once")]
    bridge.client.answer_callback_query.assert_called_with("query_99", "Выбор: ✅ Разрешено (однократно)")
    bridge.client.edit_message_text.assert_called_once()


def test_telegram_event_consumer():
    """Test event consumer accumulating content and emitting final message."""
    set_telegram_chat_id("111222")
    session = FakeSession()
    session._events = [
        {"type": "content", "text": "Шаг 1: Готово. "},
        {"type": "content", "text": "Шаг 2: Все тесты пройдены!"},
        {"type": "done"},
    ]
    bridge = TelegramBridge(session=session)
    bridge.client = MagicMock()

    bridge._event_consumer_worker("111222")

    bridge.client.send_message.assert_called_once()
    final_text = bridge.client.send_message.call_args[0][1]
    assert "Шаг 1: Готово. Шаг 2: Все тесты пройдены!" in final_text


def test_telegram_mode_config():
    """Test get/set telegram mode."""
    from config import get_telegram_mode, set_telegram_mode
    set_telegram_mode("shared")
    assert get_telegram_mode() == "shared"

    set_telegram_mode("isolated")
    assert get_telegram_mode() == "isolated"

    with pytest.raises(ValueError):
        set_telegram_mode("invalid_mode")


def test_telegram_mode_switch_command():
    """Test switching modes via Telegram slash command."""
    from config import get_telegram_mode, set_telegram_mode
    set_telegram_chat_id("111222")
    set_telegram_mode("isolated")

    bridge = TelegramBridge()
    bridge.client = MagicMock()

    # Switch to shared
    bridge._handle_command("111222", "/mode shared")
    assert get_telegram_mode() == "shared"
    bridge.client.send_message.assert_called()
    msg = bridge.client.send_message.call_args[0][1]
    assert "Компаньон" in msg

    # Switch to isolated
    bridge._handle_command("111222", "/mode isolated")
    assert get_telegram_mode() == "isolated"
    msg2 = bridge.client.send_message.call_args[0][1]
    assert "Изолированный" in msg2


def test_telegram_shared_vs_isolated_session():
    """Test attaching terminal agent and verifying session routing."""
    from config import set_telegram_mode

    terminal_agent = MagicMock()
    terminal_agent.messages = [{"role": "system", "content": "term"}]

    bridge = TelegramBridge()
    bridge.attach_terminal_agent(terminal_agent)

    # When shared: session agent is the terminal agent
    set_telegram_mode("shared")
    assert bridge.session.agent is terminal_agent

    # When isolated: session agent is independent
    set_telegram_mode("isolated")
    assert bridge.session.agent is not terminal_agent


def test_terminal_approval_relay_from_telegram():
    """Test that a terminal approval is relayed to Telegram and answered via Telegram callback."""
    set_telegram_chat_id("111222")
    bridge = TelegramBridge()
    bridge.client = MagicMock()
    bridge.client.send_message.return_value = {"message_id": 777}

    # Simulate terminal approval waiting while a background Telegram callback query approves it
    def _delayed_telegram_callback():
        time.sleep(0.05)
        # Find the pending approval ID
        with bridge._lock:
            aids = list(bridge._pending_terminal_approvals.keys())
        assert len(aids) == 1
        aid = aids[0]
        bridge._handle_callback_query({
            "id": "cb_term_1",
            "from": {"id": 111222},
            "message": {"chat": {"id": 111222}, "message_id": 777, "text": "Action"},
            "data": f"appr:{aid}:once",
        })

    threading.Thread(target=_delayed_telegram_callback, daemon=True).start()

    def _never_called_terminal(action, destructive, grant_key):
        time.sleep(1.0)
        return "deny"

    decision = bridge.relay_terminal_approval("rm -rf build/", True, None, _never_called_terminal)
    assert decision == "once"
    bridge.client.answer_callback_query.assert_called_with("cb_term_1", "Выбор: ✅ Разрешено (однократно)")
    bridge.client.edit_message_text.assert_called_once()


def test_terminal_turn_completion_notification():
    """Test sending completion notification to Telegram after terminal turn."""
    set_telegram_chat_id("111222")
    bridge = TelegramBridge()
    bridge.client = MagicMock()

    bridge.notify_terminal_turn_completed("Запусти тесты", "Все 5 тестов успешно пройдены!")
    bridge.client.send_message.assert_called_once()
    msg = bridge.client.send_message.call_args[0][1]
    assert "[Терминал] Задача завершена" in msg
    assert "Запусти тесты" in msg
    assert "Все 5 тестов успешно пройдены!" in msg


def test_telegram_stop_button_and_command():
    """Test that stop command and inline stop button interrupt the terminal agent."""
    set_telegram_chat_id("111222")
    terminal_agent = MagicMock()
    bridge = TelegramBridge()
    bridge.attach_terminal_agent(terminal_agent)
    bridge.client = MagicMock()

    # 1. Test /stop command
    bridge._handle_command("111222", "/stop")
    terminal_agent.request_stop.assert_called_once()

    # 2. Test stop_turn button callback
    terminal_agent.reset_mock()
    bridge._handle_callback_query({
        "id": "cb_stop_1",
        "from": {"id": 111222},
        "message": {"chat": {"id": 111222}, "message_id": 999, "text": "Task in progress"},
        "data": "stop_turn",
    })
    terminal_agent.request_stop.assert_called_once()
    bridge.client.answer_callback_query.assert_called_with("cb_stop_1", "⏹️ Прерывание выполнения...")


def test_terminal_turn_started_notification_has_stop_button():
    """Test that notify_terminal_turn_started attaches an inline stop button."""
    set_telegram_chat_id("111222")
    bridge = TelegramBridge()
    bridge.client = MagicMock()
    bridge.client.send_message.return_value = {"message_id": 123}

    bridge.notify_terminal_turn_started("Напиши сложный алгоритм")
    bridge.client.send_message.assert_called_once()
    kwargs = bridge.client.send_message.call_args[1]
    reply_markup = kwargs.get("reply_markup")
    assert reply_markup is not None
    assert "inline_keyboard" in reply_markup
    btn = reply_markup["inline_keyboard"][0][0]
    assert btn["callback_data"] == "stop_turn"
    assert "Прервать" in btn["text"]



