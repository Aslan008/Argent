"""The Telegram bridge's rendering, live status, queue, files and PC mirroring.

Everything here runs against a fake client: no network, no real agent.
"""

import re
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.telegram import render
from src.telegram.bot import (
    BOT_COMMANDS,
    MAX_QUEUE,
    REPLY_BUTTONS,
    TelegramBridge,
    approval_keyboard,
    reply_keyboard,
)
from src.telegram.client import TelegramClient
from src.telegram.live import LiveStatus, _looks_failed
from src.telegram.render import (
    MAX_HTML,
    diff_counts,
    fmt_duration,
    format_html,
    format_markdown_for_telegram,
    repair_html,
    split_html,
    strip_html,
    tool_label,
    turn_footer,
)


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "cfg.json")
    monkeypatch.setattr(config, "_CONFIG_CACHE", None)
    config.set_telegram_chat_id("111222")
    config.set_telegram_mode("isolated")
    yield
    config._CONFIG_CACHE = None


_TAG = re.compile(r"<(/?)([a-z-]+)[^<>]*>")


def balanced(s: str) -> bool:
    stack = []
    for m in _TAG.finditer(s):
        if m.group(1):
            if not stack or stack.pop() != m.group(2):
                return False
        else:
            stack.append(m.group(2))
    return not stack


class FakeSession:
    def __init__(self):
        self.busy = False
        self.vibe = False
        self.events = []
        self.started = []
        self.replies = []
        self.cancelled = False

    def start(self, text):
        self.busy = True
        self.started.append(text)

    def get_event(self):
        ev = self.events.pop(0) if self.events else {"type": "done"}
        if ev.get("type") == "done":
            self.busy = False
        return ev

    def reply_approval(self, aid, decision):
        self.replies.append((aid, decision))

    def cancel(self):
        self.cancelled, self.busy = True, False

    def state(self):
        return {"model": "m", "provider": "p", "tier": "t", "context": {"tokens": 50, "max": 100, "percent": 50.0}, "vibe": self.vibe}

    def set_vibe(self, on):
        self.vibe = bool(on)
        return self.vibe


def make_bridge(session=None, **client_kw):
    bridge = TelegramBridge(session=session or FakeSession())
    bridge.client = MagicMock(**client_kw)
    bridge.client.send_message.return_value = {"message_id": 7}
    return bridge


# ─── Markdown -> HTML ─────────────────────────────────────────────────────────

class TestFormatHtml:
    def test_tables_become_a_grid_when_narrow(self):
        out = format_html("| Имя | Код |\n|---|---|\n| foo | 1 |\n| bar | 22 |")
        assert out.startswith("<pre>") and "foo" in out and "──" in out

    def test_wide_tables_become_cards(self):
        row = "| " + " | ".join(["колонка"] * 5) + " |"
        sep = "|" + "---|" * 5
        out = format_html(f"{row}\n{sep}\n{row}")
        assert "<pre>" not in out and "<b>" in out

    def test_blockquote_and_long_quote_collapses(self):
        assert "<blockquote>" in format_html("> цитата")
        assert "<blockquote expandable>" in format_html("\n".join(["> " + "x" * 80] * 8))

    def test_nested_lists_task_lists_and_rules(self):
        out = format_html("- a\n  - b\n    - c\n- [x] done\n- [ ] todo\n---")
        assert "• a" in out and "◦ b" in out and "▪ c" in out
        assert "☑ done" in out and "☐ todo" in out and "──────" in out

    def test_snake_case_is_not_italic(self):
        assert "<i>" not in format_html("use snake_case_name and file_name.py here")

    def test_unclosed_fence_still_balanced(self):
        out = format_html("text\n```python\nprint(1)")
        assert "<pre>" in out and balanced(out)

    def test_inline_code_is_escaped_and_protected_from_markdown(self):
        out = format_html("run `a **b** <c>` now")
        assert "<code>a **b** &lt;c&gt;</code>" in out

    def test_header_inside_bold_does_not_nest(self):
        assert "<b><b>" not in format_html("## **Title**")

    @pytest.mark.parametrize("text", [
        "**bold _both** ital_",
        "<b>raw html</b> & <script>",
        "*a **b* c**",
        "```\n```",
        "1. one\n2. two\n   - sub",
        "[link](https://x.y/a?b=1&c=2) and ![img](https://x.y/i.png)",
        "~~gone~~ and ||spoiler||",
        "# H1\n## H2\n### H3\ntext",
    ])
    def test_always_balanced(self, text):
        assert balanced(format_html(text))

    def test_repair_html_renests_crossed_tags(self):
        assert repair_html("<b>a<i>b</b>c</i>") == "<b>a<i>b</i></b><i>c</i>"
        assert repair_html("x</b>y<i>z") == "xy<i>z</i>"

    def test_strip_html_gives_readable_text(self):
        assert strip_html("<b>a</b> &lt;b&gt; <code>c</code>") == "a <b> c"


class TestSplitHtml:
    def test_pieces_fit_and_are_balanced_and_keep_all_text(self):
        body = "<b>head</b>\n" + "\n".join(f"line {i} <code>x{i}</code> " + "w " * 30 for i in range(400))
        pieces = split_html(body, 1000)
        assert len(pieces) > 5
        for p in pieces:
            assert len(p) <= 1000 and balanced(p)
        plain = lambda s: re.sub(r"\s+", "", strip_html(s))
        assert plain("".join(pieces)) == plain(body)

    def test_a_code_block_across_pieces_is_closed_and_reopened(self):
        code = "\n".join(f"line_{i} = {i}" for i in range(600))
        pieces = split_html(format_html(f"```python\n{code}\n```"), 1500)
        assert len(pieces) > 3
        for p in pieces:
            assert p.startswith("<pre><code") and p.endswith("</code></pre>") and len(p) <= 1500

    def test_never_cuts_inside_a_tag_or_entity(self):
        body = "&amp;" * 3000 + "<a href='https://x'>t</a>" * 100
        for p in split_html(body, 500):
            assert len(p) <= 500
            assert not re.search(r"&\w{0,4}$", p) and p.count("<") == p.count(">")

    def test_markdown_chunking_prefers_paragraphs(self):
        text = "\n\n".join("para %d " % i + "word " * 100 for i in range(20))
        chunks = format_markdown_for_telegram(text, limit=1200)
        assert all(len(c) <= 1200 for c in chunks)
        assert all(c.rstrip().rsplit("\n", 1)[-1].strip() for c in chunks)

    def test_html_expansion_cannot_exceed_the_hard_limit(self):
        chunks = format_markdown_for_telegram("&<>" * 3000, limit=5000)
        assert all(len(c) <= MAX_HTML for c in chunks)


class TestSmallFormatters:
    def test_duration(self):
        assert fmt_duration(7) == "7 с" and fmt_duration(72) == "1 мин 12 с" and fmt_duration(3900) == "1 ч 05 мин"

    def test_tool_labels_are_human(self):
        assert tool_label("read_file", {"file_path": "C:\\a\\b\\c\\d\\e\\f\\g.py"})[0] == "📖"
        assert tool_label("run_command", {"command": "pytest -q"})[2] == "pytest -q"
        assert tool_label("something_new", {})[1] == "something new"

    def test_every_real_tool_has_a_label(self):
        import tools.schemas as schemas
        names = set()
        for v in vars(schemas).values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                names |= {t.get("function", t)["name"] for t in v if isinstance(t, dict)}
        unknown = {n for n in names if n not in render._TOOLS}
        assert unknown <= {"ask_user_questions"} or not unknown, f"tools without a label: {sorted(unknown)}"

    def test_footer(self):
        f = turn_footer({"total_s": 72, "tool_calls": 3, "prompt_tokens": 12300, "completion_tokens": 450, "ctx_percent": 34.2})
        assert "1 мин 12 с" in f and "🛠 3" in f and "12.3k" in f and "ctx 34%" in f
        assert turn_footer(None) == ""

    def test_diff_counts(self):
        assert diff_counts("--- a\n+++ b\n+x\n+y\n-z\n ctx") == (2, 1)


# ─── client ───────────────────────────────────────────────────────────────────

class TestClient:
    def _client(self, responses):
        c = TelegramClient("tok")
        it = iter(responses)
        c.session = SimpleNamespace(post=lambda *a, **k: SimpleNamespace(json=lambda: next(it)), close=lambda: None)
        return c

    def test_plain_retry_strips_the_tags(self):
        c = TelegramClient("tok")
        sent = []

        def fake(method, payload, **kw):
            sent.append(dict(payload))
            if payload.get("parse_mode"):
                c.last_code, c.last_error = 400, "can't parse entities"
                return None
            return {"message_id": 1}

        c._call = fake
        assert c.send_message(1, "<b>Привет</b> &amp; <code>x</code>")
        assert sent[1]["text"] == "Привет & x" and "parse_mode" not in sent[1]

    def test_not_modified_counts_as_success(self):
        c = self._client([{"ok": False, "error_code": 400, "description": "Bad Request: message is not modified"}])
        assert c.edit_message_text(1, 2, "same") is not None

    def test_rate_limit_is_waited_out_once(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)
        c = self._client([{"ok": False, "error_code": 429, "description": "slow", "parameters": {"retry_after": 1}},
                          {"ok": True, "result": {"message_id": 5}}])
        assert c.send_message(1, "hi")["message_id"] == 5

    def test_network_error_is_not_retried_as_plain_text(self):
        calls = []
        c = TelegramClient("tok")

        def boom(*a, **k):
            calls.append(1)
            raise OSError("down")

        c.session = SimpleNamespace(post=boom)
        assert c.send_message(1, "<b>x</b>") is None
        assert len(calls) == 1 and c.last_network and not c.last_ok

    def test_long_messages_split_on_tag_boundaries(self):
        c = TelegramClient("tok")
        seen = []
        c._call = lambda m, p, **k: seen.append(p["text"]) or {"message_id": 1}
        c.send_message(1, "<pre><code>" + "x\n" * 5000 + "</code></pre>")
        assert len(seen) > 2 and all(balanced(t) and len(t) <= MAX_HTML for t in seen)


# ─── live status ──────────────────────────────────────────────────────────────

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TestLiveStatus:
    def _live(self, **kw):
        client = MagicMock()
        client.send_message.return_value = {"message_id": 9}
        client.edit_message_text.return_value = {"message_id": 9}
        clock = Clock()
        live = LiveStatus(client, 1, "Запусти тесты", clock=clock, tick=0, **kw)
        return live, client, clock

    def test_the_panel_is_one_message_edited_in_place(self):
        live, client, clock = self._live()
        live.begin("read_file", {"file_path": "a.py"})
        assert client.send_message.call_count == 1
        for i in range(5):
            clock.t += 2
            live.end("read_file", "ok", 0.3)
            live.begin("run_command", {"command": f"cmd {i}"})
        assert client.send_message.call_count == 1 and client.edit_message_text.call_count >= 3

    def test_edits_are_throttled_and_coalesced(self):
        live, client, clock = self._live()
        live.begin("read_file", {"file_path": "a"})
        edits = client.edit_message_text.call_count
        for _ in range(20):
            live.end("read_file", "ok")
            live.begin("read_file", {"file_path": "b"})
        assert client.edit_message_text.call_count == edits       # all inside one interval

    def test_unchanged_text_is_not_resent(self):
        live, client, clock = self._live()
        live.begin("read_file", {})
        n = client.edit_message_text.call_count
        live._flush(force=True)
        live._flush(force=True)
        assert client.edit_message_text.call_count <= n + 1

    def test_only_the_last_steps_show_and_the_rest_is_counted(self):
        live, client, clock = self._live()
        for i in range(12):
            live.begin("run_command", {"command": f"c{i}"})
            live.end("run_command", "ok")
        text = live._render()
        assert "c11" in text and "c0<" not in text and "ещё 6 шагов выше" in text

    def test_finish_collapses_with_all_steps_in_an_expandable_quote(self):
        live, client, clock = self._live()
        live.begin("read_file", {"file_path": "a.py"})
        live.end("read_file", "ok")
        clock.t = 75
        live.finish("done")
        kwargs = client.edit_message_text.call_args.kwargs
        assert "Готово" in kwargs["text"] and "1 мин 15 с" in kwargs["text"] and "<blockquote expandable>" in kwargs["text"]
        assert kwargs["reply_markup"] == {"inline_keyboard": []} and balanced(kwargs["text"])

    def test_failed_step_is_marked(self):
        live, client, clock = self._live()
        live.begin("run_command", {"command": "x"})
        live.end("run_command", "Error: boom")
        assert "✗" in live._render()
        assert _looks_failed("Traceback (most recent call last)") and not _looks_failed("all fine")

    def test_plan_from_todo_write(self):
        live, client, clock = self._live()
        live.begin("todo_write", {"todos": [{"content": "A", "status": "completed"}, {"content": "B", "status": "in_progress"}, {"content": "C", "status": "pending"}]})
        text = live._render()
        assert "✅ A" in text and "▶️ B" in text and "☐ C" in text

    def test_user_text_is_escaped(self):
        live, client, clock = self._live()
        live.title = "a <b> & c"
        live.begin("run_command", {"command": "echo <x> && y"})
        assert balanced(live._render()) and "&lt;x&gt;" in live._render()

    def test_gives_up_after_repeated_edit_failures(self):
        live, client, clock = self._live()
        client.edit_message_text.return_value = None
        live.begin("read_file", {})
        for _ in range(10):
            clock.t += 5
            live.end("read_file", "ok")
            live.begin("read_file", {})
        assert client.edit_message_text.call_count <= 5

    def test_panel_start_failure_does_not_raise(self):
        live, client, clock = self._live()
        client.send_message.return_value = None
        live.begin("read_file", {})
        live.begin("read_file", {})
        live.begin("read_file", {})
        assert client.send_message.call_count == 2           # one retry, then it stops trying
        live.finish("done")

    def test_stop_ui_removes_the_button(self):
        live, client, clock = self._live()
        live.begin("read_file", {})
        clock.t += 5
        live.request_stop_ui()
        assert client.edit_message_text.call_args.kwargs["reply_markup"] is None
        assert "Останавливаю" in client.edit_message_text.call_args.kwargs["text"]


# ─── the bridge ───────────────────────────────────────────────────────────────

class TestTaskFromThePhone:
    def test_progress_panel_then_answer_with_footer(self):
        s = FakeSession()
        s.events = [
            {"type": "tool_start", "name": "read_file", "args": {"file_path": "a.py"}},
            {"type": "tool_end", "name": "read_file", "result": "ok", "duration_secs": 0.2},
            {"type": "content", "text": "Готово: **всё** работает"},
            {"type": "turn_snapshot", "data": {"total_s": 5, "tool_calls": 1, "prompt_tokens": 1500, "completion_tokens": 20, "ctx_percent": 12}},
            {"type": "done"},
        ]
        b = make_bridge(s)
        b._event_consumer_worker("111222", "прочитай файл")
        sent = [c.args[1] for c in b.client.send_message.call_args_list]
        assert "Работаю" in sent[0]                                    # the panel
        answer = sent[-1]
        assert "<b>всё</b>" in answer and "5 с" in answer and "ctx 12%" in answer
        assert b.client.edit_message_text.call_args.kwargs["text"].startswith("✅ <b>Готово</b>")

    def test_an_answer_without_tools_makes_no_panel(self):
        s = FakeSession()
        s.events = [{"type": "content", "text": "hi"}, {"type": "done"}]
        b = make_bridge(s)
        b._event_consumer_worker("111222", "привет")
        assert b.client.send_message.call_count == 1 and not b.client.edit_message_text.called

    def test_stopped_turn_says_so_and_keeps_the_partial_text(self):
        s = FakeSession()
        s.events = [{"type": "content", "text": "половина"}, {"type": "error", "text": "[System: Выполнение прервано пользователем (Ctrl+C / Telegram)]"}, {"type": "done"}]
        b = make_bridge(s)
        b._event_consumer_worker("111222")
        text = b.client.send_message.call_args.args[1]
        assert "половина" in text and "Остановлено" in text

    def test_error_only_turn_is_reported_not_swallowed(self):
        s = FakeSession()
        s.events = [{"type": "error", "text": "[System: turn failed: boom]"}, {"type": "done"}]
        b = make_bridge(s)
        b._event_consumer_worker("111222")
        assert "Не получилось" in b.client.send_message.call_args.args[1] and "boom" in b.client.send_message.call_args.args[1]

    def test_a_very_long_answer_is_attached_as_a_file_with_a_preview(self):
        s = FakeSession()
        s.events = [{"type": "content", "text": "\n\n".join("абзац %d " % i + "слово " * 80 for i in range(60))}, {"type": "done"}]
        b = make_bridge(s)
        b._event_consumer_worker("111222")
        assert b.client.send_document_bytes.called
        name, data = b.client.send_document_bytes.call_args.args[1:3]
        assert name == "answer.md" and "абзац 59" in data.decode("utf-8")
        assert "во вложении" in b.client.send_message.call_args.args[1]

    def test_approval_card_shows_the_action_and_is_rewritten_on_decision(self):
        s = FakeSession()
        b = make_bridge(s)
        b._send_approval_request("111222", {"id": 3, "action": "rm -rf <build>", "destructive": True, "grant_key": None})
        card = b.client.send_message.call_args.args[1]
        assert "Деструктивное" in card and "<pre>rm -rf &lt;build&gt;</pre>" in card
        kb = b.client.send_message.call_args.kwargs["reply_markup"]["inline_keyboard"]
        assert [btn["callback_data"] for row in kb for btn in row] == ["appr:3:once", "appr:3:deny"]      # no "always" for destructive
        b._handle_callback_query({"id": "q", "from": {"id": 111222}, "message": {"chat": {"id": 111222}, "message_id": 7, "text": "plain"}, "data": "appr:3:deny"})
        edited = b.client.edit_message_text.call_args.kwargs["text"]
        assert "<pre>rm -rf &lt;build&gt;</pre>" in edited and "Отклонено" in edited       # the HTML card, not its stripped text
        assert s.replies == [(3, "deny")]

    def test_approval_keyboard_offers_always_only_for_safe_actions(self):
        kb = approval_keyboard("appr", 1, "pytest", False)["inline_keyboard"]
        assert len(kb[0]) == 2 and kb[1][0]["callback_data"] == "appr:1:deny"


class TestQueue:
    def test_a_message_while_busy_is_queued_then_started_when_free(self):
        s = FakeSession()
        b = make_bridge(s)
        s.busy = True
        b._run_prompt_turn("111222", "вторая задача", message_id=5)
        assert len(b._queue) == 1 and not s.started
        assert "В очереди №1" in b.client.send_message.call_args.args[1]
        s.busy = False
        b._drain_queue()
        assert s.started == ["вторая задача"] and not b._queue

    def test_queue_is_bounded(self):
        s = FakeSession()
        s.busy = True
        b = make_bridge(s)
        for i in range(MAX_QUEUE + 2):
            b._run_prompt_turn("111222", f"m{i}")
        assert len(b._queue) == MAX_QUEUE
        assert "очередь заполнена" in b.client.send_message.call_args.args[1]

    def test_queue_buttons(self):
        s = FakeSession()
        s.busy = True
        b = make_bridge(s)
        b._run_prompt_turn("111222", "a")
        b._run_prompt_turn("111222", "b")
        b._handle_callback_query({"id": "q", "from": {"id": 111222}, "message": {"chat": {"id": 111222}, "message_id": 1}, "data": "q:del:0"})
        assert [i["text"] for i in b._queue] == ["b"]
        b._handle_callback_query({"id": "q", "from": {"id": 111222}, "message": {"chat": {"id": 111222}, "message_id": 1}, "data": "q:clear"})
        assert not b._queue

    def test_a_pc_task_in_progress_also_queues_phone_messages(self):
        s = FakeSession()
        b = make_bridge(s)
        b.notify_terminal_turn_started("задача на ПК")
        b._run_prompt_turn("111222", "с телефона")
        assert not s.started and len(b._queue) == 1
        b.notify_terminal_turn_completed("задача на ПК", "готово")
        assert s.started == ["с телефона"]

    def test_a_stale_pc_task_does_not_block_forever(self):
        b = make_bridge()
        b.notify_terminal_turn_started("x")
        b._turn_started_at = time.monotonic() - 8000
        assert not b._busy()


class TestPanelAndCommands:
    def _cb(self, b, data):
        b._handle_callback_query({"id": "q", "from": {"id": 111222}, "message": {"chat": {"id": 111222}, "message_id": 4, "text": "m"}, "data": data})

    def test_start_sends_welcome_with_keyboard_and_publishes_commands(self):
        b = make_bridge()
        b._handle_command("111222", "/start")
        b.client.set_my_commands.assert_called_once()
        cmds = [c["command"] for c in b.client.set_my_commands.call_args.args[0]]
        assert cmds == [c for c, _ in BOT_COMMANDS] and all(re.fullmatch(r"[a-z_]{1,32}", c) for c in cmds)
        kinds = [c.kwargs.get("reply_markup") for c in b.client.send_message.call_args_list]
        assert any(k and "keyboard" in k for k in kinds) and any(k and "inline_keyboard" in k for k in kinds)

    def test_bot_username_suffix_is_ignored(self):
        b = make_bridge()
        b._handle_command("111222", "/status@argent_bot")
        assert "Состояние" in b.client.send_message.call_args.args[1]

    def test_menu_buttons_edit_the_same_message(self):
        b = make_bridge()
        b.client.edit_message_text.return_value = {"message_id": 4}
        self._cb(b, "ui:vibe")
        assert b.session.vibe is True
        assert b.client.edit_message_text.call_args.kwargs["message_id"] == 4
        self._cb(b, "ui:status")
        assert "Состояние" in b.client.edit_message_text.call_args.kwargs["text"]

    def test_mode_change_from_the_panel(self):
        from config import get_telegram_mode
        b = make_bridge()
        self._cb(b, "ui:mode:shared")
        assert get_telegram_mode() == "shared"

    def test_new_dialog_asks_first(self):
        s = FakeSession()
        b = make_bridge(s)
        self._cb(b, "ui:new")
        assert b.session is s and "Начать новый диалог" in b.client.edit_message_text.call_args.kwargs["text"]
        self._cb(b, "ui:new:yes")
        assert b.session is not s

    def test_persistent_keyboard_buttons_map_to_commands(self):
        b = make_bridge()
        b._handle_message({"chat": {"id": 111222}, "from": {"id": 111222}, "text": "📊 Статус"})
        assert "Состояние" in b.client.send_message.call_args.args[1]
        b._handle_message({"chat": {"id": 111222}, "from": {"id": 111222}, "text": "🆕 Новый диалог"})
        assert "Начать новый диалог" in b.client.send_message.call_args.args[1]
        assert set(REPLY_BUTTONS) == {b["text"] for row in reply_keyboard()["keyboard"] for b in row}

    def test_acknowledges_with_a_reaction(self):
        b = make_bridge()
        b._handle_message({"chat": {"id": 111222}, "from": {"id": 111222}, "message_id": 33, "text": "сделай"})
        time.sleep(0.2)
        b.client.set_reaction.assert_any_call("111222", 33, "👀")

    def test_old_messages_are_not_obeyed(self):
        b = make_bridge()
        b._handle_message({"chat": {"id": 111222}, "from": {"id": 111222}, "message_id": 1, "date": time.time() - 3600, "text": "удали всё"})
        assert not b.session.started and b._stale_skipped == ["удали всё"]
        b._report_stale()
        assert "Пропущено сообщений: 1" in b.client.send_message.call_args.args[1]
        # ... but a fresh Stop always works, however old
        b._handle_message({"chat": {"id": 111222}, "from": {"id": 111222}, "message_id": 2, "date": time.time() - 3600, "text": "/stop"})

    def test_strangers_cannot_make_the_bot_download(self):
        b = make_bridge()
        b._handle_message({"chat": {"id": 999}, "from": {"id": 999}, "photo": [{"file_id": "x"}]})
        b._handle_message({"chat": {"id": 999}, "from": {"id": 999}, "document": {"file_id": "y", "file_name": "evil.sh"}})
        assert not b.client.get_file.called and not b.client.send_message.called

    def test_a_document_is_saved_and_named_in_the_prompt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = FakeSession()
        b = make_bridge(s)
        b.client.get_file.return_value = {"file_path": "docs/report.csv", "file_size": 10}
        b.client.download_file.side_effect = lambda remote, dest: (__import__("pathlib").Path(dest).parent.mkdir(parents=True, exist_ok=True), __import__("pathlib").Path(dest).write_text("a,b"), True)[2]
        b._handle_message({"chat": {"id": 111222}, "from": {"id": 111222}, "document": {"file_id": "d", "file_name": "report.csv", "mime_type": "text/csv"}, "caption": "что тут?"})
        assert "что тут?" in s.started[0] and "telegram_uploads" in s.started[0] and "report.csv" in s.started[0]

    def test_voice_gets_an_explanation(self):
        b = make_bridge()
        b._handle_message({"chat": {"id": 111222}, "from": {"id": 111222}, "voice": {"file_id": "v"}})
        assert "текстом" in b.client.send_message.call_args.args[1]


class TestFiles:
    def test_file_command_sends_project_files_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "notes.txt").write_text("hi")
        (tmp_path / ".env").write_text("KEY=1")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "id_rsa").write_text("x")
        b = make_bridge()
        b._handle_command("111222", "/file notes.txt")
        assert b.client.send_document.call_args.args[1].endswith("notes.txt")
        for bad in (".env", "sub/id_rsa", "../outside.txt", "C:/Windows/win.ini", ".git/config"):
            b.client.send_document.reset_mock()
            b._handle_command("111222", f"/file {bad}")
            assert not b.client.send_document.called, bad

    def test_directory_listing_hides_secrets(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.py").write_text("")
        (tmp_path / ".env").write_text("")
        b = make_bridge()
        b._handle_command("111222", "/file .")
        text = b.client.send_message.call_args.args[1]
        assert "a.py" in text and ".env" not in text

    def test_screenshots_taken_for_the_phone_go_back_to_it(self, tmp_path):
        shot = tmp_path / "s.png"
        shot.write_bytes(b"\x89PNG")
        b = make_bridge()
        b._maybe_send_artifact("111222", "capture_desktop_screen", {"output_path": str(shot)}, "saved")
        time.sleep(0.2)
        b.client.send_photo.assert_called_once()
        b._maybe_send_artifact("111222", "run_command", {"output_path": str(shot)}, "")
        assert b.client.send_photo.call_count == 1


class TestMirrorOfThePc:
    def _wait(self, b):
        deadline = time.time() + 3
        while time.time() < deadline and b._pump_q.unfinished_tasks:
            time.sleep(0.01)
        time.sleep(0.05)

    def test_tui_turn_shows_progress_and_a_result_on_the_phone(self):
        b = make_bridge()
        b.notify_terminal_turn_started("почини тест", source="tui", wait=False)
        for ev in [
            {"type": "tool_start", "name": "run_command", "args": {"command": "pytest"}},
            {"type": "tool_end", "name": "run_command", "result": "ok", "duration_secs": 2},
            {"type": "content", "text": "Починил."},
            {"type": "turn_snapshot", "data": {"total_s": 9, "tool_calls": 1}},
            {"type": "done"},
        ]:
            b.mirror_event(ev)
        self._wait(b)
        texts = [c.args[1] for c in b.client.send_message.call_args_list]
        assert "Выполняется на ПК" in texts[0]
        assert "[Терминал] Задача завершена" in texts[-1] and "Починил." in texts[-1] and "9 с" in texts[-1]
        assert b._term is None

    def test_mirroring_never_blocks_the_agent_thread(self):
        b = make_bridge()
        gate = threading.Event()
        b.client.send_message.side_effect = lambda *a, **k: (gate.wait(5), {"message_id": 1})[1]
        t0 = time.monotonic()
        b.notify_terminal_turn_started("x", source="tui", wait=False)
        for i in range(50):
            b.mirror_event({"type": "tool_start", "name": "read_file", "args": {}})
        assert time.monotonic() - t0 < 0.5
        gate.set()

    def test_tui_approval_can_be_answered_from_the_phone(self, monkeypatch):
        import config
        config.set_telegram_mode("shared")
        b = make_bridge()
        frontend = MagicMock()
        b.attach_terminal_frontend(frontend)
        monkeypatch.setattr(type(b), "is_running", property(lambda self: True))
        b.mirror_event({"type": "approval_request", "id": 8, "action": "git push", "destructive": False, "grant_key": "git"})
        self._wait(b)
        kb = b.client.send_message.call_args.kwargs["reply_markup"]["inline_keyboard"]
        assert kb[0][0]["callback_data"] == "tui:ap:8:once"
        b._handle_callback_query({"id": "q", "from": {"id": 111222}, "message": {"chat": {"id": 111222}, "message_id": 7}, "data": "tui:ap:8:once"})
        frontend.reply_approval.assert_called_once_with(8, "once")
        assert "с телефона" in b.client.edit_message_text.call_args.kwargs["text"]
        # a second tap is harmless
        b._handle_callback_query({"id": "q2", "from": {"id": 111222}, "message": {"chat": {"id": 111222}, "message_id": 7}, "data": "tui:ap:8:once"})
        frontend.reply_approval.assert_called_once()

    def test_pc_answering_first_closes_the_phone_card(self, monkeypatch):
        import config
        config.set_telegram_mode("shared")
        b = make_bridge()
        monkeypatch.setattr(type(b), "is_running", property(lambda self: True))
        b.mirror_event({"type": "approval_request", "id": 2, "action": "ls", "destructive": False, "grant_key": None})
        self._wait(b)
        b.mirror_resolved("ap", 2, "once")
        assert "на ПК" in b.client.edit_message_text.call_args.kwargs["text"]
        n = b.client.edit_message_text.call_count
        b.mirror_resolved("ap", 2, "once")
        assert b.client.edit_message_text.call_count == n

    def test_tui_questions_are_mirrored_with_buttons(self, monkeypatch):
        import config
        config.set_telegram_mode("shared")
        b = make_bridge()
        frontend = MagicMock()
        b.attach_terminal_frontend(frontend)
        monkeypatch.setattr(type(b), "is_running", property(lambda self: True))
        b.mirror_event({"type": "select_request", "id": "sel_1", "title": "Какой вариант?", "options": ["Один", "Два"], "multi": False})
        b.mirror_event({"type": "select_request", "id": "sel_2", "title": "Что включить?", "options": ["A", "B", "C"], "multi": True})
        b.mirror_event({"type": "select_request", "id": "sel_3", "title": "Введите имя", "options": [], "multi": False})   # free text: not mirrored
        self._wait(b)
        assert b.client.send_message.call_count == 2
        cb = lambda d: b._handle_callback_query({"id": "q", "from": {"id": 111222}, "message": {"chat": {"id": 111222}, "message_id": 7}, "data": d})
        cb("tui:sel:sel_1:1")
        frontend.reply_select.assert_called_with("sel_1", "Два")
        cb("tui:mt:sel_2:0")
        cb("tui:mt:sel_2:2")
        cb("tui:md:sel_2")
        assert frontend.reply_select.call_args.args == ("sel_2", '["A", "C"]')

    def test_nothing_is_mirrored_in_isolated_mode(self, monkeypatch):
        b = make_bridge()
        monkeypatch.setattr(type(b), "is_running", property(lambda self: True))
        assert not b.mirrors_terminal

    def test_classic_terminal_chunks_are_teed_not_changed(self):
        b = make_bridge()
        b.notify_terminal_turn_started("x")
        chunks = [{"type": "tool_start", "name": "read_file", "args": {"file_path": "a"}}, {"type": "content_stream", "content": "hi"}]
        assert list(b.tee_chunks(iter(chunks))) == chunks
        self._wait(b)
        assert b._term.live.step_count == 1 and b._term.content == ["hi"]

    def test_stop_finishes_open_panels(self):
        b = make_bridge()
        b.notify_terminal_turn_started("x")
        b.stop()
        assert b._term is None and b.client.edit_message_text.called


class TestPolling:
    def test_an_outage_backs_off_instead_of_spinning(self, monkeypatch):
        b = make_bridge()
        calls = []
        b.client.last_ok = False
        b.client.last_code = 0
        b.client.last_error = "ConnectionError"

        def fake_updates(offset=None, timeout=15):
            calls.append(time.monotonic())
            if len(calls) >= 3:
                b._stop_event.set()
            return []

        b.client.get_updates.side_effect = fake_updates
        waits = []
        monkeypatch.setattr(b._stop_event, "wait", lambda d=None: waits.append(d) or b._stop_event.is_set())
        b._polling_loop()
        assert waits[:2] == [1.0, 2.0]

    def test_recovery_resets_the_pause_and_handles_updates(self):
        b = make_bridge()
        seen = []
        b._handle_update = lambda u: seen.append(u["update_id"])
        b.client.last_ok = True
        batches = iter([[{"update_id": 5}], []])

        def fake_updates(offset=None, timeout=15):
            batch = next(batches, None)
            if batch is None:
                b._stop_event.set()
                return []
            return batch

        b.client.get_updates.side_effect = fake_updates
        b._polling_loop()
        assert seen == [5]


class TestAutomationNotice:
    def test_notification_is_html_and_escaped(self, monkeypatch):
        import json
        from src.automation import notify
        monkeypatch.setattr("config.get_telegram_token", lambda: "1:T")
        monkeypatch.setattr("config.get_telegram_chat_id", lambda: "42")
        seen = {}

        class R:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(notify.urllib.request, "urlopen", lambda req, timeout=None: seen.update(body=json.loads(req.data)) or R())
        assert notify.telegram_notify("Сбой <job>", "a & b" + "&" * 5000)
        body = seen["body"]
        assert body["parse_mode"] == "HTML" and body["text"].startswith("<b>Сбой &lt;job&gt;</b>")
        assert balanced(body["text"]) and not re.search(r"&\w{0,4}$", body["text"])
