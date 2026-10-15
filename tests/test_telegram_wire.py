"""The Telegram bridge over a real socket, against a strict local copy of the Bot API.

The other Telegram tests stub the HTTP session, so they cannot notice what only Telegram itself would refuse: an HTML tag it does
not know, a ``callback_data`` over 64 bytes, a reaction outside its list, a bot command with a capital letter, an edit that changes
nothing.  ``FakeBotAPI`` below enforces those documented rules and records every refusal in ``api.errors``; the end-to-end tests drive
the real polling loop, the live panel, approvals and the control panel through it and require that Telegram would have taken
everything the bridge sent.

It is a model of the documentation, not Telegram: a first run against the real service can still find a difference.
"""

import email.policy
import json
import re
import threading
import time
from email.parser import BytesParser
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from src.telegram import bot as tgbot
from src.telegram import client as tgclient
from src.telegram.bot import BOT_COMMANDS, TelegramBridge, approval_keyboard, reply_keyboard
from src.telegram.client import TelegramClient

CHAT = 111222

# ─── what Telegram documents ──────────────────────────────────────────────────

_HTML_TAGS = {"b", "strong", "i", "em", "u", "ins", "s", "strike", "del", "a", "code", "pre", "blockquote", "tg-spoiler", "span", "tg-emoji"}
_REACTIONS = set("👍👎❤🔥🥰👏😁🤔🤯😱🤬😢🎉🤩🤮💩🙏👌🕊🤡🥱🥴😍🐳🌚🌭💯🤣⚡🍌🏆💔🤨😐🍓🍾💋🖕😈😴😭🤓👻👀🎃🙈😇😨🤝✍🤗🫡🎅🎄☃💅🤪🗿🆒💘🙉🦄😘💊🙊😎👾🤷😡") | {"❤‍🔥", "👨‍💻", "🤷‍♂", "🤷‍♀"}
_CHAT_ACTIONS = {"typing", "upload_photo", "record_video", "upload_video", "record_voice", "upload_voice", "upload_document",
                 "choose_sticker", "find_location", "record_video_note", "upload_video_note"}


class _HtmlChecker(HTMLParser):
    """The parts of Telegram's HTML parse mode that fail a request."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.stack: List[str] = []
        self.error: Optional[str] = None
        self.visible = 0

    def _fail(self, message: str) -> None:
        if self.error is None:
            self.error = message

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        a = dict(attrs)
        if tag not in _HTML_TAGS:
            return self._fail(f'Unsupported start tag "{tag}"')
        if tag == "a" and not a.get("href"):
            return self._fail('Tag "a" needs an href')
        if tag == "span" and a.get("class") != "tg-spoiler":
            return self._fail('Tag "span" is only for class="tg-spoiler"')
        if tag == "blockquote" and "blockquote" in self.stack:
            return self._fail("blockquotes can't be nested")
        if self.stack and self.stack[-1] == "pre" and tag != "code":
            return self._fail("only code can be nested in pre")
        self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self._fail(f'Unsupported start tag "{tag}/"')

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            return self._fail(f'Can\'t find end tag corresponding to start tag "{self.stack[-1] if self.stack else tag}"')
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        if "<" in data:
            self._fail('Unsupported start tag ""')
        self.visible += len(data)

    def handle_entityref(self, name: str) -> None:
        self.visible += 1

    def handle_charref(self, name: str) -> None:
        self.visible += 1


def check_html(text: str) -> Tuple[Optional[str], int]:
    """(error or None, characters Telegram would count)."""
    p = _HtmlChecker()
    p.feed(text)
    p.close()
    if p.error is None and p.stack:
        p.error = f'Can\'t find end tag corresponding to start tag "{p.stack[-1]}"'
    return p.error, p.visible


def check_markup(markup: Any) -> Optional[str]:
    if markup is None:
        return None
    if not isinstance(markup, dict):
        return "reply markup is not an object"
    if "inline_keyboard" in markup:
        rows = markup["inline_keyboard"]
        if not isinstance(rows, list) or any(not isinstance(r, list) for r in rows):
            return "inline_keyboard must be a list of rows"
        total = 0
        for row in rows:
            if len(row) > 8:
                return "too many buttons in a row"
            for b in row:
                total += 1
                if not isinstance(b, dict) or not str(b.get("text") or "").strip():
                    return "button text is empty"
                kinds = [k for k in ("callback_data", "url", "switch_inline_query", "switch_inline_query_current_chat", "web_app") if k in b]
                if len(kinds) != 1:
                    return "BUTTON_TYPE_INVALID"
                if "callback_data" in b and not 1 <= len(str(b["callback_data"]).encode("utf-8")) <= 64:
                    return "BUTTON_DATA_INVALID"
                if "url" in b and not re.match(r"(https?|tg)://", str(b["url"])):
                    return "BUTTON_URL_INVALID"
        if total > 100:
            return "too many buttons"
        return None
    if "keyboard" in markup:
        for row in markup["keyboard"]:
            for b in row:
                if not str(b.get("text") if isinstance(b, dict) else b).strip():
                    return "keyboard button text is empty"
        if len(str(markup.get("input_field_placeholder") or "")) > 64:
            return "input_field_placeholder is too long"
        return None
    if markup.get("remove_keyboard"):
        return None
    return "reply markup is not valid"


def _parse_multipart(content_type: str, raw: bytes) -> Tuple[Dict[str, str], Dict[str, Tuple[str, bytes]]]:
    msg = BytesParser(policy=email.policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw)
    fields: Dict[str, str] = {}
    files: Dict[str, Tuple[str, bytes]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None:
            files[name] = (filename, data)
        else:
            fields[name] = data.decode("utf-8")
    return fields, files


# ─── the fake service ─────────────────────────────────────────────────────────

class FakeBotAPI:
    TOKEN = "123456:TEST-token_x"

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.errors: List[Tuple[str, str]] = []          # refusals the fake made by the rules above (not scripted ones)
        self.noops: List[int] = []                       # edits that changed nothing: Telegram says so with a 400 the client treats as success
        self.script: Dict[str, List[Any]] = {}           # method -> replies to give first: a Telegram-style dict, or ("raw", status, bytes)
        self.updates: List[Dict[str, Any]] = []
        self.files: Dict[str, bytes] = {}                # remote path -> content
        self.file_ids: Dict[str, str] = {}               # file_id -> remote path
        self.messages: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._queries: Dict[str, int] = {}               # callback_query_id -> times answered
        self._next_id = 100
        self._next_update = 1
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, payload: Any) -> None:
                raw = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html" if isinstance(payload, bytes) else "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self._send(*api.handle_post(self.path, self.headers.get("Content-Type", ""), raw))

            def do_GET(self) -> None:
                self._send(*api.handle_get(self.path))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.05), daemon=True, name="FakeBotAPI").start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    # ── what the tests do ──────────────────────────────────────────────────

    def push(self, update: Dict[str, Any]) -> int:
        with self._cv:
            update = {"update_id": self._next_update, **update}
            self._next_update += 1
            self.updates.append(update)
            self._cv.notify_all()
            return update["update_id"]

    def say(self, text: str = "", chat: int = CHAT, **extra: Any) -> int:
        message = {"message_id": self._take_id(), "date": int(time.time()), "chat": {"id": chat, "type": "private"},
                   "from": {"id": chat, "is_bot": False, "first_name": "Owner"}, **extra}
        if text:
            message["text"] = text
        self.push({"message": message})
        return message["message_id"]

    def press(self, data: str, message_id: int, chat: int = CHAT) -> str:
        qid = f"cq{self._take_id()}"
        with self._lock:
            self._queries[qid] = 0
        self.push({"callback_query": {"id": qid, "from": {"id": chat, "is_bot": False, "first_name": "Owner"}, "data": data,
                                      "message": {"message_id": message_id, "chat": {"id": chat, "type": "private"}}}})
        return qid

    def add_file(self, file_id: str, remote: str, data: bytes) -> None:
        self.file_ids[file_id] = remote
        self.files[remote] = data

    def sent(self, method: str, contains: str = "") -> List[Dict[str, Any]]:
        return [c["body"] for c in list(self.calls) if c["method"] == method and contains in json.dumps(c["body"], ensure_ascii=False)]

    def wait_for(self, predicate: Callable[[], Any], timeout: float = 10.0) -> bool:
        end = time.monotonic() + timeout
        with self._cv:
            while True:
                if predicate():
                    return True
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(min(left, 0.2))

    def idle(self, timeout: float = 10.0) -> bool:
        """Every pushed update was taken AND the bridge asked for more (so it finished handling them)."""
        return self.wait_for(lambda: not self.updates, timeout)

    # ── the service ────────────────────────────────────────────────────────

    def _take_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def _ok(self, result: Any) -> Tuple[int, Dict[str, Any]]:
        return 200, {"ok": True, "result": result}

    def _refuse(self, method: str, code: int, text: str) -> Tuple[int, Dict[str, Any]]:
        desc = f"Bad Request: {text}" if code == 400 else text
        with self._lock:
            self.errors.append((method, desc))
        return code, {"ok": False, "error_code": code, "description": desc}

    def handle_get(self, path: str) -> Tuple[int, Any]:
        m = re.fullmatch(rf"/file/bot{re.escape(self.TOKEN)}/(.+)", path)
        if m and m.group(1) in self.files:
            return 200, self.files[m.group(1)]
        return 404, {"ok": False, "error_code": 404, "description": "Not Found"}

    def handle_post(self, path: str, content_type: str, raw: bytes) -> Tuple[int, Any]:
        m = re.fullmatch(r"/bot([^/]+)/(\w+)", path)
        if not m or m.group(1) != self.TOKEN:
            return 401, {"ok": False, "error_code": 401, "description": "Unauthorized"}
        method = m.group(2)
        files: Dict[str, Tuple[str, bytes]] = {}
        if content_type.startswith("multipart/form-data"):
            body, files = _parse_multipart(content_type, raw)
        else:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        with self._cv:
            self.calls.append({"method": method, "body": body, "files": files})
            self._cv.notify_all()
            queued = self.script.get(method)
            scripted = queued.pop(0) if queued else None
        if scripted is not None:
            if isinstance(scripted, tuple):
                return scripted[1], scripted[2]
            return int(scripted.get("error_code") or 200), scripted
        handler = getattr(self, f"_m_{method}", None)
        if handler is None:
            return self._refuse(method, 404, "Not Found")
        return handler(body, files)

    def _text_problem(self, body: Dict[str, Any], key: str, limit: int) -> Optional[str]:
        text = body.get(key)
        if not isinstance(text, str) or not text.strip():
            return "message text is empty" if key == "text" else None
        visible = len(text)
        if body.get("parse_mode") == "HTML":
            error, visible = check_html(text)
            if error:
                return f"can't parse entities: {error}"
        if visible > limit:
            return "message is too long" if key == "text" else "message caption is too long"
        return None

    def _m_getMe(self, body, files):
        return self._ok({"id": 1, "is_bot": True, "first_name": "Argent", "username": "argent_test_bot"})

    def _m_deleteWebhook(self, body, files):
        return self._ok(True)

    def _m_getUpdates(self, body, files):
        with self._cv:
            offset = body.get("offset")
            if offset is not None:
                self.updates = [u for u in self.updates if u["update_id"] >= offset]
                self._cv.notify_all()
            batch = list(self.updates)
        if not batch:
            time.sleep(0.05)
        return self._ok(batch)

    def _m_setMyCommands(self, body, files):
        cmds = body.get("commands")
        if not isinstance(cmds, list) or not 0 < len(cmds) <= 100:
            return self._refuse("setMyCommands", 400, "BOT_COMMANDS_TOO_MUCH")
        for c in cmds:
            if not re.fullmatch(r"[a-z0-9_]{1,32}", str(c.get("command", ""))):
                return self._refuse("setMyCommands", 400, "BOT_COMMAND_INVALID")
            if not 1 <= len(str(c.get("description", ""))) <= 256:
                return self._refuse("setMyCommands", 400, "BOT_COMMAND_DESCRIPTION_INVALID")
        return self._ok(True)

    def _m_sendChatAction(self, body, files):
        if body.get("action") not in _CHAT_ACTIONS:
            return self._refuse("sendChatAction", 400, "wrong parameter action in request")
        return self._ok(True)

    def _m_sendMessage(self, body, files):
        if body.get("chat_id") in (None, ""):
            return self._refuse("sendMessage", 400, "chat_id is empty")
        problem = self._text_problem(body, "text", 4096) or check_markup(body.get("reply_markup"))
        if problem:
            return self._refuse("sendMessage", 400, problem)
        mid = self._take_id()
        self.messages[(str(body["chat_id"]), mid)] = {"text": body["text"], "markup": body.get("reply_markup")}
        return self._ok({"message_id": mid, "date": int(time.time()), "chat": {"id": body["chat_id"]}, "text": body["text"]})

    def _m_editMessageText(self, body, files):
        key = (str(body.get("chat_id")), body.get("message_id"))
        if key not in self.messages:
            return self._refuse("editMessageText", 400, "message to edit not found")
        problem = self._text_problem(body, "text", 4096) or check_markup(body.get("reply_markup"))
        if problem:
            return self._refuse("editMessageText", 400, problem)
        previous, markup = self.messages[key], body.get("reply_markup")
        if previous["text"] == body["text"] and (previous["markup"] or None) == (markup or None):
            self.noops.append(key[1])
            return 400, {"ok": False, "error_code": 400, "description": "Bad Request: message is not modified: specified new message content "
                         "and reply markup are exactly the same as a current content and reply markup of the message"}
        self.messages[key] = {"text": body["text"], "markup": markup}     # an edit without reply_markup takes the keyboard away
        return self._ok({"message_id": key[1], "chat": {"id": body["chat_id"]}, "text": body["text"]})

    def _m_editMessageReplyMarkup(self, body, files):
        key = (str(body.get("chat_id")), body.get("message_id"))
        if key not in self.messages:
            return self._refuse("editMessageReplyMarkup", 400, "message to edit not found")
        problem = check_markup(body.get("reply_markup"))
        if problem:
            return self._refuse("editMessageReplyMarkup", 400, problem)
        self.messages[key]["markup"] = body.get("reply_markup")
        return self._ok({"message_id": key[1]})

    def _m_deleteMessage(self, body, files):
        if self.messages.pop((str(body.get("chat_id")), body.get("message_id")), None) is None:
            return self._refuse("deleteMessage", 400, "message to delete not found")
        return self._ok(True)

    def _m_answerCallbackQuery(self, body, files):
        qid = str(body.get("callback_query_id") or "")
        with self._lock:
            known = qid in self._queries
            times = self._queries.get(qid, 0)
            if known:
                self._queries[qid] = times + 1
        if not known or times:
            return self._refuse("answerCallbackQuery", 400, "query is too old and response timeout expired or query ID is invalid")
        if len(str(body.get("text") or "")) > 200:
            return self._refuse("answerCallbackQuery", 400, "MESSAGE_TOO_LONG")
        return self._ok(True)

    def _m_setMessageReaction(self, body, files):
        reaction = body.get("reaction")
        if not isinstance(reaction, list) or len(reaction) > 1:
            return self._refuse("setMessageReaction", 400, "REACTION_INVALID")
        for r in reaction:
            if r.get("type") != "emoji" or r.get("emoji", "").replace("️", "") not in {e.replace("️", "") for e in _REACTIONS}:
                return self._refuse("setMessageReaction", 400, "REACTION_INVALID")
        return self._ok(True)

    def _m_getFile(self, body, files):
        remote = self.file_ids.get(str(body.get("file_id")))
        if remote is None:
            return self._refuse("getFile", 400, "invalid file_id")
        return self._ok({"file_id": body["file_id"], "file_path": remote, "file_size": len(self.files[remote])})

    def _upload(self, method: str, field: str, body, files):
        if not body.get("chat_id"):
            return self._refuse(method, 400, "chat_id is empty")
        if field not in files:
            return self._refuse(method, 400, "there is no %s in the request" % field)
        problem = self._text_problem(body, "caption", 1024)
        if problem:
            return self._refuse(method, 400, problem)
        name, data = files[field]
        if method == "sendPhoto" and not (data.startswith(b"\x89PNG") or data.startswith(b"\xff\xd8")):
            return self._refuse(method, 400, "IMAGE_PROCESS_FAILED")
        return self._ok({"message_id": self._take_id(), "chat": {"id": body["chat_id"]}, field: {"file_name": name, "file_size": len(data)}})

    def _m_sendDocument(self, body, files):
        return self._upload("sendDocument", "document", body, files)

    def _m_sendPhoto(self, body, files):
        return self._upload("sendPhoto", "photo", body, files)


# ─── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def api(monkeypatch, tmp_path):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    fake = FakeBotAPI()
    monkeypatch.setattr(tgclient, "TELEGRAM_API_BASE", f"{fake.url}/bot")
    monkeypatch.setattr(tgclient, "TELEGRAM_FILE_BASE", f"{fake.url}/file/bot")
    monkeypatch.chdir(tmp_path)
    yield fake
    fake.close()


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "cfg.json")
    monkeypatch.setattr(config, "_CONFIG_CACHE", None)
    config.set_telegram_chat_id(str(CHAT))
    config.set_telegram_mode("isolated")
    yield
    config._CONFIG_CACHE = None


# ─── the rules themselves (a check that cannot fail proves nothing) ───────────

class TestTheFakeEnforcesTheDocs:
    @pytest.mark.parametrize("text", ["<br>", "<b>x</i>", "<b>x", "a < b", "<span>x</span>", "<a>x</a>", "<pre><b>x</b></pre>",
                                      "<blockquote><blockquote>x</blockquote></blockquote>", "<img src='x'/>"])
    def test_bad_html_is_refused(self, text):
        assert check_html(text)[0], text

    @pytest.mark.parametrize("text", ["<b>x</b>", "a &lt; b &amp; c", "<pre><code class=\"language-python\">x = 1</code></pre>",
                                      "<blockquote expandable>x\ny</blockquote>", "<b>bold <i>both</i></b>", "<a href=\"https://x.io\">x</a>",
                                      "<span class=\"tg-spoiler\">s</span>", "<code>a &lt; b</code>"])
    def test_good_html_passes(self, text):
        assert check_html(text)[0] is None, text

    def test_markup_rules(self):
        assert check_markup({"inline_keyboard": [[{"text": "ok", "callback_data": "x" * 64}]]}) is None
        assert check_markup({"inline_keyboard": [[{"text": "ok", "callback_data": "x" * 65}]]}) == "BUTTON_DATA_INVALID"
        assert check_markup({"inline_keyboard": [[{"text": "ok", "callback_data": "я" * 33}]]}) == "BUTTON_DATA_INVALID"      # 66 bytes
        assert check_markup({"inline_keyboard": [[{"text": "ok"}]]}) == "BUTTON_TYPE_INVALID"
        assert check_markup({"inline_keyboard": []}) is None

    def test_a_scripted_refusal_is_not_counted_as_the_bridge_fault(self, api):
        api.script["sendMessage"] = [{"ok": False, "error_code": 429, "description": "slow", "parameters": {"retry_after": 600}}]
        c = TelegramClient(api.TOKEN)
        c.session.trust_env = False
        assert c._call("sendMessage", {"chat_id": 1, "text": "x"}) is None and api.errors == [] and c.last_code == 429


# ─── the client, over the wire ────────────────────────────────────────────────

class TestClientOverTheWire:
    def _client(self, api):
        c = TelegramClient(api.TOKEN)
        c.session.trust_env = False
        return c

    def test_get_me_and_a_wrong_token(self, api):
        assert self._client(api).get_me()["username"] == "argent_test_bot"
        bad = TelegramClient("999:wrong")
        bad.session.trust_env = False
        assert bad.get_me() is None and bad.last_code == 401 and not bad.last_network

    def test_send_message_carries_what_the_api_expects(self, api):
        c = self._client(api)
        res = c.send_message(CHAT, "<b>Привет</b> &amp; мир", reply_markup=reply_keyboard(), reply_to=5, disable_notification=True)
        assert res["message_id"] and not api.errors
        body = api.sent("sendMessage")[0]
        assert body["parse_mode"] == "HTML" and body["link_preview_options"] == {"is_disabled": True}
        assert body["reply_parameters"] == {"message_id": 5, "allow_sending_without_reply": True} and body["disable_notification"] is True

    def test_html_telegram_refuses_is_resent_as_plain_text(self, api):
        c = self._client(api)
        assert c.send_message(CHAT, "<b>Привет</b> <blink>мир</blink>")
        first, second = api.sent("sendMessage")
        assert first["parse_mode"] == "HTML" and "parse_mode" not in second and second["text"] == "Привет мир"
        assert [e[0] for e in api.errors] == ["sendMessage"]            # the refusal was the fake's own, and the retry was accepted

    def test_a_rate_limit_is_waited_out_once(self, api, monkeypatch):
        slept: List[float] = []
        monkeypatch.setattr(tgclient, "time", SimpleNamespace(sleep=slept.append))
        api.script["sendMessage"] = [{"ok": False, "error_code": 429, "description": "Too Many Requests: retry after 2", "parameters": {"retry_after": 2}}]
        res = self._client(api).send_message(CHAT, "hi")
        assert res and len(api.sent("sendMessage")) == 2 and slept and 2 <= slept[0] < 3

    def test_a_long_rate_limit_is_not_waited_out(self, api, monkeypatch):
        slept: List[float] = []
        monkeypatch.setattr(tgclient, "time", SimpleNamespace(sleep=slept.append))
        api.script["sendMessage"] = [{"ok": False, "error_code": 429, "description": "slow", "parameters": {"retry_after": 600}}]
        c = self._client(api)
        assert c.send_message(CHAT, "hi") is None and not slept and c.last_code == 429

    def test_editing_to_the_same_text_counts_as_success(self, api):
        c = self._client(api)
        mid = c.send_message(CHAT, "same", reply_markup={"inline_keyboard": [[{"text": "a", "callback_data": "a"}]]})["message_id"]
        out = c.edit_message_text(CHAT, mid, "same", reply_markup={"inline_keyboard": [[{"text": "a", "callback_data": "a"}]]})
        assert out == {"message_id": mid, "unchanged": True} and c.last_ok

    def test_an_edit_without_a_keyboard_takes_the_keyboard_away(self, api):
        c = self._client(api)
        mid = c.send_message(CHAT, "x", reply_markup={"inline_keyboard": [[{"text": "a", "callback_data": "a"}]]})["message_id"]
        c.edit_message_text(CHAT, mid, "y", reply_markup={"inline_keyboard": []})
        assert not api.messages[(str(CHAT), mid)]["markup"]["inline_keyboard"]

    def test_editing_a_message_that_is_gone_fails_cleanly(self, api):
        c = self._client(api)
        assert c.edit_message_text(CHAT, 424242, "x") is None and not c.last_network and c.last_code == 400

    def test_updates_are_acknowledged_by_offset(self, api):
        for n in (1, 2, 3):
            api.say(f"m{n}")
        c = self._client(api)
        got = c.get_updates(offset=None, timeout=1)
        assert [u["message"]["text"] for u in got] == ["m1", "m2", "m3"]
        assert [u["message"]["text"] for u in c.get_updates(offset=got[1]["update_id"] + 1, timeout=1)] == ["m3"]
        assert api.sent("getUpdates")[0]["allowed_updates"] == ["message", "callback_query"]

    def test_a_document_goes_up_as_multipart(self, api):
        c = self._client(api)
        assert c.send_document_bytes(CHAT, "отчёт.md", "# Привет\n".encode("utf-8"), caption="📎 <b>Файл</b>")
        call = [x for x in api.calls if x["method"] == "sendDocument"][0]
        assert call["body"]["chat_id"] == str(CHAT) and call["body"]["parse_mode"] == "HTML" and "<b>Файл</b>" in call["body"]["caption"]
        assert call["files"]["document"] == ("отчёт.md", "# Привет\n".encode("utf-8")) and not api.errors

    def test_a_bad_caption_is_retried_as_plain_text(self, api):
        c = self._client(api)
        assert c.send_document_bytes(CHAT, "a.txt", b"x", caption="<blink>hi</blink>")
        calls = [x["body"] for x in api.calls if x["method"] == "sendDocument"]
        assert calls[0].get("parse_mode") == "HTML" and "parse_mode" not in calls[1] and calls[1]["caption"] == "hi"

    def test_a_picture_that_is_not_one_travels_as_a_file(self, api, tmp_path):
        p = tmp_path / "shot.png"
        p.write_bytes(b"not really a picture")
        assert self._client(api).send_photo(CHAT, str(p))
        assert [c["method"] for c in api.calls] == ["sendPhoto", "sendDocument"]

    def test_a_real_png_goes_as_a_photo(self, api, tmp_path):
        p = tmp_path / "shot.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        assert self._client(api).send_photo(CHAT, str(p), caption="вид")
        assert [c["method"] for c in api.calls] == ["sendPhoto"] and not api.errors

    def test_too_big_an_upload_is_refused_before_it_is_sent(self, api, monkeypatch):
        monkeypatch.setattr(tgclient, "MAX_UPLOAD", 10)
        c = self._client(api)
        assert c.send_document_bytes(CHAT, "big.bin", b"x" * 11) is None and c.last_code == 413 and not api.calls

    def test_a_file_comes_down(self, api, tmp_path):
        api.add_file("F1", "documents/file_1.txt", "привет".encode("utf-8"))
        c = self._client(api)
        info = c.get_file("F1")
        dest = tmp_path / "in" / "f.txt"
        assert info["file_path"] == "documents/file_1.txt" and c.download_file(info["file_path"], str(dest))
        assert dest.read_text(encoding="utf-8") == "привет"
        assert not c.download_file("documents/missing.txt", str(tmp_path / "x"))

    def test_a_file_over_the_download_limit_is_dropped(self, api, tmp_path, monkeypatch):
        monkeypatch.setattr(tgclient, "MAX_DOWNLOAD", 3)
        api.add_file("F2", "documents/big.bin", b"12345")
        assert not self._client(api).download_file("documents/big.bin", str(tmp_path / "big.bin")) and not (tmp_path / "big.bin").exists()

    def test_a_server_that_is_gone_is_a_network_error_not_an_exception(self, api):
        c = self._client(api)
        api.close()
        assert c.send_message(CHAT, "<b>x</b>") is None and c.last_network and not c.last_ok
        assert c.get_updates(timeout=1) == []

    def test_a_html_error_page_is_a_network_error(self, api):
        api.script["getMe"] = [("raw", 502, b"<html>Bad Gateway</html>")]
        c = self._client(api)
        assert c.get_me() is None and c.last_network

    def test_set_my_commands_accepts_what_the_bridge_publishes(self, api):
        from src.telegram.bot import publish_commands
        c = self._client(api)
        assert publish_commands(c) and not api.errors
        sent = api.sent("setMyCommands")[0]["commands"]
        assert [x["command"] for x in sent] == [c for c, _ in BOT_COMMANDS]

    def test_every_approval_keyboard_fits_the_button_limit(self, api):
        long_key = "run_command:" + "x" * 80
        for aid in (7, 10 ** 9, "tui-2f6c9b1e-aa"):
            for destructive in (False, True):
                assert check_markup(approval_keyboard("appr", aid, long_key, destructive)) is None
                assert check_markup(approval_keyboard("tui", aid, None, destructive)) is None


# ─── the whole bridge, over the wire ──────────────────────────────────────────

class ScriptedSession:
    """Plays a fixed event list; "WAIT" blocks until the user answered an approval."""

    def __init__(self, events: Optional[List[Any]] = None) -> None:
        self.busy = False
        self.vibe = False
        self.started: List[str] = []
        self.replies: List[Tuple[Any, str]] = []
        self.cancelled = False
        self.script = list(events or [])
        self._answered = threading.Event()

    def start(self, text):
        self.busy = True
        self.started.append(text)

    def get_event(self):
        ev = self.script.pop(0) if self.script else {"type": "done"}
        if ev == "WAIT":
            self._answered.wait(10)
            return self.get_event()
        if ev.get("type") == "done":
            self.busy = False
        return ev

    def reply_approval(self, aid, decision):
        self.replies.append((aid, decision))
        self._answered.set()

    def cancel(self):
        self.cancelled, self.busy = True, False

    def state(self):
        return {"model": "test-model", "provider": "test", "tier": "complex", "context": {"tokens": 50, "max": 100, "percent": 50.0}, "vibe": self.vibe}

    def set_vibe(self, on):
        self.vibe = bool(on)
        return self.vibe


@pytest.fixture
def bridge(api, monkeypatch):
    made: List[TelegramBridge] = []
    monkeypatch.setattr(tgbot, "AgentSession", lambda *a, **k: ScriptedSession())

    def make(events=None) -> Tuple[TelegramBridge, ScriptedSession]:
        session = ScriptedSession(events)
        b = TelegramBridge(session=session)
        ok, message = b.start(token=api.TOKEN)
        assert ok, message
        made.append(b)
        return b, session

    yield make
    for b in made:
        b.stop()


def _all_texts(api: FakeBotAPI) -> List[str]:
    return [b["text"] for b in api.sent("sendMessage") + api.sent("editMessageText")]


class TestBridgeOverTheWire:
    def test_it_publishes_its_commands_and_clears_a_webhook_on_start(self, api, bridge):
        bridge()
        assert api.wait_for(lambda: api.sent("setMyCommands")) and api.sent("deleteWebhook")
        assert not api.errors

    def test_a_turn_from_message_to_formatted_answer(self, api, bridge):
        answer = ("Готово: **файл** прочитан. `x < y` & done\n\n```python\nprint(1)\n```\n\n| имя | код |\n|---|---|\n| foo | 1 |\n\n"
                  "> цитата\n\n- пункт\n  - вложенный\n\n[ссылка](https://example.com/a?b=1&c=2)")
        _, session = bridge([
            {"type": "tool_start", "name": "read_file", "args": {"path": "README.md"}},
            {"type": "tool_end", "name": "read_file", "result": "ok", "duration_secs": 0.4},
            {"type": "turn_snapshot", "data": {"total_s": 72, "tool_calls": 1, "prompt_tokens": 12300, "completion_tokens": 450, "ctx_percent": 34.2}},
            {"type": "content", "text": answer},
            {"type": "done"},
        ])
        mid = api.say("прочитай README")
        assert api.wait_for(lambda: api.sent("sendMessage", "ссылка")), _all_texts(api)
        assert api.wait_for(lambda: any(r["reaction"] and r["reaction"][0]["emoji"] == "👍" for r in api.sent("setMessageReaction")))
        assert session.started == ["прочитай README"]
        assert api.errors == [], api.errors
        reactions = [r["reaction"][0]["emoji"] for r in api.sent("setMessageReaction") if r["reaction"]]
        assert reactions[0] == "👀" and reactions[-1] == "👍" and all(r["message_id"] == mid for r in api.sent("setMessageReaction"))
        assert api.sent("sendChatAction")[0]["action"] == "typing"

        panel = [b for b in api.sent("sendMessage") if "Работаю" in b["text"]][0]
        assert panel["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "stop_turn"
        collapsed = [b for b in api.sent("editMessageText") if "Готово" in b["text"] and "blockquote" in b["text"]]
        assert collapsed and collapsed[-1]["reply_markup"] == {"inline_keyboard": []}

        final = api.sent("sendMessage", "ссылка")[0]["text"]
        assert "<b>файл</b>" in final and "x &lt; y" in final and "&amp; done" in final
        assert '<pre><code class="language-python">' in final and "<blockquote>" in final and "• пункт" in final
        assert '<a href="https://example.com/a?b=1&amp;c=2">ссылка</a>' in final and "1 мин 12 с" in final

    def test_a_very_long_answer_is_a_short_preview_and_a_file(self, api, bridge):
        text = "\n\n".join(f"Абзац {i}. " + "слово " * 60 for i in range(80))
        bridge([{"type": "content", "text": text}, {"type": "done"}])
        api.say("напиши много")
        assert api.wait_for(lambda: api.sent("sendDocument")), _all_texts(api)
        doc = [c for c in api.calls if c["method"] == "sendDocument"][0]
        assert doc["files"]["document"][0] == "answer.md" and doc["files"]["document"][1].decode("utf-8") == text.strip()
        assert api.errors == [], api.errors
        assert any("полный ответ — во вложении" in b["text"] for b in api.sent("sendMessage"))

    def test_an_approval_card_is_answered_from_the_phone(self, api, bridge):
        _, session = bridge([
            {"type": "approval_request", "id": 7, "action": "rm -rf build && echo <done> & more", "destructive": False, "grant_key": "run_command:rm"},
            "WAIT",
            {"type": "content", "text": "Сделано."},
            {"type": "done"},
        ])
        api.say("очисти сборку")
        assert api.wait_for(lambda: api.sent("sendMessage", "Нужно подтверждение")), _all_texts(api)
        card = api.sent("sendMessage", "Нужно подтверждение")[0]
        assert "&lt;done&gt; &amp; more" in card["text"]
        buttons = [b for row in card["reply_markup"]["inline_keyboard"] for b in row]
        assert {b["callback_data"].rsplit(":", 1)[1] for b in buttons} == {"once", "always", "deny"}
        card_id = [m for (chat, m), v in api.messages.items() if "Нужно подтверждение" in v["text"]][0]
        qid = api.press("appr:7:once", card_id)
        assert api.wait_for(lambda: session.replies) and session.replies == [(7, "once")]
        assert api.wait_for(lambda: api.sent("sendMessage", "Сделано")), _all_texts(api)
        assert api.wait_for(lambda: api._queries[qid] == 1)
        card_now = lambda: api.messages[(str(CHAT), card_id)]
        assert api.wait_for(lambda: not (card_now()["markup"] or {}).get("inline_keyboard")), "the buttons stayed on the answered card"
        assert "Разрешено" in card_now()["text"]
        assert api.errors == [], api.errors

    def test_a_stranger_gets_nothing_and_starts_nothing(self, api, bridge):
        _, session = bridge()
        api.say("удали всё", chat=999)
        assert api.idle()
        assert session.started == [] and not api.sent("sendMessage") and api.errors == []

    def test_a_stranger_cannot_press_buttons(self, api, bridge):
        _, session = bridge([{"type": "approval_request", "id": 3, "action": "x", "destructive": True}, "WAIT", {"type": "done"}])
        api.say("сделай")
        assert api.wait_for(lambda: api.sent("sendMessage", "Деструктивное"))
        card_id = [m for (chat, m), v in api.messages.items() if "Деструктивное" in v["text"]][0]
        api.press("appr:3:once", card_id, chat=999)
        assert api.idle()
        assert session.replies == []
        session.reply_approval(3, "deny")           # let the scripted turn finish

    def test_a_message_that_waited_too_long_is_not_obeyed(self, api, bridge):
        _, session = bridge()
        api.say("старое", date=int(time.time()) - 3600)
        assert api.wait_for(lambda: api.sent("sendMessage", "Пропущено сообщений"))
        assert session.started == [] and api.errors == []

    def test_every_command_and_panel_button_produces_something_telegram_accepts(self, api, bridge):
        _, session = bridge()
        for verb in ["/start", "/menu", "/help", "/status", "/queue", "/mode", "/mode shared", "/mode isolated", "/vibe", "/vibe off",
                     "/file", "/file nothing-here.txt", "/new", "/new_confirm", "/stop", "/unknown_thing"]:
            api.say(verb)
        assert api.idle(20)
        assert api.errors == [], api.errors
        menu_id = [m for (chat, m), v in api.messages.items() if "на связи" in v["text"]][0]
        for data in ["ui:status", "ui:help", "ui:queue", "ui:mode", "ui:mode:shared", "ui:mode:isolated", "ui:new", "ui:new:yes",
                     "ui:vibe", "ui:stop", "ui:menu", "q:clear", "q:del:3"]:
            api.press(data, menu_id)
        assert api.idle(20)
        assert api.errors == [], api.errors
        assert all(v >= 1 for v in api._queries.values()), "a button press was left without an answer"

    def test_a_file_the_user_sends_reaches_the_prompt(self, api, bridge, tmp_path):
        _, session = bridge([{"type": "done"}])
        api.add_file("F9", "documents/file_9.txt", "заметки\n".encode("utf-8"))
        api.say("посмотри", document={"file_id": "F9", "file_name": "заметки.txt", "file_size": 14, "mime_type": "text/plain"})
        assert api.wait_for(lambda: session.started)
        saved = list((tmp_path / ".argent" / "artifacts" / "telegram_uploads").glob("*заметки.txt"))
        assert len(saved) == 1 and saved[0].read_text(encoding="utf-8") == "заметки\n"
        assert "telegram_uploads" in session.started[0] and session.started[0].startswith("посмотри")

    def test_it_survives_an_outage_and_goes_on(self, api, bridge):
        api.script["getUpdates"] = [("raw", 502, b"<html>Bad Gateway</html>"), {"ok": False, "error_code": 409, "description": "Conflict: terminated by other getUpdates request"}]
        _, session = bridge([{"type": "content", "text": "жив"}, {"type": "done"}])
        api.say("ты здесь?")
        assert api.wait_for(lambda: api.sent("sendMessage", "жив"), timeout=20), _all_texts(api)
        assert session.started == ["ты здесь?"] and api.errors == []

    def test_a_second_message_while_busy_waits_in_the_queue(self, api, bridge):
        _, session = bridge([{"type": "tool_start", "name": "run_command", "args": {"command": "sleep"}}, "WAIT", {"type": "content", "text": "первое"}, {"type": "done"}])
        api.say("первое задание")
        assert api.wait_for(lambda: session.started)
        api.say("второе задание")
        assert api.wait_for(lambda: api.sent("sendMessage", "В очереди"))
        card = api.sent("sendMessage", "В очереди")[0]
        assert card["reply_markup"]["inline_keyboard"][0][0]["callback_data"].startswith("q:del:")
        session.reply_approval(0, "once")            # release the first turn
        assert api.wait_for(lambda: api.sent("sendMessage", "Из очереди"), timeout=15), _all_texts(api)
        assert api.errors == [], api.errors
