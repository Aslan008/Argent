"""A small Telegram Bot API client.

What it adds over a bare ``requests.post``:

* a rate-limit answer (``429``) is waited out once instead of being dropped;
* HTML that Telegram refuses is retried as plain text -- *stripped* of its tags, so
  the reader never sees ``<b>`` literally;
* ``edit_message_text`` treats "message is not modified" as success;
* ``last_ok`` / ``last_error`` / ``last_code`` say what the previous call did, so the
  polling loop can tell "nothing new" from "the network is down" without exceptions.
"""

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from logger import get_logger
from src.telegram.render import MAX_HTML, split_html, strip_html

log = get_logger("telegram_bot")

TELEGRAM_API_BASE = "https://api.telegram.org/bot"
TELEGRAM_FILE_BASE = "https://api.telegram.org/file/bot"
MAX_MSG_LEN = 4000
MAX_UPLOAD = 49 * 1024 * 1024      # the Bot API refuses uploads above 50 MB
MAX_DOWNLOAD = 20 * 1024 * 1024    # ... and downloads above 20 MB
_MAX_RETRY_AFTER = 30.0


class TelegramClient:
    """Lightweight Telegram Bot API client using standard requests."""

    def __init__(self, token: str, timeout: int = 30):
        self.token = token.strip()
        self.base_url = f"{TELEGRAM_API_BASE}{self.token}"
        self.timeout = timeout
        self.session = requests.Session()
        self.last_ok = True
        self.last_error = ""
        self.last_code = 0
        self.last_network = False
        self._lock = threading.Lock()

    # ── transport ──────────────────────────────────────────────────────────

    def _remember(self, ok: bool, error: str = "", code: int = 0, network: bool = False) -> None:
        self.last_ok, self.last_error, self.last_code, self.last_network = ok, error, code, network

    def _call(self, method: str, payload: Optional[Dict[str, Any]] = None, req_timeout: Optional[int] = None,
              files: Optional[Dict[str, Any]] = None, _retried: bool = False) -> Optional[Any]:
        url = f"{self.base_url}/{method}"
        try:
            if files:
                resp = self.session.post(url, data=payload or {}, files=files, timeout=req_timeout or 120)
            else:
                resp = self.session.post(url, json=payload or {}, timeout=req_timeout or self.timeout)
            data = resp.json()
        except Exception as e:
            self._remember(False, f"{type(e).__name__}", 0, network=True)
            log.error("Telegram API call %s failed: %s", method, type(e).__name__)
            return None
        if not isinstance(data, dict) or not data.get("ok"):
            data = data if isinstance(data, dict) else {}
            desc, code = str(data.get("description") or ""), int(data.get("error_code") or 0)
            self._remember(False, desc, code)
            wait = (data.get("parameters") or {}).get("retry_after")
            if code == 429 and wait and not _retried and float(wait) <= _MAX_RETRY_AFTER:
                log.info("Telegram asked to slow down on %s: waiting %ss", method, wait)
                time.sleep(float(wait) + 0.2)
                return self._call(method, payload, req_timeout, files, _retried=True)
            if "not modified" not in desc:
                log.warning("Telegram API %s returned error: %s", method, desc)
            return None
        self._remember(True)
        return data.get("result")

    # ── reading ────────────────────────────────────────────────────────────

    def get_me(self) -> Optional[Dict[str, Any]]:
        """Test the token and get bot metadata."""
        return self._call("getMe", req_timeout=10)

    def get_updates(self, offset: Optional[int] = None, timeout: int = 20) -> List[Dict[str, Any]]:
        """Long poll for updates."""
        payload: Dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            payload["offset"] = offset
        res = self._call("getUpdates", payload, req_timeout=timeout + 10)
        return res if isinstance(res, list) else []

    def delete_webhook(self) -> bool:
        """A webhook makes getUpdates fail with 409; polling needs it gone."""
        return self._call("deleteWebhook", {"drop_pending_updates": False}, req_timeout=10) is not None

    def get_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata from Telegram API."""
        return self._call("getFile", {"file_id": file_id})

    def download_file(self, file_path: str, destination: str) -> bool:
        """Download file content from Telegram API to disk."""
        url = f"{TELEGRAM_FILE_BASE}{self.token}/{file_path}"
        try:
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 200:
                if len(resp.content) > MAX_DOWNLOAD:
                    log.warning("Telegram file %s is larger than the Bot API allows", file_path)
                    return False
                p = Path(destination)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(resp.content)
                return True
            log.warning("Telegram download_file status %s for %s", resp.status_code, file_path)
            return False
        except Exception as e:
            log.error("Telegram download_file error: %s", type(e).__name__)
            return False

    # ── writing ────────────────────────────────────────────────────────────

    def send_chat_action(self, chat_id: str | int, action: str = "typing") -> bool:
        """Send chat action (e.g. typing)."""
        res = self._call("sendChatAction", {"chat_id": chat_id, "action": action}, req_timeout=5)
        return bool(res)

    def send_message(
        self,
        chat_id: str | int,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        parse_mode: Optional[str] = "HTML",
        disable_notification: bool = False,
        reply_to: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send a message, splitting into chunks if necessary. Falls back to plain text if formatting fails."""
        if not text:
            return None

        chunks = split_html(text, MAX_HTML - 96) if parse_mode == "HTML" else self._chunk_text(text, MAX_MSG_LEN)
        last_res = None

        for idx, chunk in enumerate(chunks):
            # Only attach reply_markup to the final chunk
            markup = reply_markup if idx == len(chunks) - 1 else None
            payload: Dict[str, Any] = {"chat_id": chat_id, "text": chunk, "link_preview_options": {"is_disabled": True}}
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if markup:
                payload["reply_markup"] = markup
            if disable_notification:
                payload["disable_notification"] = True
            if reply_to and idx == 0:
                payload["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}

            res = self._call("sendMessage", payload)
            # If HTML parsing failed, retry as plain text without parse_mode
            if res is None and parse_mode and not self.last_network and self.last_code in (0, 400):
                payload.pop("parse_mode", None)
                payload["text"] = strip_html(chunk) if parse_mode == "HTML" else chunk
                res = self._call("sendMessage", payload)

            last_res = res

        return last_res

    def edit_message_text(
        self,
        chat_id: str | int,
        message_id: int,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        parse_mode: Optional[str] = "HTML",
    ) -> Optional[Dict[str, Any]]:
        """Edit an existing message text."""
        body = text if len(text) <= MAX_HTML else (split_html(text, MAX_HTML - 96) or [""])[0]
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": body,
            "link_preview_options": {"is_disabled": True},
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        res = self._call("editMessageText", payload)
        if res is None and "not modified" in self.last_error:
            self._remember(True)
            return {"message_id": message_id, "unchanged": True}
        if res is None and parse_mode and not self.last_network and self.last_code in (0, 400):
            payload.pop("parse_mode", None)
            payload["text"] = strip_html(body) if parse_mode == "HTML" else body
            res = self._call("editMessageText", payload)
            if res is None and "not modified" in self.last_error:
                self._remember(True)
                return {"message_id": message_id, "unchanged": True}
        return res

    def edit_message_reply_markup(
        self,
        chat_id: str | int,
        message_id: int,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update or remove reply markup on a message."""
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return self._call("editMessageReplyMarkup", payload)

    def delete_message(self, chat_id: str | int, message_id: int) -> bool:
        return bool(self._call("deleteMessage", {"chat_id": chat_id, "message_id": message_id}, req_timeout=10))

    def answer_callback_query(self, callback_query_id: str, text: Optional[str] = None, alert: bool = False) -> bool:
        """Acknowledge an inline button click."""
        payload: Dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        if alert:
            payload["show_alert"] = True
        return bool(self._call("answerCallbackQuery", payload, req_timeout=5))

    def set_reaction(self, chat_id: str | int, message_id: int, emoji: Optional[str]) -> bool:
        """React to a message (👀 'seen', 👍 'done'); None clears it. Cosmetic, so never raises."""
        reaction = [{"type": "emoji", "emoji": emoji}] if emoji else []
        return self._call("setMessageReaction", {"chat_id": chat_id, "message_id": message_id, "reaction": reaction}, req_timeout=5) is not None

    def set_my_commands(self, commands: List[Dict[str, str]]) -> bool:
        """The list that pops up when the user types '/'."""
        return self._call("setMyCommands", {"commands": commands}, req_timeout=10) is not None

    def _upload(self, method: str, field: str, chat_id: str | int, name: str, data: bytes,
                caption: Optional[str], parse_mode: Optional[str]) -> Optional[Dict[str, Any]]:
        if len(data) > MAX_UPLOAD:
            self._remember(False, "file is larger than the 50 MB Bot API limit", 413)
            return None
        form: Dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            form["caption"] = caption[:1000]
            if parse_mode:
                form["parse_mode"] = parse_mode
        res = self._call(method, form, files={field: (name, data)}, req_timeout=120)
        if res is None and parse_mode and caption and not self.last_network and self.last_code in (0, 400):
            form["caption"] = strip_html(caption)[:1000]
            form.pop("parse_mode", None)
            res = self._call(method, form, files={field: (name, data)}, req_timeout=120)
        return res

    def send_document_bytes(self, chat_id: str | int, name: str, data: bytes, caption: Optional[str] = None,
                            parse_mode: Optional[str] = "HTML") -> Optional[Dict[str, Any]]:
        return self._upload("sendDocument", "document", chat_id, name, data, caption, parse_mode)

    def send_document(self, chat_id: str | int, path: str, caption: Optional[str] = None,
                      parse_mode: Optional[str] = "HTML") -> Optional[Dict[str, Any]]:
        p = Path(path)
        try:
            return self.send_document_bytes(chat_id, p.name, p.read_bytes(), caption, parse_mode)
        except OSError as e:
            self._remember(False, f"cannot read {p.name}: {type(e).__name__}", 0)
            return None

    def send_photo(self, chat_id: str | int, path: str, caption: Optional[str] = None,
                   parse_mode: Optional[str] = "HTML") -> Optional[Dict[str, Any]]:
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as e:
            self._remember(False, f"cannot read {p.name}: {type(e).__name__}", 0)
            return None
        res = self._upload("sendPhoto", "photo", chat_id, p.name, data, caption, parse_mode)
        if res is None and self.last_code in (400,):
            # Too large / odd aspect ratio for a photo: it still travels as a file.
            return self.send_document_bytes(chat_id, p.name, data, caption, parse_mode)
        return res

    @staticmethod
    def _chunk_text(text: str, limit: int = MAX_MSG_LEN) -> List[str]:
        """Splits plain text into chunks <= limit, attempting to break at newlines or spaces."""
        if len(text) <= limit:
            return [text]
        chunks = []
        rest = text
        while len(rest) > limit:
            split_idx = rest.rfind("\n", 0, limit)
            if split_idx == -1 or split_idx < limit // 2:
                split_idx = rest.rfind(" ", 0, limit)
            if split_idx == -1 or split_idx <= 0:
                split_idx = limit
            chunks.append(rest[:split_idx].strip())
            rest = rest[split_idx:].strip()
        if rest:
            chunks.append(rest)
        return [c for c in chunks if c]
