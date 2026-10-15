"""The `http` node: one request to a web address, with the answer split into fields.

For a room that has to ask a web service something (an API, a webhook, a status page) without a model
and without `curl` in a shell. The request is described, not typed as a command:

    mode      get | post | put | patch | delete | head          (default get)
    args      url        the address (http or https; {{fields}} are filled in before the call)
              query      {name: value}   appended to the address
              headers    {name: value}   `${ENV:NAME}` is replaced by the environment variable NAME
              json       an object sent as a JSON body          (post, put, patch)
              form       {name: value} sent as a form body       (post, put, patch)
              body       raw text sent as the body
              timeout    seconds, 1..120 (default 30)

What it leaves, as `<id>.<name>`: status, ok (a 2xx or 3xx answer), body (the text), type (content type),
bytes, seconds, truncated and json (the body parsed, when it is JSON). The text of the answer is the node's
output, so `sets` and `save_as` work as for every other step.

Secrets are never part of a room file or of the run's record: a header carries `${ENV:API_TOKEN}`, the real
value is read here, at the moment of the call, and is blanked out of whatever comes back. Only http and
https are reachable (no file:, no ftp:), also after a redirect.
"""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from src.rooms.models import HTTP_MODES

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120
MAX_BYTES = 2_000_000
KEPT_CHARS = 200_000          # what the run's state keeps of an answer (the state is saved at every step)
OUTPUT_CHARS = 20000
MAX_REDIRECTS = 5

_ENV_TOKEN = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)\}")
_BODY_MODES = ("post", "put", "patch", "delete")


class HttpResult(str):
    """The answer text of an http node, with the fields it produced in `.values`."""

    values: dict

    def __new__(cls, text: str, values: Optional[dict] = None):
        obj = super().__new__(cls, text)
        obj.values = dict(values or {})
        return obj


def _fail(text: str, **values) -> HttpResult:
    return HttpResult("Error: " + text, {"ok": False, "status": 0, "body": "", **values})


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follow a few redirects, and only to http(s) addresses."""
    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme not in ("http", "https"):
            raise urllib.error.URLError("redirected to a non-web address, refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fill_env(text: str, secrets: list) -> str:
    def one(m):
        name = m.group(1)
        value = os.environ.get(name)
        if value is None:
            raise KeyError(name)
        if value:
            secrets.append(value)
        return value
    return _ENV_TOKEN.sub(one, str(text))


def _decode(raw: bytes, content_type: str) -> str:
    m = re.search(r"charset=([\w-]+)", content_type or "", re.I)
    for enc in ([m.group(1)] if m else []) + ["utf-8"]:
        try:
            return raw.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", "replace")


def _scrub(text: str, secrets: list) -> str:
    for s in secrets:
        if len(s) >= 4:
            text = text.replace(s, "***")
    return text


def run(node) -> HttpResult:
    mode = (node.mode or "get").lower()
    if mode not in HTTP_MODES:
        return _fail(f"unknown method '{mode}' (use {', '.join(HTTP_MODES)})")
    args = dict(node.args or {})
    url = str(args.get("url") or "").strip()
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return _fail("the address must start with http:// or https://")
    secrets: list = []
    try:
        query = {str(k): str(v) for k, v in (args.get("query") or {}).items()}
        if query:
            sep = "&" if parts.query else "?"
            url = url + sep + urllib.parse.urlencode(query)
        headers = {str(k): _fill_env(v, secrets) for k, v in (args.get("headers") or {}).items()}
    except KeyError as e:
        return _fail(f"the environment variable {e.args[0]} named in the headers is not set")
    except AttributeError:
        return _fail("'query' and 'headers' must be {name: value} objects")

    data: Optional[bytes] = None
    if "json" in args and args["json"] not in (None, ""):
        payload = args["json"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                return _fail("'json' is not valid JSON text")
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers.setdefault("Content-Type", "application/json; charset=utf-8")
    elif args.get("form"):
        data = urllib.parse.urlencode({str(k): str(v) for k, v in args["form"].items()}).encode("utf-8")
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif args.get("body") not in (None, ""):
        data = str(args["body"]).encode("utf-8")
        headers.setdefault("Content-Type", "text/plain; charset=utf-8")
    if data is not None and mode not in _BODY_MODES:
        return _fail(f"a {mode.upper()} request carries no body; use post, put, patch or delete")
    headers.setdefault("User-Agent", "Argent-Rooms/1.0")
    headers.setdefault("Accept", "application/json, text/plain;q=0.9, */*;q=0.5")

    try:
        timeout = max(1, min(int(args.get("timeout") or DEFAULT_TIMEOUT), MAX_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT
    request = urllib.request.Request(url, data=data, headers=headers, method=mode.upper())
    opener = urllib.request.build_opener(_Redirects)
    started = time.monotonic()
    status, content_type, raw = 0, "", b""
    try:
        with opener.open(request, timeout=timeout) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(MAX_BYTES + 1) if mode != "head" else b""
    except urllib.error.HTTPError as e:          # 4xx / 5xx: an answer, not a failure of the call
        if 300 <= e.code < 400:                  # a redirect that was not followed (to a non-web address, or too many)
            return _fail(f"{parts.netloc} redirected ({e.code}) to an address that cannot be followed",
                         status=e.code, seconds=round(time.monotonic() - started, 2))
        status = e.code
        content_type = e.headers.get("Content-Type", "") if e.headers else ""
        try:
            raw = e.read(MAX_BYTES + 1)
        except Exception:
            raw = b""
    except urllib.error.URLError as e:
        return _fail(_scrub(f"cannot reach {parts.netloc}: {e.reason}", secrets), seconds=round(time.monotonic() - started, 2))
    except (TimeoutError, OSError) as e:
        return _fail(_scrub(f"{parts.netloc}: {e}", secrets), seconds=round(time.monotonic() - started, 2))
    seconds = round(time.monotonic() - started, 2)

    truncated = len(raw) > MAX_BYTES
    text = _scrub(_decode(raw[:MAX_BYTES], content_type), secrets)
    parsed = None
    if len(text) <= KEPT_CHARS and text.strip()[:1] in ("{", "["):
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
    shown = text if len(text) <= OUTPUT_CHARS else text[:OUTPUT_CHARS] + f"\n… (+{len(text) - OUTPUT_CHARS} symbols)"
    return HttpResult(shown, {"status": status, "ok": 200 <= status < 400, "body": text[:KEPT_CHARS], "type": content_type,
                              "bytes": len(raw[:MAX_BYTES]), "seconds": seconds,
                              "truncated": truncated or len(text) > KEPT_CHARS, "json": parsed})
