"""Deep mutation-killing tests for error_help.explain_error.

Every test targets a specific branch, pattern, or boundary so that a
mutant (changed regex, swapped order, dropped lower() call, etc.) is
caught.  Explanation strings are asserted verbatim — if the wording
changes the test must change too, which is the point.
"""

import pytest

from error_help import explain_error


# ── Exact explanation strings (copied from error_help._EXPLANATIONS) ──────────
SURROGATE = (
    "В тексте оказался «половинчатый» символ Unicode — обычно от обрезанного "
    "эмодзи в ответе модели. Диалог чинится сам на следующем ходу; если нет — /clear."
)
CONTEXT = (
    "Запрос не помещается в окно контекста модели. Сократите историю (/clear) "
    "или увеличьте окно в /model."
)
CONNECTION = (
    "Не удалось подключиться к провайдеру. Проверьте, запущен ли Ollama "
    "(`ollama ps`), и адрес в /provider."
)
AUTH = "Провайдер отклонил ключ. Проверьте его в /provider."
RATE_LIMIT = (
    "Провайдер ограничил частоту запросов. Подождите или смените модель в /model."
)
MODEL_NOT_FOUND = (
    "Провайдер не знает такую модель — часто бывает после смены провайдера. "
    "Выберите модель заново: /model."
)
TIMEOUT = (
    "Ответа не дождались. Если это MCP — проверьте, отвечает ли приложение; "
    "таймаут настраивается ключом mcp_call_timeout."
)
PERMISSION = (
    "Нет прав на файл или папку. Закройте программу, которая держит файл, "
    "или запустите Argent от имени пользователя с доступом."
)
DISK = "На диске нет места."
FILE_NOT_FOUND = (
    "Файл или папка не найдены — проверьте путь и текущую директорию (/cd)."
)
JSON = (
    "Ответ пришёл в неверном JSON. Обычно это модель сломала формат — "
    "повторите запрос; если повторяется, помогает модель поумнее."
)


# ── 1. Empty / None input ────────────────────────────────────────────────────
class TestEmptyAndNone:
    def test_empty_string_returns_none(self):
        assert explain_error("") is None

    def test_none_returns_none(self):
        assert explain_error(None) is None

    def test_whitespace_only_returns_none(self):
        # "not message" is falsy for whitespace-only? No — "   " is truthy,
        # so this should fall through to the loop and return None (no match).
        assert explain_error("   ") is None


# ── 2. Surrogate pattern ─────────────────────────────────────────────────────
class TestSurrogatePattern:
    def test_surrogates_not_allowed(self):
        assert explain_error("surrogates not allowed") == SURROGATE

    def test_surrogates_in_long_message(self):
        msg = "Error: 'utf-8' codec can't encode characters: surrogates not allowed"
        assert explain_error(msg) == SURROGATE

    def test_surrogates_case_insensitive(self):
        assert explain_error("SURROGATES NOT ALLOWED") == SURROGATE

    def test_surrogates_mixed_case(self):
        assert explain_error("Surrogates Not Allowed") == SURROGATE


# ── 3. Context window / length / tokens ──────────────────────────────────────
class TestContextPattern:
    def test_context_window(self):
        assert explain_error("context window exceeded") == CONTEXT

    def test_context_length(self):
        assert explain_error("this model has a context length of 8192") == CONTEXT

    def test_too_many_tokens(self):
        assert explain_error("too many tokens in prompt") == CONTEXT

    def test_maximum_context(self):
        assert explain_error("maximum context length reached") == CONTEXT

    def test_context_case_insensitive(self):
        assert explain_error("CONTEXT WINDOW") == CONTEXT


# ── 4. Connection refused ────────────────────────────────────────────────────
class TestConnectionPattern:
    def test_connection_refused(self):
        assert explain_error("connection refused") == CONNECTION

    def test_connection_refused_case_insensitive(self):
        assert explain_error("Connection Refused") == CONNECTION

    def test_failed_to_establish(self):
        assert explain_error("failed to establish connection") == CONNECTION

    def test_max_retries_exceeded(self):
        assert explain_error("max retries exceeded") == CONNECTION

    def test_connection_aborted(self):
        assert explain_error("connection aborted") == CONNECTION


# ── 5. Auth (401 / 403 / unauthorized) ───────────────────────────────────────
class TestAuthPattern:
    def test_401(self):
        assert explain_error("error code 401") == AUTH

    def test_403(self):
        assert explain_error("403 forbidden") == AUTH

    def test_unauthorized(self):
        assert explain_error("unauthorized access") == AUTH

    def test_invalid_api_key(self):
        assert explain_error("invalid api key") == AUTH

    def test_authentication_failed(self):
        assert explain_error("authentication failed") == AUTH

    def test_401_word_boundary_not_partial(self):
        # \b401\b must NOT match inside a longer number like 14012
        assert explain_error("error 14012 occurred") is None

    def test_403_word_boundary_not_partial(self):
        assert explain_error("port 4032 is open") is None


# ── 6. Rate limit (429 / rate limit / quota) ─────────────────────────────────
class TestRateLimitPattern:
    def test_429(self):
        assert explain_error("429 too many requests") == RATE_LIMIT

    def test_rate_limit(self):
        assert explain_error("rate limit exceeded") == RATE_LIMIT

    def test_quota(self):
        assert explain_error("quota exceeded") == RATE_LIMIT

    def test_429_word_boundary(self):
        assert explain_error("error 14293 happened") is None


# ── 7. Model not found (404 model / model not found) ─────────────────────────
class TestModelNotFoundPattern:
    def test_404_model(self):
        assert explain_error("404 model not found") == MODEL_NOT_FOUND

    def test_model_not_found(self):
        assert explain_error("model gpt-99 not found") == MODEL_NOT_FOUND

    def test_no_such_model(self):
        assert explain_error("no such model: llama-xyz") == MODEL_NOT_FOUND

    def test_404_without_model_word_no_match(self):
        # "404" alone (without "model") should NOT match this pattern
        assert explain_error("404 page not found") is None


# ── 8. Timeout ───────────────────────────────────────────────────────────────
class TestTimeoutPattern:
    def test_timed_out(self):
        assert explain_error("request timed out") == TIMEOUT

    def test_timeout(self):
        assert explain_error("timeout occurred") == TIMEOUT

    def test_timeout_case_insensitive(self):
        assert explain_error("TIMED OUT") == TIMEOUT


# ── 9. Permission denied ─────────────────────────────────────────────────────
class TestPermissionPattern:
    def test_permission_denied(self):
        assert explain_error("permission denied") == PERMISSION

    def test_access_is_denied(self):
        assert explain_error("access is denied") == PERMISSION

    def test_errno_13(self):
        assert explain_error("[Errno 13] Permission denied") == PERMISSION


# ── 10. Disk space ───────────────────────────────────────────────────────────
class TestDiskPattern:
    def test_no_space_left(self):
        assert explain_error("no space left on device") == DISK

    def test_not_enough_space(self):
        assert explain_error("not enough space on disk") == DISK

    def test_disk_full(self):
        assert explain_error("disk full") == DISK


# ── 11. File not found ───────────────────────────────────────────────────────
class TestFileNotFoundPattern:
    def test_no_such_file(self):
        assert explain_error("no such file or directory") == FILE_NOT_FOUND

    def test_errno_2(self):
        assert explain_error("[Errno 2] No such file") == FILE_NOT_FOUND

    def test_cannot_find_file(self):
        assert explain_error("cannot find the file") == FILE_NOT_FOUND

    def test_cannot_find_path(self):
        assert explain_error("cannot find the path specified") == FILE_NOT_FOUND


# ── 12. JSON decode ──────────────────────────────────────────────────────────
class TestJsonPattern:
    def test_json_decode(self):
        assert explain_error("json decode error") == JSON

    def test_expecting_value(self):
        assert explain_error("Expecting value: line 1 column 1") == JSON

    def test_invalid_json(self):
        assert explain_error("invalid json response") == JSON


# ── 13. No match ─────────────────────────────────────────────────────────────
class TestNoMatch:
    def test_random_message_returns_none(self):
        assert explain_error("random error message") is None

    def test_completely_unrelated_text(self):
        assert explain_error("the quick brown fox jumps over the lazy dog") is None


# ── 14. Pattern priority (first match wins) ──────────────────────────────────
class TestPatternPriority:
    def test_connection_before_timeout(self):
        # "connection refused" is listed before "timeout" in _EXPLANATIONS,
        # so a message containing both must return CONNECTION.
        msg = "connection refused: timed out waiting for response"
        assert explain_error(msg) == CONNECTION

    def test_context_before_connection(self):
        # "context window" is listed before "connection refused".
        msg = "context window connection refused"
        assert explain_error(msg) == CONTEXT

    def test_auth_before_rate_limit(self):
        # 401/403 pattern is before 429 pattern.
        msg = "401 and 429 errors both occurred"
        assert explain_error(msg) == AUTH

    def test_timeout_before_permission(self):
        # timeout is listed before permission denied.
        msg = "timeout: permission denied"
        assert explain_error(msg) == TIMEOUT


# ── 15. Partial match / word-boundary specifics ──────────────────────────────
class TestPartialAndWordBoundary:
    def test_401_embedded_in_text(self):
        assert explain_error("error code 401 from server") == AUTH

    def test_403_at_start_of_message(self):
        assert explain_error("403") == AUTH

    def test_429_at_end_of_message(self):
        assert explain_error("request failed with 429") == RATE_LIMIT

    def test_404_model_partial(self):
        # "404" must be followed by "model" (with .* between them)
        assert explain_error("404 model") == MODEL_NOT_FOUND

    def test_404_model_with_gap(self):
        assert explain_error("404 error: model not available") == MODEL_NOT_FOUND