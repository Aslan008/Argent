"""Context compression with a free model that is busy, out of quota, or dies mid-answer.

The rules (src/agent/service_llm.py): a busy model is asked again, an exhausted free
OpenRouter model is swapped for another one (automatically, or the user's choice), and
whatever happens the caller gets a summary or None in bounded time, never an exception.
"""

import threading
import time

import pytest

import src.agent.service_llm as sl
from providers import OpenRouterProvider, ProviderError, UpstreamBodyError
from src.agent.service_llm import Kind, Result, UI, classify, clean_summary, run

GOOD = "## Done\n" + "Изменён файл config.py: добавлены настройки сжатия контекста. " * 3


class HttpError(Exception):
    """Shaped like openai.APIStatusError."""

    def __init__(self, status, message, headers=None, meta_headers=None):
        super().__init__(message)
        self.status_code = status
        err = {"code": status, "message": message}
        if meta_headers:
            err["metadata"] = {"headers": meta_headers}
        self.body = {"error": err}
        self.response = type("R", (), {"headers": headers or {}})()


def provider_error(status, message, **kw):
    inner = HttpError(status, message, **kw)
    return ProviderError(f"OPENROUTER error (HTTP {status}): {message}", retryable=status >= 500, original_error=inner)


class FakeProvider:
    """sync_chat plays a script per model: a str is the answer, an Exception is raised, a
    callable is called (to hang, say)."""

    def __init__(self, script, name="openrouter", config_problem=None):
        self.script = {m: list(v) for m, v in script.items()}
        self.name = name
        self.calls = []
        self._problem = config_problem

    def validate_config(self):
        return self._problem

    def sync_chat(self, model, messages, timeout=None, **kw):
        self.calls.append(model)
        steps = self.script.get(model) or [provider_error(404, "No endpoints found for " + model)]
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if callable(step):
            return step()
        if isinstance(step, BaseException):
            raise step
        return step


class Recorder:
    def __init__(self, ask=None):
        self.notes = []
        self.events = []
        self.asked = []
        self._ask = ask

    def ui(self):
        def notify(msg):
            self.notes.append(msg)

        def ask(title, options):
            self.asked.append((title, list(options)))
            self.events.append("ask")
            return self._ask(title, options) if callable(self._ask) else self._ask

        return UI(
            notify=notify,
            ask=ask if self._ask is not None else None,
            pause=lambda: self.events.append("pause"),
            resume=lambda: self.events.append("resume"),
        )

    def said(self, fragment):
        return any(fragment in n for n in self.notes)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    sl.clear_cooldowns()
    monkeypatch.setattr(sl, "_pause", lambda seconds, started: False)
    yield
    sl.clear_cooldowns()


def go(provider, primary="a/primary:free", candidates=("b/second:free", "c/third:free"), **kw):
    rec = kw.pop("rec", None) or Recorder()
    kw.setdefault("retries", 3)
    kw.setdefault("policy", "auto")
    result = run("summarize", service=(provider, primary), candidates=lambda: list(candidates), ui=rec.ui(), **kw)
    return result, rec


# ── what kind of failure is it ───────────────────────────────────────────────

NOW = 1_800_000_000.0


@pytest.mark.parametrize("exc,kind", [
    (provider_error(503, "Provider returned error"), Kind.BUSY),
    (provider_error(502, "Bad gateway"), Kind.BUSY),
    (provider_error(408, "Request timeout"), Kind.BUSY),
    (provider_error(429, "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free requests"), Kind.EXHAUSTED),
    (provider_error(429, "Rate limit exceeded: free-models-per-min"), Kind.RATE_LIMITED),
    (provider_error(429, "m:free is temporarily rate-limited upstream. Please retry shortly"), Kind.RATE_LIMITED),
    (provider_error(402, "Insufficient credits"), Kind.EXHAUSTED),
    (provider_error(404, "No endpoints found for x/y:free"), Kind.UNFIT),
    (provider_error(400, "x/y:free is not a valid model ID"), Kind.UNFIT),
    (provider_error(400, "This model's maximum context length is 8192 tokens"), Kind.UNFIT),
    (provider_error(401, "No auth credentials found"), Kind.AUTH),
    (provider_error(403, "Forbidden"), Kind.AUTH),
    (TimeoutError("timed out"), Kind.BUSY),
    (ConnectionError("Connection reset by peer"), Kind.BUSY),
    (ProviderError("OPENROUTER error: Connection error."), Kind.BUSY),
    (ProviderError("something nobody has seen before"), Kind.OTHER),
])
def test_failures_are_told_apart(exc, kind):
    assert classify(exc, now=NOW).kind is kind


def test_an_error_inside_a_200_response_is_a_busy_model_not_a_crash():
    class Response:
        choices = None
        error = {"code": 502, "message": "Provider returned error"}

    err = UpstreamBodyError.from_response("openrouter", Response())
    assert err.retryable and err.status_code == 502
    wrapped = ProviderError(f"OPENROUTER error: {err}", retryable=err.retryable, original_error=err)
    assert classify(wrapped).kind is Kind.BUSY

    class Empty:
        choices = []

    assert classify(ProviderError("x", original_error=UpstreamBodyError.from_response("openrouter", Empty()))).kind is Kind.BUSY


def test_the_reset_header_says_when_the_quota_returns():
    soon = provider_error(429, "free-models-per-min", meta_headers={"X-RateLimit-Reset": str(int((NOW + 12) * 1000))})
    info = classify(soon, now=NOW)
    assert info.kind is Kind.RATE_LIMITED and 11 < info.retry_after < 13
    far = provider_error(429, "limit", meta_headers={"X-RateLimit-Reset": str(int((NOW + 7200) * 1000))})
    info = classify(far, now=NOW)
    assert info.kind is Kind.EXHAUSTED and info.reset_at == pytest.approx(NOW + 7200)
    assert classify(provider_error(429, "slow down", headers={"retry-after": "7"}), now=NOW).retry_after == 7.0


# ── an answer is checked before it is believed ───────────────────────────────

def test_summaries_are_cleaned_and_judged():
    assert clean_summary(GOOD) == GOOD.strip()
    assert clean_summary("<think>let me see</think>" + GOOD) == GOOD.strip()
    assert clean_summary("<think>I will start by considering the whole history and then") is None, "died inside its reasoning"
    assert clean_summary("") is None and clean_summary(None) is None
    assert clean_summary("[Compression Failed]") is None
    assert clean_summary("OK") is None, "too short to be a summary"
    assert clean_summary("Error: the model is currently overloaded, please try again later, thanks") is None
    assert clean_summary("I'm sorry, I cannot help with that request because it is outside what I may do here.") is None


# ── a busy model is asked again ──────────────────────────────────────────────

def test_a_model_that_answers_is_used_as_is():
    p = FakeProvider({"a/primary:free": [GOOD]})
    r, rec = go(p)
    assert r.text == GOOD.strip() and r.model == "a/primary:free" and not r.switched
    assert p.calls == ["a/primary:free"] and rec.notes == []


def test_a_busy_model_is_asked_again_and_then_answers():
    busy = provider_error(503, "Provider returned error")
    p = FakeProvider({"a/primary:free": [busy, busy, GOOD]})
    r, rec = go(p)
    assert r.text and not r.switched
    assert p.calls == ["a/primary:free"] * 3, "the SAME model, three times"
    assert rec.said("Повтор 2/3") and rec.said("Повтор 3/3")


def test_a_per_minute_limit_is_waited_out_on_the_same_model():
    p = FakeProvider({"a/primary:free": [provider_error(429, "free-models-per-min"), GOOD]})
    r, _ = go(p)
    assert r.text and p.calls == ["a/primary:free"] * 2


def test_the_wait_honours_what_the_server_asked_for(monkeypatch):
    waits = []
    monkeypatch.setattr(sl, "_pause", lambda seconds, started: waits.append(seconds) or False)
    p = FakeProvider({"a/primary:free": [provider_error(429, "slow", headers={"retry-after": "20"}), GOOD]})
    go(p)
    assert 20 <= waits[0] <= 22, waits


# ── a model that will not come back is replaced ──────────────────────────────

def test_a_model_that_stays_busy_is_replaced_after_its_retries():
    busy = provider_error(503, "Provider returned error")
    p = FakeProvider({"a/primary:free": [busy], "b/second:free": [GOOD]})
    r, rec = go(p)
    assert r.text and r.switched and r.model == "b/second:free"
    assert p.calls == ["a/primary:free"] * 3 + ["b/second:free"]
    assert rec.said("перехожу на b/second:free") and rec.said("сжатие выполнила b/second:free")


@pytest.mark.parametrize("status,message", [
    (429, "Rate limit exceeded: free-models-per-day"),
    (402, "Insufficient credits"),
    (404, "No endpoints found for a/primary:free"),
])
def test_an_exhausted_or_missing_model_is_not_asked_twice(status, message):
    p = FakeProvider({"a/primary:free": [provider_error(status, message)], "b/second:free": [GOOD]})
    r, _ = go(p)
    assert r.text and r.model == "b/second:free"
    assert p.calls == ["a/primary:free", "b/second:free"], "no pointless retry of a spent quota"


def test_the_exhausted_model_is_skipped_by_the_next_compression():
    p = FakeProvider({"a/primary:free": [provider_error(429, "free-models-per-day")], "b/second:free": [GOOD]})
    go(p)
    p.calls.clear()
    r, rec = go(p)
    assert p.calls == ["b/second:free"], "it did not walk into the spent model again"
    assert r.text and rec.said("недавно не отвечала")


def test_a_wrong_key_stops_everything():
    p = FakeProvider({"a/primary:free": [provider_error(401, "No auth credentials found")], "b/second:free": [GOOD]})
    r, rec = go(p)
    assert r.text is None and p.calls == ["a/primary:free"], "another model on the same key cannot help"
    assert rec.said("без LLM")


def test_when_every_model_fails_the_caller_gets_none_after_a_bounded_number_of_tries():
    dead = provider_error(429, "free-models-per-day")
    many = [f"m{i}/x:free" for i in range(20)]
    p = FakeProvider({"a/primary:free": [dead], **{m: [dead] for m in many}})
    r, rec = go(p, candidates=many)
    assert r.text is None
    assert len(p.calls) <= sl.MAX_SWITCHES + 1, p.calls
    assert rec.said("без LLM")


def test_with_no_other_model_left_the_summary_is_built_without_one():
    p = FakeProvider({"a/primary:free": [provider_error(429, "free-models-per-day")]})
    r, rec = go(p, candidates=())
    assert r.text is None and rec.said("не осталось")


def test_only_a_free_openrouter_model_has_free_siblings_to_move_to():
    dead = provider_error(429, "free-models-per-day")
    paid = FakeProvider({"a/paid": [dead], "b/second:free": [GOOD]})
    assert go(paid, primary="a/paid")[0].text is None and paid.calls == ["a/paid"]
    other = FakeProvider({"m": [dead], "b/second:free": [GOOD]}, name="zai")
    assert go(other, primary="m")[0].text is None and other.calls == ["m"]


def test_a_provider_that_is_not_openrouter_still_gets_its_retries():
    busy = provider_error(503, "overloaded")
    p = FakeProvider({"m": [busy, GOOD]}, name="deepseek")
    r, _ = go(p, primary="m")
    assert r.text and p.calls == ["m", "m"]


def test_a_missing_api_key_is_reported_not_attempted():
    p = FakeProvider({}, config_problem="OpenRouter API key is not set.")
    r, rec = go(p)
    assert r.text is None and p.calls == [] and rec.said("API key")


# ── the user picks ───────────────────────────────────────────────────────────

def test_in_ask_mode_the_user_chooses_the_next_model():
    p = FakeProvider({"a/primary:free": [provider_error(429, "free-models-per-day")], "c/third:free": [GOOD]})
    rec = Recorder(ask=lambda title, options: "c/third:free")
    r, _ = go(p, rec=rec, policy="ask")
    assert r.text and r.model == "c/third:free" and p.calls == ["a/primary:free", "c/third:free"]
    title, options = rec.asked[0]
    assert "a/primary:free" in title and "исчерпала лимит" in title
    assert options[0].startswith("↻") and options[-1].startswith("Без модели")
    assert "b/second:free" in options and "c/third:free" in options
    assert rec.events == ["pause", "ask", "resume"], "the spinner is off while a question is on screen"


def test_in_ask_mode_the_user_can_decline_or_retry():
    dead = provider_error(429, "free-models-per-day")
    p = FakeProvider({"a/primary:free": [dead]})
    r, _ = go(p, rec=Recorder(ask=lambda t, o: o[-1]), policy="ask")
    assert r.text is None and p.calls == ["a/primary:free"], "'without a model' stops here"
    sl.clear_cooldowns()
    p = FakeProvider({"a/primary:free": [dead, GOOD]})
    r, _ = go(p, rec=Recorder(ask=lambda t, o: o[0]), policy="ask")
    assert r.text and p.calls == ["a/primary:free"] * 2, "'retry' asks the same model again"
    sl.clear_cooldowns()
    p = FakeProvider({"a/primary:free": [dead], "b/second:free": [GOOD]})
    r, _ = go(p, rec=Recorder(ask=lambda t, o: None), policy="ask")
    assert r.text is None and p.calls == ["a/primary:free"], "cancelling the question is a refusal"


def test_a_question_that_blows_up_is_a_refusal_not_a_crash():
    def boom(title, options):
        raise RuntimeError("the TUI went away")

    p = FakeProvider({"a/primary:free": [provider_error(429, "free-models-per-day")], "b/second:free": [GOOD]})
    rec = Recorder(ask=boom)
    r, _ = go(p, rec=rec, policy="ask")
    assert r.text is None and rec.events == ["pause", "ask", "resume"]


def test_ask_mode_without_anyone_to_ask_moves_on_by_itself():
    p = FakeProvider({"a/primary:free": [provider_error(429, "free-models-per-day")], "b/second:free": [GOOD]})
    r, _ = go(p, policy="ask")  # a Recorder with no `ask`: a background worker
    assert r.text and r.model == "b/second:free"


# ── the model dies in the middle of it ───────────────────────────────────────

def test_an_empty_answer_counts_against_the_model():
    p = FakeProvider({"a/primary:free": [""], "b/second:free": [GOOD]})
    r, rec = go(p)
    assert r.text and r.model == "b/second:free"
    assert p.calls == ["a/primary:free"] * 2 + ["b/second:free"], "one more chance, then the next model"


def test_an_answer_cut_off_inside_its_reasoning_is_not_used():
    p = FakeProvider({"a/primary:free": ["<think>first I should look at the whole history and then"], "b/second:free": [GOOD]})
    r, _ = go(p)
    assert r.model == "b/second:free"


def test_reasoning_tags_are_removed_from_a_good_answer():
    p = FakeProvider({"a/primary:free": ["<think>hmm</think>\n" + GOOD]})
    r, _ = go(p)
    assert "<think>" not in r.text and r.text == GOOD.strip()


@pytest.fixture
def tiny_clock(monkeypatch):
    monkeypatch.setattr(sl, "_MIN_CALL", 0.05)
    monkeypatch.setattr(sl, "_SLACK", 0.05)
    monkeypatch.setattr(sl, "_MIN_LEFT", 0.05)
    monkeypatch.setattr(sl, "_MIN_SWITCH_LEFT", 0.05)


def test_a_model_that_hangs_is_abandoned_and_its_late_answer_is_ignored(tiny_clock):
    release = threading.Event()
    late = []

    def hang():
        release.wait(5)
        late.append(GOOD)
        return "LATE ANSWER " + GOOD

    p = FakeProvider({"a/primary:free": [hang], "b/second:free": [GOOD]})
    t0 = time.monotonic()
    r, rec = go(p, per_call_timeout=0.1, total_budget=5.0, retries=1)
    elapsed = time.monotonic() - t0
    release.set()
    assert r.text == GOOD.strip() and r.model == "b/second:free"
    assert elapsed < 2.0, f"the hung call held the turn for {elapsed:.1f}s"
    time.sleep(0.2)
    assert late, "the abandoned thread did finish..."
    assert "LATE ANSWER" not in r.text, "...and nothing it produced leaked into the result"


def test_the_whole_call_is_bounded_in_time(tiny_clock):
    never = threading.Event()
    p = FakeProvider({"a/primary:free": [lambda: never.wait(5) or GOOD]})
    t0 = time.monotonic()
    r, _ = go(p, per_call_timeout=0.1, total_budget=0.6, retries=6, candidates=())
    never.set()
    assert r.text is None and time.monotonic() - t0 < 3.0


def test_an_unexpected_failure_inside_the_machinery_is_contained():
    class Bomb(FakeProvider):
        def validate_config(self):
            raise RuntimeError("config store exploded")

    rec = Recorder()
    r = run("x", service=(Bomb({}), "a/primary:free"), ui=rec.ui(), retries=1, policy="auto")
    assert isinstance(r, Result) and r.text is None
    assert rec.said("внутренняя ошибка")


def test_even_a_ui_that_raises_cannot_break_the_call():
    def angry(_msg):
        raise RuntimeError("terminal gone")

    busy = provider_error(503, "overloaded")
    p = FakeProvider({"a/primary:free": [busy, GOOD]})
    r = run("x", service=(p, "a/primary:free"), ui=UI(notify=angry), retries=3, policy="auto")
    assert isinstance(r, Result)  # contained, whatever it decided


# ── the user presses stop ────────────────────────────────────────────────────

def test_a_stop_during_a_wait_ends_the_compression_at_once(monkeypatch):
    monkeypatch.undo()  # the real, interruptible pause
    busy = provider_error(503, "overloaded")
    p = FakeProvider({"a/primary:free": [busy]})
    threading.Timer(0.3, sl.request_cancel).start()
    t0 = time.monotonic()
    r = run("x", service=(p, "a/primary:free"), ui=UI(), retries=3, policy="auto", candidates=lambda: [])
    assert r.text is None and r.reason == "stopped"
    assert time.monotonic() - t0 < 2.5, "it should not sit out the 3 s pause"


def test_a_stop_from_an_earlier_turn_does_not_cancel_a_new_compression(monkeypatch):
    sl.request_cancel()
    time.sleep(0.02)
    # An answer that takes a moment, so the wait loop (where a stop is noticed) really runs.
    p = FakeProvider({"a/primary:free": [lambda: time.sleep(0.2) or GOOD]})
    r, _ = go(p)
    assert r.text, "a stop pressed before this compression started must not end it"


# ── the free-model list ──────────────────────────────────────────────────────

class _Model:
    def __init__(self, id, ctx=None, outputs=None):
        self.id = id
        extra = {}
        if ctx is not None:
            extra["context_length"] = ctx
        if outputs is not None:
            extra["architecture"] = {"output_modalities": outputs}
        self.model_extra = extra


def test_free_models_are_ranked_by_context_and_unfit_ones_left_out():
    catalogue = [
        _Model("paid/x", 200000),
        _Model("a/small:free", 8000),
        _Model("b/big:free", 128000),
        _Model("c/mid:free", 32000),
        _Model("d/image:free", 64000, outputs=["image"]),
        _Model("e/unknown:free"),
    ]
    p = OpenRouterProvider(api_key="k", base_url="http://x")
    p._get_client = lambda: type("C", (), {"models": type("M", (), {"list": staticmethod(lambda: type("L", (), {"data": catalogue})())})()})()
    assert p.list_free_models() == ["b/big:free", "c/mid:free", "e/unknown:free"]


def test_the_shortlist_stands_in_when_the_catalogue_is_unreachable():
    p = OpenRouterProvider(api_key="k", base_url="http://x")

    def broken():
        raise RuntimeError("offline")

    p._get_client = broken
    free = p.list_free_models()
    assert free and all(m.endswith(":free") for m in free)


def test_the_live_list_is_cached(monkeypatch):
    calls = []

    class P:
        name = "openrouter"

        def list_free_models(self):
            calls.append(1)
            return ["x:free"]

    assert sl.live_free_models(P()) == ["x:free"] and sl.live_free_models(P()) == ["x:free"]
    assert len(calls) == 1


# ── how long a spent model is skipped ────────────────────────────────────────

def test_a_spent_quota_is_skipped_until_it_resets():
    info = classify(provider_error(429, "limit", meta_headers={"X-RateLimit-Reset": str(int((NOW + 3600) * 1000))}), now=NOW)
    assert sl._cooldown_for(info, NOW) == pytest.approx(3600, abs=1)
    daily = classify(provider_error(429, "free-models-per-day"), now=NOW)
    left = sl._cooldown_for(daily, NOW)
    assert 300 <= left <= 86400


def test_a_busy_model_is_only_skipped_briefly():
    assert sl._cooldown_for(sl.ErrorInfo(Kind.BUSY), NOW) == 120


# ── the pieces around it ─────────────────────────────────────────────────────

def test_openrouters_429_keeps_the_reason_so_it_can_be_acted_on():
    p = OpenRouterProvider(api_key="k", base_url="http://x")
    with pytest.raises(ProviderError) as ei:
        p._handle_api_status_error(HttpError(429, "Rate limit exceeded: free-models-per-day"))
    assert "free-models-per-day" in str(ei.value)
    assert classify(ei.value).kind is Kind.EXHAUSTED


def test_sync_chat_turns_an_error_body_into_a_retryable_provider_error():
    p = OpenRouterProvider(api_key="k", base_url="http://x")

    class Resp:
        choices = None
        error = {"code": 503, "message": "Provider returned error"}

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return Resp()

    p._get_client = lambda: Client()
    import providers
    monkey = pytest.MonkeyPatch()
    monkey.setattr(providers.time, "sleep", lambda s: None)
    try:
        with pytest.raises(ProviderError) as ei:
            p.sync_chat("a/x:free", [{"role": "user", "content": "hi"}])
    finally:
        monkey.undo()
    assert ei.value.retryable
    assert classify(ei.value).kind is Kind.BUSY
