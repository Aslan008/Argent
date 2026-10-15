"""A lost connection or a busy provider does not end the turn; a crash does not eat the conversation.

Covers src/agent/netstate.py (what is a broken connection, who is reachable, how long to wait and what to say),
the retries in providers.py, the wait/resume branch of the agent loop and the autosave + restore of a session.
Every clock, sleep and network probe is injected: nothing here touches a real network or sleeps for real.
"""

import pytest

import agent as agent_module
import providers
import session
from agent import ArgentAgent, RESUME_NOTE
from providers import ProviderError, with_retry
from src.agent import netstate, todos


# ── fakes ───────────────────────────────────────────────────────────────────────────────────────────────

class Clock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class ConnectError(Exception):          # the name httpx gives it: no "Connection" in it
    pass


class _Provider:
    name = "openrouter"
    _base_url = "https://openrouter.ai/api/v1"


def _wait(clock, probe, **kw):
    kw.setdefault("provider", None)
    kw.setdefault("patience", 600.0)
    return netstate.RecoveryWait(clock=clock.now, sleep=clock.sleep, probe=probe, **kw)


# ── what is a broken connection ─────────────────────────────────────────────────────────────────────────

class TestTransportError:
    @pytest.mark.parametrize("exc", [
        ConnectionResetError("x"), TimeoutError("x"), ConnectError("boom"),
        Exception("Connection error."), Exception("HTTPSConnectionPool: Max retries exceeded: getaddrinfo failed"),
        RuntimeError("peer closed connection without sending complete message body"),
    ])
    def test_these_are_broken_connections(self, exc):
        assert netstate.is_transport_error(exc)

    @pytest.mark.parametrize("exc", [ValueError("bad value"), ProviderError("HTTP 401 invalid key"), KeyError("k")])
    def test_these_are_not(self, exc):
        assert not netstate.is_transport_error(exc)

    def test_the_cause_chain_is_followed(self):
        wrapped = ProviderError("OPENROUTER error: boom", original_error=ConnectError("boom"))
        assert netstate.is_transport_error(wrapped)
        try:
            try:
                raise ConnectionResetError("reset")
            except ConnectionResetError as inner:
                raise RuntimeError("stream failed") from inner
        except RuntimeError as outer:
            assert netstate.is_transport_error(outer)


class TestReachability:
    def test_endpoint_answers(self):
        assert netstate.reachability(_Provider(), probe=lambda h, p, t: h == "openrouter.ai") == "up"

    def test_endpoint_down_but_internet_up(self):
        assert netstate.reachability(_Provider(), probe=lambda h, p, t: h != "openrouter.ai") == "provider_down"

    def test_nothing_answers(self):
        assert netstate.reachability(_Provider(), probe=lambda h, p, t: False) == "offline"

    def test_a_local_server_is_judged_by_itself_alone(self):
        class Local:
            name = "ollama"
        asked = []

        def probe(host, port, timeout):
            asked.append((host, port))
            return False

        assert netstate.reachability(Local(), probe=probe) == "provider_down"
        assert asked == [("localhost", 11434)]           # no internet anchors were asked

    def test_endpoint_parsing(self):
        assert netstate.endpoint_of(_Provider()) == ("openrouter.ai", 443, False)

        class Lan:
            name = "koboldcpp"
            _base_url = "http://192.168.1.20:5001/v1"
        assert netstate.endpoint_of(Lan()) == ("192.168.1.20", 5001, True)


# ── waiting ─────────────────────────────────────────────────────────────────────────────────────────────

class TestRecoveryWait:
    def test_waits_for_the_connection_and_says_so(self):
        clock = Clock()
        w = _wait(clock, lambda h, p, t: clock.t >= 1010, kind="transport")
        notes = list(w.run())
        assert w.recovered and w.reason == "back"
        assert "потеряно" in notes[0] and "восстановлена" in notes[-1]
        assert 10 <= clock.t - 1000 < 20                 # it looked every few seconds, not once a minute

    def test_gives_up_when_patience_runs_out_and_reports_progress(self):
        clock = Clock()
        w = _wait(clock, lambda h, p, t: False, kind="transport", patience=20.0, progress_every=5.0)
        notes = list(w.run())
        assert not w.recovered and w.reason == "patience-exhausted"
        assert any("всё ещё нет связи" in n for n in notes)
        assert clock.t - 1000 <= 24

    def test_a_stop_request_ends_the_wait_within_a_slice(self):
        clock = Clock()
        stopped = {"v": False}

        def sleep(s):
            clock.sleep(s)
            if clock.t > 1001:
                stopped["v"] = True

        w = netstate.RecoveryWait(provider=None, kind="transport", patience=600.0, clock=clock.now, sleep=sleep,
                                  probe=lambda h, p, t: False, stop=lambda: stopped["v"])
        list(w.run())
        assert not w.recovered and w.reason == "stopped"
        assert clock.t - 1000 < 3

    def test_no_patience_means_no_waiting(self):
        clock = Clock()
        w = _wait(clock, lambda h, p, t: False, kind="transport", patience=0.0)
        assert list(w.run()) == [] and not w.recovered and clock.t == 1000.0

    def test_a_busy_provider_first_retry_is_quiet_the_next_ones_speak(self):
        clock = Clock()
        streak = netstate.Streak()
        first = _wait(clock, lambda h, p, t: True, kind="busy", streak=streak)
        assert list(first.run()) == [] and first.recovered
        second = _wait(clock, lambda h, p, t: True, kind="busy", streak=streak, detail="HTTP 503")
        notes = list(second.run())
        assert second.recovered and "попытка 2" in notes[0] and "HTTP 503" in notes[0]

    def test_a_provider_that_keeps_failing_is_not_retried_for_ever(self):
        clock = Clock()
        streak = netstate.Streak(started=1000.0, attempts=netstate.MAX_PROVIDER_ATTEMPTS)
        w = _wait(clock, lambda h, p, t: True, kind="busy", streak=streak)
        list(w.run())
        assert not w.recovered and w.reason == "too-many-attempts"

    def test_retry_after_is_honoured_up_to_a_cap(self):
        import random
        rng = random.Random(1)
        assert netstate.backoff_delay(0, retry_after=40, rng=rng) >= 34
        assert netstate.backoff_delay(0, retry_after=10_000, rng=rng) <= netstate.MAX_BACKOFF * 1.15


class TestRetryTransient:
    def test_repeats_a_dropped_connection_then_succeeds(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionResetError("reset")
            return "ok"

        assert netstate.retry_transient(fn, attempts=3, sleep=lambda s: None) == "ok" and len(calls) == 3

    def test_anything_else_is_raised_at_once(self):
        calls = []

        def fn():
            calls.append(1)
            raise ValueError("not the network")

        with pytest.raises(ValueError):
            netstate.retry_transient(fn, attempts=3, sleep=lambda s: None)
        assert len(calls) == 1


class TestWithRetry:
    def test_a_blink_of_the_connection_is_ridden_out(self, monkeypatch):
        monkeypatch.setattr(providers.time, "sleep", lambda s: None)
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 4:
                raise ConnectError("boom")             # no HTTP code anywhere in the text
            return "ok"

        assert with_retry(fn, max_retries=2) == "ok" and len(calls) == 4

    def test_a_provider_error_that_wraps_a_dropped_connection_is_retried(self, monkeypatch):
        monkeypatch.setattr(providers.time, "sleep", lambda s: None)
        calls = []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise ProviderError("OPENROUTER error: x", original_error=ConnectError("boom"))
            return "ok"

        assert with_retry(fn, max_retries=2) == "ok"

    def test_a_bad_request_is_not(self, monkeypatch):
        monkeypatch.setattr(providers.time, "sleep", lambda s: None)
        calls = []

        def fn():
            calls.append(1)
            raise ProviderError("HTTP 400 bad request")

        with pytest.raises(ProviderError):
            with_retry(fn, max_retries=3)
        assert len(calls) == 1

    def test_stream_errors_that_are_the_networks_are_marked_retryable(self):
        assert providers._is_transient_stream_error(ConnectError("boom"))
        assert not providers._is_transient_stream_error(ValueError("bad json"))


# ── the agent loop ──────────────────────────────────────────────────────────────────────────────────────

class _Strategy:
    def wants_constrained_decoding(self): return False
    def supports_native_tools(self): return True
    def wants_objective_anchor(self): return False
    def wants_tool_catalog(self): return False


@pytest.fixture
def bare(monkeypatch):
    a = ArgentAgent.__new__(ArgentAgent)
    a.messages = [{"role": "system", "content": "s"}]
    a.model_name = "test-model"
    a.provider = "openrouter"
    a.max_context_tokens = 8192
    a.strategy = _Strategy()
    a._native_tools_unsupported = False
    a.loop_guard = agent_module.LoopGuard()
    a.semantic_loop_guard = agent_module.SemanticLoopGuard()
    a._read_cache = {}
    monkeypatch.setattr(ArgentAgent, "_refresh_system_prompt", lambda self, t=None: False)
    monkeypatch.setattr(ArgentAgent, "_trim_history", lambda self: None)
    monkeypatch.setattr(agent_module.memory, "data", {"objective": "x"}, raising=False)
    # patient, and instant: no real sleeping, no real probing
    monkeypatch.setenv("ARGENT_NETWORK_PATIENCE", "60")
    monkeypatch.setattr(netstate, "POLL_S", 0.0)
    monkeypatch.setattr(netstate, "backoff_delay", lambda attempt, retry_after=None, rng=None: 0.0)
    return a


def _provider(monkeypatch, script):
    """A provider whose n-th stream_chat call runs script[n] (a generator function taking the call's kwargs)."""
    calls = []

    class _P:
        def validate_config(self): return None
        def supports_constrained_decoding(self): return False

        def stream_chat(self, **kw):
            calls.append(kw)
            return script[min(len(calls) - 1, len(script) - 1)](kw)

    monkeypatch.setattr(agent_module, "create_provider", lambda p: _P())
    return calls


def _says(text):
    def gen(kw):
        yield {"content": text}
    return gen


def _fails(exc):
    def gen(kw):
        raise exc
        yield
    return gen


def _reach(monkeypatch, *states):
    seq = iter(states)
    last = {"v": states[-1]}

    def fake(provider=None, timeout=3.0, probe=None):
        return next(seq, last["v"])

    monkeypatch.setattr(netstate, "reachability", fake)


def _run(agent, text="hi"):
    return list(agent.process_user_input(text))


def _texts(events, kind):
    return [e.get("content", "") for e in events if e.get("type") == kind]


class TestAgentWaitsAndResumes:
    def test_connection_lost_before_the_first_token(self, bare, monkeypatch):
        calls = _provider(monkeypatch, [_fails(ConnectError("boom")), _says("Готово.")])
        _reach(monkeypatch, "offline", "offline", "up")
        events = _run(bare)
        notes = " ".join(_texts(events, "error"))
        assert "потеряно" in notes and "восстановлена" in notes
        assert len(calls) == 2
        assert bare.messages[-1] == {"role": "assistant", "content": "Готово."}

    def test_a_first_blink_is_retried_without_noise(self, bare, monkeypatch):
        calls = _provider(monkeypatch, [_fails(ConnectionResetError("reset")), _says("Готово.")])
        _reach(monkeypatch, "up")
        events = _run(bare)
        assert _texts(events, "error") == [] and len(calls) == 2
        assert bare.messages[-1]["content"] == "Готово."

    def test_an_answer_cut_midway_is_continued_not_repeated(self, bare, monkeypatch):
        def cut(kw):
            yield {"content": "Привет, "}
            raise ConnectionResetError("reset")

        calls = _provider(monkeypatch, [cut, _says("мир.")])
        _reach(monkeypatch, "up")
        events = _run(bare)
        assert _texts(events, "content_stream") == ["Привет, ", "мир."]        # nothing shown twice
        retry_messages = calls[1]["messages"]
        assert retry_messages[-2] == {"role": "assistant", "content": "Привет, "}
        assert retry_messages[-1] == {"role": "user", "content": RESUME_NOTE}
        assert bare.messages[-1] == {"role": "assistant", "content": "Привет, мир."}   # one whole answer, no note stored
        assert RESUME_NOTE not in str(bare.messages)

    def test_a_busy_provider_is_waited_out(self, bare, monkeypatch):
        calls = _provider(monkeypatch, [_fails(ProviderError("HTTP 503 upstream unavailable")), _says("ok")])
        _reach(monkeypatch, "up")
        _run(bare)
        assert len(calls) == 2 and bare.messages[-1]["content"] == "ok"

    def test_when_the_connection_never_returns_the_turn_stops_cleanly(self, bare, monkeypatch):
        monkeypatch.setenv("ARGENT_NETWORK_PATIENCE", "0.05")
        calls = _provider(monkeypatch, [_fails(ConnectError("boom"))])
        _reach(monkeypatch, "offline")
        events = _run(bare, "сделай важное")
        assert "ход остановлен" in " ".join(_texts(events, "error")) and "продолжай" in " ".join(_texts(events, "error"))
        last = bare.messages[-1]
        assert last["role"] == "assistant" and "пропала связь" in last["content"]      # the user's request is answered
        assert bare.messages[-2] == {"role": "user", "content": "сделай важное"}       # and still there to continue from
        assert len(calls) == 1

    def test_esc_while_waiting_interrupts_instead_of_retrying(self, bare, monkeypatch):
        calls = _provider(monkeypatch, [_fails(ConnectError("boom")), _says("never")])

        def fake(provider=None, timeout=3.0, probe=None):
            bare._stop_requested = True
            return "offline"

        monkeypatch.setattr(netstate, "reachability", fake)
        events = _run(bare)
        assert "прервано пользователем" in " ".join(_texts(events, "error"))
        assert len(calls) == 1

    @pytest.mark.parametrize("exc", [ProviderError("HTTP 401 invalid api key"), ValueError("bug in the request")])
    def test_errors_that_waiting_cannot_fix_are_reported_at_once(self, bare, monkeypatch, exc):
        calls = _provider(monkeypatch, [_fails(exc), _says("never")])
        _reach(monkeypatch, "up")
        events = _run(bare)
        assert len(calls) == 1 and _texts(events, "error")

    def test_patience_zero_restores_the_old_behaviour(self, bare, monkeypatch):
        monkeypatch.setenv("ARGENT_NETWORK_PATIENCE", "0")
        calls = _provider(monkeypatch, [_fails(ConnectError("boom")), _says("never")])
        events = _run(bare)
        assert len(calls) == 1 and _texts(events, "error")


class TestWaitPlan:
    def test_what_is_worth_waiting_for(self):
        plan = ArgentAgent._wait_plan
        assert plan(ConnectionResetError("x"))[0] == "transport"
        assert plan(ProviderError("HTTP 503 upstream"))[0] == "busy"
        assert plan(ProviderError("HTTP 429 rate limit exceeded"))[0] == "rate_limited"

    def test_what_is_not(self):
        plan = ArgentAgent._wait_plan
        assert plan(ProviderError("HTTP 401 invalid api key")) is None
        assert plan(ProviderError("HTTP 429 free-models-per-day limit reached")) is None
        assert plan(ValueError("bug")) is None


# ── history repair, autosave, restore ───────────────────────────────────────────────────────────────────

class TestRepairHistory:
    def _agent(self, messages):
        a = ArgentAgent.__new__(ArgentAgent)
        a.messages = messages
        return a

    def test_the_note_for_a_lost_connection_says_so(self):
        a = self._agent([{"role": "system", "content": "s"}, {"role": "user", "content": "do it"}])
        assert a.repair_history(reason="network") == 1
        assert "пропала связь" in a.messages[-1]["content"]

    def test_a_whole_history_reports_zero(self):
        a = self._agent([{"role": "system", "content": "s"}, {"role": "user", "content": "x"},
                         {"role": "assistant", "content": "y"}])
        assert a.repair_history() == 0


class TestAutosave:
    def _agent(self, monkeypatch, saved):
        a = ArgentAgent.__new__(ArgentAgent)
        a.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
        a.model_name, a.provider, a.session_id = "m", "openrouter", None

        def fake_save(messages, meta=None, session_id=None, label=None, state=None):
            saved.append((len(messages), session_id, state))
            return "sid-1"

        monkeypatch.setattr(session, "save_session", fake_save)
        return a

    def test_saves_and_remembers_the_session_id(self, monkeypatch):
        saved = []
        a = self._agent(monkeypatch, saved)
        a._persist_progress()
        assert a.session_id == "sid-1" and saved == [(2, None, {})]

    def test_is_throttled_unless_forced(self, monkeypatch):
        saved = []
        a = self._agent(monkeypatch, saved)
        a._persist_progress()
        a._persist_progress()
        assert len(saved) == 1
        a._persist_progress(force=True)
        assert len(saved) == 2 and saved[1][1] == "sid-1"           # saved back over the same session

    def test_a_helper_agent_and_a_scheduled_run_do_not_save(self, monkeypatch):
        saved = []
        a = self._agent(monkeypatch, saved)
        a.autosave_enabled = False
        a._persist_progress(force=True)
        assert saved == []
        assert agent_module.ArgentSubAgent.autosave_enabled is False

    def test_a_failing_disk_never_breaks_the_turn(self, monkeypatch):
        a = self._agent(monkeypatch, [])
        monkeypatch.setattr(session, "save_session", lambda *x, **k: (_ for _ in ()).throw(OSError("disk full")))
        a._persist_progress(force=True)                              # must not raise

    def test_a_whole_turn_leaves_a_session_on_disk(self, bare, monkeypatch):
        _provider(monkeypatch, [_says("Готово.")])
        bare.session_id = None
        _run(bare, "привет")
        saved = session.load_session(bare.session_id)
        assert saved and saved["messages"][-1]["content"] == "Готово."


class TestSessionStateRoundTrip:
    def test_the_todo_list_survives_a_restart(self):
        todos.TODOS.clear("main")
        todos.write([{"content": "написать тест", "status": "completed"}, {"content": "запустить", "status": "in_progress"}])
        a = ArgentAgent.__new__(ArgentAgent)
        state = a.session_state()
        sid = session.save_session([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}],
                                   {"model": "m", "provider": "p"}, state=state)
        todos.TODOS.clear("main")
        assert todos.TODOS.get("main") == []
        todos.restore(session.load_session(sid)["state"]["todos"])
        assert [(t.content, t.status) for t in todos.TODOS.get("main")] == [
            ("написать тест", "completed"), ("запустить", "in_progress")]
        todos.TODOS.clear("main")
        session.delete_session(sid)

    def test_a_session_without_state_restores_an_empty_list(self):
        todos.TODOS.set(todos.parse_items(["stale"]), "main")
        todos.restore(None)
        assert todos.TODOS.get("main") == []
