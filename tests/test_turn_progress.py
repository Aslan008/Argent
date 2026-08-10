"""No turn cap — a progress notice instead, and the measurement that decided it.

The question was whether _run_turn needs a global ceiling on tool calls. It
does not. Measured over 1315 recorded turns in agent.log:

    0 calls    1224 turns      31-60      7
    1-5          62            60+       10   (max 127)
    6-15          7
    16-30         5

and every one of the long turns ended for a good reason — "final answer" at
127 calls, at 106, at 100; the rest stopped by the loop guard or an error. A
ceiling at any of those numbers would have cut finished work to prevent a
runaway the log does not contain.

What the log does show is turns running 30-40 minutes. The thing actually
missing was a way to notice one and decide.
"""

import agent as agent_mod


class TestTheNotice:
    def test_it_is_not_a_cap(self):
        """Nothing in the module stops a turn by tool count."""
        import inspect

        source = inspect.getsource(agent_mod)
        assert "TURN_PROGRESS_EVERY" in source
        for forbidden in ("_turn_tool_calls >", "_turn_tool_calls >=",
                          "MAX_TURN_TOOL_CALLS", "TURN_HARD_LIMIT"):
            assert forbidden not in source, f"появился потолок хода: {forbidden}"

    def test_the_interval_is_sane(self):
        assert 10 <= agent_mod.TURN_PROGRESS_EVERY <= 50

    def test_it_fires_on_the_interval(self):
        every = agent_mod.TURN_PROGRESS_EVERY
        fired = [n for n in range(1, 101) if n % every == 0]
        assert fired[0] == every
        assert 0 not in fired          # never on the first call

    def test_it_travels_as_a_system_notice(self):
        """cli_ui routes an 'error' chunk starting with a bracketed source to
        print_system, not print_error — this must not look like a failure."""
        from src.cli.cli_ui import _NOTICE_PREFIXES

        message = "[Argent: 25 вызовов инструментов, 3 мин работы над этим запросом.]"
        assert message.startswith(_NOTICE_PREFIXES)

    def test_it_says_how_to_stop(self):
        import inspect

        source = inspect.getsource(agent_mod)
        block = source.split("TURN_PROGRESS_EVERY == 0", 1)[1][:400]
        assert "Ctrl+C" in block
