"""
Blind-spot tests for usage_tracker.py.

These tests document edge-case behaviors (negative values, None, strings,
NaN/Inf, float precision, boundary conditions) that the production code
handles in ways that may be surprising or silently wrong.
"""

import math

import pytest

from usage_tracker import SessionUsage, _fmt_cost, _short


# ---------------------------------------------------------------------------
# L4×D1 — Negative values are accumulated silently (no validation)
# ---------------------------------------------------------------------------
class TestAddNegative:
    """Adding negative usage values: the code does no validation, so
    negatives are silently accumulated. This documents that blind spot."""

    def test_negative_values_accumulated(self):
        su = SessionUsage()
        su.add({"prompt": -100, "completion": -50, "cost": -0.5})
        assert su.prompt == -100
        assert su.completion == -50
        assert su.cost == -0.5
        assert su.requests == 1


# ---------------------------------------------------------------------------
# L4×D1 — None values: `None or 0` evaluates to 0 because None is falsy
# ---------------------------------------------------------------------------
class TestAddNone:
    """None in the usage dict: `int(usage.get('prompt', 0) or 0)` —
    None is falsy, so `None or 0` == 0. Same for cost with `or 0.0`."""

    def test_none_values_become_zero(self):
        su = SessionUsage()
        su.add({"prompt": None, "completion": None, "cost": None})
        assert su.prompt == 0
        assert su.completion == 0
        assert su.cost == 0.0
        assert su.requests == 1


# ---------------------------------------------------------------------------
# L4×D1 — String values: int('100') works, int('abc') raises ValueError
# ---------------------------------------------------------------------------
class TestAddStringValues:
    """String numeric values are coerced by int()/float().
    Non-numeric strings raise ValueError — no graceful handling."""

    def test_numeric_strings_coerced(self):
        su = SessionUsage()
        su.add({"prompt": "100", "completion": "50"})
        assert su.prompt == 100
        assert su.completion == 50

    def test_non_numeric_string_raises(self):
        su = SessionUsage()
        with pytest.raises(ValueError):
            su.add({"prompt": "abc", "completion": "50"})


# ---------------------------------------------------------------------------
# L4×D1 — NaN and Inf cost: NaN is truthy so it passes through `or 0.0`
# ---------------------------------------------------------------------------
class TestAddNaNInfCost:
    """NaN cost: `float('nan') or 0.0` == nan (NaN is truthy in Python).
    NaN > 0 is False, so format_last omits the cost. Inf > 0 is True,
    so _fmt_cost(inf) is called — it produces '$inf' without crashing."""

    def test_nan_cost_accumulated(self):
        su = SessionUsage()
        su.add({"cost": float("nan")})
        assert math.isnan(su.cost)

    def test_format_last_nan_cost_omitted(self):
        # NaN > 0 is False, so the cost portion is never appended.
        result = SessionUsage.format_last({"prompt": 10, "completion": 5, "cost": float("nan")})
        assert "$" not in result

    def test_fmt_cost_inf_does_not_crash(self):
        # Inf > 0 is True, so _fmt_cost is called. Just verify no exception.
        result = _fmt_cost(float("inf"))
        assert isinstance(result, str)

    def test_format_last_inf_cost_shown(self):
        # Inf > 0 is True → _fmt_cost(inf) == '$inf'
        result = SessionUsage.format_last({"prompt": 10, "completion": 5, "cost": float("inf")})
        assert "$" in result


# ---------------------------------------------------------------------------
# L2×D1 — _short(0)
# ---------------------------------------------------------------------------
class TestShortZero:
    def test_short_zero(self):
        assert _short(0) == "0"


# ---------------------------------------------------------------------------
# L4×D1 — _short with negative numbers (no abs() in the code)
# ---------------------------------------------------------------------------
class TestShortNegative:
    """BLIND SPOT: The thresholds use `n < 1000`, but ALL negative numbers
    satisfy `n < 1000` (e.g. -1234 < 1000 is True). So negative numbers
    ALWAYS take the `str(n)` branch — they never get k/M formatting.
    -1234 returns '-1234', not '-1.2k'."""

    def test_short_negative_one(self):
        assert _short(-1) == "-1"

    def test_short_negative_thousands(self):
        # -1234 < 1000 is True, so str(-1234) is returned — no k formatting.
        assert _short(-1234) == "-1234"

    def test_short_large_negative_no_m_formatting(self):
        # Even -2_000_000 < 1000 is True, so it returns str(-2000000).
        assert _short(-2_000_000) == "-2000000"


# ---------------------------------------------------------------------------
# L2×D1 — _short boundary values at 999 / 1000 / 999999
# ---------------------------------------------------------------------------
class TestShortBoundary:
    """999 < 1000 → str. 1000 → '1.0k'.replace('.0k','k') → '1k'.
    999999 → 999.999 → f'{999.999:.1f}' rounds to '1000.0' → '1000.0k'
    which contains '.0k' so .replace('.0k','k') → '1000k'."""

    def test_short_999(self):
        assert _short(999) == "999"

    def test_short_1000(self):
        assert _short(1000) == "1k"

    def test_short_999999(self):
        # 999999/1000 = 999.999, :.1f rounds to 1000.0 → "1000.0k"
        # ".0k" substring IS present → replaced → "1000k"
        assert _short(999999) == "1000k"


# ---------------------------------------------------------------------------
# L4×D1 — _fmt_cost with negative cost (no abs, no guard)
# ---------------------------------------------------------------------------
class TestFmtCostNegative:
    """_fmt_cost(-0.05): -0.05 < 0.1 is True → f'${-0.05:.4f}' == '$-0.0500'.
    The minus sign leaks into the formatted string — a blind spot."""

    def test_fmt_cost_negative_small(self):
        result = _fmt_cost(-0.05)
        assert "-" in result
        assert result == "$-0.0500"

    def test_fmt_cost_negative_large(self):
        # -5.0 < 0.1 is True → .4f format
        result = _fmt_cost(-5.0)
        assert "-" in result


# ---------------------------------------------------------------------------
# L3×D1 — Float accumulation precision (0.1 added ten times)
# ---------------------------------------------------------------------------
class TestCostFloatPrecision:
    """Adding 0.1 ten times does NOT produce exactly 1.0 due to IEEE 754.
    The code uses plain float += with no rounding."""

    def test_add_tenth_ten_times(self):
        su = SessionUsage()
        for _ in range(10):
            su.add({"cost": 0.1})
        # float accumulation of 0.1×10 gives 0.9999999999999999, not 1.0
        assert su.cost > 0.99
        assert su.cost < 1.01


# ---------------------------------------------------------------------------
# L2×D1 — Empty dict: all defaults, requests still incremented
# ---------------------------------------------------------------------------
class TestAddEmptyDict:
    def test_add_empty_dict(self):
        su = SessionUsage()
        su.add({})
        assert su.prompt == 0
        assert su.completion == 0
        assert su.cost == 0.0
        assert su.requests == 1


# ---------------------------------------------------------------------------
# L5×D2 — reset() restores all fields to __init__ defaults
# ---------------------------------------------------------------------------
class TestResetClearsAll:
    def test_reset_clears_all_fields(self):
        su = SessionUsage()
        su.add({"prompt": 500, "completion": 200, "cost": 0.75})
        su.add({"prompt": 100, "completion": 50, "cost": 0.25})
        assert su.prompt == 600
        assert su.completion == 250
        assert su.cost == 1.0
        assert su.requests == 2

        su.reset()
        assert su.prompt == 0
        assert su.completion == 0
        assert su.cost == 0.0
        assert su.requests == 0

    def test_reset_then_add_works(self):
        su = SessionUsage()
        su.add({"prompt": 999, "completion": 999, "cost": 1.0})
        su.reset()
        su.add({"prompt": 10, "completion": 5, "cost": 0.1})
        assert su.prompt == 10
        assert su.completion == 5
        assert su.cost == 0.1
        assert su.requests == 1
# ---------------------------------------------------------------------------
# L1×D10 — Exact format string: '->' arrow must be preserved
# ---------------------------------------------------------------------------
class TestFormatLastExactFormat:
    """The format string uses '->' between prompt and completion counts.
    Mutations that change '-' or '>' inside the f-string alter the output
    and must be caught by exact-format assertions."""

    def test_format_last_arrow_preserved(self):
        """format_last must produce 'X->Y tok' with a literal '->' arrow."""
        result = SessionUsage.format_last({"prompt": 24, "completion": 20})
        assert "->" in result
        assert result == "24->20 tok"

    def test_format_last_arrow_with_cost(self):
        """format_last with cost: 'X->Y tok · $Z.ZZZZ'."""
        result = SessionUsage.format_last({"prompt": 100, "completion": 50, "cost": 0.05})
        assert "->" in result
        assert "$0.0500" in result

    def test_format_last_zero_zero(self):
        """format_last with zero prompt and completion."""
        result = SessionUsage.format_last({"prompt": 0, "completion": 0})
        assert result == "0->0 tok"

    def test_format_last_large_numbers_short(self):
        """format_last applies _short to both prompt and completion."""
        result = SessionUsage.format_last({"prompt": 1500, "completion": 750})
        assert "->" in result
        assert "1.5k" in result
        assert "750" in result


# ---------------------------------------------------------------------------
# L1×D10 — format_session exact format
# ---------------------------------------------------------------------------
class TestFormatSessionExactFormat:
    """format_session must produce 'sum Xk tok · $Y.YY' with exact formatting."""

    def test_format_session_no_cost(self):
        su = SessionUsage()
        su.add({"prompt": 500, "completion": 300})
        result = su.format_session()
        assert result == "sum 800 tok"

    def test_format_session_with_cost(self):
        su = SessionUsage()
        su.add({"prompt": 500, "completion": 300, "cost": 1.50})
        result = su.format_session()
        assert "sum 800 tok" in result
        assert "$1.50" in result

    def test_format_session_zero_usage(self):
        su = SessionUsage()
        result = su.format_session()
        assert result == "sum 0 tok"

    def test_format_session_cost_zero_not_shown(self):
        """When cost is exactly 0, the cost portion must NOT appear."""
        su = SessionUsage()
        su.add({"prompt": 100, "completion": 50, "cost": 0.0})
        result = su.format_session()
        assert "$" not in result
        assert result == "sum 150 tok"


# ---------------------------------------------------------------------------
# L1×D10 — _fmt_cost boundary at 0.1: kills ZERO_TO_ONE mutation on threshold
# ---------------------------------------------------------------------------
class TestFmtCostBoundary:
    """_fmt_cost uses `cost < 0.1` to decide between 4-decimal and 2-decimal
    format. A ZERO_TO_ONE mutation changes 0.1 to 1.1, so costs in [0.1, 1.1)
    must be tested to ensure the 2-decimal branch is taken."""

    def test_fmt_cost_above_threshold_uses_2_decimals(self):
        """0.5 >= 0.1 → '$0.50' (2 decimals). Mutated: 0.5 < 1.1 → '$0.5000'."""
        assert _fmt_cost(0.5) == "$0.50"

    def test_fmt_cost_at_threshold_uses_2_decimals(self):
        """0.1 is NOT < 0.1, so 2-decimal format. Mutated: 0.1 < 1.1 → 4-decimal."""
        assert _fmt_cost(0.1) == "$0.10"

    def test_fmt_cost_just_above_threshold(self):
        """0.11 >= 0.1 → 2 decimals."""
        assert _fmt_cost(0.11) == "$0.11"

    def test_fmt_cost_just_below_threshold(self):
        """0.09 < 0.1 → 4 decimals. This confirms the boundary direction."""
        assert _fmt_cost(0.09) == "$0.0900"

    def test_fmt_cost_one_dollar_uses_2_decimals(self):
        """1.0 >= 0.1 → 2 decimals. Mutated: 1.0 < 1.1 → 4 decimals."""
        assert _fmt_cost(1.0) == "$1.00"