"""Blind-spot tests for tools/symbolic_math.py.

Covers edge cases the happy-path suite (test_symbolic_math.py) does not exercise:
the power guard, the string-in-arithmetic trap, ternary rejection, empty input,
caret-to-power rewriting, matrix methods/properties, kwargs rejection, short-
name undefined functions, constants, boolean True in Piecewise, and direct unit
tests of the internal ``_guard_power`` and ``_binop`` helpers.
"""

import ast

import pytest
import sympy

from tools.symbolic_math import (
    SymbolicError,
    _binop,
    _guard_power,
    evaluate_symbolic,
)


# ---------------------------------------------------------------------------
# 1. Basic arithmetic
# ---------------------------------------------------------------------------
class TestBasicArithmetic:
    def test_two_plus_three_collapses_to_five(self):
        assert evaluate_symbolic("2 + 3") == "5"

    def test_polynomial_stays_symbolic(self):
        out = evaluate_symbolic("x**2 - 4")
        assert "x**2" in out and "- 4" in out

    def test_float_is_rationalised(self):
        # floats are converted to exact rationals via nsimplify(rational=True)
        out = evaluate_symbolic("0.5 + 0.25")
        assert "3/4" in out


# ---------------------------------------------------------------------------
# 2. Calculus
# ---------------------------------------------------------------------------
class TestCalculus:
    def test_indefinite_integral(self):
        assert evaluate_symbolic("integrate(x**2, x)") == "x**3/3"

    def test_derivative_product_rule(self):
        out = evaluate_symbolic("diff(sin(x)*x, x)")
        assert "sin(x)" in out and "cos(x)" in out

    def test_classic_limit(self):
        assert evaluate_symbolic("limit(sin(x)/x, x, 0)") == "1"


# ---------------------------------------------------------------------------
# 3. Equations
# ---------------------------------------------------------------------------
class TestEquations:
    def test_solve_quadratic_returns_both_roots(self):
        assert evaluate_symbolic("solve(x**2 - 4, x)") == "[-2, 2]"

    def test_eq_constructor_works(self):
        # Eq(x + 1, 5) is a Call so no auto-simplify; just verify it parses.
        out = evaluate_symbolic("Eq(x + 1, 5)")
        assert "Eq" in out and "x + 1" in out and "5" in out

    def test_double_equals_comparison_works(self):
        # x + 1 == 5 becomes Eq(x + 1, 5) via the Compare handler.
        out = evaluate_symbolic("x + 1 == 5")
        assert "Eq" in out and "x + 1" in out

    def test_solve_with_eq_constructor(self):
        assert evaluate_symbolic("solve(Eq(x**2, 4), x)") == "[-2, 2]"


# ---------------------------------------------------------------------------
# 4. Power guard
# ---------------------------------------------------------------------------
class TestPowerGuard:
    def test_huge_exponent_rejected(self):
        with pytest.raises(SymbolicError) as exc:
            evaluate_symbolic("2**99999")
        assert "exponent too large" in str(exc.value)

    def test_tower_of_powers_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("9**9**9")

    def test_moderate_exponent_allowed(self):
        # 2**10 = 1024 — well within limits
        assert evaluate_symbolic("2**10") == "1024"


# ---------------------------------------------------------------------------
# 5. Security — disallowed function calls
# ---------------------------------------------------------------------------
class TestSecurity:
    def test_dunder_import_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic('__import__("os")')

    def test_eval_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic('eval("1+1")')

    def test_system_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic('system("ls")')

    def test_open_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("open('secret.txt')")

    def test_exec_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("exec('x=1')")


# ---------------------------------------------------------------------------
# 6. String literal in arithmetic
# ---------------------------------------------------------------------------
class TestStringInArithmetic:
    def test_string_operand_in_addition_rejected(self):
        with pytest.raises(SymbolicError) as exc:
            evaluate_symbolic('a + "b"')
        assert "string" in str(exc.value).lower()

    def test_string_repetition_via_mult_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("'a' * 3")


# ---------------------------------------------------------------------------
# 7. Unsupported operator (ternary)
# ---------------------------------------------------------------------------
class TestUnsupportedOperator:
    def test_ternary_ifexp_rejected(self):
        with pytest.raises(SymbolicError) as exc:
            evaluate_symbolic("x if True else y")
        assert "IfExp" in str(exc.value) or "unsupported" in str(exc.value).lower()

    def test_lambda_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("lambda: 1")


# ---------------------------------------------------------------------------
# 8. Empty expression
# ---------------------------------------------------------------------------
class TestEmptyExpression:
    def test_empty_string_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("   ")

    def test_none_input_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 9. Caret to power
# ---------------------------------------------------------------------------
class TestCaretToPower:
    def test_caret_replaced_with_double_star(self):
        assert evaluate_symbolic("2^3") == "8"

    def test_caret_in_polynomial(self):
        assert evaluate_symbolic("x^2 + 1").startswith("x**2")


# ---------------------------------------------------------------------------
# 10. Matrix
# ---------------------------------------------------------------------------
class TestMatrix:
    def test_matrix_literal_works(self):
        out = evaluate_symbolic("Matrix([[1,2],[3,4]])")
        assert "Matrix" in out and "1" in out and "4" in out

    def test_determinant_returns_minus_two(self):
        assert evaluate_symbolic("det(Matrix([[1,2],[3,4]]))") == "-2"

    def test_matrix_multiplication(self):
        out = evaluate_symbolic("Matrix([[1,2],[3,4]]) * Matrix([[5,6],[7,8]])")
        assert "19" in out and "50" in out


# ---------------------------------------------------------------------------
# 11. Method calls on matrices
# ---------------------------------------------------------------------------
class TestMatrixMethods:
    def test_transpose_property(self):
        out = evaluate_symbolic("Matrix([[1,2],[3,4]]).T")
        assert "Matrix" in out
        # Transpose of [[1,2],[3,4]] is [[1,3],[2,4]]
        assert "1, 3" in out or "1,3" in out

    def test_eigenvals_method(self):
        out = evaluate_symbolic("Matrix([[1,2],[3,4]]).eigenvals()")
        # Eigenvalues involve sqrt(33); just verify it returned a dict-like result
        assert "sqrt" in out or ":" in out

    def test_subs_method_on_integral(self):
        assert evaluate_symbolic("integrate(x**2, x).subs(x, 3)") == "9"


# ---------------------------------------------------------------------------
# 12. Disallowed method (__class__)
# ---------------------------------------------------------------------------
class TestDisallowedMethod:
    def test_dunder_method_on_matrix_rejected(self):
        with pytest.raises(SymbolicError) as exc:
            evaluate_symbolic("Matrix([[1,2],[3,4]]).__class__")
        # __class__ is an attribute, checked against _ALLOWED_PROPERTIES
        assert "not allowed" in str(exc.value) or "attribute" in str(exc.value).lower()

    def test_non_whitelisted_method_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("x.as_coefficients_dict()")


# ---------------------------------------------------------------------------
# 13. Disallowed attribute
# ---------------------------------------------------------------------------
class TestDisallowedAttribute:
    def test_dunder_attribute_on_symbol_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("x.__class__")

    def test_globals_escape_chain_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("x.__class__.__mro__")


# ---------------------------------------------------------------------------
# 14. Kwargs
# ---------------------------------------------------------------------------
class TestKwargs:
    def test_solve_with_eq_works(self):
        assert evaluate_symbolic("solve(Eq(x**2, 4), x)") == "[-2, 2]"

    def test_double_star_kwargs_rejected(self):
        with pytest.raises(SymbolicError) as exc:
            evaluate_symbolic("sin(**x)")
        assert "kwargs" in str(exc.value).lower()

    def test_double_star_kwargs_on_method_rejected(self):
        with pytest.raises(SymbolicError) as exc:
            evaluate_symbolic("Matrix([[1,2],[3,4]]).subs(**d)")
        assert "kwargs" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# 15. Short function names (undefined functions for ODEs)
# ---------------------------------------------------------------------------
class TestShortFunctionNames:
    def test_single_char_function_becomes_undefined_function(self):
        out = evaluate_symbolic("f(x)")
        assert "f" in out and "x" in out

    def test_two_char_function_becomes_undefined_function(self):
        out = evaluate_symbolic("gy(x)")
        assert "gy" in out

    def test_long_function_name_rejected(self):
        with pytest.raises(SymbolicError) as exc:
            evaluate_symbolic("hack(x)")
        assert "unknown function" in str(exc.value)

    def test_underscore_name_not_treated_as_short_function(self):
        # __import__ contains underscores so isalpha() is False → rejected
        with pytest.raises(SymbolicError):
            evaluate_symbolic("__import__('os')")


# ---------------------------------------------------------------------------
# 16. Constants
# ---------------------------------------------------------------------------
class TestConstants:
    def test_pi(self):
        out = evaluate_symbolic("pi")
        assert out.startswith("pi")

    def test_euler_E(self):
        out = evaluate_symbolic("E")
        assert out.startswith("E")

    def test_lowercase_e_is_E(self):
        out = evaluate_symbolic("e")
        assert out.startswith("E")

    def test_infinity(self):
        assert evaluate_symbolic("oo") == "oo"

    def test_imaginary_unit(self):
        out = evaluate_symbolic("I")
        assert "I" in out


# ---------------------------------------------------------------------------
# 17. Boolean True in Piecewise
# ---------------------------------------------------------------------------
class TestBooleanInPiecewise:
    def test_piecewise_with_true_catchall(self):
        out = evaluate_symbolic("Piecewise((x, x < 1), (2 - x, True))")
        assert "Piecewise" in out
        assert "x < 1" in out
        # True should appear as the catch-all condition
        assert "True" in out

    def test_piecewise_integral_with_true(self):
        assert evaluate_symbolic(
            "integrate(Piecewise((x, x < 1), (2 - x, True)), (x, 0, 2))"
        ) == "1"


# ---------------------------------------------------------------------------
# 18. _guard_power — direct unit tests
# ---------------------------------------------------------------------------
class TestGuardPowerDirect:
    def test_numeric_within_limits_does_not_raise(self):
        _guard_power(sympy.Integer(2), sympy.Integer(10))
        _guard_power(sympy.Integer(100), sympy.Integer(100))

    def test_huge_exponent_raises(self):
        with pytest.raises(SymbolicError) as exc:
            _guard_power(sympy.Integer(2), sympy.Integer(99999))
        assert "exponent too large" in str(exc.value)

    def test_huge_base_with_moderate_exponent_raises(self):
        # b > _MAX_BASE_FOR_BIG_EXP (1e6) and e > 100
        with pytest.raises(SymbolicError):
            _guard_power(sympy.Integer(10**7), sympy.Integer(101))

    def test_symbolic_base_does_not_raise(self):
        # Symbol has is_number=False → guard returns early
        _guard_power(sympy.Symbol("x"), sympy.Integer(99999))

    def test_symbolic_exponent_does_not_raise(self):
        _guard_power(sympy.Integer(2), sympy.Symbol("n"))

    def test_float_within_limits_does_not_raise(self):
        _guard_power(sympy.Float(2.5), sympy.Float(3.5))


# ---------------------------------------------------------------------------
# 19. _binop — direct unit tests
# ---------------------------------------------------------------------------
class TestBinopDirect:
    def test_string_left_operand_raises(self):
        with pytest.raises(SymbolicError) as exc:
            _binop(sympy, ast.Add(), "hello", sympy.Integer(1))
        assert "string" in str(exc.value).lower()

    def test_string_right_operand_raises(self):
        with pytest.raises(SymbolicError) as exc:
            _binop(sympy, ast.Add(), sympy.Integer(1), "world")
        assert "string" in str(exc.value).lower()

    def test_bitand_on_relations(self):
        x = sympy.Symbol("x")
        result = _binop(sympy, ast.BitAnd(), sympy.Gt(x, 0), sympy.Lt(x, 5))
        assert "And" in sympy.sstr(result) or "&" in sympy.sstr(result)

    def test_bitor_on_intervals(self):
        result = _binop(
            sympy,
            ast.BitOr(),
            sympy.Interval(0, 2),
            sympy.Interval(3, 4),
        )
        assert "Union" in sympy.sstr(result)

    def test_bitxor_on_booleans(self):
        result = _binop(sympy, ast.BitXor(), sympy.true, sympy.false)
        assert result is True or result == sympy.true

    def test_add_works(self):
        result = _binop(sympy, ast.Add(), sympy.Integer(2), sympy.Integer(3))
        assert result == 5

    def test_pow_calls_guard(self):
        result = _binop(sympy, ast.Pow(), sympy.Integer(2), sympy.Integer(3))
        assert result == 8

    def test_unsupported_operator_raises(self):
        with pytest.raises(SymbolicError) as exc:
            _binop(sympy, ast.LShift(), sympy.Integer(1), sympy.Integer(2))
        assert "unsupported operator" in str(exc.value)


# ---------------------------------------------------------------------------
# 20. SymbolicError is a ValueError subclass
# ---------------------------------------------------------------------------
class TestSymbolicErrorType:
    def test_symbolic_error_is_value_error_subclass(self):
        assert issubclass(SymbolicError, ValueError)

    def test_symbolic_error_can_be_caught_as_value_error(self):
        try:
            raise SymbolicError("test")
        except ValueError:
            pass  # caught as ValueError — success
        else:
            pytest.fail("SymbolicError was not caught as ValueError")

    def test_symbolic_error_message_preserved(self):
        try:
            raise SymbolicError("my message")
        except SymbolicError as exc:
            assert str(exc) == "my message"