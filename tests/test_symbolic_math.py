"""Symbolic calculator: calculus and algebra, still with zero code execution."""

import pytest

from tools.misc_tools import calculate
from tools.symbolic_math import SymbolicError, evaluate_symbolic


class TestCalculus:
    def test_indefinite_integral(self):
        assert evaluate_symbolic("integrate(x**2, x)") == "x**3/3"

    def test_definite_integral(self):
        assert evaluate_symbolic("integrate(x**2, (x, 0, 3))") == "9"

    def test_improper_integral_to_infinity(self):
        assert evaluate_symbolic("integrate(exp(-x), (x, 0, oo))") == "1"

    def test_derivative(self):
        out = evaluate_symbolic("diff(sin(x)*x, x)")
        assert "sin(x)" in out and "cos(x)" in out

    def test_second_derivative(self):
        assert evaluate_symbolic("diff(x**4, x, 2)") == "12*x**2"

    def test_classic_limit(self):
        assert evaluate_symbolic("limit(sin(x)/x, x, 0)") == "1"

    def test_one_sided_limit(self):
        assert evaluate_symbolic("limit(1/x, x, 0, '+')") == "oo"

    def test_limit_at_infinity_is_e(self):
        out = evaluate_symbolic("limit((1 + 1/n)**n, n, oo)")
        assert out.startswith("E") and "≈ 2.718281828" in out

    def test_summation(self):
        assert evaluate_symbolic("summation(k, (k, 1, 10))") == "55"

    def test_symbolic_summation(self):
        out = evaluate_symbolic("summation(k, (k, 1, n))")
        assert "n" in out          # closed form n*(n+1)/2

    def test_series_expansion(self):
        out = evaluate_symbolic("series(exp(x), x, 0, 4)")
        assert "x**2/2" in out and "O(x**4)" in out


class TestAlgebra:
    def test_solve_quadratic(self):
        out = evaluate_symbolic("solve(x**2 - 4, x)")
        assert "-2" in out and "2" in out

    def test_solve_equation_form(self):
        assert evaluate_symbolic("solve(Eq(x + 1, 5), x)") == "[4]"

    def test_solve_with_double_equals(self):
        assert evaluate_symbolic("solve(x**2 == 9, x)") == "[-3, 3]"

    def test_expand(self):
        assert evaluate_symbolic("expand((x + 1)**2)") == "x**2 + 2*x + 1"

    def test_factor(self):
        assert evaluate_symbolic("factor(x**2 - 1)") == "(x - 1)*(x + 1)"

    def test_simplify_trig_identity(self):
        assert evaluate_symbolic("simplify(sin(x)**2 + cos(x)**2)") == "1"


class TestInequalities:
    def test_quadratic_inequality(self):
        out = evaluate_symbolic("solve(x**2 > 4, x)")
        assert "-2" in out and "2" in out

    def test_system_of_inequalities(self):
        assert evaluate_symbolic("reduce_inequalities([x + 3 < 7, x > 0], [x])") \
            == "(0 < x) & (x < 4)"

    def test_not_equal_relation(self):
        assert "Ne" in evaluate_symbolic("Ne(x, 2)") or "!=" in evaluate_symbolic("Ne(x, 2)")


class TestDifferentialEquations:
    def test_first_order_ode(self):
        assert evaluate_symbolic("dsolve(Derivative(f(x), x) - f(x), f(x))") \
            == "Eq(f(x), C1*exp(x))"

    def test_second_order_ode(self):
        out = evaluate_symbolic("dsolve(Derivative(y(x), x, 2) + y(x), y(x))")
        assert "sin(x)" in out and "cos(x)" in out

    def test_undefined_function_is_limited_to_short_names(self):
        # f(x)/y(t) become undefined functions; a long name stays an error so
        # hallucinated calls are still caught.
        with pytest.raises(SymbolicError):
            evaluate_symbolic("malicious(x)")


class TestLinearAlgebra:
    def test_determinant(self):
        assert evaluate_symbolic("det(Matrix([[1, 2], [3, 4]]))") == "-2"

    def test_matrix_product(self):
        out = evaluate_symbolic("Matrix([[1,2],[3,4]]) * Matrix([[5,6],[7,8]])")
        assert "19" in out and "50" in out

    def test_matrix_inverse_via_power(self):
        out = evaluate_symbolic("Matrix([[1, 2], [3, 4]])**-1")
        assert "3/2" in out

    def test_trace(self):
        assert evaluate_symbolic("trace(Matrix([[1,2],[3,4]]))") == "5"


class TestNumericCollapse:
    def test_complex_power_collapses_to_a_number(self):
        assert evaluate_symbolic("(1 + I)**8") == "16"

    def test_integer_factorization(self):
        assert evaluate_symbolic("factorint(360)") == "{2: 3, 3: 2, 5: 1}"


class TestExactValuesAndApproximation:
    def test_exact_radical_with_decimal_hint(self):
        out = evaluate_symbolic("sqrt(8)")
        assert out.startswith("2*sqrt(2)") and "≈ 2.8284271" in out

    def test_exact_rational(self):
        assert evaluate_symbolic("Rational(1, 3) + Rational(1, 6)") == "1/2"

    def test_big_factorial_is_exact(self):
        assert evaluate_symbolic("factorial(25)").startswith("15511210043330985984000000")


class TestSafety:
    @pytest.mark.parametrize("expr", [
        "__import__('os').system('echo pwned')",
        "eval('1+1')",
        "exec('x=1')",
        "open('secret.txt')",
        "(1).__class__",
        "sympy.__dict__",
        "[].__class__.__bases__",
        "lambda: 1",
        "'a' * 3",
    ])
    def test_dangerous_input_rejected(self, expr):
        with pytest.raises((SymbolicError, Exception)) as exc:
            evaluate_symbolic(expr)
        assert not isinstance(exc.value, SystemExit)

    def test_no_code_execution_through_calculate(self, tmp_path):
        marker = tmp_path / "pwned.txt"
        result = calculate(f"__import__('pathlib').Path(r'{marker}').touch()")
        assert result.startswith("Error")
        assert not marker.exists()

    def test_attribute_access_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("x.__class__")

    def test_unknown_function_rejected(self):
        with pytest.raises(SymbolicError) as e:
            evaluate_symbolic("hack(x)")
        assert "unknown function" in str(e.value)

    def test_exponent_bomb_guarded(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("9**9**9")

    def test_empty_rejected(self):
        with pytest.raises(SymbolicError):
            evaluate_symbolic("   ")


class TestCalculateIntegration:
    def test_arithmetic_still_uses_the_fast_path(self):
        assert calculate("2 + 2") == "2 + 2 = 4"

    def test_integral_through_calculate(self):
        assert calculate("integrate(x**2, x)").endswith("= x**3/3")

    def test_limit_through_calculate(self):
        assert calculate("limit(sin(x)/x, x, 0)").endswith("= 1")

    def test_caret_power_works_symbolically(self):
        assert calculate("integrate(x^2, x)").endswith("= x**3/3")

    def test_division_by_zero_keeps_its_message(self):
        assert "division by zero" in calculate("1 / 0")

    def test_invalid_input_explains_both_modes(self):
        out = calculate("2 +* 3")
        assert out.startswith("Error")
        assert "arithmetic expression" in out and "integrate" in out
