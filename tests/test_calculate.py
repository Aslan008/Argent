from argent.tools.misc_tools import calculate


class TestCalculateBasics:
    def test_addition(self):
        assert calculate("2 + 2") == "2 + 2 = 4"

    def test_operator_precedence(self):
        assert calculate("2 + 3 * 4").endswith("= 14")

    def test_float_result(self):
        assert calculate("7 / 2").endswith("= 3.5")

    def test_integral_float_collapses_to_int(self):
        assert calculate("10 / 2").endswith("= 5")

    def test_caret_means_power(self):
        assert calculate("2^10").endswith("= 1024")

    def test_unary_minus(self):
        assert calculate("-5 + 3").endswith("= -2")

    def test_parentheses(self):
        assert calculate("(1847 * 0.15) + 0.05").endswith("= 277.1")


class TestCalculateFunctions:
    def test_sqrt(self):
        assert calculate("sqrt(144)").endswith("= 12")

    def test_constants(self):
        assert calculate("round(pi, 2)").endswith("= 3.14")

    def test_nested_functions(self):
        assert calculate("max(min(5, 10), 2)").endswith("= 5")

    def test_factorial(self):
        assert calculate("factorial(5)").endswith("= 120")


class TestCalculateSafety:
    def test_no_code_execution(self):
        result = calculate("__import__('os').system('echo pwned')")
        assert result.startswith("Error")

    def test_attribute_access_rejected(self):
        assert calculate("(1).__class__").startswith("Error")

    def test_free_variable_is_symbolic_not_an_error(self):
        # Changed intent: with symbolic support a free variable is valid input
        # (it used to be rejected as an "unknown identifier").
        assert calculate("x + 1").endswith("= x + 1")

    def test_unknown_function_rejected(self):
        assert calculate("eval('1+1')").startswith("Error")

    def test_string_constant_rejected(self):
        assert calculate("'a' * 3").startswith("Error")

    def test_huge_exponent_guarded(self):
        assert calculate("9**9**9").startswith("Error")

    def test_division_by_zero(self):
        assert "division by zero" in calculate("1 / 0")

    def test_empty_expression(self):
        assert calculate("").startswith("Error")
        assert calculate("   ").startswith("Error")

    def test_syntax_error_has_guidance(self):
        result = calculate("2 +* 3")
        assert result.startswith("Error")
        assert "arithmetic expression" in result


class TestCalculateExactness:
    """Exact arithmetic: no float drift and no silent precision loss."""

    def test_large_integer_division_is_exact(self):
        # The old float path printed 3.333333333333333e+29 (wrong digits past
        # the 16th). The exact rational must survive.
        out = calculate("10**30/3")
        assert "1000000000000000000000000000000/3" in out

    def test_scalar_and_list_agree(self):
        # The same ratio must render the same alone and inside a list.
        assert calculate("7/2").endswith("= 3.5")
        assert calculate("[7/2]").endswith("= [3.5]")

    def test_nonterminating_ratio_shows_approx(self):
        # A bare 4456151/99 hides its magnitude; an "≈" decimal is appended.
        out = calculate("4456151/99")
        assert "4456151/99" in out
        assert "≈" in out and "45011.62" in out

    def test_decimal_literals_are_exact(self):
        # 0.1 and 0.2 are taken as decimal 1/10 and 2/10, not binary floats.
        assert calculate("0.1 + 0.2").endswith("= 0.3")

    def test_float_expression_unchanged(self):
        assert calculate("(1847 * 0.15) + 0.05").endswith("= 277.1")

    def test_original_ratio_list_is_exact(self):
        out = calculate(
            "[(4656151 - 1*200000)/99, (5161983 - 3*200000)/97, "
            "((4656151-200000)/99) / ((5161983-600000)/97)]"
        )
        assert "4456151/99" in out
        assert "432246647/451636317" in out


class TestCalculateUndefinedResults:
    """A pole collapses to a clear error, not a cryptic zoo/nan value."""

    def test_factorial_negative_is_error_not_zoo(self):
        out = calculate("factorial(-1)")
        assert out.startswith("Error")
        assert "zoo" not in out

    def test_inf_minus_inf_is_error_not_nan(self):
        assert calculate("inf - inf").startswith("Error")

    def test_complex_sqrt_still_works(self):
        # sqrt(-1) has a legitimate complex answer and must NOT regress.
        assert "I" in calculate("sqrt(-1)")


class TestCalculateNeverCrashes:
    """Extreme magnitudes must return a string, never raise an exception."""

    def test_float_overflow_ratio_does_not_crash(self):
        # float(10**400/3) overflows; the exact fraction must still come back.
        out = calculate("10**400/3")
        assert out.startswith("10**400/3 =")
        assert out.endswith("/3")  # exact form kept, no float ≈

    def test_huge_terminating_fraction_does_not_crash(self):
        # (3/2)**5000 exceeds the 4300-digit int→str limit; degrade safely.
        out = calculate("(3/2)**5000")
        assert out.startswith("(3/2)**5000 =")
        assert "Error" not in out

    def test_huge_integer_is_labelled_not_dumped(self):
        assert "digit integer" in calculate("factorial(4000)")

    def test_big_multiplication_is_labelled(self):
        assert "digit integer" in calculate("10**2000*10**2000*10**2000")

    def test_factorial_size_capped(self):
        assert calculate("factorial(100000)").startswith("Error")


class TestCalculateImplicitMultiplication:
    """Common implicit products are accepted; scientific notation stays intact."""

    def test_number_variable(self):
        assert calculate("2x").endswith("= 2*x")

    def test_number_paren(self):
        assert calculate("2(x+1)").endswith("= 2*x + 2")

    def test_paren_paren(self):
        assert calculate("(x+1)(x-1)").endswith("= x**2 - 1")

    def test_function_paren_product(self):
        assert calculate("2(3)(4)").endswith("= 24")

    def test_scientific_notation_untouched(self):
        assert calculate("2e5").endswith("= 200000")
        assert calculate("1e-3").endswith("= 0.001")

    def test_identifier_ending_in_digit_not_split(self):
        assert calculate("log10(100)").endswith("= 2")

    def test_bare_identifier_is_not_a_product(self):
        assert calculate("xy").endswith("= xy")
