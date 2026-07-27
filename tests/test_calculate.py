from tools.misc_tools import calculate


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
