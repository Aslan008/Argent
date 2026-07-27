"""Symbolic mathematics for the calculator: integrals, limits, derivatives,
equations, series and sums.

Safety works the same way as the numeric evaluator in misc_tools: the input is
parsed to an AST and walked by an explicit whitelist. Nothing is ever eval'd.
This matters more here than usual — SymPy's own ``sympify``/``parse_expr`` call
eval internally, so handing them raw model output would be a code-execution
hole. Instead we build the SymPy expression node by node, so only whitelisted
functions, operators and plain symbols can ever exist.

sympy is imported lazily: it is a heavy import and must not slow Argent's start.
"""

import ast

# Callables the model may use, resolved against the sympy module at call time.
_ALLOWED_FUNCTIONS = {
    # calculus
    "integrate", "diff", "limit", "series", "summation", "product",
    "Sum", "Product", "Derivative", "Integral",
    # algebra
    "solve", "solveset", "simplify", "expand", "factor", "cancel", "apart",
    "together", "collect", "nsolve", "roots", "Eq", "degree", "div", "rem",
    "Ne", "Lt", "Le", "Gt", "Ge", "solve_univariate_inequality", "reduce_inequalities",
    # differential equations
    "dsolve", "Function", "Derivative", "checkodesol", "classify_ode",
    # linear algebra (methods need attributes, so the functional forms are exposed)
    "Matrix", "det", "trace", "transpose", "eye", "zeros", "ones", "diag",
    "nsimplify", "factorint",
    # elementary
    "sqrt", "cbrt", "root", "exp", "log", "ln", "Abs", "sign",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "atan2", "acot",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "floor", "ceiling", "factorial", "binomial", "gcd", "lcm",
    "Min", "Max", "re", "im", "conjugate", "Rational", "Float", "Integer",
    # number theory / misc
    "isprime", "prime", "primerange", "totient", "mod_inverse", "divisors",
}

# Names that resolve to sympy constants rather than free symbols.
_CONSTANT_NAMES = {
    "pi": "pi", "E": "E", "e": "E", "I": "I", "oo": "oo", "inf": "oo",
    "infinity": "oo", "zoo": "zoo", "nan": "nan", "true": "true", "false": "false",
}

# Only these string literals may appear (limit direction). Everything else is
# rejected so a string can never be smuggled into a sympy parser.
_ALLOWED_STRINGS = {"+", "-", "+-", "-+"}

# Guard against expression bombs like 9**9**9 that would hang the process.
_MAX_EXPONENT = 10000
_MAX_BASE_FOR_BIG_EXP = 1e6


class SymbolicError(ValueError):
    """Invalid or disallowed symbolic expression."""


def _binop(sympy, op, left, right):
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div):
        return left / right
    if isinstance(op, ast.FloorDiv):
        return sympy.floor(left / right)
    if isinstance(op, ast.Mod):
        return sympy.Mod(left, right)
    if isinstance(op, ast.Pow):
        _guard_power(left, right)
        return left ** right
    raise SymbolicError(f"unsupported operator: {type(op).__name__}")


def _guard_power(base, exponent):
    """Refuse numeric powers big enough to hang the process."""
    try:
        if not (getattr(base, "is_number", False) and getattr(exponent, "is_number", False)):
            return
        b, e = abs(float(base)), abs(float(exponent))
    except (TypeError, ValueError, OverflowError):
        return
    if e > _MAX_EXPONENT or (b > _MAX_BASE_FOR_BIG_EXP and e > 100):
        raise SymbolicError("exponent too large to evaluate")


def _eval_node(node, sympy, symbols):
    """Recursively turn a whitelisted AST node into a SymPy object."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            raise SymbolicError("booleans are not valid input")
        if isinstance(value, (int, float)):
            return sympy.nsimplify(value, rational=True) if isinstance(value, float) else sympy.Integer(value)
        if isinstance(value, str):
            if value in _ALLOWED_STRINGS:
                return value          # e.g. limit(..., dir='+')
            raise SymbolicError(f"string literals are not allowed: {value!r}")
        raise SymbolicError(f"unsupported constant: {value!r}")

    if isinstance(node, ast.Name):
        key = node.id
        if key in _CONSTANT_NAMES:
            return getattr(sympy, _CONSTANT_NAMES[key])
        if key in _ALLOWED_FUNCTIONS:            # bare function reference, e.g. diff(f, x)
            attr = getattr(sympy, key, None)
            if attr is not None:
                return attr
        # Anything else is a plain variable (x, n, theta…).
        return symbols.setdefault(key, sympy.Symbol(key))

    if isinstance(node, ast.BinOp):
        return _binop(sympy, node.op,
                      _eval_node(node.left, sympy, symbols),
                      _eval_node(node.right, sympy, symbols))

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, sympy, symbols)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise SymbolicError(f"unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        # A single comparison becomes a relation: x**2 == 4 -> Eq(x**2, 4),
        # x**2 > 4 -> StrictGreaterThan, so inequalities can be solved too.
        if len(node.ops) != 1:
            raise SymbolicError("only a single comparison is supported")
        relations = {
            ast.Eq: sympy.Eq, ast.NotEq: sympy.Ne,
            ast.Lt: sympy.Lt, ast.LtE: sympy.Le,
            ast.Gt: sympy.Gt, ast.GtE: sympy.Ge,
        }
        builder = relations.get(type(node.ops[0]))
        if builder is None:
            raise SymbolicError(f"unsupported comparison: {type(node.ops[0]).__name__}")
        return builder(_eval_node(node.left, sympy, symbols),
                       _eval_node(node.comparators[0], sympy, symbols))

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(el, sympy, symbols) for el in node.elts)

    if isinstance(node, ast.List):
        return [_eval_node(el, sympy, symbols) for el in node.elts]

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise SymbolicError("only direct calls to known functions are allowed")
        fname = node.func.id
        if fname not in _ALLOWED_FUNCTIONS:
            # A short, single-letter-ish name is the standard notation for the
            # UNKNOWN function of a differential equation — f(x), y(t), g(x) —
            # so it becomes an undefined SymPy Function instead of an error.
            # Longer names stay rejected, which still catches hallucinated
            # calls like hack(x) or eval(...).
            if len(fname) <= 2 and fname.isalpha():
                undefined = symbols.setdefault(f"__fn_{fname}", sympy.Function(fname))
                return undefined(*[_eval_node(a, sympy, symbols) for a in node.args])
            raise SymbolicError(
                f"unknown function '{fname}'. Allowed: {', '.join(sorted(_ALLOWED_FUNCTIONS))}")
        func = getattr(sympy, "log" if fname == "ln" else fname, None)
        if func is None:
            raise SymbolicError(f"function '{fname}' is unavailable in this sympy build")
        args = [_eval_node(a, sympy, symbols) for a in node.args]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise SymbolicError("**kwargs are not allowed")
            kwargs[kw.arg] = _eval_node(kw.value, sympy, symbols)
        return func(*args, **kwargs)

    raise SymbolicError(f"unsupported expression element: {type(node).__name__}")


def _format(sympy, result) -> str:
    """Readable answer, with a decimal approximation only when it adds
    information — i.e. for irrational values (sqrt(2), pi, E). Integers and
    rationals are already exact and readable, so "9 ≈ 9.000000000" is noise."""
    text = sympy.sstr(result)
    try:
        if (isinstance(result, sympy.Expr) and not result.free_symbols
                and result.is_number and result.is_rational is False
                and result.is_finite is not False):
            text += f"  ≈ {sympy.N(result, 10)}"
    except Exception:
        pass
    return text


def evaluate_symbolic(expression: str) -> str:
    """Evaluate a symbolic expression. Raises SymbolicError on invalid input."""
    try:
        import sympy
    except ImportError:
        raise SymbolicError(
            "symbolic math needs the 'sympy' package (pip install sympy)")

    normalized = (expression or "").strip().replace("^", "**")
    if not normalized:
        raise SymbolicError("expression is empty")

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as e:
        raise SymbolicError(f"could not parse the expression: {e}")

    result = _eval_node(tree.body, sympy, {})

    # A BARE expression is more useful reduced: with free symbols it gets
    # simplified, without them it collapses to a value ((1 + I)**8 -> 16).
    # An explicit call is never second-guessed: factor(x**2 - 1) must stay
    # factored even though x**2 - 1 is "simpler" by operation count.
    try:
        if not isinstance(tree.body, ast.Call) and isinstance(result, sympy.Expr):
            if result.free_symbols:
                simplified = sympy.simplify(result)
                if sympy.count_ops(simplified) <= sympy.count_ops(result):
                    result = simplified
            else:
                result = sympy.simplify(result)
    except Exception:
        pass

    return _format(sympy, result)
