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
import re

# Names that live in a sympy SUBMODULE rather than the top level. They are
# imported on demand so a heavy namespace (sympy.stats) costs nothing unless
# the expression actually uses it.
_SUBMODULE_NAMES = {
    # probability & statistics
    "Normal": "sympy.stats", "LogNormal": "sympy.stats", "Poisson": "sympy.stats",
    "Binomial": "sympy.stats", "Bernoulli": "sympy.stats", "Uniform": "sympy.stats",
    "Exponential": "sympy.stats", "Gamma": "sympy.stats", "Beta": "sympy.stats",
    "ChiSquared": "sympy.stats", "StudentT": "sympy.stats", "Geometric": "sympy.stats",
    "Die": "sympy.stats", "Coin": "sympy.stats", "DiscreteUniform": "sympy.stats",
    "density": "sympy.stats", "variance": "sympy.stats", "covariance": "sympy.stats",
    "std": "sympy.stats", "cdf": "sympy.stats", "skewness": "sympy.stats",
    "quantile": "sympy.stats", "correlation": "sympy.stats", "median": "sympy.stats",
    "expectation": "sympy.stats", "probability": "sympy.stats",
    # units
    "convert_to": "sympy.physics.units",
}

# Unit symbols, resolved from sympy.physics.units on demand.
_UNIT_NAMES = {
    "meter", "meters", "metre", "second", "seconds", "kilogram", "kilograms",
    "gram", "grams", "foot", "feet", "inch", "inches", "mile", "miles",
    "hour", "hours", "minute", "minutes", "day", "days", "year", "years",
    "pound", "pounds", "kelvin", "joule", "joules", "watt", "watts",
    "newton", "newtons", "liter", "liters", "litre", "km", "cm", "mm",
    "speed_of_light", "gravitational_constant",
}

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
    "nextprime", "prevprime", "primepi", "fibonacci", "lucas", "catalan",
    "harmonic", "bernoulli", "euler", "subfactorial", "continued_fraction",
    "npartitions", "multiplicity", "perfect_power", "integer_nthroot", "Poly",
    # special functions
    "gamma", "beta", "erf", "erfc", "zeta", "polygamma", "digamma", "LambertW",
    "besselj", "bessely", "airyai", "hyper", "meijerg",
    # integral transforms
    "laplace_transform", "inverse_laplace_transform", "fourier_transform",
    "inverse_fourier_transform", "mellin_transform", "fft", "ifft",
    # geometry
    "Point", "Point2D", "Point3D", "Line", "Segment", "Ray", "Circle", "Ellipse",
    "Polygon", "Triangle", "RegularPolygon", "intersection",
    # logic
    "And", "Or", "Not", "Xor", "Implies", "Equivalent", "satisfiable",
    "simplify_logic", "to_cnf", "to_dnf",
    # sets & piecewise
    "Interval", "FiniteSet", "Union", "Intersection", "Complement", "ProductSet",
    "EmptySet", "Range", "imageset", "Piecewise",
    # rewriting helpers
    "trigsimp", "powsimp", "radsimp", "logcombine", "ratsimp",
    "expand_trig", "expand_log", "expand_func",
} | set(_SUBMODULE_NAMES)

# Methods callable on a SymPy object, e.g. expr.subs(x, 2) or M.eigenvals().
# Attribute access is otherwise forbidden — it is the classic sandbox escape
# (__class__ -> __globals__). Safety rests on two checks that must BOTH hold:
# the name appears in this explicit list (so no dunder can ever be named), and
# the receiver is a SymPy object (so str/list/module internals are unreachable).
_ALLOWED_METHODS = {
    "subs", "doit", "evalf", "n", "simplify", "expand", "factor", "cancel",
    "apart", "together", "trigsimp", "radsimp", "powsimp", "nsimplify",
    "diff", "integrate", "limit", "series", "rewrite", "coeff", "as_poly",
    "det", "inv", "rank", "rref", "eigenvals", "eigenvects", "norm",
    "transpose", "adjugate", "nullspace", "columnspace", "row", "col",
    "conjugate", "equals", "round", "as_real_imag", "expand_trig",
}

# Attributes readable without a call: M.T (transpose), eq.lhs / eq.rhs.
_ALLOWED_PROPERTIES = {
    "T", "lhs", "rhs", "free_symbols", "args", "shape", "is_number",
    # geometry: Circle(...).area, Segment(...).length, Triangle(...).angles
    "area", "perimeter", "circumference", "radius", "center", "length",
    "vertices", "sides", "angles", "midpoint", "slope", "equation", "bounds",
    "incenter", "circumcenter", "centroid", "start", "end", "real", "imag",
}

# Names that resolve to sympy constants rather than free symbols.
_CONSTANT_NAMES = {
    "pi": "pi", "E": "E", "e": "E", "I": "I", "oo": "oo", "inf": "oo",
    "infinity": "oo", "zoo": "zoo", "nan": "nan", "true": "true", "false": "false",
}

# String literals are restricted to two harmless shapes: a limit direction and
# an identifier-like name (random variables are named: Normal("X", 0, 1)).
# Anything else is rejected so a string can never be smuggled into a SymPy
# parser that would eval it.
_ALLOWED_STRINGS = {"+", "-", "+-", "-+"}
_NAME_STRING_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,15}$")

# Guard against expression bombs like 9**9**9 that would hang the process.
_MAX_EXPONENT = 10000
_MAX_BASE_FOR_BIG_EXP = 1e6


class SymbolicError(ValueError):
    """Invalid or disallowed symbolic expression."""


def _resolve_name(sympy, name):
    """Find a whitelisted name in sympy, importing its submodule on demand."""
    if name in _SUBMODULE_NAMES:
        import importlib
        module = importlib.import_module(_SUBMODULE_NAMES[name])
        # sympy.stats spells expectation/probability as E and P.
        alias = {"expectation": "E", "probability": "P"}.get(name, name)
        return getattr(module, alias, None)
    if name in _UNIT_NAMES:
        import importlib
        units = importlib.import_module("sympy.physics.units")
        return getattr(units, name, None)
    return getattr(sympy, "log" if name == "ln" else name, None)


def _is_sympy_object(sympy, obj) -> bool:
    """Only SymPy values may expose attributes to the expression language."""
    from sympy.matrices.matrixbase import MatrixBase
    return isinstance(obj, (sympy.Basic, MatrixBase, tuple, list))


def _binop(sympy, op, left, right):
    # A string is only ever a NAME passed to a function (a limit direction, a
    # random-variable label). Letting one into arithmetic would make 'a' * 3
    # mean Python string repetition instead of an error.
    if isinstance(left, str) or isinstance(right, str):
        raise SymbolicError("string literals cannot be used in arithmetic")
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
    # SymPy overloads &, | and ^ for logic and set algebra:
    # (x > 0) & (x < 5), Interval(0, 2) | Interval(3, 4).
    if isinstance(op, ast.BitAnd):
        return left & right
    if isinstance(op, ast.BitOr):
        return left | right
    if isinstance(op, ast.BitXor):
        return left ^ right
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
            # Piecewise's catch-all condition is literally True.
            return sympy.true if value else sympy.false
        if isinstance(value, (int, float)):
            return sympy.nsimplify(value, rational=True) if isinstance(value, float) else sympy.Integer(value)
        if isinstance(value, str):
            if value in _ALLOWED_STRINGS or _NAME_STRING_RE.match(value):
                return value          # limit direction, or a random-variable name
            raise SymbolicError(
                f"string literals must be a name or a limit direction, got {value!r}")
        raise SymbolicError(f"unsupported constant: {value!r}")

    if isinstance(node, ast.Name):
        key = node.id
        if key in _CONSTANT_NAMES:
            return getattr(sympy, _CONSTANT_NAMES[key])
        if key in _UNIT_NAMES or key in _ALLOWED_FUNCTIONS:
            # A unit symbol (5*meter) or a bare function reference (diff(f, x)).
            attr = _resolve_name(sympy, key)
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
        if isinstance(operand, str):
            raise SymbolicError("string literals cannot be used in arithmetic")
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

    if isinstance(node, ast.Attribute):
        # Bare property read: M.T, equation.lhs. Same two-part guard as methods.
        if node.attr not in _ALLOWED_PROPERTIES:
            raise SymbolicError(
                f"attribute '{node.attr}' is not allowed. Readable: "
                f"{', '.join(sorted(_ALLOWED_PROPERTIES))}")
        target = _eval_node(node.value, sympy, symbols)
        if not _is_sympy_object(sympy, target):
            raise SymbolicError("attributes are only readable on symbolic values")
        return getattr(target, node.attr)

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            # Method call: expr.subs(x, 2), M.eigenvals(). Allowed only when the
            # NAME is whitelisted (no dunder can be named) AND the receiver is a
            # SymPy value (so str/list/module internals stay unreachable).
            method = node.func.attr
            if method not in _ALLOWED_METHODS:
                raise SymbolicError(
                    f"method '{method}' is not allowed. Callable: "
                    f"{', '.join(sorted(_ALLOWED_METHODS))}")
            target = _eval_node(node.func.value, sympy, symbols)
            if not _is_sympy_object(sympy, target):
                raise SymbolicError("methods are only callable on symbolic values")
            bound = getattr(target, method, None)
            if bound is None or not callable(bound):
                raise SymbolicError(f"'{method}' is not available on this value")
            args = [_eval_node(a, sympy, symbols) for a in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None:
                    raise SymbolicError("**kwargs are not allowed")
                kwargs[kw.arg] = _eval_node(kw.value, sympy, symbols)
            return bound(*args, **kwargs)

        if not isinstance(node.func, ast.Name):
            # Calling the RESULT of an expression, e.g. density(Normal(...))(x).
            # The callee was produced by this same whitelisted evaluator, so it
            # can only be a SymPy value — nothing external is reachable here.
            callee = _eval_node(node.func, sympy, symbols)
            if not callable(callee):
                raise SymbolicError("this value is not callable")
            return callee(*[_eval_node(a, sympy, symbols) for a in node.args])
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
        func = _resolve_name(sympy, fname)
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
    if isinstance(result, str):
        raise SymbolicError("a bare string is not a mathematical expression")

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
