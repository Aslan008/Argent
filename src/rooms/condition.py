"""Safe condition DSL for edge transitions.

Edge conditions are expressions over run state, e.g. ``state.tests_passed``,
``state.retries < 3``, ``state.error_type in ["timeout", "network"]``. They are
NEVER evaluated with eval(): a whitelisted AST walker allows only comparisons,
boolean logic, ``not``, literals, lists, and ``state.<field>`` reads. Anything
else — function calls, attribute chains, subscripts, arbitrary names — is
rejected, so a condition can neither run code nor reach outside the declared
state. ``validate`` checks a condition statically (for the RoomValidator);
``evaluate`` runs it against a state's data.
"""

import ast
import operator

_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


class ConditionError(ValueError):
    """A condition string is malformed or uses a forbidden construct."""


def _read_state_field(node: ast.Attribute) -> str:
    if isinstance(node.value, ast.Name) and node.value.id == "state":
        return node.attr
    raise ConditionError("only 'state.<field>' attribute access is allowed")


def _eval(node: ast.AST, data: dict):
    if isinstance(node, ast.Expression):
        return _eval(node.body, data)
    if isinstance(node, ast.BoolOp):
        vals = (_eval(v, data) for v in node.values)
        if isinstance(node.op, ast.And):
            return all(vals)
        return any(vals)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval(node.operand, data)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, data)
        for op, comparator in zip(node.ops, node.comparators):
            fn = _COMPARE_OPS.get(type(op))
            if fn is None:
                raise ConditionError(f"comparison {type(op).__name__} not allowed")
            right = _eval(comparator, data)
            if not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, data) for e in node.elts]
    if isinstance(node, ast.Attribute):
        return data.get(_read_state_field(node))
    raise ConditionError(f"{type(node).__name__} is not allowed in a condition")


def _check(node: ast.AST) -> None:
    """Static walk: raise ConditionError on any disallowed construct."""
    if isinstance(node, ast.Expression):
        return _check(node.body)
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            _check(v)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.Not):
            raise ConditionError(f"unary {type(node.op).__name__} not allowed")
        return _check(node.operand)
    if isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _COMPARE_OPS:
                raise ConditionError(f"comparison {type(op).__name__} not allowed")
        _check(node.left)
        for c in node.comparators:
            _check(c)
        return
    if isinstance(node, ast.Constant):
        return
    if isinstance(node, (ast.List, ast.Tuple)):
        for e in node.elts:
            _check(e)
        return
    if isinstance(node, ast.Attribute):
        _read_state_field(node)  # raises if not state.<field>
        return
    raise ConditionError(f"{type(node).__name__} is not allowed in a condition")


def validate(expr: str) -> None:
    """Raise ConditionError if `expr` is not a well-formed, in-grammar condition."""
    if not expr or not expr.strip():
        raise ConditionError("empty condition")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ConditionError(f"syntax error: {e.msg}")
    _check(tree)


def evaluate(expr: str, state_data: dict) -> bool:
    """Evaluate `expr` against `state_data`, returning a bool."""
    validate(expr)
    return bool(_eval(ast.parse(expr, mode="eval"), state_data or {}))
