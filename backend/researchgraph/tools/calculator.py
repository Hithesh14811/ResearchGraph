"""A safe arithmetic evaluator (no ``eval``): parses the expression into an AST and only
interprets numeric literals and a whitelist of operators/functions, with size bounds."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 64
MAX_MAGNITUDE = 1e15

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


class CalculatorError(ValueError):
    pass


def _check(value: float) -> float:
    if isinstance(value, complex) or not math.isfinite(value) or abs(value) > MAX_MAGNITUDE:
        raise CalculatorError("Result out of range")
    return value


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        return _check(float(node.value))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError("Exponent too large")
        try:
            return _check(_BIN_OPS[type(node.op)](left, right))
        except ZeroDivisionError as exc:
            raise CalculatorError("Division by zero") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _check(_UNARY_OPS[type(node.op)](_eval(node.operand)))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
        and len(node.args) <= 4
    ):
        try:
            return _check(float(_FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args))))
        except (ValueError, TypeError) as exc:
            raise CalculatorError(str(exc)) from exc
    raise CalculatorError(f"Unsupported expression element: {type(node).__name__}")


def evaluate_expression(expression: str) -> float:
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculatorError("Expression too long")
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError("Invalid expression") from exc
    return _eval(tree)
