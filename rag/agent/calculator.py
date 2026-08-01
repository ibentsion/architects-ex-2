"""Safe arithmetic evaluator — the agent's ``calculate`` tool.

Numbers the user asks about are NEVER computed by an LLM; the orchestrator
LLM only writes an expression, and this AST-whitelist evaluator computes it.
Supported: int/float literals, + - * / // % **, unary ±, parentheses, and
round/abs/min/max. Everything else (names, attributes, calls beyond the
whitelist, subscripts, comprehensions...) raises CalculationError.
"""
from __future__ import annotations

import ast
import operator

MAX_EXPRESSION_LENGTH = 500
MAX_POWER_EXPONENT = 100  # blocks 9**9**9-style CPU/memory bombs

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {"round": round, "abs": abs, "min": min, "max": max}


class CalculationError(ValueError):
    """Invalid or unsafe expression; message is safe to feed back to the LLM."""


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression string to a float."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculationError(f"expression longer than {MAX_EXPRESSION_LENGTH} chars")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"not a valid expression: {exc.msg}") from None
    try:
        result = _eval(tree.body)
    except ZeroDivisionError:
        raise CalculationError("division by zero") from None
    if not isinstance(result, (int, float)) or isinstance(result, bool):
        raise CalculationError(f"expression did not evaluate to a number: {result!r}")
    return float(result)


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalculationError(f"only numeric literals allowed, got {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER_EXPONENT:
            raise CalculationError(f"exponent magnitude capped at {MAX_POWER_EXPONENT}")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args))
        raise CalculationError("only round/abs/min/max calls are allowed")
    raise CalculationError(f"unsupported syntax: {type(node).__name__}")
