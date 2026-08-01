"""Agent harness tests: calculator safety, engine routing (fast paths vs
tool loop), concurrency plumbing, and degradation. All backends/LLMs faked.
"""
from __future__ import annotations

import pytest

from rag.agent.calculator import CalculationError, calculate


# --------------------------------------------------------------------------- #
# Calculator — tool computes, LLM never does arithmetic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("0.15 * 2340", 351.0),
        ("(1500 + 300) / 2", 900.0),
        ("2 ** 10", 1024.0),
        ("17 // 5", 3.0),
        ("17 % 5", 2.0),
        ("-5 + +3", -2.0),
        ("round(3.14159, 2)", 3.14),
        ("min(3, 1, 2)", 1.0),
        ("max(3, 1, 2)", 3.0),
        ("abs(-7.5)", 7.5),
    ],
)
def test_calculate_supported_arithmetic(expression, expected):
    assert calculate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('rm -rf /')",  # call of a name
        "().__class__",  # attribute access
        "x + 1",  # free variable
        "[1,2][0]",  # subscript/list
        "'a' * 3",  # string operand
        "1 if True else 2",  # conditional
        "lambda: 1",  # lambda
        "1; 2",  # not an expression
        "round(1.5, ndigits=0)",  # keyword args rejected
        "9 ** 999",  # exponent bomb
        "1 / 0",  # division by zero
        "True + 1",  # bool literal
    ],
)
def test_calculate_rejects_unsafe_or_invalid(expression):
    with pytest.raises(CalculationError):
        calculate(expression)


def test_calculate_rejects_overlong_expression():
    with pytest.raises(CalculationError, match="longer"):
        calculate("1+" * 300 + "1")
