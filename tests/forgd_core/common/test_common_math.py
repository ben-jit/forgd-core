import pytest

from decimal import Decimal

from forgd_core.common.math import decimal_approx_equal


@pytest.mark.parametrize(
    "a, b, tol, expected",
    [
        (Decimal("1.000"), Decimal("1.000"), Decimal("0.0001"), True),

        (Decimal("1.0000"), Decimal("1.0001"), Decimal("0.001"), True),

        (Decimal("1.0000"), Decimal("1.0010"), Decimal("0.001"), False),

        (Decimal("1.0000"), Decimal("1.0008"), Decimal("0.001"), True),

        (Decimal("2.000"), Decimal("2.020"), Decimal("0.001"), False),

        (Decimal("-1.000"), Decimal("-0.999"), Decimal("0.01"), True),

        (Decimal("0.000"), Decimal("0.000"), Decimal("0.000"), False),
    ]
)
def test_decimal_approx_equal(a, b, tol, expected):
    """
    Test the decimal_approx_equal function with various inputs
    """
    result = decimal_approx_equal(a, b, tol)
    assert result == expected, f"Expected {expected} for a={a}, b={b}, tol={tol}, got {result}"
