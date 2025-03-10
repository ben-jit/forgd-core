from decimal import Decimal


def decimal_approx_equal(a: Decimal, b: Decimal, tol: Decimal = Decimal(10**14)) -> bool:
    return abs(a - b) < tol
