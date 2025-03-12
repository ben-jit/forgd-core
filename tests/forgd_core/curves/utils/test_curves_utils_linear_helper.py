import pytest

from datetime import datetime, timedelta
from decimal import Decimal

from forgd_core.common.enums import BondingCurveDistribution
from forgd_core.common.math import decimal_approx_equal
from forgd_core.curves.utils.linear_curve_helper import LinearCurveHelper


@pytest.mark.parametrize(
    "start, end, i, m, expected",
    [
        # 1) Simple case: zero slope (m=0), intercept i=1
        #    cost = i*(end - start) = 1*(1 - 0) = 1
        (Decimal("0"), Decimal("1"), Decimal("1"), Decimal("0"), Decimal("1")),

        # 2) Non-zero slope (m=1), intercept i=1
        #    cost = i*(4-2) + 0.5*(4^2 - 2^2)
        #          = 1*2 + 0.5*(16 - 4)
        #          = 2 + 0.5*12
        #          = 2 + 6 = 8
        (Decimal("2"), Decimal("4"), Decimal("1"), Decimal("1"), Decimal("8")),

        # 3) start == end => no distance to integrate => cost should be 0
        (Decimal("5"), Decimal("5"), Decimal("2"), Decimal("3"), Decimal("0")),

        # 4) Larger decimals
        #    cost = i*(end - start) + (m/2)*(end^2 - start^2)
        #    Choose i=2.5, m=0.75, start=10, end=10.5
        #    difference = 10.5 - 10 = 0.5
        #    i_term = 2.5 * 0.5 = 1.25
        #    end^2=110.25, start^2 = 100 => difference=10.25
        #    slope_term = 0.75/2 * 10.25 = 0.375 * 10.25 = 3.84375
        #    total= 1.25 + 3.84375 = 5.09375
        (Decimal("10"), Decimal("10.5"), Decimal("2.5"), Decimal("0.75"), Decimal("5.09375")),

        # 5) Negative intercept or slope,
        #    not necessarily expected in typical usage, but it shows the formula still works mathematically.
        #    start=0, end=2, i=-1, m=-0.5
        #    i_term = -1*(2 - 0)= -2
        #    end^2-start^2=4 - 0=4 => slope_term = -0.5/2 * 4= -0.25*4=-1
        #    total = -2 + -1 = -3
        (Decimal("0"), Decimal("2"), Decimal("-1"), Decimal("-0.5"), Decimal("-3")),
    ],
)
def test_cost_between(start, end, i, m, expected):
    """
    Test various scenarios for the linear cost_between function
    """
    result = LinearCurveHelper.cost_between(start, end, i, m)
    assert decimal_approx_equal(result, expected)


def test_apply_time_decay_approach_1_no_decay_rate(monkeypatch):
    """
    If time_decay_rate = 0, no change should be applied regardless of how many days have passed.
    """
    fixed_now = datetime(2025, 3, 10, 12, 0, 0)

    # Mock datetime.now() to return fixed_now
    class MockDateTime:
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr("datetime.datetime", MockDateTime)

    # last_timestamp is 2 days before fixed_now
    last_timestamp = fixed_now - timedelta(days=2)

    current_i = Decimal("1.0")
    current_m = Decimal("0.5")
    time_decay_rate = Decimal("0.0")  # 0 => no decay

    new_i, new_m, new_timestamp = LinearCurveHelper.apply_time_decay(
        current_i, current_m, last_timestamp, time_decay_rate, approach=1
    )

    # Expect no changes
    assert new_i == current_i
    assert new_m == current_m
    assert new_timestamp == last_timestamp, "Timestamp should remain unchanged if no decay is applied."


def test_apply_time_decay_approach_1_positive(monkeypatch):
    """
    If time_decay_rate > 0, approach = 1 => new_m = current_m + (elapsed_days * decay_rate).
    """
    fixed_now = datetime(2025, 3, 10, 12, 0, 0)

    class MockDateTime:
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr("forgd_core.curves.utils.linear_curve_helper.datetime", MockDateTime)

    last_timestamp = fixed_now - timedelta(days=2)  # 2 days earlier
    elapsed_days = Decimal("2")

    current_i = Decimal("1.0")
    current_m = Decimal("0.5")
    time_decay_rate = Decimal("0.1")

    new_i, new_m, new_timestamp = LinearCurveHelper.apply_time_decay(
        current_i, current_m, last_timestamp, time_decay_rate, approach=1
    )

    expected_m = current_m + elapsed_days * time_decay_rate  # 0.5 + (2 * 0.1) = 0.7
    assert new_i == current_i, "Approach=1 does not change the intercept."
    assert decimal_approx_equal(new_m, expected_m, Decimal("0.0000001")), (
        f"Expected slope to be {expected_m} but got {new_m}"
    )
    assert new_timestamp == fixed_now, "Timestamp should be updated to 'now'."


def test_apply_time_decay_approach_2_positive(monkeypatch):
    """
    If time_decay_rate > 0, approach=2 => new_m = current_m * (1 + (elapsed_days * decay_rate)).
    """
    fixed_now = datetime(2025, 3, 10, 12, 0, 0)

    class MockDateTime:
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr("forgd_core.curves.utils.linear_curve_helper.datetime", MockDateTime)

    last_timestamp = fixed_now - timedelta(days=3)  # 3 days earlier
    elapsed_days = Decimal("3")

    current_i = Decimal("2.0")
    current_m = Decimal("1.0")
    time_decay_rate = Decimal("0.1")

    new_i, new_m, new_timestamp = LinearCurveHelper.apply_time_decay(
        current_i, current_m, last_timestamp, time_decay_rate, approach=2
    )

    # new_m = 1.0 * (1 + (3 * 0.1)) = 1.3
    expected_m = current_m * (Decimal("1") + elapsed_days * time_decay_rate)
    assert new_i == current_i, "Approach=2 does not change the intercept."
    assert decimal_approx_equal(new_m, expected_m, Decimal("0.0000001")), (
        f"Expected slope to be {expected_m} but got {new_m}"
    )
    assert new_timestamp == fixed_now, "Timestamp should be updated to 'now'."


def test_apply_time_decay_approach_default(monkeypatch):
    """
    If approach is not recognized or default, there's no slope/intercept change
    but the timestamp *does* update if time_decay_rate != 0 and time has passed.
    """
    fixed_now = datetime(2025, 3, 10, 12, 0, 0)

    class MockDateTime:
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr("forgd_core.curves.utils.linear_curve_helper.datetime", MockDateTime)

    last_timestamp = fixed_now - timedelta(days=2)

    current_i = Decimal("1.0")
    current_m = Decimal("1.0")
    time_decay_rate = Decimal("0.2")

    new_i, new_m, new_timestamp = LinearCurveHelper.apply_time_decay(
        current_i, current_m, last_timestamp, time_decay_rate, approach=999
    )

    # Because approach is default triggers the else-branch => no i/m change
    assert new_i == current_i
    assert new_m == current_m
    # But time_decay_rate != 0 and elapsed_days>0 => new_timestamp = now
    assert new_timestamp == fixed_now


def test_apply_time_decay_zero_elapsed(monkeypatch):
    """
    If there's no elapsed time (last_timestamp == now), the function should not update slope or intercept.
    """
    fixed_now = datetime(2025, 3, 10, 12, 0, 0)

    class MockDateTime:
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr("forgd_core.curves.utils.linear_curve_helper.datetime", MockDateTime)

    # No time has passed
    last_timestamp = fixed_now

    current_i = Decimal("1.0")
    current_m = Decimal("1.0")
    time_decay_rate = Decimal("0.2")

    new_i, new_m, new_timestamp = LinearCurveHelper.apply_time_decay(
        current_i, current_m, last_timestamp, time_decay_rate, approach=1
    )

    assert new_i == current_i, "No change if elapsed time = 0."
    assert new_m == current_m, "No change if elapsed time = 0."
    assert new_timestamp == last_timestamp, "Timestamp should remain the same if no time passed."


def test_apply_time_decay_negative_rate(monkeypatch):
    """
    Demonstrate that a negative time_decay_rate reduces the slope for approach=1 or approach=2.
    """
    fixed_now = datetime(2025, 3, 10, 12, 0, 0)

    class MockDateTime:
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr("forgd_core.curves.utils.linear_curve_helper.datetime", MockDateTime)

    last_timestamp = fixed_now - timedelta(days=1)
    elapsed_days = Decimal("1")

    current_i = Decimal("1.0")
    current_m = Decimal("1.0")
    time_decay_rate = Decimal("-0.1")  # negative

    # Approach=1: new_m = current_m + (elapsed_days * decay_rate) = 1.0 + (1 * -0.1) = 0.9
    new_i, new_m, new_timestamp = LinearCurveHelper.apply_time_decay(
        current_i, current_m, last_timestamp, time_decay_rate, approach=1
    )
    expected_m = Decimal("0.9")
    assert decimal_approx_equal(new_m, expected_m, Decimal("0.0000001"))
    assert new_timestamp == fixed_now

    # Approach=2: new_m = current_m * (1 + (elapsed_days * decay_rate))
    #           = 1.0 * (1 + 1 * -0.1) = 0.9
    new_i2, new_m2, new_timestamp2 = LinearCurveHelper.apply_time_decay(
        current_i, current_m, last_timestamp, time_decay_rate, approach=2
    )
    expected_m2 = current_m * (Decimal("1") + elapsed_days * time_decay_rate)  # 0.9
    assert decimal_approx_equal(new_m2, expected_m2, Decimal("0.0000001"))
    assert new_timestamp2 == fixed_now

    # Intercept remains unchanged
    assert new_i == current_i
    assert new_i2 == current_i
