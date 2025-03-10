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


@pytest.mark.parametrize(
    "user_requested, available, approach, expected",
    [
        # 1) user_requested < available => no partial fill needed
        (Decimal("10"), Decimal("20"), 1, Decimal("10")),
        (Decimal("10"), Decimal("20"), 2, Decimal("10")),
        (Decimal("10"), Decimal("20"), 999, Decimal("10")),

        # 2) user_requested == available => returns user_requested
        (Decimal("20"), Decimal("20"), 1, Decimal("20")),
        (Decimal("20"), Decimal("20"), 2, Decimal("20")),
        (Decimal("20"), Decimal("20"), 3, Decimal("20")),

        # 3) user_requested > available => approach=1 => hard cap
        (Decimal("50"), Decimal("30"), 1, Decimal("30")),

        # 4) user_requested > available => approach=2 => ratio-based => effectively available
        #    fraction = 30/50=0.6 => final=0.6*50=30
        (Decimal("50"), Decimal("30"), 2, Decimal("30")),

        # 5) user_requested > available => unknown approach => fallback => available
        (Decimal("50"), Decimal("30"), 999, Decimal("30")),

        # 6) user_requested=0 => approach=2 => fraction=0 => partial=0
        (Decimal("0"), Decimal("30"), 2, Decimal("0")),
    ]
)
def test_partial_fill(user_requested, available, approach, expected):
    """
    Tests various scenarios for partial_fill()
    """
    result = LinearCurveHelper.partial_fill(user_requested, available, approach)
    assert decimal_approx_equal(result, expected), (
        f"Expected {expected} but got {result} "
        f"(user_requested={user_requested}, available={available}, approach={approach})"
    )


@pytest.mark.parametrize(
    "raw_value, expected_value, slip_tolerance, is_buy, approach, desc, expected_result, expected_revert",
    [
        # 1) Slippage tolerance <= 0 => no slippage logic, so raw returned as is, revert=False
        (Decimal("100"), Decimal("95"), Decimal("0"), True, 1, "zero slip (buy)", Decimal("100"), False),
        (Decimal("50"), Decimal("100"), Decimal("-0.1"), False, 2, "negative slip (sell)", Decimal("50"), False),

        # -------------------------
        # BUY scenarios
        # If raw_cost <= expected*(1+slip), no changes
        # If raw_cost > threshold => approach logic
        # threshold = expected_value*(1+slip)

        # 2) Buy within tolerance (raw=100, exp=95, slip=0.1 => threshold=95*(1+0.1)=104.5 => raw=100 <=104.5 => no changes
        (Decimal("100"), Decimal("95"), Decimal("0.1"), True, 1, "buy within tolerance", Decimal("100"), False),

        # 3) Buy exactly threshold => still no changes (raw=104.5, threshold=104.5)
        (Decimal("104.5"), Decimal("95"), Decimal("0.1"), True, 1, "buy exactly threshold", Decimal("104.5"), False),

        # 4) Buy above threshold => approach=1 => partial fill
        #    fraction= threshold/raw= (95*(1+0.1))/110= 104.5/110= ~0.95 => new_value= 110*0.95= ~104.5
        (Decimal("110"), Decimal("95"), Decimal("0.1"), True, 1, "buy above threshold approach=1", Decimal("104.5"), False),

        # 5) Buy above threshold => approach=2 => revert => (0, True)
        (Decimal("110"), Decimal("95"), Decimal("0.1"), True, 2, "buy above threshold approach=2 revert", Decimal("0"), True),

        # 6) Buy above threshold => unknown approach => no partial fill / no revert => raw_value
        (Decimal("110"), Decimal("95"), Decimal("0.1"), True, 999, "buy above threshold unknown approach", Decimal("110"), False),

        # -------------------------
        # SELL scenarios
        # If raw_return >= threshold => no changes
        # threshold= expected_value*(1-slip)
        # If raw_return < threshold => approach logic

        # 7) Sell within tolerance => raw=80, expected=100 => slip=0.2 => threshold=100*(1-0.2)=80 => raw=80 => no changes
        (Decimal("80"), Decimal("100"), Decimal("0.2"), False, 1, "sell within tolerance", Decimal("80"), False),

        # 8) Sell exactly threshold => raw=80 => threshold=80 => no changes
        (Decimal("80"), Decimal("100"), Decimal("0.2"), False, 2, "sell exactly threshold", Decimal("80"), False),

        # 9) Sell above threshold => raw=85 => threshold=80 => raw>=80 => no changes
        (Decimal("85"), Decimal("100"), Decimal("0.2"), False, 1, "sell above threshold => no partial fill", Decimal("85"), False),

        # 10) Sell below threshold => approach=1 => partial => fraction= threshold/raw => new_val= raw*fraction
        #     raw=70 => threshold=80 => fraction=80/70=1.142857 => new_value= ~80
        #     Because raw< threshold, fraction>1 => new_value= raw*(>1)= bigger than raw
        #     This scenario can be weird if partial fill is supposed to reduce the final. But the code is symmetrical:
        #     new_value= 70*1.142857 ~80 => revert=False
        (Decimal("70"), Decimal("100"), Decimal("0.2"), False, 1, "sell below threshold approach=1 partial fill", Decimal("80"), False),

        # 11) Sell below threshold => approach=2 => revert => (0, True)
        (Decimal("70"), Decimal("100"), Decimal("0.2"), False, 2, "sell below threshold approach=2 revert", Decimal("0"), True),

        # 12) Sell below threshold => unknown approach => no partial fill, no revert => raw_value
        (Decimal("70"), Decimal("100"), Decimal("0.2"), False, 999, "sell below threshold unknown approach", Decimal("70"), False),

        # 13) Edge case => raw_value=0 => approach=1 => fraction= threshold/0 => guard => fraction=0 => new_val=0
        (Decimal("0"), Decimal("100"), Decimal("0.2"), True, 1, "edge raw=0 buy => partial fill => 0", Decimal("0"), False),
        (Decimal("0"), Decimal("100"), Decimal("0.2"), False, 1, "edge raw=0 sell => partial fill => 0", Decimal("0"), False),
    ]
)
def test_scale_for_slippage(raw_value,
                            expected_value,
                            slip_tolerance,
                            is_buy,
                            approach,
                            desc,
                            expected_result,
                            expected_revert,
                            ):
    result_value, revert_flag = LinearCurveHelper.scale_for_slippage(
        raw_cost_or_return=raw_value,
        expected_cost_or_return=expected_value,
        slippage_tolerance=slip_tolerance,
        is_buy=is_buy,
        approach=approach,
    )

    # Check revert first
    assert revert_flag == expected_revert, (
        f"{desc} => Expected revert_flag={expected_revert}, got {revert_flag}"
    )

    # Check final_value
    # These are mostly exact calculations or zero.
    # However, partial fill might produce a repeating decimal
    assert decimal_approx_equal(result_value, expected_result), (
        f"{desc} => Expected final_value={expected_result}, got {result_value}"
    )


@pytest.mark.parametrize(
    "raw_value, risk_profile, is_buy, expected",
    [
        # 1) No risk profile -> no change
        (Decimal("100"), None, True, Decimal("100")),
        (Decimal("200"), None, False, Decimal("200")),

        # 2) Conservative
        #    Buy => raw * 1.20
        (Decimal("100"), BondingCurveDistribution.CONSERVATIVE, True, Decimal("120")),
        #    Sell => raw * 0.85
        (Decimal("100"), BondingCurveDistribution.CONSERVATIVE, False, Decimal("85")),

        # 3) Moderate
        #    Buy => raw * 1.10
        (Decimal("100"), BondingCurveDistribution.MODERATE, True, Decimal("110")),
        #    Sell => raw * 0.90
        (Decimal("100"), BondingCurveDistribution.MODERATE, False, Decimal("90")),

        # 4) Aggressive
        #    Buy => raw * 0.95
        (Decimal("100"), BondingCurveDistribution.AGGRESSIVE, True, Decimal("95")),
        #    Sell => raw * 1.01
        (Decimal("100"), BondingCurveDistribution.AGGRESSIVE, False, Decimal("101")),

        # Additional checks with different raw_value
        (Decimal("250.5"), BondingCurveDistribution.CONSERVATIVE, True, Decimal("300.60")),  # 250.5 * 1.20
        (Decimal("250.5"), BondingCurveDistribution.MODERATE, True, Decimal("275.55")),     # 250.5 * 1.10
        (Decimal("250.5"), BondingCurveDistribution.AGGRESSIVE, True, Decimal("237.975")),  # 250.5 * 0.95

        # Edge scenario: low raw_value, sell
        (Decimal("10"), BondingCurveDistribution.CONSERVATIVE, False, Decimal("8.5")),    # 10 * 0.85
        (Decimal("10"), BondingCurveDistribution.MODERATE, False, Decimal("9.0")),       # 10 * 0.9
        (Decimal("10"), BondingCurveDistribution.AGGRESSIVE, False, Decimal("10.1")),    # 10 * 1.01
    ]
)
def test_apply_risk_profile(raw_value, risk_profile, is_buy, expected):
    result = LinearCurveHelper.apply_risk_profile(raw_value, risk_profile, is_buy)
    assert decimal_approx_equal(result, expected), (
        f"Expected {expected} for raw_value={raw_value},"
        f" risk_profile={risk_profile}, is_buy={is_buy}, got {result}"
    )


@pytest.mark.parametrize(
    "raw_value, fee_rate, is_buy, expected",
    [
        # 1) fee_rate <= 0 => no changes
        (Decimal("100"), Decimal("0"), True, Decimal("100")),
        (Decimal("100"), Decimal("-0.05"), False, Decimal("100")),

        # 2) buy => user pays raw_value + fee
        #    fee_rate=0.05 => 5%
        (Decimal("100"), Decimal("0.05"), True, Decimal("105")),
        (Decimal("200.50"), Decimal("0.01"), True, Decimal("202.505")),

        # 3) sell => user receives raw_value - fee
        #    fee_rate=0.05 => 5%
        (Decimal("100"), Decimal("0.05"), False, Decimal("95")),
        (Decimal("200.50"), Decimal("0.01"), False, Decimal("198.495")),

        # Additional edge case: very small or large raw values
        (Decimal("0.00"), Decimal("0.10"), True, Decimal("0")),  # 10% of 0 => 0
        (Decimal("1000.1234"), Decimal("0.001"), False, Decimal("999.1232766")),  # ~ 1.0001234 fee
    ]
)
def test_apply_transaction_fee(raw_value, fee_rate, is_buy, expected):
    result = LinearCurveHelper.apply_transaction_fee(raw_value, fee_rate, is_buy)

    assert decimal_approx_equal(result, expected), (
        f"Expected {expected} for raw_value={raw_value}, fee_rate={fee_rate}, is_buy={is_buy}, got {result}"
    )
