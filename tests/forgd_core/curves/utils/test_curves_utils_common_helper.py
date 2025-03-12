import pytest

from decimal import Decimal

from forgd_core.common.enums import BondingCurveDistribution
from forgd_core.common.math import decimal_approx_equal
from forgd_core.curves.utils.common_curve_helper import CommonCurveHelper


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
    result = CommonCurveHelper.partial_fill(user_requested, available, approach)
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
    result_value, revert_flag = CommonCurveHelper.scale_for_slippage(
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
    result = CommonCurveHelper.apply_risk_profile(raw_value, risk_profile, is_buy)
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
    result = CommonCurveHelper.apply_transaction_fee(raw_value, fee_rate, is_buy)

    assert decimal_approx_equal(result, expected), (
        f"Expected {expected} for raw_value={raw_value}, fee_rate={fee_rate}, is_buy={is_buy}, got {result}"
    )

