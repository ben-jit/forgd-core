import pytest

from datetime import datetime, timedelta
from decimal import Decimal

from forgd_core.common.math import decimal_approx_equal
from forgd_core.curves.helpers.exponential import ExponentialCurveHelper



@pytest.mark.parametrize(
    "p0, alpha",
    [
        (Decimal("0.0001"), Decimal("0")),     # alpha=0 => allowed
        (Decimal("1"), Decimal("1")),         # valid
        (Decimal("10"), Decimal("5.5")),      # larger valid values
    ]
)
def test_validate_exponential_params_valid(p0, alpha):
    """
    If p0>0 and alpha>=0, no error should be raised.
    """
    # Should not raise
    ExponentialCurveHelper.validate_exponential_params(p0, alpha)


@pytest.mark.parametrize(
    "p0, alpha, expected_error_msg",
    [
        (Decimal("0"), Decimal("0"), "Exponential curve requires p0 > 0."),
        (Decimal("-1"), Decimal("1"), "Exponential curve requires p0 > 0."),
        (Decimal("1"), Decimal("-0.0001"), "Negative alpha is not supported"),
    ]
)
def test_validate_exponential_params_invalid(p0, alpha, expected_error_msg):
    """
    If p0<=0 or alpha<0, a ValueError should be raised with the correct message.
    """
    with pytest.raises(ValueError, match=expected_error_msg):
        ExponentialCurveHelper.validate_exponential_params(p0, alpha)


def test_apply_time_decay_simple_no_rate_or_no_elapsed(monkeypatch):
    """
    If time_decay_rate<=0 or elapsed_days<=0 => no changes returned.
    We'll fix now==last_timestamp to produce elapsed_days=0.
    """
    fixed_now = datetime(2026, 1, 1, 12, 0, 0)

    class MockDateTime:
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr("forgd_core.curves.helpers.exponential.datetime", MockDateTime)

    # current_p0=1.0, alpha=0.5 => time_decay_rate=0 => no changes
    new_p0, new_alpha, new_ts = ExponentialCurveHelper.apply_time_decay_simple(
        current_p0=Decimal("1.0"),
        current_alpha=Decimal("0.5"),
        last_timestamp=fixed_now,  # => elapsed_days=0
        time_decay_rate=Decimal("0"),
        approach=1,
    )
    # no changes
    assert new_p0 == Decimal("1.0")
    assert new_alpha == Decimal("0.5")
    assert new_ts == fixed_now


def test_apply_time_decay_simple_positive_approach_1(monkeypatch):
    """
    If time_decay_rate>0 and approach=1 => alpha is multiplied by (1+ rate*elapsed_days).
    """
    old_ts = datetime(2026, 1, 1, 12, 0, 0)
    new_ts = old_ts + timedelta(days=2)  # => 2 days

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.helpers.exponential.datetime", MockDateTime)

    # time_decay_rate=0.1 => elapsed_days=2 => factor=0.2 => alpha *=1.2
    new_p0, new_alpha, updated_ts = ExponentialCurveHelper.apply_time_decay_simple(
        current_p0=Decimal("1.0"),
        current_alpha=Decimal("0.5"),
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0.1"),
        approach=1
    )
    assert updated_ts == new_ts

    # approach=1 => alpha => alpha=0.5*(1+0.2)=0.6 => p0 unchanged
    assert new_p0 == Decimal("1.0")
    # Because 0.5*(1.2)=0.60
    assert new_alpha == Decimal("0.60")


def test_apply_time_decay_simple_positive_approach_2(monkeypatch):
    """
    If time_decay_rate>0 and approach=2 => p0 is multiplied by (1+ rate*elapsed_days).
    """
    old_ts = datetime(2026, 2, 10, 10, 0, 0)
    new_ts = old_ts + timedelta(days=3)  # 3 days

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.helpers.exponential.datetime", MockDateTime)

    # time_decay_rate=0.05 => elapsed_days=3 => factor=0.15 => p0 *=1.15 => alpha unchanged
    new_p0, new_alpha, updated_ts = ExponentialCurveHelper.apply_time_decay_simple(
        current_p0=Decimal("2.0"),
        current_alpha=Decimal("1.0"),
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0.05"),
        approach=2
    )
    assert updated_ts == new_ts
    # p0 => 2.0*(1+0.15)=2.3 => alpha=1.0
    assert new_p0 == Decimal("2.30")
    assert new_alpha == Decimal("1.0")


def test_apply_time_decay_simple_unknown_approach(monkeypatch):
    """
    If approach !=1 or !=2 => code does no changes to p0 or alpha, but updates timestamp.
    """
    old_ts = datetime(2026, 3, 1, 12, 0, 0)
    new_ts = old_ts + timedelta(days=1)  # 1 day

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.helpers.exponential.datetime", MockDateTime)

    new_p0, new_alpha, updated_ts = ExponentialCurveHelper.apply_time_decay_simple(
        current_p0=Decimal("10"),
        current_alpha=Decimal("1"),
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0.1"),
        approach=999  # unknown => pass
    )
    # p0, alpha unchanged, but new_ts updated
    assert new_p0 == Decimal("10")
    assert new_alpha == Decimal("1")
    assert updated_ts == new_ts


def test_apply_time_decay_simple_negative_elapsed(monkeypatch):
    """
    If now < last_timestamp => negative elapsed => no changes.
    """
    old_ts = datetime(2026, 1, 5, 12, 0, 0)
    new_ts = old_ts - timedelta(days=2)  # 2 days earlier => negative elapsed

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.helpers.exponential.datetime", MockDateTime)

    new_p0, new_alpha, updated_ts = ExponentialCurveHelper.apply_time_decay_simple(
        current_p0=Decimal("5"),
        current_alpha=Decimal("2"),
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0.1"),
        approach=1
    )
    # no changes
    assert new_p0 == Decimal("5")
    assert new_alpha == Decimal("2")
    assert updated_ts == old_ts


def test_apply_time_decay_advanced_no_op(monkeypatch):
    """
    If time_decay_rate <= 0 or elapsed_days <= 0 => no changes to p0, alpha, or last_timestamp.
    We'll fix now == last_timestamp => elapsed=0 => no changes.
    """
    fixed_now = datetime(2030, 1, 1, 12, 0, 0)

    class MockDateTime:
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr("forgd_core.curves.helpers.exponential.datetime", MockDateTime)

    new_p0, new_alpha, new_ts = ExponentialCurveHelper.apply_time_decay_advanced(
        current_p0=Decimal("1"),
        current_alpha=Decimal("0.5"),
        last_timestamp=fixed_now,     # => 0 days
        time_decay_rate=Decimal("0"), # no rate => no changes
        approach=1
    )
    # No changes
    assert new_p0 == Decimal("1")
    assert new_alpha == Decimal("0.5")
    assert new_ts == fixed_now


def test_apply_time_decay_advanced_approach1(monkeypatch):
    """
    approach=1 => both alpha & p0 multiply by (1 + rate*elapsed).
    e.g. time_decay_rate=0.1 => 2 days => factor=0.2 => scale=1.2 => alpha= alpha*1.2 => p0= p0*1.2
    """
    old_ts = datetime(2030, 2, 1, 12, 0, 0)
    new_ts = old_ts + timedelta(days=2)

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.helpers.exponential.datetime", MockDateTime)

    # time_decay_rate=0.1 => 2 days => factor=0.2 => scale=1.2
    new_p0, new_alpha, updated_ts = ExponentialCurveHelper.apply_time_decay_advanced(
        current_p0=Decimal("2"),
        current_alpha=Decimal("1"),
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0.1"),
        approach=1
    )
    assert updated_ts == new_ts
    # p0= 2*(1.2)=2.4 => alpha=1*(1.2)=1.2
    assert new_p0 == Decimal("2.4")
    assert new_alpha == Decimal("1.2")


def test_apply_time_decay_advanced_approach2(monkeypatch):
    """
    approach=2 => alpha+= additional_factor, p0*= (1 + additional_factor/2).
    e.g. time_decay_rate=0.05 => 3 days => factor=0.15 => alpha= alpha+0.15 => p0= p0*(1+0.075).
    """
    old_ts = datetime(2030, 3, 10, 10, 0, 0)
    new_ts = old_ts + timedelta(days=3)

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.helpers.exponential.datetime", MockDateTime)

    new_p0, new_alpha, updated_ts = ExponentialCurveHelper.apply_time_decay_advanced(
        current_p0=Decimal("10"),
        current_alpha=Decimal("2"),
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0.05"),
        approach=2
    )
    assert updated_ts == new_ts
    # factor= 0.05*3=0.15 => alpha=2+0.15=2.15 => p0= 10*(1+0.15/2)= 10*(1+0.075)=10*1.075=10.75
    assert new_alpha == Decimal("2.15")
    assert new_p0 == Decimal("10.75")


def test_apply_time_decay_advanced_unknown_approach(monkeypatch):
    """
    If approach !=1 or 2 => the snippet doesn't define anything => so no code modifies p0/alpha,
    but we do update the timestamp to now.
    """
    old_ts = datetime(2030, 4, 1, 12, 0, 0)
    new_ts = old_ts + timedelta(days=1)

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.helpers.exponential.datetime", MockDateTime)

    new_p0, new_alpha, updated_ts = ExponentialCurveHelper.apply_time_decay_advanced(
        current_p0=Decimal("5"),
        current_alpha=Decimal("2"),
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0.1"),
        approach=999  # no change code
    )
    # p0/alpha unchanged, but timestamp updated
    assert new_p0 == Decimal("5")
    assert new_alpha == Decimal("2")
    assert updated_ts == new_ts


def test_clamped_exponential_integral_start_gte_end():
    """
    If start >= end, the function returns 0 immediately.
    """
    result = ExponentialCurveHelper._clamped_exponential_integral(
        start=Decimal("10"),
        end=Decimal("10"),
        p0=Decimal("1"),
        alpha=Decimal("0.1"),
        min_price=None,
        max_price=None
    )
    assert result == Decimal("0")

    result2 = ExponentialCurveHelper._clamped_exponential_integral(
        start=Decimal("12"),
        end=Decimal("10"),
        p0=Decimal("2"),
        alpha=Decimal("0.2"),
        min_price=None,
        max_price=None
    )
    assert result2 == Decimal("0")


def test_clamped_exponential_integral_alpha_zero_no_clamp():
    """
    If alpha=0 => the price is constant p0. No clamp => just p0*(end-start).
    """
    start = Decimal("5")
    end = Decimal("10")
    p0 = Decimal("2")
    alpha = Decimal("0")

    result = ExponentialCurveHelper._clamped_exponential_integral(
        start, end, p0, alpha, min_price=None, max_price=None
    )
    # cost => 2*(10-5)= 10
    assert result == Decimal("10")


def test_clamped_exponential_integral_alpha_zero_with_clamp():
    """
    alpha=0 => price=p0 => we clamp that price into [min_price, max_price].
    We'll do p0=2 => min_price=3 => max_price=10 => final clamped price=3 => area= 3*(end-start).
    """
    start = Decimal("0")
    end = Decimal("5")
    p0 = Decimal("2")
    alpha = Decimal("0")
    min_price = Decimal("3")
    max_price = Decimal("10")

    # Because p0=2 < min_price=3 => entire subrange is at clamped price=3
    result = ExponentialCurveHelper._clamped_exponential_integral(
        start, end, p0, alpha, min_price, max_price
    )
    # area => 3*(5-0)=15
    assert result == Decimal("15")


def test_clamped_exponential_integral_no_clamp_alpha_positive():
    """
    No clamps => pure exponential integral => (p0/alpha)*[exp(alpha*end)- exp(alpha*start)].
    We'll check with a small scenario.
    """
    start = Decimal("10")
    end = Decimal("12")
    p0 = Decimal("1")
    alpha = Decimal("0.05")

    # We'll compute expected manually:
    # area= (p0/alpha)* [ exp(alpha*end)- exp(alpha*start) ]
    # => (1/0.05)* [ exp(0.05*12) - exp(0.05*10) ]
    exp_start = (alpha*start).exp()
    exp_end = (alpha*end).exp()
    expected = (p0/alpha)*(exp_end - exp_start)

    result = ExponentialCurveHelper._clamped_exponential_integral(
        start, end, p0, alpha, None, None
    )
    assert decimal_approx_equal(result, expected, tol=Decimal(10**7)), f"Expected ~{expected}, got {result}"


def test_clamped_exponential_integral_min_price_cross():
    """
    Only min_price => Suppose p0 e^(alpha*x) starts below min_price, then crosses it.
    We'll do an example where p0*(alpha*x).exp is < min_price at x<someX, then above after someX.
    """
    # p0=1, alpha=0.1 => price= e^(0.1*x)
    # min_price=2 => xMin => e^(0.1*x)=2 => 0.1*x=ln(2) => x= (ln(2))/0.1 ~ 6.93147
    # We'll define start=5, end=10 => so the price is below 2 in [5..6.93...), above 2 in [6.93..10].
    start = Decimal("5")
    end = Decimal("10")
    p0 = Decimal("1")
    alpha = Decimal("0.1")
    min_price = Decimal("2")
    max_price = None

    # We'll do a piecewise approach:
    # sub-interval [5..6.93...] => clamp at min_price=2 => area=2*(6.93-5)
    # sub-interval [6.93..10] => real integral e^(0.1*x)
    # We'll do an approximate check with the function result.

    result = ExponentialCurveHelper._clamped_exponential_integral(
        start, end, p0, alpha, min_price, max_price
    )

    # xMin ~ 6.93147
    x_min = (min_price/p0).ln()/alpha  # ln(2)/0.1
    # area1 => clamp => 2*(xMin-5)
    area1 = 2*(x_min - 5)
    # area2 => integral e^(0.1*x) from xMin..10 => (1/0.1)*[ exp(0.1*10) - exp(0.1*xMin)]
    exp_min = (alpha*x_min).exp()
    exp_end = (alpha*end).exp()
    area2 = (p0/alpha)*(exp_end - exp_min)

    expected = area1 + area2

    assert decimal_approx_equal(result, expected, tol=Decimal(10**7)), f"Expected ~{expected}, got {result}"


def test_clamped_exponential_integral_max_price_cross():
    """
    Only max_price => Suppose p0 e^(alpha*x) crosses that from below to above in the interval.
    E.g. p0=1, alpha=0.05 => max_price=2 => find x_max => e^(0.05*x)=2 => x= ln(2)/0.05= ~13.8629
    We'll do start=12, end=16 => crossing at ~13.86
    """
    start = Decimal("12")
    end = Decimal("16")
    p0 = Decimal("1")
    alpha = Decimal("0.05")
    min_price = None
    max_price = Decimal("2")

    # We'll do the piecewise approach:
    # sub-interval [12..13.86] => normal integral e^(0.05*x)
    # sub-interval [13.86..16] => clamp => price=2 => area=2*(16-13.86)
    result= ExponentialCurveHelper._clamped_exponential_integral(
        start, end, p0, alpha, min_price, max_price
    )

    # manual approximate:
    x_max = (max_price/p0).ln()/alpha
    # area1 => integral from 12..x_max => (1/0.05)* [exp(0.05*x_max)- exp(0.05*12)]
    alpha_start = alpha*start
    alpha_xMax = alpha*x_max
    area1 = (p0/alpha)*(alpha_xMax.exp() - alpha_start.exp())

    area2 = max_price * (end - x_max)

    expected = area1 + area2
    assert decimal_approx_equal(result, expected, tol=Decimal(10**7)), f"Expected ~{expected}, got {result}"


def test_clamped_exponential_integral_both_min_max():
    """
    Both min_price and max_price => Suppose we cross min_price, then cross max_price in the same interval.
    We'll pick a scenario: p0 e^(alpha*x) starts < min_price, then goes above max_price. This yields 3 sub-intervals.
    """
    # p0=1, alpha=0.2 => let's pick start=0, end=5 => min_price=1.5 => max_price=4
    # We'll find x_min => e^(0.2*x)=1.5 => x_min= ln(1.5)/0.2
    # We'll find x_max => e^(0.2*x)=4 => x_max= ln(4)/0.2
    # Then piecewise:
    #   [start.. x_min] => clamp=1.5
    #   [x_min.. x_max] => actual integral
    #   [x_max.. end ] => clamp=4
    start = Decimal("0")
    end = Decimal("5")
    p0 = Decimal("1")
    alpha = Decimal("0.2")
    min_price = Decimal("1.5")
    max_price = Decimal("4")

    result= ExponentialCurveHelper._clamped_exponential_integral(
        start, end, p0, alpha, min_price, max_price
    )

    # manual approximate:
    x_min = (min_price/p0).ln()/ alpha  # ln(1.5)/0.2
    x_max = (max_price/p0).ln()/ alpha  # ln(4)/0.2

    # be sure to clamp x_min, x_max in [0..5]
    x_min = max(start, x_min)
    x_min = min(end, x_min)
    x_max = max(start, x_max)
    x_max = min(end, x_max)

    # area1 => clamp min => 1.5*(x_min-0) if x_min>0
    area1 = Decimal("0")
    if x_min > start:
        area1 = min_price*(x_min - start)

    # area2 => integral => [x_min.. x_max], if x_max> x_min
    area2 = Decimal("0")
    if x_max > x_min:
        alphaA = alpha*x_min
        alphaB = alpha*x_max
        area2 = (p0/ alpha)* (alphaB.exp() - alphaA.exp())

    # area3 => clamp max => [ x_max.. end]
    area3 = Decimal("0")
    if end > x_max:
        area3 = max_price*(end - x_max)

    expected = area1 + area2 + area3
    assert decimal_approx_equal(result, expected, tol=Decimal(10**7)), f"Expected ~{expected}, got {result}"


def test_clamped_exponential_integral_negative_alpha():
    """
    Negative alpha => decaying exponent => piecewise logic same. We'll do no clamps for simplicity.
    We'll just compare to the pure integral formula.
    """
    start = Decimal("2")
    end = Decimal("5")
    p0 = Decimal("3")
    alpha = Decimal("-0.1")
    min_price = None
    max_price = None

    result = ExponentialCurveHelper._clamped_exponential_integral(
        start, end, p0, alpha, min_price, max_price
    )
    # pure integral => area= (p0/ alpha)* [exp(alpha*end)- exp(alpha*start)]
    # => (3/ -0.1)* [ exp(-0.1*5)- exp(-0.1*2)]
    alpha_start = alpha*start
    alpha_end = alpha*end
    expected = (p0/ alpha)*( alpha_end.exp() - alpha_start.exp())

    assert decimal_approx_equal(result, expected, tol=Decimal(10**7)), f"Expected ~{expected}, got {result}"


@pytest.mark.parametrize(
    "current_supply, amount, p0, alpha, expected_cost",
    [
        # 1) amount <= 0 => cost=0
        (Decimal("100"), Decimal("0"), Decimal("1"), Decimal("0.1"), Decimal("0")),
        (Decimal("100"), Decimal("-5"), Decimal("2"), Decimal("0"), Decimal("0")),

        # 2) alpha=0 => cost= p0*amount
        (Decimal("100"), Decimal("10"), Decimal("1.5"), Decimal("0"), Decimal("15")),
        # => 1.5*10=15

        # 3a) alpha>0 => let's pick alpha=0.1 =>
        #    cost= (p0/alpha)*[exp(alpha*(start+amount))-exp(alpha*start)]
        #    s=100 => end=110 => alpha=0.1 => p0=1
        #    cost= (1/0.1)*[exp(0.1*110)-exp(0.1*100)]
        # We'll approximate the result
        (Decimal("100"), Decimal("10"), Decimal("1"), Decimal("0.1"), "formula"),

        # 3b) alpha<0 => same formula, let's see a small scenario
        (Decimal("0"), Decimal("5"), Decimal("2"), Decimal("-0.05"), "formula_neg"),
    ]
)
def test_exponential_cost_for_purchase(current_supply, amount, p0, alpha, expected_cost):
    """
    Tests exponential_cost_for_purchase with various scenarios:
     - amount<=0 => cost=0
     - alpha=0 => cost=p0*amount
     - alpha>0 => integral formula
     - alpha<0 => same integral formula (decaying exponent)
    We'll do approximate checks for formula cases.
    """
    cost = ExponentialCurveHelper.exponential_cost_for_purchase(
        current_supply, amount, p0, alpha
    )

    if expected_cost == "formula":
        # We'll compute the integral ourselves or do an approximate check.
        # cost= (p0/alpha)*[exp(alpha*(s+Δs)) - exp(alpha*s)]
        s = current_supply
        end = s + amount
        # cost= (1/0.1)*[exp(0.1*110) - exp(0.1*100)]
        # Let's compute an approximate:
        from decimal import getcontext
        # Possibly set precision if needed:
        # getcontext().prec=28  # default 28 is often fine

        # We'll do the manual formula:
        alpha_end = alpha * end
        alpha_s = alpha * s
        part = alpha_end.exp() - alpha_s.exp()
        expected = (p0 / alpha) * part

        # We'll do an approximate check
        assert decimal_approx_equal(cost, expected, tol=Decimal(10**7)), f"Expected ~{expected}, got {cost}"

    elif expected_cost == "formula_neg":
        # alpha<0 scenario
        s = current_supply
        end = s + amount
        part = (alpha * end).exp() - (alpha * s).exp()
        expected = (p0 / alpha)* part

        assert decimal_approx_equal(cost, expected, tol=Decimal(10**7)), f"Expected ~{expected}, got {cost}"

    else:
        # direct compare
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"


@pytest.mark.parametrize(
    "current_supply, amount, p0, alpha, expected",
    [
        # 1) amount<=0 => 0
        (Decimal("100"), Decimal("0"), Decimal("1"), Decimal("0.1"), Decimal("0")),
        (Decimal("100"), Decimal("-5"), Decimal("2"), Decimal("0"), Decimal("0")),

        # 2) alpha=0 => return= p0*amount
        (Decimal("100"), Decimal("10"), Decimal("1.5"), Decimal("0"), Decimal("15")),
        # => 1.5*10=15

        # 3a) alpha>0 => use formula => return= (p0/alpha)*[exp(alpha*end) - exp(alpha*start)]
        #   where start= s-Δs, end=s
        #   let's do s=110 => amount=10 => start=100 => alpha=0.05 => p0=1 => see formula
        (Decimal("110"), Decimal("10"), Decimal("1"), Decimal("0.05"), "formula_pos"),

        # 3b) alpha<0 => negative exponent => e.g. s=10 => amount=5 => start=5 => alpha=-0.1 => p0=2 => see formula
        (Decimal("10"), Decimal("5"), Decimal("2"), Decimal("-0.1"), "formula_neg"),
    ]
)
def test_exponential_return_for_sale(current_supply, amount, p0, alpha, expected):
    """
    Tests exponential_return_for_sale with various scenarios:
      - amount<=0 => returns 0
      - alpha=0 => p0*amount
      - alpha>0 => integral formula
      - alpha<0 => integral formula
    We'll do approximate checks for formula-based cases.
    """
    if (current_supply - amount) < 0 < amount:
        # We anticipate ValueError for 'Cannot sell more than current supply.'
        # But let's see if we param that specifically in a separate test.
        with pytest.raises(ValueError, match="Cannot sell more tokens"):
            ExponentialCurveHelper.exponential_return_for_sale(
                current_supply, amount, p0, alpha
            )
        return

    returned = ExponentialCurveHelper.exponential_return_for_sale(
        current_supply, amount, p0, alpha
    )

    if expected == "formula_pos":
        # Let's compute approximate integral ourselves:
        # start= s-amount => end=s => cost= (p0/alpha)*[exp(alpha*end)-exp(alpha*start)]
        s = current_supply
        start = s-amount
        end = s
        alpha_end = alpha * end
        alpha_start = alpha * start
        part = alpha_end.exp() - alpha_start.exp()
        expected_val = (p0 / alpha)* part
        # We'll do isclose check
        assert decimal_approx_equal(returned, expected_val, tol=Decimal(10**7)), (
            f"Expected ~{expected_val}, got {returned}"
        )

    elif expected == "formula_neg":
        # same formula => negative alpha => s=10 => amount=5 => start=5 => alpha=-0.1 => p0=2
        s = current_supply
        start = s-amount
        end = s
        part = (alpha * end).exp() - (alpha * start).exp()
        expected_val = (p0 / alpha)* part
        assert decimal_approx_equal(returned, expected_val, tol=Decimal(10**7)), (
            f"Expected ~{expected_val}, got {returned}"
        )

    else:
        # direct numeric
        assert returned == expected, f"Expected {expected}, got {returned}"


def test_exponential_return_for_sale_more_than_supply():
    """
    If start<0 => we raise ValueError("Cannot sell more tokens than the current supply.")
    """
    with pytest.raises(ValueError, match="Cannot sell more tokens"):
        ExponentialCurveHelper.exponential_return_for_sale(
            current_supply=Decimal("10"),
            amount=Decimal("15"),
            p0=Decimal("1"),
            alpha=Decimal("0.1")
        )


def test_exponential_cost_for_purchase_no_clamp():
    """
    clamped=False => same old logic: if alpha=0 => p0*amount, else integral => (p0/alpha)*[exp(...) - exp(...)].
    We'll do a small check as before.
    """
    s = Decimal("100")
    amount = Decimal("10")
    p0 = Decimal("1")
    alpha = Decimal("0.05")

    cost = ExponentialCurveHelper.exponential_cost_for_purchase(
        current_supply=s,
        amount=amount,
        p0=p0,
        alpha=alpha,
        clamped=False,
        min_price=None,
        max_price=None
    )
    # Compare with old integral formula
    end = s+amount
    expected = (p0/alpha)* ((alpha*end).exp() - (alpha*s).exp())

    assert decimal_approx_equal(cost, expected, tol=Decimal(10**7)), f"Expected ~{expected}, got {cost}"


def test_exponential_cost_for_purchase_alpha_zero_clamp():
    """
    If alpha=0 => cost = p0*amount => but we also clamp p0 if clamped=True => effectively cost= clampedPrice*(amount).
    E.g. p0=2 => min_price=3 => final cost => 3*(end-start).
    """
    s = Decimal("50")
    amount = Decimal("10")
    p0 = Decimal("2")
    alpha = Decimal("0")
    min_price = Decimal("3")
    max_price = Decimal("5")
    cost = ExponentialCurveHelper.exponential_cost_for_purchase(
        current_supply=s,
        amount=amount,
        p0=p0,
        alpha=alpha,
        clamped=True,
        min_price=min_price,
        max_price=max_price
    )
    # Because p0=2 < min_price=3 => entire range => price=3 => cost= 3*10=30
    assert cost == Decimal("30")


@pytest.mark.parametrize(
    "s, amount, p0, alpha, min_p, max_p, desc",
    [
        # min_price only => partial region below min, partial region normal
        (Decimal("5"), Decimal("5"), Decimal("1"), Decimal("0.1"), Decimal("2"), None, "min clamp cross"),
        # max_price only => partial region normal, partial region above max
        (Decimal("10"), Decimal("10"), Decimal("1"), Decimal("0.05"), None, Decimal("2"), "max clamp cross"),
        # both min & max
        (Decimal("0"), Decimal("10"), Decimal("1"), Decimal("0.2"), Decimal("1.5"), Decimal("4"), "both clamp cross"),
        # negative alpha scenario
        (Decimal("2"), Decimal("6"), Decimal("2"), Decimal("-0.1"), Decimal("1.5"), Decimal("3"), "negative alpha clamp"),
    ]
)
def test_exponential_cost_for_purchase_clamped_piecewise(s, amount, p0, alpha, min_p, max_p, desc):
    """
    We can compare with a direct call to _clamped_exponential_integral or do a piecewise manual check.
    We'll do a direct call to the helper's _clamped_exponential_integral to see if they match.
    """
    if amount<=0:
        # cost=0 => trivial
        cost = ExponentialCurveHelper.exponential_cost_for_purchase(
            current_supply=s,
            amount=amount,
            p0=p0,
            alpha=alpha,
            clamped=True,
            min_price=min_p,
            max_price=max_p
        )
        assert cost==0
        return

    start = s
    end = s + amount

    expected = ExponentialCurveHelper._clamped_exponential_integral(
        start, end, p0, alpha, min_p, max_p
    )

    cost = ExponentialCurveHelper.exponential_cost_for_purchase(
        current_supply=s,
        amount=amount,
        p0=p0,
        alpha=alpha,
        clamped=True,
        min_price=min_p,
        max_price=max_p
    )
    # Compare
    # We'll do isclose because of exponent decimal
    assert decimal_approx_equal(cost, expected, tol=Decimal(10**7)), (
        f"[{desc}] Expected clamp cost ~{expected}, got {cost}"
    )
