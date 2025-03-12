import pytest

from datetime import datetime, timedelta
from decimal import Decimal

from forgd_core.common.model import StepConfig
from forgd_core.curves.utils.stepwise_curve_helper import StepwiseCurveHelper


@pytest.fixture
def sample_steps():
    """
    Provides a small list of StepConfig objects for testing.
    """
    return [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("10")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("20")),
    ]


def test_validate_and_sort_steps_empty():
    """
    If steps is empty, should return an empty list (no exception).
    """
    steps = []
    result = StepwiseCurveHelper.validate_and_sort_steps(steps)
    assert result == [], "Expected empty list when input is empty."


def test_validate_and_sort_steps_single():
    """
    A single step should be returned as is (no sorting needed, no error).
    """
    steps = [StepConfig(supply_threshold=Decimal("100"), price=Decimal("2"))]
    result = StepwiseCurveHelper.validate_and_sort_steps(steps)
    assert len(result) == 1
    assert result[0].supply_threshold == Decimal("100")


def test_validate_and_sort_steps_already_sorted():
    """
    Multiple valid steps in strictly ascending order => return them as is.
    """
    steps = [
        StepConfig(supply_threshold=Decimal("50"), price=Decimal("1.5")),
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("150"), price=Decimal("2.5")),
    ]
    result = StepwiseCurveHelper.validate_and_sort_steps(steps)
    assert len(result) == 3
    # Should preserve the same order since they're already ascending
    assert result[0].supply_threshold == Decimal("50")
    assert result[1].supply_threshold == Decimal("100")
    assert result[2].supply_threshold == Decimal("150")


def test_validate_and_sort_steps_unsorted():
    """
    Steps not in ascending order should be sorted by ascending supply_threshold.
    """
    steps = [
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3.0")),
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1.0")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2.0")),
    ]
    result = StepwiseCurveHelper.validate_and_sort_steps(steps)
    # After sorting, we expect [100, 200, 300] in ascending order
    assert result[0].supply_threshold == Decimal("100")
    assert result[1].supply_threshold == Decimal("200")
    assert result[2].supply_threshold == Decimal("300")


def test_validate_and_sort_steps_non_strict_raises():
    """
    If any threshold is less than or equal to the previous threshold, raises ValueError.
    """
    # Example: two steps have the same threshold=100 => not strictly ascending
    steps = [
        StepConfig(supply_threshold=Decimal("50"), price=Decimal("1.5")),
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("2.5")),  # same as previous
    ]
    with pytest.raises(ValueError, match="must have strictly ascending"):
        StepwiseCurveHelper.validate_and_sort_steps(steps)


def test_validate_and_sort_steps_negative_threshold_raises():
    """
    Negative threshold is not allowed => raises ValueError.
    """
    steps = [
        StepConfig(supply_threshold=Decimal("-10"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("2")),
    ]
    with pytest.raises(ValueError, match="Negative thresholds are not allowed"):
        StepwiseCurveHelper.validate_and_sort_steps(steps)

def test_apply_time_decay_to_steps_no_elapsed_time(monkeypatch, sample_steps):
    """
    If there's no elapsed time (now == last_timestamp),
    function should return the same steps, same last_timestamp.
    """
    fixed_now = datetime(2025, 3, 10, 12, 0, 0)

    class MockDateTime:
        @classmethod
        def now(cls):
            return fixed_now

    monkeypatch.setattr("forgd_core.curves.utils.stepwise_curve_helper.datetime", MockDateTime)

    result_steps, new_ts = StepwiseCurveHelper.apply_time_decay_to_steps(
        steps=sample_steps,
        last_timestamp=fixed_now,             # no elapsed time
        time_decay_rate=Decimal("0.05"),
        approach=1
    )
    # No changes expected
    assert result_steps == sample_steps, "No price change if elapsed time=0."
    assert new_ts == fixed_now, "Timestamp should remain unchanged if no time passed."


def test_apply_time_decay_to_steps_zero_rate(monkeypatch, sample_steps):
    """
    If time_decay_rate=0 => no changes regardless of elapsed time.
    """
    old_ts = datetime(2025, 3, 10, 12, 0, 0)
    new_ts = old_ts + timedelta(days=5)  # 5 days later

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.utils.stepwise_curve_helper.datetime", MockDateTime)

    result_steps, updated_ts = StepwiseCurveHelper.apply_time_decay_to_steps(
        steps=sample_steps,
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0"),  # zero => no changes
        approach=1
    )
    assert result_steps == sample_steps, "No price change if decay_rate=0."
    assert updated_ts == old_ts, "Timestamp remains old_ts if no decay is applied."


def test_apply_time_decay_to_steps_approach_1_positive(monkeypatch, sample_steps):
    """
    Approach 1 => price *= (1 + (time_decay_rate * elapsed_days)).
    Let's test with time_decay_rate=0.1, elapsed_days=2 => factor=0.2 => multiplier=1.2
    """
    old_ts = datetime(2025, 3, 1, 0, 0, 0)
    new_ts = old_ts + timedelta(days=2)  # 2 days difference

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.utils.stepwise_curve_helper.datetime", MockDateTime)

    # time_decay_rate=0.1 => additional_factor= 2 * 0.1=0.2 => multiplier=1.2
    result_steps, updated_ts = StepwiseCurveHelper.apply_time_decay_to_steps(
        steps=sample_steps,
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0.1"),
        approach=1
    )

    assert updated_ts == new_ts, "Timestamp should be updated to new_ts"
    # Check each step's price => old_price * 1.2
    for i, original in enumerate(sample_steps):
        expected_price = original.price * Decimal("1.2")
        assert result_steps[i].supply_threshold == original.supply_threshold
        assert result_steps[i].price == expected_price


def test_apply_time_decay_to_steps_approach_2_positive(monkeypatch, sample_steps):
    """
    Approach 2 => price = old_price + (time_decay_rate * elapsed_days).
    For time_decay_rate=0.1, elapsed_days=3 => add 0.3
    """
    old_ts = datetime(2025, 3, 1, 0, 0, 0)
    new_ts = old_ts + timedelta(days=3)  # 3 days difference

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.utils.stepwise_curve_helper.datetime", MockDateTime)

    # time_decay_rate=0.1 => add= 3 * 0.1=0.3
    result_steps, updated_ts = StepwiseCurveHelper.apply_time_decay_to_steps(
        steps=sample_steps,
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0.1"),
        approach=2
    )

    assert updated_ts == new_ts
    for i, original in enumerate(sample_steps):
        expected_price = original.price + Decimal("0.3")
        assert result_steps[i].supply_threshold == original.supply_threshold
        assert result_steps[i].price == expected_price


def test_apply_time_decay_to_steps_unknown_approach(monkeypatch, sample_steps):
    """
    If approach !=1 and !=2, there's no explicit else.
    The code doesn't update the price. So it remains the same as original.
    """
    old_ts = datetime(2025, 3, 1, 0, 0, 0)
    new_ts = old_ts + timedelta(days=2)  # 2 days difference

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.utils.stepwise_curve_helper.datetime", MockDateTime)

    result_steps, updated_ts = StepwiseCurveHelper.apply_time_decay_to_steps(
        steps=sample_steps,
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0.05"),
        approach=999  # unknown
    )
    # The code sets new_price=step.price if approach isn't 1 or 2.
    # So the prices remain the same, though timestamp updates
    assert updated_ts == new_ts
    for i, original in enumerate(sample_steps):
        assert result_steps[i].supply_threshold == original.supply_threshold
        # no change in price
        assert result_steps[i].price == original.price


def test_apply_time_decay_to_steps_negative_elapsed(monkeypatch, sample_steps):
    """
    If now < last_timestamp => negative elapsed_days => no changes returned.
    """
    old_ts = datetime(2025, 3, 10, 12, 0, 0)
    new_ts = old_ts - timedelta(days=1)  # this is 1 day earlier => negative elapsed

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.utils.stepwise_curve_helper.datetime", MockDateTime)

    result_steps, updated_ts = StepwiseCurveHelper.apply_time_decay_to_steps(
        steps=sample_steps,
        last_timestamp=old_ts,
        time_decay_rate=Decimal("0.1"),
        approach=1
    )

    # Because elapsed_days < 0 => no change
    assert result_steps == sample_steps, "No changes if elapsed_days<0"
    assert updated_ts == old_ts


def test_apply_time_decay_to_steps_negative_rate(monkeypatch, sample_steps):
    """
    Demonstrate what happens if time_decay_rate <0 => price is decreased for approach=1 or approach=2.
    """
    old_ts = datetime(2025, 3, 1, 0, 0, 0)
    new_ts = old_ts + timedelta(days=2)  # 2 days difference

    class MockDateTime:
        @classmethod
        def now(cls):
            return new_ts

    monkeypatch.setattr("forgd_core.curves.utils.stepwise_curve_helper.datetime", MockDateTime)

    # time_decay_rate=-0.05 => approach=1 => factor= 2 * -0.05= -0.1 => multiplier=0.9 => price*0.9
    result_steps, updated_ts = StepwiseCurveHelper.apply_time_decay_to_steps(
        steps=sample_steps,
        last_timestamp=old_ts,
        time_decay_rate=Decimal("-0.05"),
        approach=1
    )
    for i, original in enumerate(sample_steps):
        expected_price = original.price * Decimal("0.9")
        assert result_steps[i].price == expected_price


def test_stepwise_cost_for_purchase_zero_amount():
    """
    If the user requests amount<=0 => cost=0 immediately.
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    cost = StepwiseCurveHelper.stepwise_cost_for_purchase(
        current_supply=Decimal("50"),
        amount=Decimal("0"),
        steps=steps,
        allow_partial_fill=True
    )
    assert cost == Decimal("0")

    # Also test negative amount => cost=0
    cost_neg = StepwiseCurveHelper.stepwise_cost_for_purchase(
        current_supply=Decimal("50"),
        amount=Decimal("-10"),
        steps=steps,
        allow_partial_fill=True
    )
    assert cost_neg == Decimal("0")


def test_stepwise_cost_for_purchase_current_supply_in_second_tier():
    """
    If current_supply=150, that means we've already passed the first tier (0-100).
    So we start in tier2 => price=2 until threshold=200, then tier3=price=3.
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    cost = StepwiseCurveHelper.stepwise_cost_for_purchase(
        current_supply=Decimal("150"),  # we are in the second tier (100-200)
        amount=Decimal("25"),
        steps=steps,
        allow_partial_fill=True
    )
    # Explanation:
    # Tier2 range= [100..200). current_supply=150 => user has 50 tokens capacity left in tier2
    # The user wants 25 => all at price=2 => total=25*2=50
    assert cost == Decimal("50")


def test_stepwise_cost_for_purchase_cross_multiple_tiers():
    """
    A purchase that spans multiple tiers:
      current_supply=90, user buys 50 =>
      => 10 tokens in tier1 (up to threshold=100)
      => remaining 40 tokens in tier2 (price=2, up to threshold=200)
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    cost = StepwiseCurveHelper.stepwise_cost_for_purchase(
        current_supply=Decimal("90"),
        amount=Decimal("50"),
        steps=steps,
        allow_partial_fill=True
    )
    # Calculation:
    # - Tier1 capacity from supply=90..100 => capacity=10 => price=1 => cost=10*1=10
    # - Then we have 40 left to buy => Tier2 => price=2 => cost=40*2=80
    # total= 10 + 80=90
    assert cost == Decimal("90")


def test_stepwise_cost_for_purchase_exhaust_all_tiers_partial_allowed():
    """
    User tries to buy beyond the last tier capacity:
      Tiers: 0-100(price=1), 100-200(price=2), 200-300(price=3)
      current_supply=250 => half in tier3 => capacity=50 left in tier3 => price=3
      user wants=100 => only 50 available => partial fill => cost= 50*3=150
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    cost = StepwiseCurveHelper.stepwise_cost_for_purchase(
        current_supply=Decimal("250"),
        amount=Decimal("100"),
        steps=steps,
        allow_partial_fill=True
    )
    # The only tier left is Tier3 => threshold=300 => capacity= 300-250=50 => price=3
    # user wants 100 => can only buy 50 => cost=50*3=150
    assert cost == Decimal("150")


def test_stepwise_cost_for_purchase_exhaust_all_tiers_partial_not_allowed():
    """
    Same scenario, but allow_partial_fill=False => raise ValueError if we exceed final tier capacity.
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    with pytest.raises(ValueError, match="exceeds final tier capacity"):
        StepwiseCurveHelper.stepwise_cost_for_purchase(
            current_supply=Decimal("250"),
            amount=Decimal("100"),
            steps=steps,
            allow_partial_fill=False
        )


def test_stepwise_cost_for_purchase_start_on_tier_boundary():
    """
    current_supply exactly = 100 => user is just entering tier2 => price=2.
    Buying 20 => all in tier2 => cost= 20*2=40.
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    cost = StepwiseCurveHelper.stepwise_cost_for_purchase(
        current_supply=Decimal("100"),
        amount=Decimal("20"),
        steps=steps,
        allow_partial_fill=True
    )
    assert cost == Decimal("40")


def test_stepwise_cost_for_purchase_no_tiers():
    """
    If steps is empty => no capacity => partial_fill => cost=0, or raise error if partial_fill=False.
    """
    # partial_fill=True => user just gets 0 => cost=0
    cost = StepwiseCurveHelper.stepwise_cost_for_purchase(
        current_supply=Decimal("50"),
        amount=Decimal("10"),
        steps=[],
        allow_partial_fill=True
    )
    assert cost == Decimal("0"), "No tiers => no capacity => partial fill => 0"

    # partial_fill=False => raise ValueError
    with pytest.raises(ValueError, match="exceeds final tier capacity"):
        StepwiseCurveHelper.stepwise_cost_for_purchase(
            current_supply=Decimal("50"),
            amount=Decimal("10"),
            steps=[],
            allow_partial_fill=False
        )


def test_stepwise_return_for_sale_zero_negative():
    """
    If amount <= 0 => returns 0 immediately.
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    ret_zero = StepwiseCurveHelper.stepwise_return_for_sale(
        current_supply=Decimal("150"),
        amount=Decimal("0"),
        steps=steps,
        allow_partial_fill=True
    )
    assert ret_zero == Decimal("0")

    ret_neg = StepwiseCurveHelper.stepwise_return_for_sale(
        current_supply=Decimal("150"),
        amount=Decimal("-10"),
        steps=steps,
        allow_partial_fill=True
    )
    assert ret_neg == Decimal("0")


def test_stepwise_return_for_sale_current_supply_in_tier2():
    """
    current_supply=150 => That falls in tier2 => range [100..200), price=2.
    Selling 25 => up to 50 tokens available at tier2 from supply=150..100 => 50 tokens "in" tier2
    user only sells 25 => all at price=2 => total=25*2=50
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    result = StepwiseCurveHelper.stepwise_return_for_sale(
        current_supply=Decimal("150"),
        amount=Decimal("25"),
        steps=steps,
        allow_partial_fill=True
    )
    assert result == Decimal("50")  # 25 tokens at price=2 => 50 total.


def test_stepwise_return_for_sale_cross_multiple_tiers():
    """
    Example:
     - current_supply=210 => that starts in tier3 => range [200..300), price=3
       in_this_tier=210-200=10
     - user wants 30 => we sell 10 in tier3 => remainder=20 => supply=200
     - now tier2 => range [100..200), price=2 => can sell 100 tokens in that tier
       user sells the remaining 20 => total => 10*3 + 20*2= 30+40=70
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    result = StepwiseCurveHelper.stepwise_return_for_sale(
        current_supply=Decimal("210"),
        amount=Decimal("30"),
        steps=steps,
        allow_partial_fill=True
    )
    # Calculation:
    # Tier3 => [200..300), supply=210 => in_this_tier=10 => sells min(10,30)=10 => ret=10*3=30 => leftover=20 => supply=200
    # Tier2 => [100..200), now supply=200 => in_this_tier= (200-100)=100 => user has 20 left => sells 20 => ret=20*2=40
    # total=30+40=70
    assert result == Decimal("70")


def test_stepwise_return_for_sale_exhaust_tiers_partial_allowed():
    """
    If user tries to sell more than the coverage from current_supply down to 0 in all tiers,
    partial_fill=True => just sells as much as possible.

    Suppose supply=50 => that is in tier1 => user wants 80 =>
    we have only 50 tokens total from supply=50..0 => tier1 covers [0..100).
    sells 50 => remainder=30 => no more tiers? partial fill => returns cost= 50*1=50
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    result = StepwiseCurveHelper.stepwise_return_for_sale(
        current_supply=Decimal("50"),
        amount=Decimal("80"),
        steps=steps,
        allow_partial_fill=True
    )
    # Only 50 tokens exist from supply=50..0 in tier1 => price=1 => ret=50
    assert result == Decimal("50")


def test_stepwise_return_for_sale_exhaust_tiers_partial_not_allowed():
    """
    Same scenario but partial_fill=False => raise ValueError if user tries to sell more
    than total coverage from the current_supply down to 0 in all tiers.
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    with pytest.raises(ValueError, match="exceeds tier coverage"):
        StepwiseCurveHelper.stepwise_return_for_sale(
            current_supply=Decimal("50"),
            amount=Decimal("80"),
            steps=steps,
            allow_partial_fill=False
        )


def test_stepwise_return_for_sale_start_tier_boundary():
    """
    current_supply=200 => that is right on tier3 boundary =>
     - actually tier2 covers [100..200), so if supply=200, that means we are
       at the threshold for tier3 => let's see how the code picks it:
       the code checks if supply < steps[idx].supply_threshold.
       supply=200 => not <200 => idx-- => we end up in tier2.
    So user sells 20 => all in tier2 => 20*2=40
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    # Actually see how the code does the boundary logic:
    result = StepwiseCurveHelper.stepwise_return_for_sale(
        current_supply=Decimal("200"),
        amount=Decimal("20"),
        steps=steps,
        allow_partial_fill=True
    )
    # Because "if current_supply < steps[idx].supply_threshold" => 200<200? false => idx=1 => tier2 => price=2 => capacity=?
    # The capacity in tier2 => supply range= 200..100 => that might be negative if we do direct math.
    # The code sets in_this_tier= supply - max(prev_threshold,0)= 200-100=100 => sells 20 => total=20*2=40
    assert result == Decimal("40")


def test_stepwise_return_for_sale_no_tiers():
    """
    If steps is empty => no coverage => partial fill => 0 or revert if partial_fill=False
    """
    # partial_fill => user just sells 0
    ret = StepwiseCurveHelper.stepwise_return_for_sale(
        current_supply=Decimal("200"),
        amount=Decimal("50"),
        steps=[],
        allow_partial_fill=True
    )
    assert ret == Decimal("0")

    # partial_fill=False => revert
    with pytest.raises(ValueError, match="Exceeded final tier capacity. Reverting."):
        StepwiseCurveHelper.stepwise_return_for_sale(
            current_supply=Decimal("200"),
            amount=Decimal("50"),
            steps=[],
            allow_partial_fill=False
        )
