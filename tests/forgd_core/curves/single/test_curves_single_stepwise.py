import pytest

from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

from forgd_core.common.enums import BondingCurveType, OrderSide
from forgd_core.common.model import (
    BondingCurveParams,
    BondingCurveState,
    StepConfig, Token, Liquidity, TransactionRequest
)
from forgd_core.curves.single.stepwise import StepwiseBondingCurve
from forgd_core.curves.helpers.common import CommonCurveHelper as common_helper
from forgd_core.curves.helpers.stepwise import StepwiseCurveHelper as stepwise_helper


@pytest.fixture
def basic_params():
    """
    Returns a minimal BondingCurveParams object.
    """
    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
    ]
    return BondingCurveParams(curve_type=BondingCurveType.STEPWISE, steps=steps)


@pytest.fixture
def stepwise_curve():
    """
    Returns a StepwiseCurve with a minimal set of steps and a known state.
    You can override 'time_decay_rate' in tests if needed.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.STEPWISE,
        steps=[
            StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
            StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        ],
    )
    state = BondingCurveState(
        current_supply=Decimal("50"),
        current_price=Decimal("1.0"),
        # We'll set an initial timestamp
        last_timestamp=datetime(2025, 1, 1, 12, 0, 0)
    )
    # Provide steps directly to the constructor
    curve = StepwiseBondingCurve(params, state=state, steps=params.steps, time_decay_rate=Decimal("0.0"))
    return curve


@pytest.fixture
def stepwise_curve_with_liquidity():
    """
    Returns a StepwiseCurve instance with a BondingCurveState that includes liquidity.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.STEPWISE,
        steps=[
            StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
            StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        ],
    )
    token = Token(name="USDC", symbol="USDC", decimals=6)
    liquidity = Liquidity(token=token, amount=Decimal("1000"))
    state = BondingCurveState(
        current_supply=Decimal("500"),
        current_price=Decimal("2"),
        liquidity=liquidity
    )
    curve = StepwiseBondingCurve(params=params, state=state, steps=[])
    return curve


@pytest.fixture
def stepwise_curve_no_liquidity():
    """
    Returns a StepwiseCurve with no liquidity object (liquidity=None).
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.STEPWISE,
        steps=[
            StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
            StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        ],
    )
    state = BondingCurveState(
        current_supply=Decimal("500"),
        current_price=Decimal("2"),
        liquidity=None
    )
    curve = StepwiseBondingCurve(params=params, state=state, steps=[])
    return curve


@pytest.fixture
def stepwise_curve_with_tiers():
    """
    Returns a StepwiseCurve with 3 tiers:
      tier0: supply<100 => price=1
      tier1: supply<200 => price=2
      tier2: supply>=200 => price=3
    """
    state = BondingCurveState(current_supply=Decimal("50"), current_price=Decimal("1.0"))

    steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
        StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
    ]
    params = BondingCurveParams(
        curve_type=BondingCurveType.STEPWISE,
        steps=steps
    )

    curve = StepwiseBondingCurve(params=params, state=state)
    return curve


def test_init_no_state(basic_params):
    """
    If no state is passed, StepwiseCurve should create a default BondingCurveState.
    Also sets self._state.last_timestamp to now if it was None.
    """
    curve = StepwiseBondingCurve(params=basic_params, state=None, steps=[])
    assert curve._state is not None, "A default BondingCurveState should be created if none is passed."
    assert curve._state.last_timestamp is not None, "last_timestamp should be set if it was None."


def test_init_with_state_no_timestamp(basic_params):
    """
    If an existing state is passed but last_timestamp=None,
    it should be set to datetime.now().
    We'll mock datetime.now() to a fixed value for verification.
    """
    mock_now = datetime(2025, 4, 1, 12, 0, 0)

    with patch("forgd_core.curves.single.stepwise.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now

        custom_state = BondingCurveState(current_supply=Decimal("100"))
        curve = StepwiseBondingCurve(params=basic_params, state=custom_state, steps=[])

        assert curve._state is custom_state, "Should keep the same state object."
        assert curve._state.last_timestamp == mock_now, "Should set last_timestamp to mocked now() if it was None."


def test_init_with_state_existing_timestamp(basic_params):
    """
    If the state already has a last_timestamp, preserve it.
    """
    existing_ts = datetime(2025, 1, 1, 15, 0, 0)
    custom_state = BondingCurveState(
        current_supply=Decimal("1000"),
        current_price=Decimal("2.0"),
        last_timestamp=existing_ts
    )
    curve = StepwiseBondingCurve(params=basic_params, state=custom_state, steps=[])
    assert curve._state.last_timestamp == existing_ts, (
        "Should not overwrite an existing last_timestamp."
    )


def test_default_options(basic_params):
    """
    Verify the default options are set correctly if no overrides are provided.
    """
    curve = StepwiseBondingCurve(params=basic_params, steps=[])
    # Check default values from the constructor
    assert curve.options["allow_buy"] is True
    assert curve.options["allow_sell"] is True
    assert curve.options["max_supply"] is None
    assert curve.options["max_liquidity"] is None
    assert curve.options["slippage_tolerance"] == Decimal("0")
    assert curve.options["txn_fee_rate"] == Decimal("0")
    assert curve.options["time_decay_rate"] == Decimal("0")
    assert curve.options["risk_profile"] is None
    assert curve.options["partial_fill_approach"] == 1
    assert curve.options["time_decay_approach"] == 1
    assert curve.options["pro_rata"] is False


def test_init_with_options_overrides(basic_params):
    """
    If recognized kwargs are passed, they override the default options.
    If unrecognized kwargs are passed, they go into self.custom_options.
    """
    curve = StepwiseBondingCurve(
        params=basic_params,
        steps=[],
        allow_buy=False,
        max_supply=Decimal("500"),
        foo="bar",  # not recognized => custom_options
        pro_rata=True
    )
    assert curve.options["allow_buy"] is False
    assert curve.options["max_supply"] == Decimal("500")
    assert curve.options["pro_rata"] is True

    assert hasattr(curve, "custom_options")
    assert curve.custom_options["foo"] == "bar"


def test_init_with_steps(basic_params):
    """
    Ensure we call stepwise_helper.validate_and_sort_steps and store the result in self._steps.
    """
    # We'll mock the helper to ensure it's called
    with patch.object(stepwise_helper, "validate_and_sort_steps", return_value=["mocked_result"]) as mock_validate:
        curve = StepwiseBondingCurve(params=basic_params)
    # validate_and_sort_steps should be called with steps_input
    mock_validate.assert_called_once_with(basic_params.steps)
    # and curve._steps should store the result
    assert curve._steps == ["mocked_result"]


def test_init_with_invalid_steps_raises(basic_params):
    """
    If validate_and_sort_steps raises an error, we expect the constructor to fail.
    """
    invalid_steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("2")),  # same threshold => invalid
    ]
    with patch.object(stepwise_helper, "validate_and_sort_steps", side_effect=ValueError("Invalid tiers")):
        with pytest.raises(ValueError, match="Invalid tiers"):
            StepwiseBondingCurve(params=basic_params, steps=invalid_steps)


def test_apply_time_decay_to_steps_no_decay(stepwise_curve):
    """
    If time_decay_rate <= 0, _apply_time_decay_to_steps should do nothing:
      - No call to stepwise_helper.apply_time_decay_to_steps
      - self._steps / last_timestamp unchanged
    """
    stepwise_curve.options["time_decay_rate"] = Decimal("0")

    original_steps = list(stepwise_curve._steps)  # copy
    original_ts = stepwise_curve._state.last_timestamp

    with patch.object(
        stepwise_helper,
        "apply_time_decay_to_steps",
        return_value=("mocked_steps", datetime(2026,1,1))) as mock_decay:
        stepwise_curve._apply_time_decay_to_steps()

    mock_decay.assert_not_called()  # because time_decay_rate=0 => no call
    assert stepwise_curve._steps == original_steps, "Steps should remain unchanged"
    assert stepwise_curve._state.last_timestamp == original_ts, "Timestamp unchanged"


def test_apply_time_decay_to_steps_positive(stepwise_curve):
    """
    If time_decay_rate > 0, we call stepwise_helper.apply_time_decay_to_steps,
    then update self._steps and self._state.last_timestamp from the result.
    """
    stepwise_curve.options["time_decay_rate"] = Decimal("0.05")
    stepwise_curve.options["time_decay_approach"] = 2

    original_ts = stepwise_curve._state.last_timestamp
    original_steps = list(stepwise_curve._steps)

    mocked_new_steps = [
        StepConfig(supply_threshold=Decimal("100"), price=Decimal("1.05")),
        StepConfig(supply_threshold=Decimal("200"), price=Decimal("2.10")),
    ]
    mocked_new_ts = datetime(2026, 1, 1, 0, 0, 0)

    with patch.object(
        stepwise_helper,
        "apply_time_decay_to_steps",
        return_value=(mocked_new_steps, mocked_new_ts)
    ) as mock_decay:
        stepwise_curve._apply_time_decay_to_steps()

    # Check the call used the *old* steps:
    mock_decay.assert_called_once_with(
        original_steps,      # The original [StepConfig(1), StepConfig(2)]
        original_ts,
        Decimal("0.05"),
        approach=2
    )

    # Confirm we stored the new steps + new timestamp
    assert stepwise_curve._steps == mocked_new_steps
    assert stepwise_curve._state.last_timestamp == mocked_new_ts


def test_apply_time_decay_to_steps_negative_rate_disallowed(stepwise_curve):
    """
    If time_decay_rate < 0, we raise a ValueError, since negative rates are not allowed.
    """
    stepwise_curve.options["time_decay_rate"] = Decimal("-0.02")  # negative

    with pytest.raises(ValueError, match="Negative time_decay_rate is not allowed"):
        stepwise_curve._apply_time_decay_to_steps()


def test_update_state_after_buy_with_liquidity(stepwise_curve_with_liquidity):
    """
    If liquidity is present, both current_supply and liquidity.amount should increase.
    """
    curve = stepwise_curve_with_liquidity
    initial_supply = curve._state.current_supply
    initial_liquidity = curve._state.liquidity.amount

    buy_amount = Decimal("50")
    total_cost = Decimal("150")  # 50 tokens at e.g. price=3 in some scenario

    curve._update_state_after_buy(buy_amount, total_cost)

    assert curve._state.current_supply == initial_supply + buy_amount, (
        "current_supply should increment by the buy_amount"
    )
    assert curve._state.liquidity.amount == initial_liquidity + total_cost, (
        "liquidity.amount should increment by total_cost"
    )


def test_update_state_after_buy_no_liquidity(stepwise_curve_no_liquidity):
    """
    If liquidity is None, we only update current_supply; no error should occur.
    """
    curve = stepwise_curve_no_liquidity
    initial_supply = curve._state.current_supply

    buy_amount = Decimal("10")
    total_cost = Decimal("30")

    curve._update_state_after_buy(buy_amount, total_cost)

    assert curve._state.current_supply == initial_supply + buy_amount, (
        "Should still update current_supply"
    )
    # liquidity is None => no increment
    assert curve._state.liquidity is None, "Still None; no error triggered"


def test_update_state_after_buy_decimal_math(stepwise_curve_with_liquidity):
    """
    Demonstrates decimal precision. For instance, partial buys or fees might produce decimal values.
    """
    curve = stepwise_curve_with_liquidity
    initial_supply = curve._state.current_supply
    initial_liquidity = curve._state.liquidity.amount

    # e.g., 12.345 tokens at a cost of 45.6789
    buy_amount = Decimal("12.345")
    total_cost = Decimal("45.6789")

    curve._update_state_after_buy(buy_amount, total_cost)

    assert curve._state.current_supply == initial_supply + Decimal("12.345"), (
        "Should handle decimal increments precisely"
    )
    assert curve._state.liquidity.amount == initial_liquidity + Decimal("45.6789"), (
        "Should increment liquidity by decimal cost"
    )

def test_update_state_after_sell_with_liquidity(stepwise_curve_with_liquidity):
    """
    If liquidity is present, both current_supply and liquidity.amount should decrease.
    """
    curve = stepwise_curve_with_liquidity
    initial_supply = curve._state.current_supply
    initial_liquidity = curve._state.liquidity.amount

    sell_amount = Decimal("50")
    total_return = Decimal("125")  # e.g. user sold 50 tokens at price=2.5

    curve._update_state_after_sell(sell_amount, total_return)

    # current_supply should be decremented by sell_amount
    assert curve._state.current_supply == initial_supply - sell_amount, (
        "current_supply should decrement by sell_amount"
    )
    # liquidity.amount should be decremented by total_return
    assert curve._state.liquidity.amount == initial_liquidity - total_return, (
        "liquidity.amount should decrement by total_return"
    )


def test_update_state_after_sell_no_liquidity(stepwise_curve_no_liquidity):
    """
    If liquidity=None, we only update current_supply and do not error.
    """
    curve = stepwise_curve_no_liquidity
    initial_supply = curve._state.current_supply

    sell_amount = Decimal("10")
    total_return = Decimal("30")

    curve._update_state_after_sell(sell_amount, total_return)

    # current_supply decremented
    assert curve._state.current_supply == initial_supply - sell_amount
    # liquidity is None => no update
    assert curve._state.liquidity is None


def test_update_state_after_sell_decimal(stepwise_curve_with_liquidity):
    """
    Demonstrates decimal arithmetic. For partial sells, fees, etc., we may get decimals.
    """
    curve = stepwise_curve_with_liquidity
    initial_supply = curve._state.current_supply
    initial_liquidity = curve._state.liquidity.amount

    sell_amount = Decimal("12.345")
    total_return = Decimal("49.9999")

    curve._update_state_after_sell(sell_amount, total_return)

    assert curve._state.current_supply == initial_supply - Decimal("12.345"), (
        "Should decrement supply by 12.345"
    )
    assert curve._state.liquidity.amount == initial_liquidity - Decimal("49.9999"), (
        "Should decrement liquidity by 49.9999"
    )


def test_get_spot_price_calls_time_decay(stepwise_curve_with_tiers):
    """
    Ensure that get_spot_price always calls _apply_time_decay_to_steps
    before looking up the tier price.
    """
    curve = stepwise_curve_with_tiers

    with patch.object(curve, "_apply_time_decay_to_steps") as mock_decay:
        price = curve.get_spot_price(Decimal("50"))

    mock_decay.assert_called_once()
    # Because supply=50 => below the first threshold(100) => price=1
    assert price == Decimal("1")



def test_get_spot_price_below_first_threshold(stepwise_curve_with_tiers):
    """
    If supply < first tier threshold => return first tier's price.
    Tiers: [100=>price=1, 200=>2, 300=>3]
    """
    curve = stepwise_curve_with_tiers
    # supply=50 => <100 => price=1
    price = curve.get_spot_price(Decimal("50"))
    assert price == Decimal("1")


def test_get_spot_price_in_middle_tier(stepwise_curve_with_tiers):
    """
    If supply is between 100..200 => tier1 => price=2.
    """
    curve = stepwise_curve_with_tiers
    # supply=150 => >=100 && <200 => price=2
    price = curve.get_spot_price(Decimal("150"))
    assert price == Decimal("2")


def test_get_spot_price_beyond_last_threshold(stepwise_curve_with_tiers):
    """
    If supply >= last threshold => return last tier's price (3).
    """
    curve = stepwise_curve_with_tiers
    # supply=250 => beyond 200 => in tier2 => 300 => price=3
    price = curve.get_spot_price(Decimal("250"))
    assert price == Decimal("3")


def test_get_spot_price_exact_boundary(stepwise_curve_with_tiers):
    """
    If supply == a threshold, we interpret that as '>= that threshold => next tier index'?
    The code does: while supply >= steps[idx].supply_threshold => idx++
    So supply=100 => idx=1 => price=2
    """
    curve = stepwise_curve_with_tiers
    price = curve.get_spot_price(Decimal("100"))
    # Because while supply >= threshold => idx++ => supply=100 >=100 => idx=1 => tier1 => price=2
    assert price == Decimal("2")

def test_calculate_purchase_cost_time_decay_called(stepwise_curve):
    """
    Ensure _apply_time_decay_to_steps is called first,
    then stepwise_cost_for_purchase is invoked with correct arguments.
    """
    # By default, stepwise_curve has:
    #   _state.current_supply=50,
    #   steps=[(100=>1),(200=>2)],
    #   time_decay_rate=0 => ignoring actual time decay but let's ensure it calls the method.

    with patch.object(stepwise_curve, "_apply_time_decay_to_steps") as mock_decay, \
         patch.object(stepwise_helper, "stepwise_cost_for_purchase", return_value=Decimal("123")) as mock_purchase:

        cost = stepwise_curve.calculate_purchase_cost(Decimal("10"))

    # 1) Confirm time decay is called
    mock_decay.assert_called_once()

    # 2) Confirm the helper is called with the correct arguments
    mock_purchase.assert_called_once_with(
        current_supply=Decimal("50"),   # from the fixture’s state
        amount=Decimal("10"),
        steps=stepwise_curve._steps,    # the fixture’s tier list
        allow_partial_fill=False        # by default, 'pro_rata' is False
    )

    # 3) The returned cost is exactly the mock's return
    assert cost == Decimal("123")


def test_calculate_purchase_cost_pro_rata_true(stepwise_curve):
    """
    If pro_rata=True, then allow_partial_fill=True is passed to the helper.
    """
    # Turn on pro_rata
    stepwise_curve.options["pro_rata"] = True

    with patch.object(stepwise_curve, "_apply_time_decay_to_steps"), \
         patch.object(stepwise_helper, "stepwise_cost_for_purchase", return_value=Decimal("45")) as mock_purchase:

        cost = stepwise_curve.calculate_purchase_cost(Decimal("20"))

    mock_purchase.assert_called_once_with(
        current_supply=Decimal("50"),
        amount=Decimal("20"),
        steps=stepwise_curve._steps,
        allow_partial_fill=True
    )
    assert cost == Decimal("45")


def test_calculate_purchase_cost_zero_amount(stepwise_curve):
    """
    If the user passes amount=0 => stepwise_cost_for_purchase typically returns 0 cost.
    """
    with patch.object(stepwise_curve, "_apply_time_decay_to_steps"), \
         patch.object(stepwise_helper, "stepwise_cost_for_purchase", return_value=Decimal("0")) as mock_purchase:

        cost = stepwise_curve.calculate_purchase_cost(Decimal("0"))

    mock_purchase.assert_called_once_with(
        current_supply=Decimal("50"),
        amount=Decimal("0"),
        steps=stepwise_curve._steps,
        allow_partial_fill=False
    )
    assert cost == Decimal("0")


def test_calculate_purchase_cost_negative_amount(stepwise_curve):
    """
    If domain logic doesn't forbid negative amounts, we simply pass them to the helper.
    Or if your domain forbids it, you might test for an exception. This test checks the current code path.
    """
    with patch.object(stepwise_curve, "_apply_time_decay_to_steps"), \
         patch.object(stepwise_helper, "stepwise_cost_for_purchase", return_value=Decimal("-1")) as mock_purchase:

        cost = stepwise_curve.calculate_purchase_cost(Decimal("-5"))

    mock_purchase.assert_called_once_with(
        current_supply=Decimal("50"),
        amount=Decimal("-5"),
        steps=stepwise_curve._steps,
        allow_partial_fill=False
    )
    assert cost == Decimal("-1"), "We just return what the helper returns"


def test_calculate_purchase_cost_return_value(stepwise_curve):
    """
    Verify the final cost returned is exactly what stepwise_cost_for_purchase returns.
    """
    with patch.object(stepwise_curve, "_apply_time_decay_to_steps"), \
         patch.object(stepwise_helper, "stepwise_cost_for_purchase", return_value=Decimal("999.99")) as mock_purchase:

        cost = stepwise_curve.calculate_purchase_cost(Decimal("25"))

    assert cost == Decimal("999.99")


def test_calculate_sale_return_time_decay_called(stepwise_curve):
    """
    Ensures _apply_time_decay_to_steps is called first, then stepwise_return_for_sale.
    """
    with patch.object(stepwise_curve, "_apply_time_decay_to_steps") as mock_decay, \
        patch.object(stepwise_helper, "stepwise_return_for_sale", return_value=Decimal("321")) as mock_sale:
        returned = stepwise_curve.calculate_sale_return(Decimal("10"))

    mock_decay.assert_called_once()
    mock_sale.assert_called_once_with(
        current_supply=Decimal("50"),  # from fixture’s BondingCurveState
        amount=Decimal("10"),
        steps=stepwise_curve._steps,  # fixture’s steps
        allow_partial_fill=False  # default: pro_rata=False
    )
    assert returned == Decimal("321")


def test_calculate_sale_return_pro_rata_true(stepwise_curve):
    """
    If pro_rata=True => allow_partial_fill=True in the helper call.
    """
    stepwise_curve.options["pro_rata"] = True

    with patch.object(stepwise_curve, "_apply_time_decay_to_steps"), \
        patch.object(stepwise_helper, "stepwise_return_for_sale", return_value=Decimal("88")) as mock_sale:
        returned = stepwise_curve.calculate_sale_return(Decimal("20"))

    mock_sale.assert_called_once_with(
        current_supply=Decimal("50"),
        amount=Decimal("20"),
        steps=stepwise_curve._steps,
        allow_partial_fill=True
    )
    assert returned == Decimal("88")


def test_calculate_sale_return_zero_amount(stepwise_curve):
    """
    If user calls with amount=0 => typically returns 0.
    """
    with patch.object(stepwise_curve, "_apply_time_decay_to_steps"), \
        patch.object(stepwise_helper, "stepwise_return_for_sale", return_value=Decimal("0")) as mock_sale:
        returned = stepwise_curve.calculate_sale_return(Decimal("0"))

    mock_sale.assert_called_once_with(
        current_supply=Decimal("50"),
        amount=Decimal("0"),
        steps=stepwise_curve._steps,
        allow_partial_fill=False
    )
    assert returned == Decimal("0")


def test_calculate_sale_return_negative_amount(stepwise_curve):
    """
    If negative amounts aren't explicitly forbidden, we just pass them to the helper.
    Or domain logic might forbid => we test an exception.
    For now, we check the code path just passes negative to the helper.
    """
    with patch.object(stepwise_curve, "_apply_time_decay_to_steps"), \
        patch.object(stepwise_helper, "stepwise_return_for_sale", return_value=Decimal("-5")) as mock_sale:
        returned = stepwise_curve.calculate_sale_return(Decimal("-10"))

    mock_sale.assert_called_once_with(
        current_supply=Decimal("50"),
        amount=Decimal("-10"),
        steps=stepwise_curve._steps,
        allow_partial_fill=False
    )
    assert returned == Decimal("-5")


def test_calculate_sale_return_value(stepwise_curve):
    """
    Verify we return exactly what stepwise_return_for_sale returns.
    """
    with patch.object(stepwise_curve, "_apply_time_decay_to_steps"), \
        patch.object(stepwise_helper, "stepwise_return_for_sale", return_value=Decimal("999.99")) as mock_sale:
        returned = stepwise_curve.calculate_sale_return(Decimal("25"))

    assert returned == Decimal("999.99")


def test_buy_disabled(stepwise_curve_with_liquidity):
    """
    If allow_buy=False, an attempt to buy should raise ValueError.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["allow_buy"] = False

    req = TransactionRequest(
        token=Token("T", "T", 18),
        order_type=OrderSide.BUY,
        amount=Decimal("10")
    )
    with pytest.raises(ValueError, match="Buys are disabled"):
        curve.buy(req)


def test_buy_exceeds_max_supply_no_pro_rata(stepwise_curve_with_liquidity):
    """
    If final_amount > max_supply - current_supply and pro_rata=False => revert.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["max_supply"] = Decimal("550")  # current_supply=500 => capacity=50
    curve.options["pro_rata"] = False

    req = TransactionRequest(
        token=Token("T", "T", 18),
        amount=Decimal("60"),  # more than 50
        order_type=OrderSide.BUY
    )
    with pytest.raises(ValueError, match="exceeds max supply"):
        curve.buy(req)


def test_buy_exceeds_max_supply_pro_rata(stepwise_curve_with_liquidity):
    """
    If final_amount > remaining capacity but pro_rata=True => partial fill => final_amount is partial.
    We'll confirm partial fill logic by mocking 'partial_fill' from common_helper.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["max_supply"] = Decimal("550")  # capacity=50
    curve.options["pro_rata"] = True

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("60"))

    with patch.object(common_helper, "partial_fill", return_value=Decimal("30")) as mock_pf, \
        patch.object(curve, "calculate_purchase_cost", return_value=Decimal("100")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("100")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("105")) as mock_fee:
        result = curve.buy(req)

    # partial_fill should be called to reduce from 60 to 30
    mock_pf.assert_called_once_with(Decimal("60"), Decimal("50"))
    assert result.executed_amount == Decimal("30")
    assert result.total_cost == Decimal("105")
    # supply => 500 + 30=530
    assert curve._state.current_supply == Decimal("530")
    # liquidity => 1000 + 105=1105
    assert curve._state.liquidity.amount == Decimal("1105")


def test_buy_final_amount_zero(stepwise_curve_with_liquidity):
    """
    If final_amount=0 after partial fill or max supply=0 => trivial transaction => returns 0 cost
    and does not update state.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["max_supply"] = Decimal("500")  # current_supply=500 => capacity=0
    curve.options["pro_rata"] = True  # => final_amount=0

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    result = curve.buy(req)
    assert result.executed_amount == Decimal("0")
    assert result.total_cost == Decimal("0")
    assert result.new_supply == Decimal("500")  # no change
    # liquidity remains unchanged
    assert curve._state.liquidity.amount == Decimal("1000")


def test_buy_calculate_purchase_cost(stepwise_curve_with_liquidity):
    """
    Normal scenario => calls 'calculate_purchase_cost'. We can mock the result from that
    and check flow.
    """
    curve = stepwise_curve_with_liquidity
    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("5"))

    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("50")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("50")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("52")) as mock_fee:
        result = curve.buy(req)

    mock_calc.assert_called_once_with(Decimal("5"))
    # final amount => 5 => total cost => 52 => supply=505 => liquidity= 1000+52=1052
    assert result.executed_amount == Decimal("5")
    assert result.total_cost == Decimal("52")
    assert curve._state.current_supply == Decimal("505")
    assert curve._state.liquidity.amount == Decimal("1052")


def test_buy_apply_risk_profile(stepwise_curve_with_liquidity):
    """
    If risk_profile is set => apply_risk_profile is invoked on the raw_cost.
    We'll confirm we pass is_buy=True.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["risk_profile"] = "AGGRESSIVE"  # or enum
    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("100")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("95")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("97")) as mock_fee:
        result = curve.buy(req)

    mock_risk.assert_called_once_with(Decimal("100"), "AGGRESSIVE", is_buy=True)
    assert result.total_cost == Decimal("97")  # after fee
    assert result.executed_amount == Decimal("10")


def test_buy_exceeds_max_liquidity_no_pro_rata(stepwise_curve_with_liquidity):
    """
    If adding 'raw_cost' to current liquidity exceeds max_liquidity and pro_rata=False => revert.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["max_liquidity"] = Decimal("1050")  # only 50 space left
    curve.options["pro_rata"] = False

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))
    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("60")) as mock_calc:
        with pytest.raises(ValueError, match="Max liquidity reached"):
            curve.buy(req)


def test_buy_exceeds_max_liquidity_pro_rata(stepwise_curve_with_liquidity):
    """
    If raw_cost + liquidity.amount > max_liquidity but pro_rata=True => partial fill.
    We scale final_amount proportionally. Then the leftover is unsold.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["max_liquidity"] = Decimal("1050")  # 50 space left
    curve.options["pro_rata"] = True

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    # Suppose raw_cost=60 => we only have 50 space => fraction= 50/60=0.833..., final_amount= 10*0.833..=8.333..., cost=50
    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("60")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("60")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("50")) as mock_fee:
        result = curve.buy(req)

    # final_amount => ~8.333..., total_cost => 50 => supply= 508.3333..., liquidity=1050
    assert result.executed_amount < Decimal("10")
    assert result.total_cost == Decimal("50")
    assert curve._state.liquidity.amount == Decimal("1050")
    assert curve._state.current_supply > Decimal("500")


def test_buy_slippage_revert(stepwise_curve_with_liquidity):
    """
    If raw_cost > baseline_cost*(1+slippage) => revert if partial_fill_approach=2
    or partial fill if approach=1. We'll do revert scenario.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["slippage_tolerance"] = Decimal("0.05")  # 5%
    curve.options["partial_fill_approach"] = 2  # => revert approach

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    # raw_cost=110, baseline=100 => threshold=100*(1+0.05)=105 => 110>105 => revert
    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("110")) as mock_calc, \
        patch.object(stepwise_helper, "stepwise_cost_for_purchase", return_value=Decimal("100")) as mock_baseline:
        with pytest.raises(ValueError, match="Slippage tolerance exceeded"):
            curve.buy(req)


def test_buy_slippage_partial_fill(stepwise_curve_with_liquidity):
    """
    If raw_cost > baseline*(1+slip) => partial fill if partial_fill_approach=1 => scale_for_slippage => returns (some_new_cost, False).
    """
    curve = stepwise_curve_with_liquidity
    curve.options["slippage_tolerance"] = Decimal("0.05")
    curve.options["partial_fill_approach"] = 1

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    # raw_cost=110, baseline=100 => threshold=105 => partial => returns final_cost=105 => revert_flag=False
    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("110")), \
        patch.object(stepwise_helper, "stepwise_cost_for_purchase", return_value=Decimal("100")) as mock_baseline, \
        patch.object(common_helper, "scale_for_slippage", return_value=(Decimal("105"), False)) as mock_scale, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("106")) as mock_fee:
        result = curve.buy(req)

    mock_scale.assert_called_once_with(
        Decimal("110"),
        Decimal("100"),
        Decimal("0.05"),
        is_buy=True,
        approach=1
    )
    # final cost => 106 => executed_amount => still 10 => we didn't recalc final_amount in scale_for_slippage
    # domain might expect partial fill => final_amount <10 if we truly scale.
    # The code doesn't recalc final_amount though. Let's assume your code lumps that in the cost alone.
    assert result.executed_amount == Decimal("10")
    assert result.total_cost == Decimal("106")


def test_buy_transaction_fee(stepwise_curve_with_liquidity):
    """
    If txn_fee_rate>0 => final_cost includes the fee.
    We'll confirm apply_transaction_fee is called with is_buy=True
    """
    curve = stepwise_curve_with_liquidity
    curve.options["txn_fee_rate"] = Decimal("0.02")  # 2%

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("5"))

    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("100")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("100")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("102")) as mock_fee:
        result = curve.buy(req)

    mock_fee.assert_called_once_with(Decimal("100"), Decimal("0.02"), is_buy=True)
    assert result.total_cost == Decimal("102")
    assert curve._state.current_supply == Decimal("505")
    assert curve._state.liquidity.amount == Decimal("1102")


def test_buy_success_minimal(stepwise_curve_with_liquidity):
    """
    A simpler scenario with no max_supply, no max_liquidity, no slippage => end-to-end with actual cost logic
    or a simpler partial mock if needed. We'll do partial mocking to confirm final state is updated.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["pro_rata"] = False  # no partial
    # by default no max_supply, no max_liquidity, no slippage => no revert

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("50")) as mock_calc, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("52")) as mock_fee:
        result = curve.buy(req)

    # final amount => 10 => total cost => 52 => supply => 510 => liquidity => 1000+52=1052
    assert result.executed_amount == Decimal("10")
    assert result.total_cost == Decimal("52")
    assert result.new_supply == Decimal("510")
    assert curve._state.current_supply == Decimal("510")
    assert curve._state.liquidity.amount == Decimal("1052")
    # average_price => 52/10= 5.2
    assert result.average_price == Decimal("5.2")
    assert result.timestamp is not None


def test_sell_disabled(stepwise_curve_with_liquidity):
    """
    If allow_sell=False, calling sell(...) should raise ValueError.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["allow_sell"] = False

    req = TransactionRequest(
        token=Token("S", "S", 18),
        order_type=OrderSide.SELL,
        amount=Decimal("10")
    )
    with pytest.raises(ValueError, match="Sells are disabled"):
        curve.sell(req)


def test_sell_normal_scenario(stepwise_curve_with_liquidity):
    """
    A typical scenario: we mock calculate_sale_return, then apply risk, fees, etc.
    Confirm the final state is updated.
    """
    curve = stepwise_curve_with_liquidity
    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("80")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("85")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("83")) as mock_fee:
        result = curve.sell(req)

    # final_amount => 10 => final_return => 83 => supply => 490 => liquidity => 1000-83=917
    assert result.executed_amount == Decimal("10")
    assert result.total_cost == Decimal("83")
    assert curve._state.current_supply == Decimal("490")
    assert curve._state.liquidity.amount == Decimal("917")
    # average_price => total_return / tokens => 83/10=8.3
    assert result.average_price == Decimal("8.3")
    assert result.timestamp is not None


def test_sell_insufficient_liquidity_no_pro_rata(stepwise_curve_with_liquidity):
    """
    If raw_return > current_liquidity and pro_rata=False => revert.
    """
    curve = stepwise_curve_with_liquidity
    # current_liquidity => 1000 by default, let's pretend raw_return=1500 => not enough
    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))
    with patch.object(curve, "calculate_sale_return", return_value=Decimal("1500")):
        with pytest.raises(ValueError, match="Insufficient liquidity"):
            curve.sell(req)


def test_sell_insufficient_liquidity_pro_rata(stepwise_curve_with_liquidity):
    """
    If raw_return > current_liquidity but pro_rata=True => partial fill => fraction= partial_fill(...).
    Then final_amount & raw_return are scaled down.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["pro_rata"] = True
    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))

    # Suppose raw_return=1200 => current_liquid=1000 => partial => fraction=1000/1200=0.8333 => final_amount=10*0.8333=8.3333 => raw_return=1000
    with patch.object(curve, "calculate_sale_return", return_value=Decimal("1200")), \
        patch.object(common_helper, "partial_fill", return_value=Decimal("1000")) as mock_pf, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("990")) as mock_fee:
        result = curve.sell(req)

    mock_pf.assert_called_once_with(Decimal("1200"), Decimal("1000"))
    # final_amount => 8.333..., final_return => 990 => supply => 491.666..., liquidity => 1000-990=10
    # The code calculates fraction_of_tokens by fraction/ raw_return => fraction=1000 => raw_return=1200 => fraction_of_tokens= 1000/1200=0.8333
    assert result.executed_amount < Decimal("10")
    assert result.total_cost == Decimal("990")
    assert curve._state.current_supply < Decimal("500")
    assert curve._state.liquidity.amount == Decimal("10")


def test_sell_slippage_revert(stepwise_curve_with_liquidity):
    """
    If raw_return < baseline*(1 - slip) => revert if partial_fill_approach=2 => no partial fill.
    We'll do a scenario with approach=2 => revert.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["slippage_tolerance"] = Decimal("0.05")  # 5%
    curve.options["partial_fill_approach"] = 2  # revert approach

    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))

    # raw_return=80, baseline=100 => threshold= 100*(1-0.05)=95 => 80<95 => revert
    with patch.object(curve, "calculate_sale_return", return_value=Decimal("80")) as mock_calc, \
        patch.object(stepwise_helper, "stepwise_return_for_sale", return_value=Decimal("100")):
        with pytest.raises(ValueError, match="Slippage tolerance not met"):
            curve.sell(req)


def test_sell_slippage_partial_fill(stepwise_curve_with_liquidity):
    """
    If raw_return < baseline*(1 - slip) => partial fill if partial_fill_approach=1 => scale_for_slippage => returns (some_new_ret,False).
    """
    curve = stepwise_curve_with_liquidity
    curve.options["slippage_tolerance"] = Decimal("0.05")
    curve.options["partial_fill_approach"] = 1

    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))

    # raw_return=80, baseline=100 => threshold=95 => 80<95 => partial => final_ret=90 => revert_flag=False => code sets raw_return=90
    with patch.object(curve, "calculate_sale_return", return_value=Decimal("80")), \
        patch.object(stepwise_helper, "stepwise_return_for_sale", return_value=Decimal("100")), \
        patch.object(common_helper, "scale_for_slippage", return_value=(Decimal("90"), False)), \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("88")):
        result = curve.sell(req)

    # final return => 88 => supply => 490 +some fraction? Actually the code sets final_amount if partial fill or insufficient liquidity
    # This test checks we accept the scaled final_ret=90 => then fee => 88 => final_amount is not updated here unless partial fill from insufficient liq.
    assert result.total_cost == Decimal("88")


def test_sell_risk_profile(stepwise_curve_with_liquidity):
    """
    If risk_profile => we call apply_risk_profile(raw_return, risk_profile, is_buy=False).
    """
    curve = stepwise_curve_with_liquidity
    curve.options["risk_profile"] = "CONSERVATIVE"

    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("5"))

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("50")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("42.5")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("40")) as mock_fee:
        result = curve.sell(req)

    mock_risk.assert_called_once_with(Decimal("50"), "CONSERVATIVE", is_buy=False)
    assert result.total_cost == Decimal("40")


def test_sell_transaction_fee(stepwise_curve_with_liquidity):
    """
    If txn_fee_rate>0 => final_return = raw_return - fee. We'll confirm apply_transaction_fee is called with is_buy=False.
    """
    curve = stepwise_curve_with_liquidity
    curve.options["txn_fee_rate"] = Decimal("0.02")

    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("5"))

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("100")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("100")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("98")) as mock_fee:
        result = curve.sell(req)

    mock_fee.assert_called_once_with(Decimal("100"), Decimal("0.02"), is_buy=False)
    assert result.total_cost == Decimal("98")
    # supply => 495 => liquidity => 902
    assert curve._state.current_supply == Decimal("495")
    assert curve._state.liquidity.amount == Decimal("902")


def test_sell_success_minimal(stepwise_curve_with_liquidity):
    """
    A simpler end-to-end scenario with no partial fill, no max liquidity => normal sell => updates state.
    """
    curve = stepwise_curve_with_liquidity
    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("40")) as mock_calc, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("38")) as mock_fee:
        result = curve.sell(req)

    assert result.executed_amount == Decimal("10")
    assert result.total_cost == Decimal("38")
    # supply => 500-10=490 => liquidity=> 1000-38=962
    assert result.new_supply == Decimal("490")
    assert curve._state.current_supply == Decimal("490")
    assert curve._state.liquidity.amount == Decimal("962")
    assert result.average_price == Decimal("3.8")  # 38/10
    assert result.timestamp is not None
