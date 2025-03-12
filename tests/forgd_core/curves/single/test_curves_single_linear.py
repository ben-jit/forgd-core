import pytest

from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

from forgd_core.common.model import BondingCurveParams, BondingCurveState, Token, Liquidity, TransactionRequest
from forgd_core.common.enums import BondingCurveType, OrderSide, BondingCurveDistribution
from forgd_core.curves.single.base import BondingCurve
from forgd_core.curves.single.linear import LinearBondingCurve
from forgd_core.curves.helpers.common import CommonCurveHelper as common_helper
from forgd_core.curves.helpers.linear import LinearCurveHelper as linear_helper


@pytest.fixture
def linear_params():
    """
    Returns a minimal valid BondingCurveParams object for a linear curve.
    """
    return BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.1")
    )


def _make_linear_curve_with_state_liquidity(
    initial_supply=Decimal("100"),
    initial_liquidity=Decimal("1000")
) -> LinearBondingCurve:
    """
    Helper to create a LinearBondingCurve with a BondingCurveState
    that may or may not include liquidity.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.1"),
    )

    # If initial_liquidity is not None, create a Liquidity object
    liquidity_obj = None
    if initial_liquidity is not None:
        token = Token(name="TestToken", symbol="TT", decimals=18)
        liquidity_obj = Liquidity(token=token, amount=initial_liquidity)

    state = BondingCurveState(
        current_supply=initial_supply,
        current_price=Decimal("2.0"),
        liquidity=liquidity_obj
    )
    return LinearBondingCurve(params, state)



def _make_linear_curve_simple(current_supply=Decimal("100")) -> LinearBondingCurve:
    """
    Small helper to create a LinearBondingCurve with a given current_supply.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.1")
    )
    state = BondingCurveState(current_supply=current_supply)
    return LinearBondingCurve(params, state)


def _make_linear_curve_buy(
    current_supply=Decimal("100"),
    liquidity_amount=Decimal("1000"),
    allow_buy=True,
    pro_rata=True,
    max_supply=None,
    max_liquidity=None,
    slippage_tolerance=Decimal("0"),
    txn_fee_rate=Decimal("0"),
    time_decay_rate=Decimal("0"),
    risk_profile=None,
    partial_fill_approach=1,
):
    """
    Helper to create a LinearBondingCurve with various default or overridden options.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1"),
        slope=Decimal("0.1"),
    )

    # Build a Liquidity object if liquidity_amount is not None
    liquidity_obj = None
    if liquidity_amount is not None:
        token = Token("MockToken", "MTK", 18)
        liquidity_obj = Liquidity(token=token, amount=liquidity_amount)

    state = BondingCurveState(
        current_supply=current_supply,
        current_price=Decimal("2"),
        liquidity=liquidity_obj
    )

    curve = LinearBondingCurve(params, state,
        allow_buy=allow_buy,
        pro_rata=pro_rata,
        max_supply=max_supply,
        max_liquidity=max_liquidity,
        slippage_tolerance=slippage_tolerance,
        txn_fee_rate=txn_fee_rate,
        time_decay_rate=time_decay_rate,
        risk_profile=risk_profile,
        partial_fill_approach=partial_fill_approach
    )
    return curve


def _make_linear_curve_sell(
    current_supply=Decimal("100"),
    liquidity_amount=Decimal("1000"),
    allow_sell=True,
    pro_rata=False,
    slippage_tolerance=Decimal("0"),
    txn_fee_rate=Decimal("0"),
    risk_profile=None,
    partial_fill_approach=1,
):
    """
    Helper to create a LinearBondingCurve with default or overridden options for selling.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.1"),
    )

    # Create liquidity if liquidity_amount is not None
    liquidity_obj = None
    if liquidity_amount is not None:
        token = Token("MockToken", "MTK", 18)
        liquidity_obj = Liquidity(token=token, amount=liquidity_amount)

    state = BondingCurveState(
        current_supply=current_supply,
        current_price=Decimal("2.0"),
        liquidity=liquidity_obj
    )

    curve = LinearBondingCurve(params, state,
        allow_sell=allow_sell,
        pro_rata=pro_rata,
        slippage_tolerance=slippage_tolerance,
        txn_fee_rate=txn_fee_rate,
        risk_profile=risk_profile,
        partial_fill_approach=partial_fill_approach
    )
    return curve


def test_init_no_state(linear_params):
    """
    If no state is passed, BondingCurveState should be automatically created.
    """
    curve = LinearBondingCurve(params=linear_params)
    assert curve._state is not None, "A default BondingCurveState should be created if none is passed."
    assert curve._state.last_timestamp is not None, "last_timestamp should be set if it was None."


def test_init_with_state_no_last_timestamp(linear_params):
    """
    If a state is passed without last_timestamp, it should be set to now().
    We'll mock datetime.now() to a fixed value to test this.
    """
    mock_now = datetime(2025, 3, 10, 12, 0, 0)
    with patch("forgd_core.curves.single.linear.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_now

        custom_state = BondingCurveState(current_supply=Decimal("100"), current_price=Decimal("2.0"))
        assert custom_state.last_timestamp is None, "Precondition: state has no timestamp."

        curve = LinearBondingCurve(params=linear_params, state=custom_state)
        assert curve._state is not None
        assert curve._state.current_supply == Decimal("100")
        assert curve._state.current_price == Decimal("2.0")

        # last_timestamp should now be set to mock_now
        assert curve._state.last_timestamp == mock_now


def test_init_with_state_existing_timestamp(linear_params):
    """
    If a state is passed with an existing last_timestamp, it should remain unchanged.
    """
    existing_ts = datetime(2025, 1, 1, 12, 0, 0)
    custom_state = BondingCurveState(
        current_supply=Decimal("500"),
        current_price=Decimal("3.0"),
        last_timestamp=existing_ts
    )

    curve = LinearBondingCurve(params=linear_params, state=custom_state)
    assert curve._state.last_timestamp == existing_ts, "Should preserve existing timestamp."


def test_init_with_options(linear_params):
    """
    If recognized kwargs are passed, they should override defaults in self.options.
    Unrecognized kwargs should be stored in 'custom'.
    """
    curve = LinearBondingCurve(
        params=linear_params,
        state=None,
        pro_rata=True,            # recognized
        max_supply=Decimal("1000"),  # recognized
        foo="bar",                # unrecognized => goes into custom
        slippage_tolerance=Decimal("0.05"),  # recognized
    )

    # Check recognized fields
    assert curve.options["pro_rata"] is True
    assert curve.options["max_supply"] == Decimal("1000")
    assert curve.options["slippage_tolerance"] == Decimal("0.05")

    # Check unrecognized field
    assert "custom" in curve.options, "A 'custom' sub-dict should exist for unrecognized keys."
    assert curve.options["custom"]["foo"] == "bar", "Unrecognized option 'foo' should be in custom sub-dict."


def test_init_default_options(linear_params):
    """
    Verify the default options are set correctly if no overrides are provided.
    """
    curve = LinearBondingCurve(params=linear_params)
    assert curve.options["pro_rata"] is False
    assert curve.options["max_supply"] is None
    assert curve.options["allow_buy"] is True
    assert curve.options["allow_sell"] is True
    assert curve.options["slippage_tolerance"] == Decimal("0")
    assert curve.options["txn_fee_rate"] == Decimal("0")
    assert curve.options["time_decay_rate"] == Decimal("0")
    assert curve.options["risk_profile"] is None
    assert curve.options["partial_fill_approach"] == 1
    assert curve.options["time_decay_approach"] == 1
    assert "custom" not in curve.options, "No custom sub-dict unless we supply unknown kwargs."


def test_time_decay_no_decay():
    """
    Scenario: time_decay_rate = 0 => No change to slope or intercept.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.1"),
    )
    old_ts = datetime(2025, 3, 10, 12, 0)
    state = BondingCurveState(
        current_supply=Decimal("100"),
        current_price=Decimal("2.0"),
        last_timestamp=old_ts
    )

    # Initialize curve with zero time_decay_rate
    curve = LinearBondingCurve(params, state, time_decay_rate=Decimal("0"), time_decay_approach=1)

    # Suppose helper returns exactly the same slope/intercept/timestamp
    mock_return = (Decimal("1.0"), Decimal("0.1"), old_ts)

    with patch.object(linear_helper, "apply_time_decay", return_value=mock_return) as mock_helper:
        new_i, new_m = curve._apply_time_decay()

    # Check the call arguments
    mock_helper.assert_called_once()
    args, kwargs = mock_helper.call_args
    assert args[0] == Decimal("1.0")  # initial_price
    assert args[1] == Decimal("0.1")  # slope
    assert args[2] == old_ts         # old timestamp from state
    assert args[3] == Decimal("0")   # time_decay_rate
    assert kwargs["approach"] == 1   # time_decay_approach

    # Check the resulting updates
    # Because mock_return has the same old_ts, we expect last_timestamp is unchanged
    assert curve._state.last_timestamp == old_ts
    assert new_i == Decimal("1.0") and new_m == Decimal("0.1")


def test_time_decay_approach1_positive():
    """
    Scenario: time_decay_rate > 0, approach=1 => typical additive approach.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.1"),
    )
    old_ts = datetime(2025, 3, 10, 12, 0)
    state = BondingCurveState(
        current_supply=Decimal("100"),
        current_price=Decimal("2.0"),
        last_timestamp=old_ts
    )

    curve = LinearBondingCurve(
        params,
        state,
        time_decay_rate=Decimal("0.05"),
        time_decay_approach=1
    )

    # The helper says new intercept=1.2, new slope=0.15, new timestamp=2025-03-10 13:00
    mock_return = (
        Decimal("1.2"),
        Decimal("0.15"),
        datetime(2025, 3, 10, 13, 0)
    )

    with patch.object(linear_helper, "apply_time_decay", return_value=mock_return) as mock_helper:
        new_i, new_m = curve._apply_time_decay()

    mock_helper.assert_called_once()
    args, kwargs = mock_helper.call_args
    # Verify old timestamp
    assert args[2] == old_ts

    # The new timestamp is the third item from mock_return
    assert curve._state.last_timestamp == mock_return[2]
    assert new_i == Decimal("1.2")
    assert new_m == Decimal("0.15")


def test_time_decay_approach2_positive():
    """
    Scenario: time_decay_rate > 0, approach=2 => multiplicative approach.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("2.0"),
        slope=Decimal("0.5"),
    )
    old_ts = datetime(2025, 3, 11, 12, 0)
    state = BondingCurveState(
        current_supply=Decimal("1000"),
        current_price=Decimal("2.0"),
        last_timestamp=old_ts
    )

    curve = LinearBondingCurve(
        params,
        state,
        time_decay_rate=Decimal("0.1"),
        time_decay_approach=2
    )

    # Suppose the helper returns new_i=2.1, new_m=0.55, and a brand-new timestamp
    mock_return = (
        Decimal("2.1"),
        Decimal("0.55"),
        datetime(2026, 1, 1, 0, 0, 0)
    )

    with patch.object(linear_helper, "apply_time_decay", return_value=mock_return) as mock_helper:
        new_i, new_m = curve._apply_time_decay()

    args, kwargs = mock_helper.call_args
    # The old timestamp is still 2025-03-11
    assert args[2] == old_ts

    assert curve._state.last_timestamp == mock_return[2]
    assert new_i == Decimal("2.1")
    assert new_m == Decimal("0.55")


def test_time_decay_approach_unrecognized():
    """
    Scenario: time_decay_rate != 0, approach=999 => the helper might do no slope change,
    but still updates the timestamp.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.2"),
    )
    old_ts = datetime(2025, 6, 1, 10, 0)
    state = BondingCurveState(
        current_supply=Decimal("500"),
        current_price=Decimal("1.5"),
        last_timestamp=old_ts
    )

    curve = LinearBondingCurve(
        params,
        state,
        time_decay_rate=Decimal("0.2"),
        time_decay_approach=999
    )

    # Suppose approach=999 leads the helper to basically keep same slope, but updates timestamp anyway
    mock_return = (
        Decimal("1.0"),   # intercept
        Decimal("0.2"),   # slope unchanged
        datetime(2025, 6, 2, 12, 0)  # new timestamp
    )

    with patch.object(linear_helper, "apply_time_decay", return_value=mock_return) as mock_helper:
        new_i, new_m = curve._apply_time_decay()

    args, kwargs = mock_helper.call_args
    assert args[2] == old_ts

    assert curve._state.last_timestamp == mock_return[2]
    # The slope didn't change, so new_i=1.0, new_m=0.2
    assert new_i == Decimal("1.0")
    assert new_m == Decimal("0.2")


def test_update_state_after_buy_with_liquidity():
    """
    Test that _update_state_after_buy correctly increments:
      1) current_supply
      2) liquidity.amount
    if liquidity is present.
    """
    curve = _make_linear_curve_with_state_liquidity(
        initial_supply=Decimal("100"),
        initial_liquidity=Decimal("1000")
    )

    buy_amount = Decimal("10")
    total_cost = Decimal("50")

    curve._update_state_after_buy(amount=buy_amount, total_cost=total_cost)

    # Check current_supply increment
    assert curve._state.current_supply == Decimal("110"), (
        "current_supply should increase by buy_amount"
    )
    # Check liquidity increment
    assert curve._state.liquidity.amount == Decimal("1050"), (
        "liquidity.amount should increase by total_cost"
    )


def test_update_state_after_buy_no_liquidity():
    """
    If self._state.liquidity is None, only current_supply should be updated.
    """
    curve = _make_linear_curve_with_state_liquidity(
        initial_supply=Decimal("100"),
        initial_liquidity=None  # No liquidity object
    )

    buy_amount = Decimal("5")
    total_cost = Decimal("20")

    curve._update_state_after_buy(amount=buy_amount, total_cost=total_cost)

    # Check current_supply increment
    assert curve._state.current_supply == Decimal("105"), (
        "current_supply should increase by buy_amount"
    )
    # Liquidity is None => no error, no update
    assert curve._state.liquidity is None


def test_update_state_after_sell_with_liquidity():
    """
    Test that _update_state_after_sell correctly decrements:
      1) current_supply
      2) liquidity.amount
    if liquidity is present.
    """
    curve = _make_linear_curve_with_state_liquidity(
        initial_supply=Decimal("100"),
        initial_liquidity=Decimal("1000")
    )

    sell_amount = Decimal("25")
    total_return = Decimal("70")

    curve._update_state_after_sell(amount=sell_amount, total_return=total_return)

    # Check current_supply decrement
    assert curve._state.current_supply == Decimal("75"), (
        "current_supply should decrease by sell_amount"
    )
    # Check liquidity decrement
    assert curve._state.liquidity.amount == Decimal("930"), (
        "liquidity.amount should decrease by total_return"
    )


def test_update_state_after_sell_no_liquidity():
    """
    If self._state.liquidity is None, only current_supply should be updated.
    """
    curve = _make_linear_curve_with_state_liquidity(
        initial_supply=Decimal("100"),
        initial_liquidity=None
    )

    sell_amount = Decimal("10")
    total_return = Decimal("25")

    curve._update_state_after_sell(amount=sell_amount, total_return=total_return)

    # Check current_supply decrement
    assert curve._state.current_supply == Decimal("90"), (
        "current_supply should decrease by sell_amount"
    )
    # Liquidity is None => no error, no update
    assert curve._state.liquidity is None


def test_get_spot_price_simple():
    """
    Test get_spot_price with a simple scenario:
    _apply_time_decay() is mocked to return (i=1, m=0.1).
    Suppose supply=100 => spot_price=1 + (0.1*100)=1+10=11.
    """
    # Prepare a minimal curve
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("0"),  # Will be overridden by mock
        slope=Decimal("0")          # Will be overridden by mock
    )
    state = BondingCurveState(
        current_supply=Decimal("100"),
        current_price=Decimal("0")    # Not used in this function
    )
    curve = LinearBondingCurve(params, state)

    # Mock _apply_time_decay to return (i=1, m=0.1)
    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("1"), Decimal("0.1"))) as mock_decay:
        spot_price = curve.get_spot_price(Decimal("100"))

    mock_decay.assert_called_once()
    assert spot_price == Decimal("11"), "Expected 1 + (0.1*100)=11"


def test_get_spot_price_different_values():
    """
    Another scenario:
    _apply_time_decay() => (i=2.5, m=0.05)
    supply=10 => 2.5 + (0.05*10)=3.0
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("0"),  # Will be overridden by mock
        slope=Decimal("0")
    )
    state = BondingCurveState(
        current_supply=Decimal("500"),  # Not directly used here
        current_price=Decimal("2.0")
    )
    curve = LinearBondingCurve(params, state)

    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("2.5"), Decimal("0.05"))) as mock_decay:
        spot_price = curve.get_spot_price(Decimal("10"))

    mock_decay.assert_called_once()
    # Calculation: 2.5 + 0.05*10= 2.5+0.5= 3.0
    assert spot_price == Decimal("3.0"), "Expected 3.0 for i=2.5, m=0.05, supply=10."


def test_get_spot_price_zero_supply():
    """
    Test with supply=0:
    The result should just be i if supply=0.
    """
    params = BondingCurveParams(curve_type=BondingCurveType.LINEAR)
    state = BondingCurveState()

    curve = LinearBondingCurve(params, state)

    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("5"), Decimal("10"))) as mock_decay:
        spot_price = curve.get_spot_price(Decimal("0"))

    mock_decay.assert_called_once()
    # Calculation: 5 + (10*0)=5
    assert spot_price == Decimal("5"), "If supply=0, spot_price should be i=5."


def test_get_spot_price_negative_supply():
    """
    If negative supply is passed, the formula is still i + m*supply mathematically.
    Not typical, but let's confirm the code just handles the arithmetic
    (or decide if your domain logic disallows it).
    """
    params = BondingCurveParams(curve_type=BondingCurveType.LINEAR)
    state = BondingCurveState()

    curve = LinearBondingCurve(params, state)

    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("2"), Decimal("1"))) as mock_decay:
        # supply= -10 => 2 + (1*-10)=2-10= -8
        spot_price = curve.get_spot_price(Decimal("-10"))

    mock_decay.assert_called_once()
    assert spot_price == Decimal("-8"), (
        "Mathematically 2 + (1 * -10)= -8. Domain logic might forbid negative supply, but let's test it anyway."
    )


def test_calculate_purchase_cost_simple():
    """
    Scenario: _apply_time_decay returns (i=1, m=0.1).
    current_supply=100, amount=10 => start=100, end=110.
    We expect cost_between(100, 110, 1, 0.1) to be called.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.1"),
    )
    state = BondingCurveState(current_supply=Decimal("100"))
    curve = LinearBondingCurve(params, state)

    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("1"), Decimal("0.1"))) as mock_decay, \
        patch.object(linear_helper, "cost_between", return_value=Decimal("999")) as mock_cost:
        cost = curve.calculate_purchase_cost(Decimal("10"))

    mock_decay.assert_called_once(), "Should call _apply_time_decay once."
    # Check cost_between was called with start=100, end=110, i=1, m=0.1
    mock_cost.assert_called_once_with(Decimal("100"), Decimal("110"), Decimal("1"), Decimal("0.1"))
    assert cost == Decimal("999"), "Should return the value from cost_between."


def test_calculate_purchase_cost_zero_amount():
    """
    If amount=0 => start_supply=end_supply => cost_between(100,100,i,m) => presumably 0 cost.
    """
    params = BondingCurveParams(curve_type=BondingCurveType.LINEAR)
    state = BondingCurveState(current_supply=Decimal("100"))
    curve = LinearBondingCurve(params, state)

    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("2"), Decimal("0.5"))) as mock_decay, \
        patch.object(linear_helper, "cost_between", return_value=Decimal("0")) as mock_cost:
        cost = curve.calculate_purchase_cost(Decimal("0"))

    mock_decay.assert_called_once()
    mock_cost.assert_called_once_with(Decimal("100"), Decimal("100"), Decimal("2"), Decimal("0.5"))
    assert cost == Decimal("0")


def test_calculate_purchase_cost_negative_amount():
    """
    If a negative amount is given, the formula results in end_supply < start_supply.
    Not typical in a 'purchase', but let's see if the code just calls cost_between
    or if domain logic forbids it.
    """
    params = BondingCurveParams(curve_type=BondingCurveType.LINEAR)
    state = BondingCurveState(current_supply=Decimal("1000"))
    curve = LinearBondingCurve(params, state)

    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("2"), Decimal("0.1"))) as mock_decay, \
        patch.object(linear_helper, "cost_between", return_value=Decimal("-123")) as mock_cost:
        # -50 => start=1000, end=950 => cost_between(1000,950,2,0.1).
        cost = curve.calculate_purchase_cost(Decimal("-50"))

    mock_decay.assert_called_once()
    mock_cost.assert_called_once_with(Decimal("1000"), Decimal("950"), Decimal("2"), Decimal("0.1"))
    assert cost == Decimal("-123"), "Returns whatever cost_between returns, even if negative."


def test_calculate_purchase_cost_large_amount():
    """
    Large purchase scenario: amount=5000, just to ensure the math is scaled up.
    """
    params = BondingCurveParams(curve_type=BondingCurveType.LINEAR)
    state = BondingCurveState(current_supply=Decimal("100"))
    curve = LinearBondingCurve(params, state)

    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("5"), Decimal("1"))) as mock_decay, \
        patch.object(linear_helper, "cost_between", return_value=Decimal("999999")) as mock_cost:
        cost = curve.calculate_purchase_cost(Decimal("5000"))

    mock_decay.assert_called_once()
    # cost_between(100,5100,5,1)
    mock_cost.assert_called_once_with(Decimal("100"), Decimal("5100"), Decimal("5"), Decimal("1"))
    assert cost == Decimal("999999")


def test_calculate_sale_return_normal():
    """
    Normal scenario where amount <= current_supply.
    If current_supply=100, amount=10 => start=90, end=100 => cost_between(90,100,i,m).
    """
    curve = _make_linear_curve_simple(Decimal("100"))

    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("2"), Decimal("0.1"))) as mock_decay, \
         patch.object(linear_helper, "cost_between", return_value=Decimal("50")) as mock_cost:

        sale_return = curve.calculate_sale_return(Decimal("10"))

    mock_decay.assert_called_once()
    mock_cost.assert_called_once_with(
        Decimal("90"),  # start_supply=100 - 10
        Decimal("100"), # end_supply=100
        Decimal("2"),   # i
        Decimal("0.1")  # m
    )
    assert sale_return == Decimal("50"), "Should return the value from cost_between."


def test_calculate_sale_return_zero():
    """
    Selling zero tokens => start=end => cost_between(100,100,i,m)=0 typically.
    """
    curve = _make_linear_curve_simple(Decimal("100"))

    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("5"), Decimal("0.2"))) as mock_decay, \
         patch.object(linear_helper, "cost_between", return_value=Decimal("0")) as mock_cost:

        sale_return = curve.calculate_sale_return(Decimal("0"))

    mock_decay.assert_called_once()
    mock_cost.assert_called_once_with(
        Decimal("100"), Decimal("100"), Decimal("5"), Decimal("0.2")
    )
    assert sale_return == Decimal("0")


def test_calculate_sale_return_full_supply():
    """
    Selling the entire supply => start=0, end=100 => cost_between(0,100,i,m).
    """
    curve = _make_linear_curve_simple(Decimal("100"))

    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("1"), Decimal("0.05"))) as mock_decay, \
         patch.object(linear_helper, "cost_between", return_value=Decimal("999")) as mock_cost:

        sale_return = curve.calculate_sale_return(Decimal("100"))

    mock_decay.assert_called_once()
    mock_cost.assert_called_once_with(
        Decimal("0"),   # start_supply=100-100=0
        Decimal("100"), # end_supply=100
        Decimal("1"),   # i
        Decimal("0.05") # m
    )
    assert sale_return == Decimal("999")


def test_calculate_sale_return_exceed_supply_raises():
    """
    Attempting to sell more tokens than current_supply => raises ValueError.
    """
    curve = _make_linear_curve_simple(Decimal("50"))

    with pytest.raises(ValueError, match="Cannot sell more tokens than current supply"):
        curve.calculate_sale_return(Decimal("51"))


def test_calculate_sale_return_negative_amount():
    """
    Negative amount => code does not forbid, so start_supply= 100 - (-10)=110, end=100.
    => cost_between(110,100,i,m).
    """
    curve = _make_linear_curve_simple(Decimal("100"))

    with patch.object(curve, "_apply_time_decay", return_value=(Decimal("2"), Decimal("1"))) as mock_decay, \
         patch.object(linear_helper, "cost_between", return_value=Decimal("-999")) as mock_cost:

        sale_return = curve.calculate_sale_return(Decimal("-10"))

    mock_decay.assert_called_once()
    # start=110, end=100
    mock_cost.assert_called_once_with(Decimal("110"), Decimal("100"), Decimal("2"), Decimal("1"))
    assert sale_return == Decimal("-999")


def test_buy_disabled():
    """
    If 'allow_buy=False', an attempt to buy should raise ValueError.
    """
    curve = _make_linear_curve_buy(allow_buy=False)
    req = TransactionRequest(
        token=Token("T", "T", 18),
        order_type=OrderSide.BUY,
        amount=Decimal("10")
    )

    with pytest.raises(ValueError, match="Buys are disabled"):
        curve.buy(req)


def test_buy_exceeds_max_supply_no_pro_rata():
    """
    If the buy request exceeds max_supply and pro_rata=False,
    a ValueError should be raised.
    """
    curve = _make_linear_curve_buy(
        current_supply=Decimal("100"),
        max_supply=Decimal("105"),  # Only 5 left
        pro_rata=False
    )
    req = TransactionRequest(
        token=Token("T", "T", 18),
        order_type=OrderSide.BUY,
        amount=Decimal("10")  # Attempting to buy 10, but only 5 left
    )
    with pytest.raises(ValueError, match="exceeds max supply"):
        curve.buy(req)


def test_buy_exceeds_max_supply_pro_rata():
    """
    If the buy exceeds max_supply but pro_rata=True,
    we partially fill (final_amount < requested_amount).
    """
    curve = _make_linear_curve_buy(
        current_supply=Decimal("100"),
        max_supply=Decimal("105"),  # Only 5 left
        pro_rata=True
    )
    req = TransactionRequest(
        token=Token("T", "T", 18),
        order_type=OrderSide.BUY,
        amount=Decimal("10")
    )

    # We'll mock partial_fill to return 3. (Pretend it did some ratio.)
    with patch.object(common_helper, "partial_fill", return_value=Decimal("3")) as mock_pf, \
        patch.object(curve, "calculate_purchase_cost", return_value=Decimal("100")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("100")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("102")) as mock_fee:
        result = curve.buy(req)

    mock_pf.assert_called_once_with(Decimal("10"), Decimal("5"), approach=1)
    assert result.executed_amount == Decimal("3"), "We used partial_fill's 3 tokens."
    assert result.total_cost == Decimal("102")
    assert curve._state.current_supply == Decimal("103")  # 100 + 3
    assert curve._state.liquidity.amount == Decimal("1102")  # 1000 + 102


def test_buy_no_tokens_final_amount_zero():
    """
    If final_amount ends up 0 (e.g. no supply left),
    we return a trivial TransactionResult with executed_amount=0, total_cost=0.
    """
    curve = _make_linear_curve_buy(current_supply=Decimal("100"), max_supply=Decimal("100"))
    req = TransactionRequest(
        token=Token("T", "T", 18),
        order_type=OrderSide.BUY,
        amount=Decimal("10")
    )

    # Because max_supply=100 and current_supply=100 => remaining=0 => final_amount=0
    # The method should return a trivial TransactionResult.
    result = curve.buy(req)
    assert result.executed_amount == Decimal("0")
    assert result.total_cost == Decimal("0")
    assert result.new_supply == Decimal("100"), "No change in supply"
    # Confirm state didn't update liquidity
    assert curve._state.liquidity.amount == Decimal("1000"), "No cost added"


def test_buy_risk_profile_aggressive():
    """
    If risk_profile=AGGRESSIVE => we apply a discount (currently 0.95 in example).
    That means the raw_cost is multiplied by 0.95.
    """
    curve = _make_linear_curve_buy(risk_profile=BondingCurveDistribution.AGGRESSIVE)
    req = TransactionRequest(
        token=Token("T", "T", 18),
        order_type=OrderSide.BUY,
        amount=Decimal("10")
    )
    # We'll mock cost & see if it gets multiplied by 0.95
    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("100")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("95")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("95")) as mock_fee:
        result = curve.buy(req)

    mock_calc.assert_called_once_with(Decimal("10"))
    mock_risk.assert_called_once_with(Decimal("100"), BondingCurveDistribution.AGGRESSIVE, is_buy=True)
    assert result.total_cost == Decimal("95"), "After discount, cost=95"
    assert result.executed_amount == Decimal("10")


def test_buy_exceeds_max_liquidity_revert():
    """
    If adding 'raw_cost' to current liquidity would exceed max_liquidity,
    and pro_rata=False => revert.
    """
    curve = _make_linear_curve_buy(
        liquidity_amount=Decimal("1000"),
        max_liquidity=Decimal("1050"),  # Only 50 'space' left
        pro_rata=False
    )
    req = TransactionRequest(
        token=Token("T", "T", 18),
        amount=Decimal("10"),
        order_type=OrderSide.BUY
    )
    # Suppose cost=60 => 1000+60=1060 => exceeds 1050
    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("60")):
        with pytest.raises(ValueError, match="Max liquidity reached"):
            curve.buy(req)


def test_buy_exceeds_max_liquidity_pro_rata():
    """
    If adding 'raw_cost' to current liquidity exceeds max_liquidity, but pro_rata=True,
    we scale down the final_amount & cost proportionally.
    """
    curve = _make_linear_curve_buy(
        liquidity_amount=Decimal("1000"),
        max_liquidity=Decimal("1050"),  # 50 space left
        pro_rata=True
    )
    req = TransactionRequest(
        token=Token("T", "T", 18),
        amount=Decimal("10"),
        order_type=OrderSide.BUY
    )
    # raw_cost=60 => overrun= (1000+60)-1050=10 => fraction=(60-10)/60=50/60=0.8333
    # final_amount= 10*0.8333=8.3333 => cost= 60*0.8333=50
    # We'll just mock to ensure the partial fill logic is used
    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("60")), \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("60")), \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("50")) as mock_fee:
        result = curve.buy(req)

    # final_amount= 8.3333..., cost=50 => new supply= 108.3333, liquidity= 1050
    # We'll allow minor rounding or you can do exact decimal logic
    assert result.executed_amount < Decimal("10"), "Should partial fill"
    assert result.new_supply > Decimal("100"), "Supply increased but not by the full 10"
    # total_cost= final cost after partial fill
    assert result.total_cost == Decimal("50"), "Mocked final cost from apply_transaction_fee"
    assert curve._state.liquidity.amount == Decimal("1050")


def test_buy_slippage_revert():
    """
    If raw_cost > baseline_cost*(1+slip) => revert if partial_fill_approach=2.
    We'll mock:
      - curve's own calculate_purchase_cost => 110
      - parent class's calculate_purchase_cost => 100
      => threshold= 100*(1+0.05)=105 => 110>105 => revert
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1"),
        slope=Decimal("0.1")
    )
    state = BondingCurveState(current_supply=Decimal("100"))
    curve = LinearBondingCurve(
        params=params,
        state=state,
        allow_buy=True,
        pro_rata=False,
        max_supply=None,
        max_liquidity=None,
        slippage_tolerance=Decimal("0.05"),
        partial_fill_approach=2
    )

    req = TransactionRequest(
        token=Token("T", "T", 18),
        amount=Decimal("10"),
        order_type=OrderSide.BUY
    )

    # 1) patch curve.calculate_purchase_cost => 110
    # 2) patch BONDINGCURVE (parent).calculate_purchase_cost => 100
    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("110")) as mock_curve_calc, \
        patch.object(BondingCurve, "calculate_purchase_cost", return_value=Decimal("100")) as mock_parent_calc:
        # Attempt the buy => expect revert
        with pytest.raises(ValueError, match="Slippage tolerance exceeded"):
            curve.buy(req)

    mock_curve_calc.assert_called_once_with(Decimal("10"))
    mock_parent_calc.assert_called_once_with(Decimal("10"))


def test_buy_slippage_partial_fill():
    """
    If raw_cost > baseline_cost*(1+slip) => partial fill if partial_fill_approach=1.
    We'll do a simpler scenario: final_cost is scaled, revert_flag=False
    """
    token = Token(name="TestCoin", symbol="TST", decimals=18)
    initial_liquidity = Liquidity(token=token, amount=Decimal("1000"))

    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1"),
        slope=Decimal("0.1")
    )
    state = BondingCurveState(current_supply=Decimal("100"), liquidity=initial_liquidity)
    curve = LinearBondingCurve(
        params=params,
        state=state,
        slippage_tolerance=Decimal("0.05"),
        partial_fill_approach=1,  # approach=1 => partial fill
    )

    req = TransactionRequest(
        token=Token("T", "T", 18),
        amount=Decimal("10"),
        order_type=OrderSide.BUY
    )

    # We'll mock calls:
    #  1) Child's calculate_purchase_cost => 110 (raw_cost)
    #  2) Parent's calculate_purchase_cost => 100 (baseline_cost)
    #
    # Then we mock scale_for_slippage => (Decimal("105"), False)
    # => final_cost=105, revert_flag=False => partial fill logic
    #
    # Finally, any transaction fee is 0 by default, or we can patch apply_transaction_fee => 106

    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("110")) as mock_curve_calc, \
        patch.object(BondingCurve, "calculate_purchase_cost", return_value=Decimal("100")) as mock_parent_calc, \
        patch.object(common_helper, "scale_for_slippage", return_value=(Decimal("105"), False)) as mock_scale, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("106")) as mock_fee:
        result = curve.buy(req)

    # raw_cost = 110
    # baseline_cost=100 => threshold=100*(1+0.05)=105 => raw>105 => partial fill => scale_for_slippage => (105,False)
    # final_cost=105 => apply_transaction_fee => 106 => total_cost=106 => executed_amount=?
    # (In reality, you'd want to confirm final_amount changed; you might need to check the supply changed.
    #  But for this test, let's just confirm the cost logic.)

    mock_scale.assert_called_once_with(
        Decimal("110"),  # raw_cost
        Decimal("100"),  # baseline_cost
        Decimal("0.05"),  # slip
        is_buy=True,  # is_buy
        approach=1
    )
    assert result.total_cost == Decimal("106"), "After partial fill + fee"
    # If you want to confirm the final_amount is 10, note the code doesn't recalc final_amount in scale_for_slippage by default.
    # The method might do it if partial fill is truly implemented.

    # Supply & liquidity checks
    assert curve._state.current_supply == Decimal(
        "110"), "supply incremented by the original 10 in this simplified scenario"
    assert curve._state.liquidity.amount == Decimal("1106")


def test_buy_transaction_fee():
    """
    If txn_fee_rate > 0, final_cost= raw_cost + fee (for a buy).
    """
    curve = _make_linear_curve_buy(txn_fee_rate=Decimal("0.02"))  # 2% fee
    req = TransactionRequest(token=Token("T", "T", 18), amount=Decimal("5"), order_type=OrderSide.BUY)

    # We'll mock intermediate steps => raw_cost=100 => apply_risk_profile=100 => transaction_fee => 102
    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("100")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("100")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("102")) as mock_fee:
        result = curve.buy(req)

    assert result.executed_amount == Decimal("5")
    assert result.total_cost == Decimal("102")
    assert curve._state.current_supply == Decimal("105"), "100 + 5"
    assert curve._state.liquidity.amount == Decimal("1102")


def test_buy_success_minimal():
    """
    A minimal scenario where everything is default, no partial fill, no revert.
    We won't mock subcalls => test a small integration scenario.
    """
    curve = _make_linear_curve_buy(
        current_supply=Decimal("100"),
        liquidity_amount=Decimal("1000"),
        max_supply=None,  # no limit
        max_liquidity=None,  # no limit
        slippage_tolerance=Decimal("0"),  # no slip
        txn_fee_rate=Decimal("0"),  # no fee
        risk_profile=None
    )
    req = TransactionRequest(
        token=Token("T", "T", 18),
        amount=Decimal("10"),
        order_type=OrderSide.BUY
    )

    # The cost is from: cost_between(100->110, i=1,slope=0.1)
    # cost= i*(110-100) + (slope/2)*(110^2 - 100^2)
    #    = 1*10 + 0.05*(12100-10000)
    #    = 10 + 0.05*2100= 10+105=115
    # No risk, no fee => final=115
    result = curve.buy(req)

    assert result.executed_amount == Decimal("10")
    assert result.total_cost == Decimal("115")
    assert result.new_supply == Decimal("110")
    assert result.average_price == Decimal("11.5")  # 115/10
    # Check state
    assert curve._state.current_supply == Decimal("110")
    assert curve._state.liquidity.amount == Decimal("1115")  # 1000 + 115
    assert result.timestamp is not None


def test_sell_disabled():
    """
    If allow_sell=False, calling sell(...) should raise ValueError.
    """
    curve = _make_linear_curve_sell(allow_sell=False)
    req = TransactionRequest(
        token=Token("T", "T", 18),
        order_type=OrderSide.SELL,
        amount=Decimal("10"),
    )
    with pytest.raises(ValueError, match="Sells are disabled"):
        curve.sell(req)


def test_sell_sufficient_liquidity():
    """
    Normal scenario: there's enough liquidity to pay the user the full 'raw_return'.
    We patch calculate_sale_return => 50 => no partial fill needed.
    """
    curve = _make_linear_curve_sell(
        current_supply=Decimal("100"),
        liquidity_amount=Decimal("1000")  # plenty of liquidity
    )
    req = TransactionRequest(
        token=Token("T", "T", 18),
        order_type=OrderSide.SELL,
        amount=Decimal("10")
    )

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("50")) as mock_calc, \
         patch.object(common_helper, "apply_risk_profile", return_value=Decimal("50")) as mock_risk, \
         patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("49")) as mock_fee:

        result = curve.sell(req)

    # We sold 10 tokens => new supply=90
    # user receives final_return=49 after fees
    assert result.executed_amount == Decimal("10")
    assert result.total_cost == Decimal("49")
    assert result.new_supply == Decimal("90")
    # liquidity is decreased by 49 => 1000-49=951
    assert curve._state.liquidity.amount == Decimal("951")


def test_sell_insufficient_liquidity_no_pro_rata():
    """
    If raw_return > current_liquid and pro_rata=False => revert.
    """
    curve = _make_linear_curve_sell(
        liquidity_amount=Decimal("40"),  # not enough
        pro_rata=False
    )
    req = TransactionRequest(
        token=Token("T", "T", 18),
        order_type=OrderSide.SELL,
        amount=Decimal("10")
    )

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("50")):
        with pytest.raises(ValueError, match="Insufficient liquidity"):
            curve.sell(req)

def test_sell_insufficient_liquidity_pro_rata():
    """
    If raw_return> current_liquid but pro_rata=True => partial fill.
    final_amount is scaled so user receives exactly current_liquid.
    """
    curve = _make_linear_curve_sell(
        current_supply=Decimal("100"),
        liquidity_amount=Decimal("40"),  # not enough
        pro_rata=True
    )
    req = TransactionRequest(
        token=Token("T", "T", 18),
        order_type=OrderSide.SELL,
        amount=Decimal("10")
    )

    # raw_return=50 => user can't get 50, only 40 => fraction= 40/50=0.8
    # => final_amount= 10*0.8=8 => raw_return=40 => then fees? We'll just mock
    with patch.object(curve, "calculate_sale_return", return_value=Decimal("50")) as mock_calc, \
         patch.object(common_helper, "apply_risk_profile", return_value=Decimal("50")) as mock_risk, \
         patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("39")) as mock_fee:

        result = curve.sell(req)

    assert result.executed_amount == Decimal("8"), "Scaled down from 10 to 8"
    assert result.total_cost == Decimal("39"), "After transaction fee"
    # New supply => 100 - 8=92
    assert curve._state.current_supply == Decimal("92")
    # Liquidity => 40-39=1
    assert curve._state.liquidity.amount == Decimal("1")


def test_sell_risk_profile_conservative():
    """
    If risk_profile=CONSERVATIVE => sells get discounted (e.g. raw_value *= 0.85).
    We'll mock the raw_return => check if apply_risk_profile was called => then final.
    """
    curve = _make_linear_curve_sell(risk_profile=BondingCurveDistribution.CONSERVATIVE)
    req = TransactionRequest(
        token=Token("T","T",18),
        order_type=OrderSide.SELL,
        amount=Decimal("5")
    )

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("100")) as mock_calc, \
         patch.object(common_helper, "apply_risk_profile", return_value=Decimal("85")) as mock_risk, \
         patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("83")) as mock_fee:

        result = curve.sell(req)

    mock_risk.assert_called_once_with(Decimal("100"), BondingCurveDistribution.CONSERVATIVE, is_buy=False)
    # Final return => 83
    assert result.total_cost == Decimal("83")
    assert result.executed_amount == Decimal("5")


def test_sell_slippage_revert():
    """
    If raw_return < baseline_return*(1 - slip) => revert if partial_fill_approach=2 => approach=2 => revert
    e.g. raw=80 < threshold= (100*(1-0.05)=95)
    """
    curve = _make_linear_curve_sell(
        slippage_tolerance=Decimal("0.05"),
        partial_fill_approach=2,  # revert
    )
    req = TransactionRequest(
        token=Token("T","T",18),
        order_type=OrderSide.SELL,
        amount=Decimal("10")
    )

    # child's calculate_sale_return => 80
    # parent's => 100 => threshold=100*(1-0.05)=95 => 80<95 => revert
    with patch.object(curve, "calculate_sale_return", return_value=Decimal("80")) as mock_child, \
         patch.object(BondingCurve, "calculate_sale_return", return_value=Decimal("100")) as mock_parent:
        with pytest.raises(ValueError, match="Slippage tolerance not met"):
            curve.sell(req)


def test_sell_slippage_partial_fill():
    """
    If raw_return < baseline_return*(1-slip) => partial fill if approach=1 =>
    We'll mock scale_for_slippage => returns (90, False).
    This means final_return=90 after partial fill, revert_flag=False.
    """
    curve = _make_linear_curve_sell(
        slippage_tolerance=Decimal("0.05"),
        partial_fill_approach=1
    )
    req = TransactionRequest(
        token=Token("T","T",18),
        order_type=OrderSide.SELL,
        amount=Decimal("10")
    )

    # child => raw=80
    # parent => baseline=100 => threshold=95 => 80<95 => partial fill => scale_for_slippage =>(90,False)
    with patch.object(curve, "calculate_sale_return", return_value=Decimal("80")) as mock_child, \
         patch.object(BondingCurve, "calculate_sale_return", return_value=Decimal("100")) as mock_parent, \
         patch.object(common_helper, "scale_for_slippage", return_value=(Decimal("90"), False)) as mock_scale, \
         patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("88")) as mock_fee:

        result = curve.sell(req)

    mock_scale.assert_called_once_with(
        Decimal("80"),   # raw_return
        Decimal("100"),  # baseline_return
        Decimal("0.05"), # slip
        is_buy=False,
        approach=1
    )
    assert result.total_cost == Decimal("88"), "After partial fill & fee"
    # By default, the code doesn't recalc final_amount in scale_for_slippage for sells, but if you do partial fill
    # you'd expect final_amount < original. That depends on your logic.
    # Let's confirm supply changed by 10 or not, etc., if your domain logic modifies it.


def test_sell_transaction_fee():
    """
    If txn_fee_rate>0 => final_return= raw_return - fee.
    """
    curve = _make_linear_curve_sell(txn_fee_rate=Decimal("0.02"))
    req = TransactionRequest(
        token=Token("T","T",18),
        order_type=OrderSide.SELL,
        amount=Decimal("5")
    )

    # Suppose raw_return=100 => risk_profile=100 => fee => 98
    with patch.object(curve, "calculate_sale_return", return_value=Decimal("100")) as mock_calc, \
         patch.object(common_helper, "apply_risk_profile", return_value=Decimal("100")) as mock_risk, \
         patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("98")) as mock_fee:

        result = curve.sell(req)

    assert result.executed_amount == Decimal("5")
    assert result.total_cost == Decimal("98")
    # supply => 95, liquidity => 1000 - 98=902
    assert curve._state.current_supply == Decimal("95")
    assert curve._state.liquidity.amount == Decimal("902")


def test_sell_success_minimal():
    """
    An end-to-end scenario with no partial fill, no revert, no fees.
    We'll do the actual cost math for a small example.
    """
    curve = _make_linear_curve_sell(
        current_supply=Decimal("100"),
        liquidity_amount=Decimal("1000"),
        pro_rata=False,
        slippage_tolerance=Decimal("0"),
        txn_fee_rate=Decimal("0"),
        risk_profile=None,
        partial_fill_approach=1
    )
    req = TransactionRequest(
        token=Token("T","T",18),
        amount=Decimal("10"),
        order_type=OrderSide.SELL
    )

    # We do the real math from calculate_sale_return:
    # sale_return= cost_between(90->100, i=1,slope=0.1)
    # => integral= i*(end-start)+ (m/2)*(end^2 - start^2)
    # = 1*(100-90) + 0.05*(100^2 - 90^2)
    # = 10 + 0.05*(10000-8100)
    # = 10 + 0.05*1900
    # = 10+95=105
    result = curve.sell(req)

    # final_return=105 => no risk markup => no fees => user gets 105
    assert result.executed_amount == Decimal("10")
    assert result.total_cost == Decimal("105")
    # average_price= total_return / tokens => 105/10=10.5
    assert result.average_price == Decimal("10.5")
    # new supply => 90 => liquidity => 1000 - 105= 895
    assert result.new_supply == Decimal("90")
    assert curve._state.current_supply == Decimal("90")
    assert curve._state.liquidity.amount == Decimal("895")
    assert result.timestamp is not None
