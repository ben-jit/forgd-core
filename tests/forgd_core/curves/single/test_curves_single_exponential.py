import pytest
from unittest.mock import patch
from datetime import datetime
from decimal import Decimal

from forgd_core.common.math import decimal_approx_equal
from forgd_core.common.model import (
    BondingCurveParams,
    BondingCurveState, TransactionRequest, Token, Liquidity,
)
from forgd_core.common.enums import BondingCurveType, OrderSide
from forgd_core.curves.single.exponential import ExponentialBondingCurve
from forgd_core.curves.helpers.common import CommonCurveHelper as common_helper
from forgd_core.curves.helpers.exponential import ExponentialCurveHelper as exponential_helper


@pytest.fixture
def exponential_curve_fixture():
    """
    Returns an ExponentialBondingCurve with known p0, alpha, and a known last_timestamp.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.EXPONENTIAL
    )
    params.initial_price = Decimal("1.0")  # p0
    params.exponential = Decimal("0.1")    # alpha

    state = BondingCurveState(
        current_supply=Decimal("100"),
        last_timestamp=datetime(2026,1,1,12,0,0),
    )
    curve = ExponentialBondingCurve(params=params, state=state, time_decay_rate=Decimal("0"))
    return curve


@pytest.fixture
def exponential_curve_fixture_with_liquidity():
    """
    Returns an ExponentialBondingCurve with default p0=1.0, alpha=0.1,
    supply=100, and liquidity=1000 USDC in the state.
    """
    # 1) Create your BondingCurveParams with p0 & alpha
    params = BondingCurveParams(curve_type=BondingCurveType.EXPONENTIAL)
    params.initial_price = Decimal("1.0")  # p0
    params.exponential = Decimal("0.1")    # alpha

    # 2) Create a Liquidity object
    token = Token(name="USDC", symbol="USDC", decimals=6)
    liquidity = Liquidity(token=token, amount=Decimal("1000"))

    # 3) Create a BondingCurveState
    state = BondingCurveState(
        current_supply=Decimal("100"),
        current_price=Decimal("1"),
        liquidity=liquidity,  # not None
        last_timestamp=datetime(2026, 1, 1, 12, 0, 0)
    )

    # 4) Construct the curve, optionally setting time_decay_rate=0 by default
    curve = ExponentialBondingCurve(
        params=params,
        state=state,
        time_decay_rate=Decimal("0")  # or other overrides
    )
    return curve


@pytest.fixture
def default_params():
    """
    Returns a BondingCurveParams with p0 & alpha set to minimal valid values.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.EXPONENTIAL
    )
    # We'll store p0, alpha in the same fields your code references:
    params.initial_price = Decimal("1.0")  # p0
    params.exponential = Decimal("0.1")    # alpha
    return params


def test_init_no_state(default_params):
    """
    If no BondingCurveState is passed, we create one. Also check last_timestamp is set.
    """
    curve = ExponentialBondingCurve(params=default_params, state=None)
    assert curve._state is not None
    assert curve._state.last_timestamp is not None, "Should set last_timestamp if it was None"


def test_init_with_state_no_timestamp(default_params):
    """
    If a state is passed but last_timestamp=None, we set it to datetime.now().
    We'll mock datetime.now() to confirm it's used.
    """
    mock_now = datetime(2026, 1, 1, 12, 0, 0)
    with patch("forgd_core.curves.single.exponential.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now

        st = BondingCurveState(current_supply=Decimal("100"), last_timestamp=None)
        curve = ExponentialBondingCurve(params=default_params, state=st)

        assert curve._state is st
        assert curve._state.last_timestamp == mock_now


def test_init_with_state_existing_timestamp(default_params):
    """
    If the state has an existing last_timestamp, preserve it.
    """
    existing_ts = datetime(2025, 3, 10, 12, 0, 0)
    st = BondingCurveState(current_supply=Decimal("50"), last_timestamp=existing_ts)
    curve = ExponentialBondingCurve(params=default_params, state=st)

    assert curve._state.last_timestamp == existing_ts, "Should not overwrite existing timestamp."


def test_init_default_options(default_params):
    """
    Verify the default options dictionary is set if no overrides are provided.
    """
    curve = ExponentialBondingCurve(params=default_params)
    opts = curve.options
    assert opts["allow_buy"] is True
    assert opts["allow_sell"] is True
    assert opts["max_supply"] is None
    assert opts["max_liquidity"] is None
    assert opts["slippage_tolerance"] == Decimal("0")
    assert opts["txn_fee_rate"] == Decimal("0")
    assert opts["risk_profile"] is None
    assert opts["partial_fill_approach"] == 1
    assert opts["time_decay_rate"] == Decimal("0")
    assert opts["time_decay_approach"] == 1
    assert opts["pro_rata"] is False
    assert opts["clamp_prices"] is False
    assert opts["min_price"] is None
    assert opts["max_price"] is None


def test_init_with_overrides(default_params):
    """
    If recognized kwargs are passed, override defaults.
    Unrecognized => store in custom_options.
    """
    curve = ExponentialBondingCurve(
        params=default_params,
        state=None,
        allow_buy=False,
        max_supply=Decimal("1000"),
        clamp_prices=True,
        min_price=Decimal("1.5"),
        foo="bar"
    )
    opts = curve.options
    assert opts["allow_buy"] is False
    assert opts["max_supply"] == Decimal("1000")
    assert opts["clamp_prices"] is True
    assert opts["min_price"] == Decimal("1.5")

    # Unrecognized => custom_options
    assert hasattr(curve, "custom_options")
    assert curve.custom_options["foo"] == "bar"


def test_init_p0_alpha_validation(default_params):
    """
    p0>0, alpha>=0 => valid. We'll patch validate_exponential_params to confirm it's called.
    If p0<=0 or alpha<0 => ValueError from the helper.
    """
    # 1) valid => no error
    with patch.object(exponential_helper, "validate_exponential_params") as mock_val:
        curve = ExponentialBondingCurve(params=default_params)
    mock_val.assert_called_once_with(Decimal("1.0"), Decimal("0.1"))

    # 2) invalid => p0<=0 or alpha<0
    invalid_params = BondingCurveParams(curve_type=BondingCurveType.EXPONENTIAL)
    invalid_params.initial_price = Decimal("-1")  # p0<=0
    invalid_params.exponential = Decimal("0")

    with pytest.raises(ValueError, match="p0 > 0"):
        ExponentialBondingCurve(params=invalid_params)


def test_apply_time_decay_no_rate(exponential_curve_fixture):
    """
    If time_decay_rate=0 => no changes, no call to apply_time_decay_simple.
    """
    curve = exponential_curve_fixture
    curve.options["time_decay_rate"] = Decimal("0")

    old_p0 = curve._p0
    old_alpha = curve._alpha
    old_ts = curve._state.last_timestamp

    with patch.object(
        exponential_helper,
        "apply_time_decay_simple",
        return_value=(Decimal("999"), Decimal("999"), old_ts)
    ) as mock_decay:
        curve._apply_time_decay()

    # Should not call the helper
    mock_decay.assert_not_called()

    # No changes
    assert curve._p0 == old_p0
    assert curve._alpha == old_alpha
    assert curve._state.last_timestamp == old_ts


def test_apply_time_decay_positive_rate_approach1(exponential_curve_fixture):
    """
    If time_decay_rate>0 => calls apply_time_decay_simple with approach=whatever is in options.
    We'll do approach=1 => confirm we store the returned p0, alpha, last_timestamp.
    """
    curve = exponential_curve_fixture
    curve.options["time_decay_rate"] = Decimal("0.05")
    curve.options["time_decay_approach"] = 1

    old_ts = curve._state.last_timestamp

    with patch.object(
        exponential_helper,
        "apply_time_decay_simple",
        return_value=(Decimal("2.5"), Decimal("0.25"), datetime(2026,2,1,12,0,0))
    ) as mock_decay:
        curve._apply_time_decay()

    # Verify the helper was called with correct arguments
    mock_decay.assert_called_once_with(
        Decimal("1.0"),       # old p0
        Decimal("0.1"),       # old alpha
        old_ts,               # last_timestamp
        Decimal("0.05"),      # time_decay_rate
        approach=1
    )

    # Check we updated p0, alpha, and last_timestamp
    assert curve._p0 == Decimal("2.5")
    assert curve._alpha == Decimal("0.25")
    assert curve._state.last_timestamp == datetime(2026,2,1,12,0,0)


def test_apply_time_decay_positive_rate_approach2(exponential_curve_fixture):
    """
    Similar to approach1 test, but approach=2 => just confirm it passes approach=2
    and updates from the returned values.
    """
    curve = exponential_curve_fixture
    curve.options["time_decay_rate"] = Decimal("0.05")
    curve.options["time_decay_approach"] = 2

    old_ts = curve._state.last_timestamp

    new_ts = datetime(2027,3,1,12,0,0)

    with patch.object(
        exponential_helper,
        "apply_time_decay_simple",
        return_value=(Decimal("10"), Decimal("1.0"), new_ts)
    ) as mock_decay:
        curve._apply_time_decay()

    mock_decay.assert_called_once_with(
        Decimal("1.0"),       # old p0
        Decimal("0.1"),       # old alpha
        old_ts,
        Decimal("0.05"),
        approach=2
    )

    assert curve._p0 == Decimal("10")
    assert curve._alpha == Decimal("1.0")
    assert curve._state.last_timestamp == new_ts


def test_get_spot_price_calls_time_decay(exponential_curve_fixture):
    """
    Ensures _apply_time_decay is called once before computing the final spot price.
    """
    curve = exponential_curve_fixture
    with patch.object(curve, "_apply_time_decay") as mock_decay:
        price = curve.get_spot_price(Decimal("50"))

    mock_decay.assert_called_once()
    # For alpha=0.1, p0=1.0 => raw_price= 1.0 * exp(0.1*50)= ...
    # We'll do a quick approximate check:
    alpha_s = curve._alpha * Decimal("50")
    expected_price = curve._p0 * alpha_s.exp()
    assert decimal_approx_equal(price, expected_price, tol=Decimal(10**7)), f"Expected ~{expected_price}, got {price}"


def test_get_spot_price_alpha_zero(exponential_curve_fixture):
    """
    If alpha=0 => price = p0. We'll just set alpha=0 and check.
    """
    curve = exponential_curve_fixture
    curve._alpha = Decimal("0")

    price = curve.get_spot_price(Decimal("100"))
    # alpha=0 => price= p0=1.0, ignoring supply
    assert price == curve._p0


def test_get_spot_price_exponential(exponential_curve_fixture):
    """
    alpha>0 => price= p0 * exp(alpha*s). We'll do a direct numeric check for a supply.
    """
    curve = exponential_curve_fixture
    # By default, p0=1, alpha=0.1 => let's pick supply=20 => price= 1* exp(0.1*20)
    supply = Decimal("20")

    price = curve.get_spot_price(supply)
    alpha_s = curve._alpha * supply
    expected = curve._p0 * alpha_s.exp()

    assert decimal_approx_equal(price, expected, tol=Decimal(10**7)), f"Expected ~{expected}, got {price}"


def test_get_spot_price_clamped_min(exponential_curve_fixture):
    """
    If clamp_prices=True, and raw_price < min_price => final= min_price.
    We'll pick a scenario where exp(...) < min_price.
    """
    curve = exponential_curve_fixture
    curve.options["clamp_prices"] = True
    curve.options["min_price"] = Decimal("5")
    curve.options["max_price"] = None

    # We'll choose a supply that yields raw_price < 5 => e.g. supply=0 => raw_price= p0 * exp(0)=1
    # So final => 5
    price = curve.get_spot_price(Decimal("0"))
    assert price == Decimal("5")


def test_get_spot_price_clamped_max(exponential_curve_fixture):
    """
    If clamp_prices=True, and raw_price > max_price => final= max_price.
    We'll pick a scenario where exp(...) => large => clamp to max.
    """
    curve = exponential_curve_fixture
    curve.options["clamp_prices"] = True
    curve.options["min_price"] = None
    curve.options["max_price"] = Decimal("10")

    # We'll pick a big supply => s=100 => raw_price= 1*exp(0.1*100)= exp(10)= ~22026.46579 => clamp to 10
    price = curve.get_spot_price(Decimal("100"))
    assert price == Decimal("10")


def test_get_spot_price_clamped_both(exponential_curve_fixture):
    """
    clamp_prices=True, both min_price and max_price set.
    We'll pick a supply that yields raw_price in the middle => no clamping.
    Then a supply that's below min. Then one above max.
    """
    curve = exponential_curve_fixture
    curve.options["clamp_prices"] = True
    curve.options["min_price"] = Decimal("2")
    curve.options["max_price"] = Decimal("8")

    # 1) raw_price in [2..8] => no clamp => e.g. supply=10 => alpha=0.1 => raw= exp(1)= ~2.71828 => p0=1 => 2.71828 => in range
    price1= curve.get_spot_price(Decimal("10"))
    assert Decimal("2") < price1 < Decimal("8"), f"Expected 2 <{price1}<8"

    # 2) raw_price <2 => e.g. supply=2 => alpha=0.1 => raw= exp(0.2)= ~1.2214 => clamp to 2
    price2= curve.get_spot_price(Decimal("2"))
    assert price2== Decimal("2")

    # 3) raw_price>8 => e.g. supply=50 => raw= exp(5)=148.413... => clamp=8
    price3= curve.get_spot_price(Decimal("50"))
    assert price3== Decimal("8")


def test_calculate_purchase_cost_amount_zero(exponential_curve_fixture):
    """
    If amount <= 0 => returns 0 immediately, no call to _apply_time_decay or the helper.
    """
    curve = exponential_curve_fixture
    # We'll confirm that no time decay or cost function is called
    with patch.object(curve, "_apply_time_decay") as mock_decay, \
        patch.object(exponential_helper, "exponential_cost_for_purchase") as mock_ec:
        cost = curve.calculate_purchase_cost(Decimal("0"))

    mock_decay.assert_not_called()
    mock_ec.assert_not_called()
    assert cost == Decimal("0")


def test_calculate_purchase_cost_positive(exponential_curve_fixture):
    """
    If amount>0 => we call _apply_time_decay, then exponential_cost_for_purchase,
    passing clamp_prices, min_price, max_price from options.
    """
    curve = exponential_curve_fixture
    curve.options["clamp_prices"] = True
    curve.options["min_price"] = Decimal("2")
    curve.options["max_price"] = Decimal("10")

    with patch.object(curve, "_apply_time_decay") as mock_decay, \
        patch.object(exponential_helper, "exponential_cost_for_purchase", return_value=Decimal("123")) as mock_ec:
        cost = curve.calculate_purchase_cost(Decimal("15"))

    mock_decay.assert_called_once()
    mock_ec.assert_called_once_with(
        current_supply=curve._state.current_supply,  # by default =100
        amount=Decimal("15"),
        p0=curve._p0,  # 1.0
        alpha=curve._alpha,  # 0.1
        clamped=True,
        min_price=Decimal("2"),
        max_price=Decimal("10")
    )
    assert cost == Decimal("123")


def test_calculate_purchase_cost_numeric(exponential_curve_fixture):
    """
    End-to-end numeric test without mocking the helper, for a small scenario
    verifying the integral (if you like).
    We'll just confirm alpha>0 => cost ~ formula, ignoring clamp since default is off.
    """
    curve = exponential_curve_fixture
    # p0=1.0, alpha=0.1 => supply=100 => let's buy 10 => cost= (p0/alpha)* [exp(alpha*(110)) - exp(alpha*100)]
    amount = Decimal("10")
    cost = curve.calculate_purchase_cost(amount)

    from math import isclose
    # expected= (1/0.1)* [ exp(0.1*(110)) - exp(0.1*(100)) ]
    alpha = curve._alpha
    s = curve._state.current_supply
    end = s + amount
    expected = (curve._p0 / alpha) * ((alpha * end).exp() - (alpha * s).exp())

    assert decimal_approx_equal(cost, expected, tol=Decimal(10**7)), f"Expected ~{expected}, got {cost}"


def test_calculate_sale_return_amount_zero(exponential_curve_fixture):
    """
    If amount <= 0 => returns 0, no time decay or helper call.
    """
    curve = exponential_curve_fixture
    with patch.object(curve, "_apply_time_decay") as mock_decay, \
         patch.object(exponential_helper, "exponential_return_for_sale") as mock_ret:

        returned = curve.calculate_sale_return(Decimal("0"))

    mock_decay.assert_not_called()
    mock_ret.assert_not_called()
    assert returned == Decimal("0")


def test_calculate_sale_return_positive(exponential_curve_fixture):
    """
    If amount>0 => calls _apply_time_decay, then exponential_return_for_sale,
    passing clamp_prices, min_price, max_price, etc.
    """
    curve = exponential_curve_fixture
    curve.options["clamp_prices"] = True
    curve.options["min_price"] = Decimal("2")
    curve.options["max_price"] = Decimal("10")

    with patch.object(curve, "_apply_time_decay") as mock_decay, \
         patch.object(exponential_helper, "exponential_return_for_sale", return_value=Decimal("123")) as mock_ret:

        returned = curve.calculate_sale_return(Decimal("7"))

    mock_decay.assert_called_once()
    mock_ret.assert_called_once_with(
        current_supply=curve._state.current_supply,  # default=100
        amount=Decimal("7"),
        p0=curve._p0,           # 1.0
        alpha=curve._alpha,     # 0.1
        clamped=True,
        min_price=Decimal("2"),
        max_price=Decimal("10")
    )
    assert returned == Decimal("123")


def test_calculate_sale_return_numeric(exponential_curve_fixture):
    """
    End-to-end numeric test with no mocking, verifying integral ~ (p0/alpha)*[exp(alpha*s) - exp(alpha*(s-amount))].
    We'll ignore clamp since default is off, alpha>0 => normal formula.
    """
    curve = exponential_curve_fixture
    # e.g. supply=100 => selling 10 => range=[90..100], alpha=0.1 =>
    # return= (1/0.1)* [exp(0.1*100)- exp(0.1*90)]
    # We'll do isclose check
    amount= Decimal("10")
    returned= curve.calculate_sale_return(amount)

    s= curve._state.current_supply
    alpha= curve._alpha
    p0= curve._p0
    start= s- amount
    end= s
    expected= (p0/ alpha)* ((alpha*end).exp()- (alpha*start).exp())

    assert decimal_approx_equal(returned, expected, tol=Decimal(10**7)), f"Expected ~{expected}, got {returned}"


def test_buy_disabled(exponential_curve_fixture):
    """
    If allow_buy=False, raise ValueError immediately.
    """
    curve = exponential_curve_fixture
    curve.options["allow_buy"] = False

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))
    with pytest.raises(ValueError, match="Buys are disabled"):
        curve.buy(req)


def test_buy_exceeds_max_supply_no_pro_rata(exponential_curve_fixture_with_liquidity):
    """
    If final_amount > max_supply - current_supply and pro_rata=False => revert.
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["max_supply"] = Decimal("150")  # current_supply=100 => capacity=50
    curve.options["pro_rata"] = False

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("60"))
    with pytest.raises(ValueError, match="exceeds max supply"):
        curve.buy(req)


def test_buy_exceeds_max_supply_pro_rata(exponential_curve_fixture_with_liquidity):
    """
    If final_amount > remaining capacity but pro_rata=True => partial fill => final_amount is partial.
    We'll confirm partial fill logic by mocking 'common_helper.partial_fill'.
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["max_supply"] = Decimal("150")  # capacity=50
    curve.options["pro_rata"] = True

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("60"))

    # We'll patch partial_fill => returns 30 => new final_amount=30
    with patch.object(common_helper, "partial_fill", return_value=Decimal("30")) as mock_pf, \
        patch.object(curve, "calculate_purchase_cost", return_value=Decimal("100")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("100")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("105")) as mock_fee:
        result = curve.buy(req)

    mock_pf.assert_called_once_with(
        Decimal("60"), Decimal("50"),
        approach=curve.options["partial_fill_approach"]  # default=1
    )
    # We used partial fill => final_amount=30 => total cost=105 => supply=130 => liquidity=1000+105=1105
    assert result.executed_amount == Decimal("30")
    assert result.total_cost == Decimal("105")
    assert curve._state.current_supply == Decimal("130")
    assert curve._state.liquidity.amount == Decimal("1105")


def test_buy_final_amount_zero(exponential_curve_fixture_with_liquidity):
    """
    If final_amount=0 => trivial transaction => no cost, no supply change.
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["max_supply"] = Decimal("100")  # => capacity=0 => final_amount=0 if pro_rata=True
    curve.options["pro_rata"] = True

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))
    result = curve.buy(req)

    assert result.executed_amount == Decimal("0")
    assert result.total_cost == Decimal("0")
    assert result.average_price == Decimal("0")
    # no state changes
    assert curve._state.current_supply == Decimal("100")
    assert curve._state.liquidity.amount == Decimal("1000")


def test_buy_calculate_purchase_cost(exponential_curve_fixture_with_liquidity):
    """
    Normal scenario => calls 'calculate_purchase_cost'. We'll do partial mocking to confirm flow.
    """
    curve = exponential_curve_fixture_with_liquidity
    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("5"))

    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("50")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("50")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("52")) as mock_fee:
        result = curve.buy(req)

    mock_calc.assert_called_once_with(Decimal("5"))
    # final_amount => 5 => total_cost => 52 => supply=105 => liquidity=1052
    assert result.executed_amount == Decimal("5")
    assert result.total_cost == Decimal("52")
    assert curve._state.current_supply == Decimal("105")
    assert curve._state.liquidity.amount == Decimal("1052")


def test_buy_apply_risk_profile(exponential_curve_fixture_with_liquidity):
    """
    If risk_profile is set => we call apply_risk_profile(raw_cost, risk_profile, is_buy=True).
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["risk_profile"] = "AGGRESSIVE"

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("100")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("95")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("97")) as mock_fee:
        result = curve.buy(req)

    mock_risk.assert_called_once_with(Decimal("100"), "AGGRESSIVE", is_buy=True)
    assert result.total_cost == Decimal("97")


def test_buy_exceeds_max_liquidity_no_pro_rata(exponential_curve_fixture_with_liquidity):
    """
    If raw_cost + current_liquidity > max_liquidity => revert if pro_rata=False.
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["max_liquidity"] = Decimal("1050")  # => 50 space left
    curve.options["pro_rata"] = False

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("60")):
        with pytest.raises(ValueError, match="Max liquidity reached"):
            curve.buy(req)


def test_buy_exceeds_max_liquidity_pro_rata(exponential_curve_fixture_with_liquidity):
    """
    If raw_cost + liquidity> max_liquidity => partial fill => we scale final_amount & cost proportionally.
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["max_liquidity"] = Decimal("1050")  # => 50 space left
    curve.options["pro_rata"] = True

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    # Suppose raw_cost=60 => overrun= 60+1000 -1050=10 => fraction= (60-10)/60= 50/60=0.8333 => final_amount= 10*0.8333 =>8.33 => cost=50
    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("60")), \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("60")), \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("50")):
        result = curve.buy(req)

    assert result.executed_amount < Decimal("10")
    assert result.total_cost == Decimal("50")
    assert curve._state.liquidity.amount == Decimal("1050")  # fully used
    assert curve._state.current_supply > Decimal("100")


def test_buy_slippage_revert(exponential_curve_fixture_with_liquidity):
    """
    If raw_cost> baseline*(1+slip) => revert if partial_fill_approach=2 => no partial.
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["slippage_tolerance"] = Decimal("0.05")
    curve.options["partial_fill_approach"] = 2

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    # raw_cost=110, baseline=100 => threshold=105 => 110>105 => revert
    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("110")) as mock_calc, \
        patch.object(exponential_helper, "exponential_cost_for_purchase", return_value=Decimal("100")) as mock_baseline:
        with pytest.raises(ValueError, match="Slippage tolerance exceeded"):
            curve.buy(req)


def test_buy_slippage_partial_fill(exponential_curve_fixture_with_liquidity):
    """
    If raw_cost> baseline*(1+ slip) => partial fill if approach=1 => scale_for_slippage => returns (some_new_cost, False).
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["slippage_tolerance"] = Decimal("0.05")
    curve.options["partial_fill_approach"] = 1

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("110")), \
        patch.object(exponential_helper, "exponential_cost_for_purchase", return_value=Decimal("100")) as mock_base, \
        patch.object(common_helper, "scale_for_slippage", return_value=(Decimal("105"), False)) as mock_scale, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("106")):
        result = curve.buy(req)

    mock_scale.assert_called_once_with(
        Decimal("110"), Decimal("100"), Decimal("0.05"), is_buy=True, approach=1
    )
    # The code doesn't recalc final_amount in scale_for_slippage unless you explicitly do it.
    # We'll check total_cost=106 => executed_amount=10 => supply=110 => liquidity= 1000+106=1106
    assert result.total_cost == Decimal("106")
    assert result.executed_amount == Decimal("10")


def test_buy_transaction_fee(exponential_curve_fixture_with_liquidity):
    """
    If txn_fee_rate>0 => final_cost= raw_cost + fee => confirm we call apply_transaction_fee with is_buy=True.
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["txn_fee_rate"] = Decimal("0.02")

    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("5"))

    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("100")), \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("100")), \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("102")) as mock_fee:
        result = curve.buy(req)

    mock_fee.assert_called_once_with(Decimal("100"), Decimal("0.02"), is_buy=True)
    assert result.total_cost == Decimal("102")
    # supply => 105 => liquidity => 1102


def test_buy_success_minimal(exponential_curve_fixture_with_liquidity):
    """
    End-to-end minimal scenario => no partial fill, no max supply, no slippage => everything straightforward.
    We'll do partial mocks for cost & fee => check final state.
    """
    curve = exponential_curve_fixture_with_liquidity
    req = TransactionRequest(token=None, order_type=OrderSide.BUY, amount=Decimal("10"))

    with patch.object(curve, "calculate_purchase_cost", return_value=Decimal("50")), \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("52")):
        result = curve.buy(req)

    # final_amount=10 => total_cost=52 => supply=110 => liquidity=1052
    assert result.executed_amount == Decimal("10")
    assert result.total_cost == Decimal("52")
    assert result.new_supply == Decimal("110")
    assert curve._state.current_supply == Decimal("110")
    assert curve._state.liquidity.amount == Decimal("1052")
    assert result.average_price == Decimal("5.2")
    assert result.timestamp is not None


def test_sell_disabled(exponential_curve_fixture_with_liquidity):
    """
    If allow_sell=False => raise ValueError immediately.
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["allow_sell"] = False

    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))
    with pytest.raises(ValueError, match="Sells are disabled"):
        curve.sell(req)


def test_sell_normal_scenario(exponential_curve_fixture_with_liquidity):
    """
    Typical scenario => calls calculate_sale_return => risk profile => partial fill if needed => fee => ...
    We'll do partial mocking for simplicity.
    """
    curve = exponential_curve_fixture_with_liquidity
    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("5"))

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("80")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("85")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("82")) as mock_fee:
        result = curve.sell(req)

    # final_amount=>5 => final_return=>82 => supply => 95 => liquidity=> 1000-82=918
    assert result.executed_amount == Decimal("5")
    assert result.total_cost == Decimal("82")
    assert curve._state.current_supply == Decimal("95")
    assert curve._state.liquidity.amount == Decimal("918")
    # average_price => 82/5=16.4
    assert result.average_price == Decimal("16.4")
    assert result.timestamp is not None


def test_sell_insufficient_liquidity_no_pro_rata(exponential_curve_fixture_with_liquidity):
    """
    If raw_return> liquidity => revert if pro_rata=False.
    """
    curve = exponential_curve_fixture_with_liquidity
    # current_liquidity=1000 => let's produce raw_return=1200 => revert
    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("1200")):
        with pytest.raises(ValueError, match="Insufficient liquidity"):
            curve.sell(req)


def test_sell_insufficient_liquidity_pro_rata(exponential_curve_fixture_with_liquidity):
    """
    If raw_return> liquidity => partial fill => fraction= partial_fill(raw_return, liquidity).
    => final_amount= final_amount*(fraction_of_tokens).
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["pro_rata"] = True

    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))

    # Suppose raw_return=1200 => liquidity=1000 => fraction= partial_fill(1200,1000)= 1000 => fraction_of_tokens=1000/1200=0.8333 => final_amount=8.333 => raw_return=1000
    with patch.object(curve, "calculate_sale_return", return_value=Decimal("1200")), \
        patch.object(common_helper, "partial_fill", return_value=Decimal("1000")) as mock_pf, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("990")) as mock_fee:
        result = curve.sell(req)

    mock_pf.assert_called_once_with(Decimal("1200"), Decimal("1000"))
    assert result.executed_amount < Decimal("10")
    assert result.total_cost == Decimal("990")
    # supply => 100 - 8.333=91.667 => liquidity=> 1000-990=10
    assert curve._state.current_supply < Decimal("100")
    assert curve._state.liquidity.amount == Decimal("10")


def test_sell_slippage_revert(exponential_curve_fixture_with_liquidity):
    """
    If raw_return< baseline*(1-slippage) => revert if partial_fill_approach=2 => no partial fill.
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["slippage_tolerance"] = Decimal("0.05")
    curve.options["partial_fill_approach"] = 2

    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))

    # raw_return=80, baseline=100 => threshold= 100*(1-0.05)=95 => 80<95 => revert
    with patch.object(curve, "calculate_sale_return", return_value=Decimal("80")) as mock_calc, \
        patch.object(exponential_helper, "exponential_return_for_sale", return_value=Decimal("100")):
        with pytest.raises(ValueError, match="Slippage tolerance not met"):
            curve.sell(req)


def test_sell_slippage_partial_fill(exponential_curve_fixture_with_liquidity):
    """
    If raw_return< baseline*(1- slip) => partial fill if partial_fill_approach=1 => scale_for_slippage => returns (some_new_ret,False).
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["slippage_tolerance"] = Decimal("0.05")
    curve.options["partial_fill_approach"] = 1

    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))

    # raw=80, baseline=100 => threshold=95 => partial => final_ret=90 => revert_flag=False => code sets raw_return=90 => then fee?
    with patch.object(curve, "calculate_sale_return", return_value=Decimal("80")), \
        patch.object(exponential_helper, "exponential_return_for_sale", return_value=Decimal("100")), \
        patch.object(common_helper, "scale_for_slippage", return_value=(Decimal("90"), False)), \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("88")):
        result = curve.sell(req)

    # final return => 88 => final_amount=10 => supply => 90 => liquidity => 912...
    # We confirm final total=88 => no recalc final_amount unless partial fill from insufficient liq.
    assert result.total_cost == Decimal("88")


def test_sell_risk_profile(exponential_curve_fixture_with_liquidity):
    """
    If risk_profile => apply_risk_profile(raw_return, risk_profile, is_buy=False).
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["risk_profile"] = "CONSERVATIVE"

    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("5"))

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("40")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("34")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("32")) as mock_fee:
        result = curve.sell(req)

    mock_risk.assert_called_once_with(Decimal("40"), "CONSERVATIVE", is_buy=False)
    assert result.total_cost == Decimal("32")


def test_sell_transaction_fee(exponential_curve_fixture_with_liquidity):
    """
    If txn_fee_rate>0 => final_return= raw_return - fee => confirm we call apply_transaction_fee(..., is_buy=False).
    """
    curve = exponential_curve_fixture_with_liquidity
    curve.options["txn_fee_rate"] = Decimal("0.02")

    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("5"))

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("100")) as mock_calc, \
        patch.object(common_helper, "apply_risk_profile", return_value=Decimal("95")) as mock_risk, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("93")) as mock_fee:
        result = curve.sell(req)

    mock_fee.assert_called_once_with(Decimal("95"), Decimal("0.02"), is_buy=False)
    assert result.total_cost == Decimal("93")
    assert curve._state.current_supply == Decimal("95")
    assert curve._state.liquidity.amount == Decimal("907")  # 1000-93=907


def test_sell_success_minimal(exponential_curve_fixture_with_liquidity):
    """
    A simpler scenario => no partial fill, no max liquidity => normal sell => partial mocking for final check.
    """
    curve = exponential_curve_fixture_with_liquidity
    req = TransactionRequest(token=None, order_type=OrderSide.SELL, amount=Decimal("10"))

    with patch.object(curve, "calculate_sale_return", return_value=Decimal("50")) as mock_calc, \
        patch.object(common_helper, "apply_transaction_fee", return_value=Decimal("48")) as mock_fee:
        result = curve.sell(req)

    # final_amount=10 => final_return=48 => supply=90 => liquidity=1000-48=952
    assert result.executed_amount == Decimal("10")
    assert result.total_cost == Decimal("48")
    assert result.new_supply == Decimal("90")
    assert curve._state.current_supply == Decimal("90")
    assert curve._state.liquidity.amount == Decimal("952")
    # average_price => 48/10=4.8
    assert result.average_price == Decimal("4.8")
    assert result.timestamp is not None
