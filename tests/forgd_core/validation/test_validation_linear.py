import pytest

from decimal import Decimal
from unittest.mock import patch

from forgd_core.common.enums import BondingCurveType, OrderSide
from forgd_core.common.model import BondingCurveParams, Liquidity, Token, TransactionResult, TransactionRequest, \
    BondingCurveState
from forgd_core.curves.single.linear import LinearBondingCurve
from forgd_core.validation.linear_validator import LinearCurveValidator


@pytest.fixture
def valid_params():
    """
    Returns a BondingCurveParams for a linear curve with valid initial_price>0, slope>=0.
    """
    return BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.1"),
    )


@pytest.fixture
def valid_options():
    """
    Returns a dictionary of valid options:
      - max_supply >=0
      - max_liquidity >=0
      - txn_fee_rate >=0
    """
    return {
        "max_supply": Decimal("1000"),
        "max_liquidity": Decimal("5000"),
        "txn_fee_rate": Decimal("0.01"),  # 1% fee
    }


@pytest.fixture
def linear_curve_fixture():
    """
    Returns a minimal LinearBondingCurve instance for testing boundary_tests.
    We assume it has valid initial_price>0, slope>=0, etc.
    (Or you can param if you have a standard fixture.)
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.1")
    )
    liquidity = Liquidity(token=Token("USDC", "USDC", 6), amount=Decimal("10000"))
    state = BondingCurveState(
        current_supply=Decimal("500"),
        current_price=Decimal("2"),
        liquidity=liquidity
    )
    curve = LinearBondingCurve(params=params, state=state)
    curve.options["max_supply"] = Decimal("1000")
    curve.options["max_liquidity"] = Decimal("100000")
    return curve


@pytest.fixture
def valid_curve_and_params():
    """
    Returns a (curve, curve_params) pair that is fully valid for a linear curve.
    """
    from forgd_core.common.model import BondingCurveState, Liquidity, Token
    # Minimal valid BondingCurveParams
    params = BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.1")
    )
    # Minimal valid LinearBondingCurve
    state = BondingCurveState(
        current_supply=Decimal("500"),
        # Provide liquidity so buy/sell won't error
        liquidity=Liquidity(token=Token("USD", "USD", 6), amount=Decimal("10000"))
    )
    curve = LinearBondingCurve(params=params, state=state)
    curve.options["max_supply"] = Decimal("1000")
    curve.options["max_liquidity"] = Decimal("100000")
    return curve, params


def test_validate_params_all_valid(valid_params, valid_options):
    """
    If all parameters are valid => no errors, no warnings, info includes param_summary.
    """
    result = LinearCurveValidator.validate_params(valid_params, valid_options)
    assert "errors" in result
    assert "warnings" in result
    assert "info" in result

    assert result["errors"] == [], f"Expected no errors, got {result['errors']}"
    assert result["warnings"] == [], f"Expected no warnings, got {result['warnings']}"
    assert "param_summary" in result["info"], "Expected param_summary in info"

    param_summary = result["info"]["param_summary"]
    assert param_summary["initial_price"] == "1.0"
    assert param_summary["slope"] == "0.1"
    assert param_summary["max_supply"] == "1000"
    assert param_summary["max_liquidity"] == "5000"
    assert param_summary["txn_fee_rate"] == "0.01"


@pytest.mark.parametrize(
    "field, value, expected_msg",
    [
        # 1) initial_price <=0
        ("initial_price", Decimal("0"), "LinearCurve: 'initial_price' must be > 0."),
        ("initial_price", Decimal("-1"), "LinearCurve: 'initial_price' must be > 0."),
        # 2) slope<0
        ("slope", Decimal("-0.001"), "LinearCurve: 'slope' must be >= 0 (cannot be negative)."),
    ]
)
def test_validate_params_invalid_params(field, value, expected_msg, valid_params, valid_options):
    """
    If certain fields in BondingCurveParams are invalid => we get an error message.
    """
    setattr(valid_params, field, value)  # override that field
    result = LinearCurveValidator.validate_params(valid_params, valid_options)
    assert expected_msg in result["errors"], f"Expected error '{expected_msg}', got {result['errors']}"


@pytest.mark.parametrize(
    "opt_key, opt_value, expected_msg",
    [
        # max_supply<0 => error
        ("max_supply", Decimal("-10"), "LinearCurve: 'max_supply' cannot be negative."),
        # max_liquidity<0 => error
        ("max_liquidity", Decimal("-1"), "LinearCurve: 'max_liquidity' cannot be negative."),
        # txn_fee_rate<0 => error
        ("txn_fee_rate", Decimal("-0.01"), "LinearCurve: 'txn_fee_rate' cannot be negative."),
    ]
)
def test_validate_params_invalid_options(opt_key, opt_value, expected_msg, valid_params, valid_options):
    """
    If certain fields in the options are invalid => we get an error message.
    """
    valid_options[opt_key] = opt_value  # override
    result = LinearCurveValidator.validate_params(valid_params, valid_options)
    assert expected_msg in result["errors"], f"Expected error '{expected_msg}', got {result['errors']}"


def test_validate_params_missing_options(valid_params):
    """
    If max_supply, max_liquidity, or other required fields are missing => we see the relevant error.
    We can pass an empty dict or incomplete dict for options.
    """
    incomplete_options = {}  # missing everything
    result = LinearCurveValidator.validate_params(valid_params, incomplete_options)

    # Expect "max_supply cannot be negative" and "max_liquidity cannot be negative" because it's None => <0 check
    assert "LinearCurve: 'max_supply' cannot be negative." in result["errors"]
    assert "LinearCurve: 'max_liquidity' cannot be negative." in result["errors"]

    # txn_fee_rate is by default 0 if missing, so no error unless negative.


def test_validate_params_multiple_errors(valid_params, valid_options):
    """
    If multiple fields are invalid at once => accumulate errors in the list.
    """
    valid_params.initial_price = Decimal("-5")  # invalid
    valid_options["max_liquidity"] = Decimal("-1")  # invalid

    result = LinearCurveValidator.validate_params(valid_params, valid_options)
    assert len(result["errors"]) == 2, f"Expected 2 errors, got {result['errors']}"
    assert "LinearCurve: 'initial_price' must be > 0." in result["errors"]
    assert "LinearCurve: 'max_liquidity' cannot be negative." in result["errors"]


def test_boundary_tests_all_good(linear_curve_fixture):
    """
    If spot price at 0 >=0, cost to buy 0=0, large supply price>=0 => no errors/warnings.
    """
    # We do not mock so we rely on the linear logic:
    #   spot_price(0) => 1.0 + 0.1*0=1 => >=0
    #   cost(0)=>0, large supply => e.g. 1.0 + 0.1*1e6 => positive => no error
    result = LinearCurveValidator.boundary_tests(linear_curve_fixture)
    assert "errors" in result
    assert "warnings" in result
    assert "info" in result
    assert result["errors"] == []
    assert result["warnings"] == []
    assert result["info"].get("boundary_tests_run") is True


def test_boundary_tests_spot_price_negative_zero(linear_curve_fixture):
    """
    If get_spot_price(0) returns negative => we have an error appended.
    """
    with patch.object(linear_curve_fixture, "get_spot_price", return_value=Decimal("-1")):
        result = LinearCurveValidator.boundary_tests(linear_curve_fixture)
    assert "Spot price is negative at supply=0." in result["errors"]
    assert result["warnings"] == []


def test_boundary_tests_cost_zero_nonzero_warning(linear_curve_fixture):
    """
    If calculate_purchase_cost(0) != 0 => that yields a warning.
    """
    with patch.object(linear_curve_fixture, "calculate_purchase_cost", return_value=Decimal("5")):
        result = LinearCurveValidator.boundary_tests(linear_curve_fixture)
    assert result["errors"] == []
    assert "Cost to buy 0 tokens is not zero: got 5" in result["warnings"]


def test_boundary_tests_large_supply_exception(linear_curve_fixture):
    """
    get_spot_price(1e6) raises an exception => it's caught => yields a warning.
    We let get_spot_price(0) succeed normally.
    """
    def spot_price_side_effect(supply):
        # if supply=0 => return some valid price
        if supply == Decimal("0"):
            return Decimal("1")   # or some positive number
        # if supply=1e6 => raise
        elif supply == Decimal("1000000"):
            raise Exception("some error")
        # For any other supply if needed:
        return Decimal("10")

    with patch.object(
        linear_curve_fixture,
        "get_spot_price",
        side_effect=spot_price_side_effect
    ):
        result = LinearCurveValidator.boundary_tests(linear_curve_fixture)

    # Now step#1: get_spot_price(0) => returns 1 => no error
    # step#2: calculate_purchase_cost(0) => default code => returns 0 => no error
    # step#3: get_spot_price(1e6) => raises => put into warnings
    assert result["errors"] == [], f"Unexpected errors: {result['errors']}"
    # We expect the warning about "Exception calling get_spot_price(1e6): some error"
    assert any("Exception calling get_spot_price(1e6): some error" in w for w in result["warnings"])


def test_boundary_tests_spot_price_0_exception(linear_curve_fixture):
    """
    get_spot_price(0) raises an exception => error is appended.
    """
    with patch.object(linear_curve_fixture, "get_spot_price", side_effect=Exception("test error zero supply")):
        result = LinearCurveValidator.boundary_tests(linear_curve_fixture)
    # "Exception calling get_spot_price(0): test error zero supply"
    assert any("Exception calling get_spot_price(0): test error zero supply" in e for e in result["errors"])


def test_boundary_tests_cost_zero_exception(linear_curve_fixture):
    """
    If calculate_purchase_cost(0) raises => it's an error in boundary_tests.
    """
    with patch.object(linear_curve_fixture, "calculate_purchase_cost", side_effect=Exception("test cost error")):
        result = LinearCurveValidator.boundary_tests(linear_curve_fixture)
    # "Exception calling calculate_purchase_cost(0): test cost error"
    assert any("Exception calling calculate_purchase_cost(0): test cost error" in e for e in result["errors"])


def test_scenario_tests_all_good(linear_curve_fixture):
    """
    No exceptions, no negative cost or supply => no errors, no warnings.
    We'll not mock => rely on normal linear logic for small scenario.
    But if your domain logic would produce large costs, you can
    partially mock 'buy'/'sell' calls if you prefer.
    """
    result = LinearCurveValidator.scenario_tests(linear_curve_fixture)
    assert "errors" in result
    assert "warnings" in result
    assert "info" in result
    assert result["errors"] == []
    assert result["warnings"] == []
    # final_supply_after_scenario is recorded
    assert "final_supply_after_scenario" in result["info"]


def test_scenario_tests_buy1_exception(linear_curve_fixture):
    """
    If buy(100) raises => we record that in errors =>
    but the scenario code still proceeds to buy(200) and sell(50).
    """

    def side_effect_buy(request):
        if request.amount == Decimal("100"):
            raise Exception("mocked buy(100) error")
        # For buy(200) or any other buy => return a normal TransactionResult
        return TransactionResult(
            executed_amount=request.amount,
            total_cost=Decimal("100"),
            average_price=Decimal("1.0"),
            new_supply=linear_curve_fixture._state.current_supply + request.amount,
            timestamp=None
        )

    with patch.object(linear_curve_fixture, "buy", side_effect=side_effect_buy) as mock_buy, \
        patch.object(linear_curve_fixture, "sell") as mock_sell:
        result = LinearCurveValidator.scenario_tests(linear_curve_fixture)

    # We expect an error about "Exception in scenario step buy(100): mocked buy(100) error"
    assert any(
        "Exception in scenario step buy(100): mocked buy(100) error" in e
        for e in result["errors"]
    ), f"Errors were: {result['errors']}"

    # The scenario continues, so buy(200) and sell(50) are presumably called.
    # If you want to confirm, you can do:
    # mock_buy was called 2 times total: once for buy(100) (which raised), once for buy(200).
    assert mock_buy.call_count == 2, f"Expected 2 calls to buy(...) but got {mock_buy.call_count}"

    # The scenario also calls sell(50) in step 3
    mock_sell.assert_called_once_with(
        TransactionRequest(
            token=Token("TEST", "Test Token", 18),
            order_type=OrderSide.SELL,
            amount=Decimal("50"),
            user_id=None
        )
    )


def test_scenario_tests_buy2_negative_cost(linear_curve_fixture):
    """
    Suppose buy(100) is fine, but buy(200) yields negative total_cost => that triggers an error message.
    We'll mock the 'curve.buy' calls so the first returns a normal result, second returns negative cost.
    """

    def side_effect_buy(request):
        if request.amount == Decimal("100"):
            # normal
            return TransactionResult(
                executed_amount=Decimal("100"),
                total_cost=Decimal("200"),
                average_price=Decimal("2"),
                new_supply=linear_curve_fixture._state.current_supply + Decimal("100"),
                timestamp=None
            )
        elif request.amount == Decimal("200"):
            # negative cost => triggers "Buying 200 tokens => negative total cost."
            return TransactionResult(
                executed_amount=Decimal("200"),
                total_cost=Decimal("-1"),
                average_price=Decimal("-0.005"),
                new_supply=linear_curve_fixture._state.current_supply + Decimal("300"),
                timestamp=None
            )
        else:
            # fallback
            return TransactionResult(
                Decimal("0"),
                Decimal("0"),
                Decimal("0"),
                linear_curve_fixture._state.current_supply,
                None
            )

    with patch.object(linear_curve_fixture, "buy", side_effect=side_effect_buy), \
        patch.object(linear_curve_fixture, "sell", return_value=TransactionResult(
            executed_amount=Decimal("50"),
            total_cost=Decimal("100"),
            average_price=Decimal("2"),
            new_supply=Decimal("850"),
            timestamp=None
        )) as mock_sell:
        result = LinearCurveValidator.scenario_tests(linear_curve_fixture)

    # We expect an error: "Buying 200 tokens => negative total cost."
    assert any("Buying 200 tokens => negative total cost." in e for e in result["errors"])
    # presumably the scenario code does attempt the sell(50) step => your code's flow might skip or might continue
    # If it continues, mock_sell is called once
    mock_sell.assert_called_once()


def test_scenario_tests_sell_exception(linear_curve_fixture):
    """
    If sell(50) raises => we record it in errors. We'll let buy(100) and buy(200) succeed normally.
    """

    # We'll return normal results for buy calls, raise on sell(50)
    def side_effect_sell(request):
        if request.amount == Decimal("50"):
            raise Exception("mocked sell(50) failure")
        return TransactionResult(
            executed_amount=request.amount,
            total_cost=Decimal("0"),
            average_price=Decimal("0"),
            new_supply=linear_curve_fixture._state.current_supply - request.amount,
            timestamp=None
        )

    with patch.object(linear_curve_fixture, "buy", side_effect=[
        # first buy(100)
        TransactionResult(
            executed_amount=Decimal("100"),
            total_cost=Decimal("100"),
            average_price=Decimal("1"),
            new_supply=Decimal("600"),
            timestamp=None
        ),
        # second buy(200)
        TransactionResult(
            executed_amount=Decimal("200"),
            total_cost=Decimal("400"),
            average_price=Decimal("2"),
            new_supply=Decimal("800"),
            timestamp=None
        )
    ]) as mock_buys, \
        patch.object(linear_curve_fixture, "sell", side_effect=side_effect_sell):
        result = LinearCurveValidator.scenario_tests(linear_curve_fixture)

    # We expect "Exception in scenario step sell(50): mocked sell(50) failure"
    assert any("Exception in scenario step sell(50): mocked sell(50) failure" in e for e in result["errors"])


def test_scenario_tests_sell_negative_supply(linear_curve_fixture):
    """
    Suppose first two buys are normal, but the sell(50) yields a negative new_supply => triggers an error.
    """

    def side_effect_sell(request):
        # negative new_supply => e.g. -100
        return TransactionResult(
            executed_amount=Decimal("50"),
            total_cost=Decimal("100"),
            average_price=Decimal("2"),
            new_supply=Decimal("-100"),
            timestamp=None
        )

    with patch.object(linear_curve_fixture, "buy", side_effect=[
        # successful buy(100)
        TransactionResult(
            executed_amount=Decimal("100"),
            total_cost=Decimal("100"),
            average_price=Decimal("1"),
            new_supply=Decimal("600"),
            timestamp=None
        ),
        # successful buy(200)
        TransactionResult(
            executed_amount=Decimal("200"),
            total_cost=Decimal("300"),
            average_price=Decimal("1.5"),
            new_supply=Decimal("800"),
            timestamp=None
        )
    ]), \
        patch.object(linear_curve_fixture, "sell", side_effect=side_effect_sell):
        result = LinearCurveValidator.scenario_tests(linear_curve_fixture)

    assert any("Supply is negative after sell(50)." in e for e in result["errors"])
    # no mention of negative cost => so no other error for that step


def test_scenario_tests_multiple_issues(linear_curve_fixture):
    """
    If multiple steps produce issues => we gather multiple errors.
    E.g. buy(100) negative cost, buy(200) negative supply, sell(50) raise exception.
    """

    def side_effect_buy(request):
        if request.amount == Decimal("100"):
            return TransactionResult(Decimal("100"), Decimal("-50"), Decimal("-0.5"), Decimal("450"), None)  # negative cost
        elif request.amount == Decimal("200"):
            return TransactionResult(Decimal("200"), Decimal("200"), Decimal("1"), Decimal("-100"), None)  # negative supply
        else:
            return TransactionResult(Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), None)

    def side_effect_sell(request):
        raise Exception("some sell error")

    with patch.object(linear_curve_fixture, "buy", side_effect=side_effect_buy), \
        patch.object(linear_curve_fixture, "sell", side_effect=side_effect_sell):
        result = LinearCurveValidator.scenario_tests(linear_curve_fixture)

    # Expect 3 errors:
    # 1) "Buying 100 tokens => negative total cost."
    # 2) "Supply is negative after buy(200)."
    # 3) "Exception in scenario step sell(50): some sell error"
    assert len(result["errors"]) == 3
    assert "Buying 100 tokens => negative total cost." in result["errors"]
    assert "Supply is negative after buy(200)." in result["errors"]
    assert any("Exception in scenario step sell(50): some sell error" in e for e in result["errors"])
    # no warnings by default
    assert result["warnings"] == []


def test_run_all_validations_non_linear_curve():
    """
    If 'curve' is not a LinearBondingCurve => raise ValueError.
    """
    from forgd_core.curves.single.exponential import ExponentialBondingCurve
    # Example: pass an ExponentialBondingCurve instead
    # or just pass a random object
    with pytest.raises(ValueError, match="Invalid curve type for LinearCurveValidator"):
        # we can pass a fake or an exponential curve
        LinearCurveValidator.run_all_validations(curve="notALinearCurve", curve_params=None)


def test_run_all_validations_aggregates_subcalls(valid_curve_and_params):
    """
    We mock validate_params, boundary_tests, scenario_tests => each returns some errors/warnings/info.
    Then run_all_validations => ensures they are aggregated properly in results.
    """
    curve, params = valid_curve_and_params

    # Mock sub-checks
    with patch.object(LinearCurveValidator, "validate_params", return_value={
        "errors": ["param error1", "param error2"],
        "warnings": ["param warn1"],
        "info": {"param_info_key": "param_info_value"}
    }) as mock_params, \
        patch.object(LinearCurveValidator, "boundary_tests", return_value={
            "errors": ["boundary err"],
            "warnings": [],
            "info": {"boundary_info_key": True}
        }) as mock_boundary, \
        patch.object(LinearCurveValidator, "scenario_tests", return_value={
            "errors": [],
            "warnings": ["scenario warn"],
            "info": {"scenario_info_key": 123}
        }) as mock_scenario:
        results = LinearCurveValidator.run_all_validations(curve, params)

    # Confirm each sub-check called once
    mock_params.assert_called_once_with(params, curve.options)
    mock_boundary.assert_called_once_with(curve)
    mock_scenario.assert_called_once_with(curve)

    # Aggregated errors => param error1, error2, plus boundary err
    assert len(results["errors"]) == 3
    assert "param error1" in results["errors"]
    assert "param error2" in results["errors"]
    assert "boundary err" in results["errors"]

    # Aggregated warnings => param warn1 + scenario warn
    assert len(results["warnings"]) == 2
    assert "param warn1" in results["warnings"]
    assert "scenario warn" in results["warnings"]

    # Aggregated info => from param_info, boundary_info, scenario_info
    assert "param_info_key" in results["info"]
    assert results["info"]["param_info_key"] == "param_info_value"
    assert "boundary_info_key" in results["info"]
    assert results["info"]["boundary_info_key"] is True
    assert "scenario_info_key" in results["info"]
    assert results["info"]["scenario_info_key"] == 123


def test_run_all_validations_all_good(valid_curve_and_params):
    """
    Unmocked calls => if everything is valid => no errors/warnings.
    We rely on the real sub-calls. If your curve is truly valid & the scenario is not extreme,
    we expect no issues.
    """
    curve, params = valid_curve_and_params

    results = LinearCurveValidator.run_all_validations(curve, params)
    assert results["errors"] == []
    assert results["warnings"] == []
    assert "info" in results
    # e.g. "param_summary", "boundary_tests_run", "final_supply_after_scenario", etc.
