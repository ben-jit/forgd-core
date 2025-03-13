import pytest

from decimal import Decimal
from unittest.mock import patch

from forgd_core.common.model import BondingCurveParams, StepConfig, BondingCurveState, TransactionResult
from forgd_core.common.enums import BondingCurveType
from forgd_core.curves.single.stepwise import StepwiseBondingCurve
from forgd_core.validation.stepwise_validator import StepwiseCurveValidator


@pytest.fixture
def valid_stepwise_params():
    """
    Returns a BondingCurveParams with a minimal valid set of steps for a stepwise curve.
    Each step has ascending supply_threshold and non-negative prices.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.STEPWISE,
        steps=[
            StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
            StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
            StepConfig(supply_threshold=Decimal("300"), price=Decimal("3")),
        ]
    )
    return params


@pytest.fixture
def valid_options():
    """
    A minimal valid options dict, with no negative values.
    """
    return {
        "max_supply": Decimal("1000"),
        "max_liquidity": Decimal("5000"),
        "txn_fee_rate": Decimal("0.01")
    }


@pytest.fixture
def stepwise_curve_fixture():
    """
    Returns a minimal StepwiseBondingCurve with valid steps, liquidity, etc.
    Adjust as needed or adapt an existing fixture if you have one.
    """
    from forgd_core.common.enums import BondingCurveType
    from forgd_core.common.model import Liquidity, Token

    # 1) BondingCurveParams with some valid steps
    params = BondingCurveParams(
        curve_type=BondingCurveType.STEPWISE,
        steps=[
            StepConfig(supply_threshold=Decimal("100"), price=Decimal("1")),
            StepConfig(supply_threshold=Decimal("200"), price=Decimal("2")),
            StepConfig(supply_threshold=Decimal("11000"), price=Decimal("3")),
        ]
    )
    # 2) Provide liquidity so buy won't fail
    token = Token("USDC", "USDC", 6)
    liquidity = Liquidity(token=token, amount=Decimal("12000"))
    # 3) BondingCurveState
    state = BondingCurveState(
        current_supply=Decimal("150"),
        current_price=Decimal("2"),
        liquidity=liquidity
    )
    # 4) Create StepwiseBondingCurve
    curve = StepwiseBondingCurve(params=params, state=state)
    return curve


def test_validate_params_all_valid(valid_stepwise_params, valid_options):
    """
    If steps are non-empty, strictly ascending, prices>=0,
    and options do not contain negative fields => no errors/warnings.
    """
    result = StepwiseCurveValidator.validate_params(valid_stepwise_params, valid_options)
    assert "errors" in result
    assert "warnings" in result
    assert "info" in result

    assert not result["errors"], f"Expected no errors, got {result['errors']}"
    assert not result["warnings"], f"Expected no warnings, got {result['warnings']}"

    info = result["info"]
    # param_summary => "num_steps", "max_supply", "max_liquidity", "txn_fee_rate"
    assert info["param_summary"]["num_steps"] == 3, "We have 3 valid steps."
    assert info["param_summary"]["max_supply"] == "1000"
    assert info["param_summary"]["max_liquidity"] == "5000"
    assert info["param_summary"]["txn_fee_rate"] == "0.01"


def test_validate_params_negative_supply_threshold(valid_stepwise_params, valid_options):
    """
    If any step has a negative supply_threshold => error.
    """
    valid_stepwise_params.steps[1].supply_threshold = Decimal("-1")  # negative
    result = StepwiseCurveValidator.validate_params(valid_stepwise_params, valid_options)
    assert "Step 1 has negative supply_threshold -1." in result["errors"]


def test_validate_params_negative_price(valid_stepwise_params, valid_options):
    """
    If any step has a negative price => error.
    """
    valid_stepwise_params.steps[0].price = Decimal("-0.5")
    result = StepwiseCurveValidator.validate_params(valid_stepwise_params, valid_options)
    assert "Step 0 has negative price -0.5." in result["errors"]


def test_validate_params_non_strict_ascending(valid_stepwise_params, valid_options):
    """
    If steps[i].supply_threshold <= steps[i-1].supply_threshold => error about ascending thresholds.
    """
    # Make step1 threshold == 100 => not strictly ascending from step0(100)
    valid_stepwise_params.steps[1].supply_threshold = Decimal("100")
    result = StepwiseCurveValidator.validate_params(valid_stepwise_params, valid_options)
    assert any("Steps must be strictly ascending." in e for e in result["errors"])


@pytest.mark.parametrize(
    "opt_key,opt_value,expected_msg",
    [
        ("max_supply", Decimal("-10"), "StepwiseCurve: 'max_supply' cannot be negative."),
        ("max_liquidity", Decimal("-5"), "StepwiseCurve: 'max_liquidity' cannot be negative."),
        ("txn_fee_rate", Decimal("-0.01"), "StepwiseCurve: 'txn_fee_rate' cannot be negative."),
    ]
)
def test_validate_params_negative_options(valid_stepwise_params, valid_options, opt_key, opt_value, expected_msg):
    """
    If max_supply<0 or max_liquidity<0 or txn_fee_rate<0 => error.
    """
    valid_options[opt_key] = opt_value
    result = StepwiseCurveValidator.validate_params(valid_stepwise_params, valid_options)
    assert expected_msg in result["errors"], f"Expected error: {expected_msg}, got {result['errors']}"


def test_boundary_tests_all_good(stepwise_curve_fixture):
    """
    get_spot_price(0) >=0, cost(0)=0, cost(10000) >=0 => no errors, no warnings.
    """
    result = StepwiseCurveValidator.boundary_tests(stepwise_curve_fixture)
    assert "errors" in result
    assert "warnings" in result
    assert "info" in result

    # Typically no errors, warnings if the curve is valid,
    # and the large buy doesn't produce negative cost.
    assert result["errors"] == []
    assert result["warnings"] == []
    assert result["info"].get("boundary_tests_run") is True


def test_boundary_tests_spot_price_zero_negative(stepwise_curve_fixture):
    """
    get_spot_price(0) => negative => we add an error:
    'Spot price is negative at supply=0.'
    """
    with patch.object(stepwise_curve_fixture, "get_spot_price", return_value=Decimal("-1")):
        result = StepwiseCurveValidator.boundary_tests(stepwise_curve_fixture)
    assert "Spot price is negative at supply=0." in result["errors"]
    # No other errors or warnings
    assert len(result["errors"]) == 1


def test_boundary_tests_spot_price_zero_exception(stepwise_curve_fixture):
    """
    get_spot_price(0) => raises => we add a warning:
    'Exception calling get_spot_price(0): ...'
    """
    with patch.object(stepwise_curve_fixture, "get_spot_price", side_effect=Exception("test error")):
        result = StepwiseCurveValidator.boundary_tests(stepwise_curve_fixture)
    # We expect a warning about that exception
    assert any("Exception calling get_spot_price(0): test error" in w for w in result["warnings"])
    # No error about negative price
    assert result["errors"] == []


def test_boundary_tests_cost_zero_nonzero_warning(stepwise_curve_fixture):
    """
    If calculate_purchase_cost(0) => nonzero => we get a warning:
    'Cost to buy 0 tokens is not zero: got X'
    """
    with patch.object(stepwise_curve_fixture, "calculate_purchase_cost", return_value=Decimal("5")):
        result = StepwiseCurveValidator.boundary_tests(stepwise_curve_fixture)
    assert any("Cost to buy 0 tokens is not zero: got 5" in w for w in result["warnings"])
    # No errors
    assert result["errors"] == []


def test_boundary_tests_cost_zero_exception(stepwise_curve_fixture):
    """
    If calculate_purchase_cost(0) => raises => we add an error:
    'Exception calling calculate_purchase_cost(0): ...'
    """

    def side_effect_cost(amount):
        if amount == Decimal("0"):
            raise Exception("test cost error")  # only for zero
        # For 10000 or anything else => normal
        return Decimal("9999")

    with patch.object(stepwise_curve_fixture, "calculate_purchase_cost", side_effect=side_effect_cost):
        result = StepwiseCurveValidator.boundary_tests(stepwise_curve_fixture)

    assert any("Exception calling calculate_purchase_cost(0): test cost error" in e for e in result["errors"])
    # No mention of 10000 => so you get no warnings
    assert result["warnings"] == []


def test_boundary_tests_large_buy_negative(stepwise_curve_fixture):
    """
    If calculate_purchase_cost(10000) => negative => we add an error:
    'Cost to buy 10k tokens is negative.'
    """
    def side_effect_cost(amount):
        if amount == Decimal("10000"):
            return Decimal("-500")
        return Decimal("0")

    with patch.object(stepwise_curve_fixture, "calculate_purchase_cost", side_effect=side_effect_cost):
        result = StepwiseCurveValidator.boundary_tests(stepwise_curve_fixture)
    # We expect "Cost to buy 10k tokens is negative." in errors
    assert any("Cost to buy 10k tokens is negative." in e for e in result["errors"])
    # no warnings
    assert result["warnings"] == []


def test_boundary_tests_large_buy_exception(stepwise_curve_fixture):
    """
    If calculate_purchase_cost(10000) => raises => we add a warning:
    'Exception calling calculate_purchase_cost(10000): ...'
    """
    def side_effect_cost(amount):
        if amount== Decimal("10000"):
            raise Exception("large buy error")
        return Decimal("10")

    with patch.object(stepwise_curve_fixture, "calculate_purchase_cost", side_effect=side_effect_cost):
        result= StepwiseCurveValidator.boundary_tests(stepwise_curve_fixture)
    # a warning => "Exception calling calculate_purchase_cost(10000): large buy error"
    assert any("Exception calling calculate_purchase_cost(10000): large buy error" in w for w in result["warnings"])
    assert result["errors"]==[]


def test_scenario_tests_all_good(stepwise_curve_fixture):
    """
    No exceptions, no negative cost or supply => no errors, no warnings.
    """
    result = StepwiseCurveValidator.scenario_tests(stepwise_curve_fixture)
    assert "errors" in result
    assert "warnings" in result
    assert "info" in result

    assert result["errors"] == []
    assert result["warnings"] == []
    # final_supply_after_scenario in info
    assert "final_supply_after_scenario" in result["info"]


def test_scenario_tests_buy1_exception(stepwise_curve_fixture):
    """
    If buy(100) raises => we record that in errors =>
    code presumably continues to buy(200), sell(50).
    """
    def side_effect_buy(request):
        if request.amount == Decimal("100"):
            raise Exception("mocked buy(100) error")
        else:
            # normal result
            return TransactionResult(
                executed_amount=request.amount,
                total_cost=Decimal("100"),
                average_price=Decimal("1"),
                new_supply=stepwise_curve_fixture._state.current_supply + request.amount,
                timestamp=None
            )

    with patch.object(stepwise_curve_fixture, "buy", side_effect=side_effect_buy) as mock_buy, \
         patch.object(stepwise_curve_fixture, "sell") as mock_sell:

        result = StepwiseCurveValidator.scenario_tests(stepwise_curve_fixture)

    # "Exception in scenario step buy(100): mocked buy(100) error" in errors
    assert any("Exception in scenario step buy(100): mocked buy(100) error" in e for e in result["errors"])
    # the scenario code attempts buy(200) and sell(50) too
    # confirm the final supply is recorded
    assert "final_supply_after_scenario" in result["info"]


def test_scenario_tests_buy2_negative_cost(stepwise_curve_fixture):
    """
    The first buy(100) is normal, second buy(200) yields negative total_cost => triggers an error.
    """
    def side_effect_buy(request):
        if request.amount == Decimal("100"):
            # normal
            return TransactionResult(
                executed_amount=Decimal("100"),
                total_cost=Decimal("100"),
                average_price=Decimal("1"),
                new_supply=stepwise_curve_fixture._state.current_supply + Decimal("100"),
                timestamp=None
            )
        elif request.amount == Decimal("200"):
            # negative cost => "buy(200) => negative total cost."
            return TransactionResult(
                executed_amount=Decimal("200"),
                total_cost=Decimal("-50"),
                average_price=Decimal("-0.25"),
                new_supply=stepwise_curve_fixture._state.current_supply + Decimal("300"),
                timestamp=None
            )
        return TransactionResult(
            Decimal("0"), Decimal("0"), Decimal("0"), stepwise_curve_fixture._state.current_supply, None
        )

    with patch.object(stepwise_curve_fixture, "buy", side_effect=side_effect_buy), \
         patch.object(stepwise_curve_fixture, "sell", return_value=TransactionResult(
            executed_amount=Decimal("50"),
            total_cost=Decimal("80"),
            average_price=Decimal("1.6"),
            new_supply=Decimal("350"),
            timestamp=None
         )) as mock_sell:
        result = StepwiseCurveValidator.scenario_tests(stepwise_curve_fixture)

    assert any("buy(200) => negative total cost." in e for e in result["errors"])
    # scenario presumably calls sell(50) => mock_sell
    mock_sell.assert_called_once()


def test_scenario_tests_sell_exception(stepwise_curve_fixture):
    """
    If sell(50) raises => we record it => scenario code logs an error.
    """
    def side_effect_buy(request):
        # normal results for buy(100) and buy(200)
        return TransactionResult(
            executed_amount=request.amount,
            total_cost=Decimal("100"),
            average_price=Decimal("1"),
            new_supply=stepwise_curve_fixture._state.current_supply + request.amount,
            timestamp=None
        )

    def side_effect_sell(request):
        raise Exception("mocked sell(50) failure")

    with patch.object(stepwise_curve_fixture, "buy", side_effect=side_effect_buy), \
         patch.object(stepwise_curve_fixture, "sell", side_effect=side_effect_sell):

        result = StepwiseCurveValidator.scenario_tests(stepwise_curve_fixture)

    assert any("Exception in scenario step sell(50): mocked sell(50) failure" in e for e in result["errors"])


def test_scenario_tests_sell_negative_supply(stepwise_curve_fixture):
    """
    If sell(50) results in negative new_supply => triggers an error.
    """
    def side_effect_buy(request):
        # normal for buy(100) & buy(200)
        return TransactionResult(
            executed_amount=request.amount,
            total_cost=Decimal("150"),
            average_price=Decimal("1.5"),
            new_supply=stepwise_curve_fixture._state.current_supply + request.amount,
            timestamp=None
        )

    def side_effect_sell(request):
        # negative supply => e.g. -100
        return TransactionResult(
            executed_amount=Decimal("50"),
            total_cost=Decimal("80"),
            average_price=Decimal("1.6"),
            new_supply=Decimal("-100"),
            timestamp=None
        )

    with patch.object(stepwise_curve_fixture, "buy", side_effect=side_effect_buy), \
         patch.object(stepwise_curve_fixture, "sell", side_effect=side_effect_sell):
        result = StepwiseCurveValidator.scenario_tests(stepwise_curve_fixture)

    assert any("new_supply is negative after sell(50)." in e for e in result["errors"])


def test_scenario_tests_multiple_issues(stepwise_curve_fixture):
    """
    If multiple steps produce issues => we gather multiple errors in the list.
    e.g. buy(100)=> negative cost, buy(200)=> negative supply, sell(50)=> raises an exception
    """
    def side_effect_buy(request):
        if request.amount == Decimal("100"):
            return TransactionResult(
                executed_amount=Decimal("100"),
                total_cost=Decimal("-10"),  # negative cost
                average_price=Decimal("-0.1"),
                new_supply=stepwise_curve_fixture._state.current_supply + Decimal("100"),
                timestamp=None
            )
        elif request.amount == Decimal("200"):
            return TransactionResult(
                executed_amount=Decimal("200"),
                total_cost=Decimal("200"),
                average_price=Decimal("1"),
                new_supply=Decimal("-500"),  # negative supply
                timestamp=None
            )
        return TransactionResult(Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), None)

    def side_effect_sell(request):
        # raise exception
        raise Exception("mocked sell error")

    with patch.object(stepwise_curve_fixture, "buy", side_effect=side_effect_buy), \
         patch.object(stepwise_curve_fixture, "sell", side_effect=side_effect_sell):
        result= StepwiseCurveValidator.scenario_tests(stepwise_curve_fixture)

    # Expect 3 errors:
    # 1) "buy(100)=> negative total cost."
    # 2) "new_supply is negative after buy(200)."
    # 3) "Exception in scenario step sell(50): mocked sell error"
    errs= result["errors"]
    assert len(errs)==3
    assert "buy(100) => negative total cost." in errs
    assert "new_supply is negative after buy(200)." in errs
    assert any("Exception in scenario step sell(50): mocked sell error" in e for e in errs)
    # No warnings
    assert result["warnings"]==[]


def test_run_all_validations_success(stepwise_curve_fixture):
    """
    Unmocked: If everything is valid => we expect no errors/warnings.

    stepwise_curve_fixture uses steps up to threshold=11000,
    and liquidity=12000, so buy(100)+buy(200)+sell(50) won't revert
    or go negative by default.
    """
    # We'll also retrieve the curve_params from the fixture if needed:
    curve_params = stepwise_curve_fixture.params  # or your code's approach

    results = StepwiseCurveValidator.run_all_validations(stepwise_curve_fixture, curve_params)
    assert "errors" in results
    assert "warnings" in results
    assert "info" in results

    # If the fixture and scenario are large enough => no negative or exceptions => no errors or warnings
    assert results["errors"] == []
    assert results["warnings"] == []
    # info might contain param_summary, boundary_tests_run, final_supply_after_scenario, etc.
    assert results["info"]


def test_run_all_validations_aggregates_subcalls(stepwise_curve_fixture):
    """
    We mock validate_params, boundary_tests, scenario_tests => each returns distinct errors/warnings/info.
    Then run_all_validations => ensures they aggregate properly.
    """
    # Suppose we load curve_params from the fixture
    curve_params = stepwise_curve_fixture.params

    with patch.object(StepwiseCurveValidator, "validate_params", return_value={
        "errors": ["param_err1"],
        "warnings": ["param_warn1"],
        "info": {"param_info_key": "param_info_val"}
    }) as mock_params, \
        patch.object(StepwiseCurveValidator, "boundary_tests", return_value={
            "errors": ["boundary_err1", "boundary_err2"],
            "warnings": [],
            "info": {"boundary_info_key": True}
        }) as mock_boundary, \
        patch.object(StepwiseCurveValidator, "scenario_tests", return_value={
            "errors": [],
            "warnings": ["scenario_warn1"],
            "info": {"scenario_info_key": 123}
        }) as mock_scenario:
        results = StepwiseCurveValidator.run_all_validations(stepwise_curve_fixture, curve_params)

    # Check each sub-call is invoked
    mock_params.assert_called_once_with(curve_params, stepwise_curve_fixture.options)
    mock_boundary.assert_called_once_with(stepwise_curve_fixture)
    mock_scenario.assert_called_once_with(stepwise_curve_fixture)

    # Aggregated errors => param_err1 + boundary_err1 + boundary_err2
    assert len(results["errors"]) == 3
    assert "param_err1" in results["errors"]
    assert "boundary_err1" in results["errors"]
    assert "boundary_err2" in results["errors"]

    # Aggregated warnings => param_warn1 + scenario_warn1
    assert len(results["warnings"]) == 2
    assert "param_warn1" in results["warnings"]
    assert "scenario_warn1" in results["warnings"]

    # Aggregated info => param_info_key, boundary_info_key, scenario_info_key
    assert results["info"]["param_info_key"] == "param_info_val"
    assert results["info"]["boundary_info_key"] is True
    assert results["info"]["scenario_info_key"] == 123


def test_run_all_validations_scenario_fail(stepwise_curve_fixture):
    """
    Suppose the scenario test fails => aggregator should collect those errors in results.
    We'll mock scenario_tests => returns some errors.
    """
    curve_params = stepwise_curve_fixture.params

    with patch.object(StepwiseCurveValidator, "validate_params", return_value={
        "errors": [],
        "warnings": [],
        "info": {"param_info": True}
    }) as mock_params, \
        patch.object(StepwiseCurveValidator, "boundary_tests", return_value={
            "errors": [],
            "warnings": [],
            "info": {"boundary_info": True}
        }) as mock_boundary, \
        patch.object(StepwiseCurveValidator, "scenario_tests", return_value={
            "errors": ["buy(100) => negative cost", "sell(50)=> exception"],
            "warnings": [],
            "info": {"scenario_info": 999}
        }) as mock_scenario:
        results = StepwiseCurveValidator.run_all_validations(stepwise_curve_fixture, curve_params)

    # we accumulate errors from scenario
    assert len(results["errors"]) == 2
    assert "buy(100) => negative cost" in results["errors"]
    assert "sell(50)=> exception" in results["errors"]


def test_run_all_validations_boundary_fail(stepwise_curve_fixture):
    """
    Suppose boundary_tests => some warnings or errors => aggregator includes them.
    """
    curve_params = stepwise_curve_fixture.params

    with patch.object(StepwiseCurveValidator, "validate_params", return_value={
        "errors": [],
        "warnings": [],
        "info": {}
    }) as mock_params, \
        patch.object(StepwiseCurveValidator, "boundary_tests", return_value={
            "errors": ["Spot price negative at 0."],
            "warnings": ["Cost to buy(0) nonzero?"],
            "info": {"boundary_done": True}
        }) as mock_boundary, \
        patch.object(StepwiseCurveValidator, "scenario_tests", return_value={
            "errors": [],
            "warnings": [],
            "info": {}
        }) as mock_scenario:
        results = StepwiseCurveValidator.run_all_validations(stepwise_curve_fixture, curve_params)

    assert len(results["errors"]) == 1
    assert "Spot price negative at 0." in results["errors"]
    assert results["warnings"] == ["Cost to buy(0) nonzero?"]
    assert results["info"]["boundary_done"] is True

