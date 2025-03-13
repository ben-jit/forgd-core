import pytest

from decimal import Decimal
from unittest.mock import patch

from forgd_core.common.model import BondingCurveParams, BondingCurveState, Liquidity, Token, TransactionResult
from forgd_core.common.enums import BondingCurveType
from forgd_core.curves.single.exponential import ExponentialBondingCurve
from forgd_core.validation.exponential_validator import ExponentialCurveValidator



@pytest.fixture
def valid_exponential_params():
    """
    Returns a BondingCurveParams for an exponential curve with p0>0, alpha>=0.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.EXPONENTIAL,
        initial_price=Decimal("1.0"),   # p0
        exponential=Decimal("0.1"),     # alpha
    )
    return params


@pytest.fixture
def valid_options():
    """
    A minimal valid options dict (max_supply, max_liquidity, txn_fee_rate all >=0 or None).
    """
    return {
        "max_supply": Decimal("1000"),
        "max_liquidity": Decimal("5000"),
        "txn_fee_rate": Decimal("0.01")
    }


@pytest.fixture
def valid_exponential_curve():
    """
    Returns a *real* ExponentialBondingCurve with valid parameters & state.
    By default:
      - p0 = 1.0
      - alpha = 0.1
      - current_supply = 100
      - liquidity.amount = 500
    These defaults ensure no negative or zero edge-cases by design.
    """
    params = BondingCurveParams(
        curve_type=BondingCurveType.EXPONENTIAL,
        initial_price=Decimal("1.0"),   # p0
        exponential=Decimal("0.1"),     # alpha
    )
    state = BondingCurveState(
        current_supply=Decimal("100"),
        current_price=Decimal("0"),
        liquidity=Liquidity(
            token=None,
            amount=Decimal("500")
        )
    )
    curve = ExponentialBondingCurve(params=params, state=state)
    return curve


@pytest.fixture
def exponential_curve_fixture():
    """
    Returns an ExponentialBondingCurve with p0>0, alpha>=0, plus liquidity, etc.
    We'll define it so 'buy(100)+buy(200)+sell(50)' doesn't fail normally.
    """
    from forgd_core.common.model import BondingCurveParams, BondingCurveState, Liquidity
    from forgd_core.common.enums import BondingCurveType
    from forgd_core.curves.single.exponential import ExponentialBondingCurve

    params = BondingCurveParams(
        curve_type=BondingCurveType.EXPONENTIAL,
        initial_price=Decimal("1.0"),   # p0
        exponential=Decimal("0.1")      # alpha
    )
    # Provide enough liquidity => no revert on buy
    liquidity = Liquidity(
        token=Token("USD", "USDC", 6),
        amount=Decimal("50000")
    )
    state = BondingCurveState(
        current_supply=Decimal("200"),
        current_price=Decimal("1.5"),
        liquidity=liquidity
    )
    # Create the curve
    curve = ExponentialBondingCurve(params, state)
    # Possibly set pro_rata or other options if needed
    return curve


def test_validate_params_all_valid(valid_exponential_params, valid_options):
    """
    If p0>0, alpha>=0, and no negative fields in options => no errors/warnings.
    """
    result = ExponentialCurveValidator.validate_params(valid_exponential_params, valid_options)
    assert "errors" in result
    assert "warnings" in result
    assert "info" in result

    assert not result["errors"], f"Expected no errors, got: {result['errors']}"
    assert not result["warnings"], f"Expected no warnings, got: {result['warnings']}"

    # param_summary => check p0, alpha, etc.
    info = result["info"]
    summary = info["param_summary"]
    assert summary["p0"] == "1.0"
    assert summary["alpha"] == "0.1"
    assert summary["max_supply"] == "1000"
    assert summary["max_liquidity"] == "5000"
    assert summary["txn_fee_rate"] == "0.01"


def test_validate_params_p0_zero_or_negative(valid_exponential_params, valid_options):
    """
    If p0<=0 => "ExponentialCurve: 'p0' (base price) must be > 0."
    """
    # 1) p0=0
    valid_exponential_params.initial_price = Decimal("0")
    result = ExponentialCurveValidator.validate_params(valid_exponential_params, valid_options)
    assert "ExponentialCurve: 'p0' (base price) must be > 0." in result["errors"]

    # 2) p0<0
    valid_exponential_params.initial_price = Decimal("-1")
    result2 = ExponentialCurveValidator.validate_params(valid_exponential_params, valid_options)
    assert "ExponentialCurve: 'p0' (base price) must be > 0." in result2["errors"]


def test_validate_params_alpha_negative(valid_exponential_params, valid_options):
    """
    If alpha<0 => "ExponentialCurve: 'alpha' must be >=0"
    """
    valid_exponential_params.exponential = Decimal("-0.01")
    result = ExponentialCurveValidator.validate_params(valid_exponential_params, valid_options)
    assert "ExponentialCurve: 'alpha' must be >= 0" in result["errors"][0]


@pytest.mark.parametrize(
    "opt_key, opt_value, expected_msg",
    [
        ("max_supply", Decimal("-1"), "ExponentialCurve: 'max_supply' cannot be negative."),
        ("max_liquidity", Decimal("-10"), "ExponentialCurve: 'max_liquidity' cannot be negative."),
        ("txn_fee_rate", Decimal("-0.05"), "ExponentialCurve: 'txn_fee_rate' cannot be negative."),
    ]
)
def test_validate_params_negative_options(valid_exponential_params, valid_options, opt_key, opt_value, expected_msg):
    """
    If the options contain negative max_supply, max_liquidity, or txn_fee_rate => error.
    """
    valid_options[opt_key] = opt_value
    result = ExponentialCurveValidator.validate_params(valid_exponential_params, valid_options)
    assert expected_msg in result["errors"], f"Expected error '{expected_msg}', got: {result['errors']}"


def test_boundary_tests_happy_path(valid_exponential_curve):
    """
    Test that boundary_tests returns no errors and no warnings
    under normal conditions:
      - get_spot_price(0) >= 0  (== p0 = 1.0 in this setup)
      - calculate_purchase_cost(0) == 0
      - large supply call does not raise or return negative
    """
    result = ExponentialCurveValidator.boundary_tests(valid_exponential_curve)
    assert "errors" in result
    assert "warnings" in result
    assert "info" in result

    assert not result["errors"], f"Expected no errors but got: {result['errors']}"
    assert not result["warnings"], f"Expected no warnings but got: {result['warnings']}"
    assert result["info"].get("boundary_tests_run") is True


def test_boundary_tests_negative_spot_price(valid_exponential_curve, monkeypatch):
    """
    Force get_spot_price(0) to return a negative value and confirm it's reported as an error.
    """
    def mock_negative_spot_price(_):
        return Decimal("-1")

    monkeypatch.setattr(valid_exponential_curve, "get_spot_price", mock_negative_spot_price)

    result = ExponentialCurveValidator.boundary_tests(valid_exponential_curve)
    assert len(result["errors"]) == 2
    assert "Spot price is negative at supply=0" in result["errors"][0]
    assert not result["warnings"]


def test_boundary_tests_nonzero_cost_for_zero_purchase(valid_exponential_curve, monkeypatch):
    """
    Force calculate_purchase_cost(0) to return non-zero and confirm it's reported as a warning.
    """
    def mock_cost_zero_purchase(_):
        return Decimal("5")

    monkeypatch.setattr(valid_exponential_curve, "calculate_purchase_cost", mock_cost_zero_purchase)

    result = ExponentialCurveValidator.boundary_tests(valid_exponential_curve)
    assert not result["errors"]
    assert len(result["warnings"]) == 1
    assert "Cost to buy 0 tokens is not zero: got 5" in result["warnings"][0]


def test_boundary_tests_exception_spot_price_zero_supply(valid_exponential_curve, monkeypatch):
    """
    Force get_spot_price(0) to raise an exception; all other calls remain normal.
    """
    original_spot_price = valid_exponential_curve.get_spot_price

    def mock_spot_price_exception(supply):
        if supply == 0:
            raise RuntimeError("Simulated spot price error")
        # For other supplies, call the original method
        return original_spot_price(supply)

    monkeypatch.setattr(valid_exponential_curve, "get_spot_price", mock_spot_price_exception)

    result = ExponentialCurveValidator.boundary_tests(valid_exponential_curve)

    # We expect exactly 1 error (for supply=0) and 0 warnings.
    assert len(result["errors"]) == 1
    assert "Exception calling get_spot_price(0): Simulated spot price error" in result["errors"][0]
    assert not result["warnings"]


def test_boundary_tests_exception_spot_price_large_supply(valid_exponential_curve, monkeypatch):
    """
    Force get_spot_price(1e6) to raise an exception and confirm it's captured in warnings.
    """
    original_get_spot_price = valid_exponential_curve.get_spot_price

    def mock_spot_price_exception(supply):
        if supply == Decimal("1000000"):
            raise ValueError("Simulated large-supply spot price error")
        return original_get_spot_price(supply)

    monkeypatch.setattr(valid_exponential_curve, "get_spot_price", mock_spot_price_exception)

    result = ExponentialCurveValidator.boundary_tests(valid_exponential_curve)
    assert not result["errors"]
    assert len(result["warnings"]) == 1
    assert "Exception calling get_spot_price(1e6): Simulated large-supply spot price error" \
           in result["warnings"][0]


def test_boundary_tests_exception_calculate_purchase_cost_zero(valid_exponential_curve, monkeypatch):
    """
    Force calculate_purchase_cost(0) to raise an exception and confirm it's captured in errors.
    """
    def mock_cost_exception(_):
        raise ArithmeticError("Simulated cost error")

    monkeypatch.setattr(valid_exponential_curve, "calculate_purchase_cost", mock_cost_exception)

    result = ExponentialCurveValidator.boundary_tests(valid_exponential_curve)
    assert len(result["errors"]) == 1
    assert "Exception calling calculate_purchase_cost(0): Simulated cost error" in result["errors"][0]
    assert not result["warnings"]


def test_scenario_tests_all_good(exponential_curve_fixture):
    """
    If buy(100), buy(200), sell(50) produce no exceptions or negative cost/supply => no errors/warnings.
    """
    result = ExponentialCurveValidator.scenario_tests(exponential_curve_fixture)
    assert "errors" in result
    assert "warnings" in result
    assert "info" in result

    assert result["errors"] == []
    assert result["warnings"] == []
    # final_supply_after_scenario in info
    assert "final_supply_after_scenario" in result["info"]


def test_scenario_tests_buy100_exception(exponential_curve_fixture):
    """
    If buy(100) raises => we record it in errors => scenario continues to buy(200), sell(50).
    """

    def side_effect_buy(request):
        if request.amount == Decimal("100"):
            raise Exception("mocked buy(100) error")
        return TransactionResult(
            executed_amount=request.amount,
            total_cost=Decimal("100"),
            average_price=Decimal("1"),
            new_supply=exponential_curve_fixture._state.current_supply + request.amount,
            timestamp=None
        )

    with patch.object(exponential_curve_fixture, "buy", side_effect=side_effect_buy) as mock_buy, \
        patch.object(exponential_curve_fixture, "sell") as mock_sell:
        result = ExponentialCurveValidator.scenario_tests(exponential_curve_fixture)

    # "Exception in scenario step buy(100): mocked buy(100) error" in errors
    assert any("Exception in scenario step buy(100): mocked buy(100) error" in e for e in result["errors"])
    # scenario presumably attempts buy(200) and sell(50) anyway
    assert "final_supply_after_scenario" in result["info"]


def test_scenario_tests_buy200_negative_cost(exponential_curve_fixture):
    """
    If buy(200) => negative total cost => 'buy(200) => negative total cost.' in errors.
    """

    def side_effect_buy(request):
        if request.amount == Decimal("100"):
            return TransactionResult(
                executed_amount=Decimal("100"),
                total_cost=Decimal("120"),
                average_price=Decimal("1.2"),
                new_supply=exponential_curve_fixture._state.current_supply + Decimal("100"),
                timestamp=None
            )
        elif request.amount == Decimal("200"):
            return TransactionResult(
                executed_amount=Decimal("200"),
                total_cost=Decimal("-10"),  # negative
                average_price=Decimal("-0.05"),
                new_supply=exponential_curve_fixture._state.current_supply + Decimal("300"),
                timestamp=None
            )
        return TransactionResult(Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), None)

    with patch.object(exponential_curve_fixture, "buy", side_effect=side_effect_buy) as mock_buy, \
        patch.object(exponential_curve_fixture, "sell", return_value=TransactionResult(
            executed_amount=Decimal("50"),
            total_cost=Decimal("80"),
            average_price=Decimal("1.6"),
            new_supply=Decimal("450"),
            timestamp=None
        )) as mock_sell:

        result = ExponentialCurveValidator.scenario_tests(exponential_curve_fixture)

    assert any("buy(200) => negative total cost." in e for e in result["errors"])
    # scenario calls sell(50) => mock_sell
    mock_sell.assert_called_once()


def test_scenario_tests_sell_exception(exponential_curve_fixture):
    """
    If sell(50) => raises => 'Exception in scenario step sell(50): ...' in errors.
    """

    def side_effect_buy(request):
        # normal
        return TransactionResult(
            executed_amount=request.amount,
            total_cost=Decimal("100"),
            average_price=Decimal("1"),
            new_supply=exponential_curve_fixture._state.current_supply + request.amount,
            timestamp=None
        )

    def side_effect_sell(request):
        raise Exception("mocked sell(50) error")

    with patch.object(exponential_curve_fixture, "buy", side_effect=side_effect_buy), \
        patch.object(exponential_curve_fixture, "sell", side_effect=side_effect_sell):
        result = ExponentialCurveValidator.scenario_tests(exponential_curve_fixture)

    assert any("Exception in scenario step sell(50): mocked sell(50) error" in e for e in result["errors"])


def test_scenario_tests_sell_negative_supply(exponential_curve_fixture):
    """
    If sell(50) => new_supply <0 => 'Supply is negative after sell(50).' in errors.
    """

    def side_effect_buy(request):
        # normal
        return TransactionResult(
            executed_amount=request.amount,
            total_cost=Decimal("150"),
            average_price=Decimal("1.5"),
            new_supply=exponential_curve_fixture._state.current_supply + request.amount,
            timestamp=None
        )

    def side_effect_sell(request):
        return TransactionResult(
            executed_amount=Decimal("50"),
            total_cost=Decimal("120"),
            average_price=Decimal("2.4"),
            new_supply=Decimal("-100"),  # negative
            timestamp=None
        )

    with patch.object(exponential_curve_fixture, "buy", side_effect=side_effect_buy), \
        patch.object(exponential_curve_fixture, "sell", side_effect=side_effect_sell):
        result = ExponentialCurveValidator.scenario_tests(exponential_curve_fixture)

    assert any("Supply is negative after sell(50)." in e for e in result["errors"])


def test_scenario_tests_multiple_issues(exponential_curve_fixture):
    """
    If multiple steps produce issues => we gather multiple errors.
    e.g. buy(100)=> negative cost, buy(200)=> negative supply, sell(50)=> raises exception
    """

    def side_effect_buy(request):
        if request.amount == Decimal("100"):
            return TransactionResult(
                executed_amount=Decimal("100"),
                total_cost=Decimal("-20"),  # negative
                average_price=Decimal("-0.2"),
                new_supply=exponential_curve_fixture._state.current_supply + Decimal("100"),
                timestamp=None
            )
        elif request.amount == Decimal("200"):
            return TransactionResult(
                executed_amount=Decimal("200"),
                total_cost=Decimal("300"),
                average_price=Decimal("1.5"),
                new_supply=Decimal("-200"),  # negative supply
                timestamp=None
            )
        return TransactionResult(Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), None)

    def side_effect_sell(request):
        # raise
        raise Exception("some sell error")

    with patch.object(exponential_curve_fixture, "buy", side_effect=side_effect_buy), \
        patch.object(exponential_curve_fixture, "sell", side_effect=side_effect_sell):
        result = ExponentialCurveValidator.scenario_tests(exponential_curve_fixture)

    errs = result["errors"]
    # 1) buy(100) => negative cost => "buy(100) => negative total cost."
    # 2) buy(200) => negative supply => "Supply is negative after buy(200)."
    # 3) sell(50) => exception => "Exception in scenario step sell(50): some sell error"
    assert len(errs) == 3
    assert any("buy(100) => negative total cost." in e for e in errs)
    assert any("Supply is negative after buy(200)." in e for e in errs)
    assert any("Exception in scenario step sell(50): some sell error" in e for e in errs)
    # no warnings
    assert result["warnings"] == []


def test_run_all_validations_success(exponential_curve_fixture):
    """
    If everything is valid => no errors/warnings.
    'exponential_curve_fixture' is presumably large enough so boundary & scenario steps
    do not fail, and param check is good.
    """
    # If you need the curve_params, you can fetch from the fixture
    curve_params = exponential_curve_fixture.params

    results = ExponentialCurveValidator.run_all_validations(exponential_curve_fixture, curve_params)
    assert "errors" in results
    assert "warnings" in results
    assert "info" in results

    assert results["errors"] == []
    assert results["warnings"] == []
    # info may contain param_summary, boundary_tests_run, final_supply_after_scenario, etc.
    assert results["info"]


def test_run_all_validations_aggregates_subcalls(exponential_curve_fixture):
    """
    We mock validate_params, boundary_tests, scenario_tests => each returns
    distinct errors/warnings/info => aggregator merges them in final results.
    """
    curve_params = exponential_curve_fixture.params

    with patch.object(ExponentialCurveValidator, "validate_params", return_value={
        "errors": ["param_err1", "param_err2"],
        "warnings": ["param_warn1"],
        "info": {"param_info_key": "param_info_val"}
    }) as mock_params, \
        patch.object(ExponentialCurveValidator, "boundary_tests", return_value={
            "errors": ["boundary_err"],
            "warnings": ["boundary_warn"],
            "info": {"boundary_info_key": True}
        }) as mock_boundary, \
        patch.object(ExponentialCurveValidator, "scenario_tests", return_value={
            "errors": [],
            "warnings": ["scenario_warn1", "scenario_warn2"],
            "info": {"scenario_info_key": 123}
        }) as mock_scenario:
        results = ExponentialCurveValidator.run_all_validations(exponential_curve_fixture, curve_params)

    # Subcalls
    mock_params.assert_called_once_with(curve_params, exponential_curve_fixture.options)
    mock_boundary.assert_called_once_with(exponential_curve_fixture)
    mock_scenario.assert_called_once_with(exponential_curve_fixture)

    # Aggregated errors => param_err1, param_err2, boundary_err
    assert len(results["errors"]) == 3
    assert "param_err1" in results["errors"]
    assert "param_err2" in results["errors"]
    assert "boundary_err" in results["errors"]

    # Aggregated warnings => param_warn1, boundary_warn, scenario_warn1, scenario_warn2
    assert len(results["warnings"]) == 4
    assert "param_warn1" in results["warnings"]
    assert "boundary_warn" in results["warnings"]
    assert "scenario_warn1" in results["warnings"]
    assert "scenario_warn2" in results["warnings"]

    # Aggregated info => param_info_key, boundary_info_key, scenario_info_key
    assert results["info"]["param_info_key"] == "param_info_val"
    assert results["info"]["boundary_info_key"] is True
    assert results["info"]["scenario_info_key"] == 123


def test_run_all_validations_param_fails(exponential_curve_fixture):
    """
    If param check fails => aggregator should include those errors
    even if boundary & scenario pass or not.
    """
    curve_params = exponential_curve_fixture.params

    with patch.object(ExponentialCurveValidator, "validate_params", return_value={
        "errors": ["ExponentialCurve: 'p0' must be >0."],
        "warnings": [],
        "info": {}
    }) as mock_params, \
        patch.object(ExponentialCurveValidator, "boundary_tests", return_value={
            "errors": [],
            "warnings": [],
            "info": {}
        }) as mock_boundary, \
        patch.object(ExponentialCurveValidator, "scenario_tests", return_value={
            "errors": [],
            "warnings": [],
            "info": {}
        }) as mock_scenario:
        results = ExponentialCurveValidator.run_all_validations(exponential_curve_fixture, curve_params)

    assert "ExponentialCurve: 'p0' must be >0." in results["errors"]
    assert results["warnings"] == []


def test_run_all_validations_scenario_fails(exponential_curve_fixture):
    """
    If scenario tests produce errors => aggregator includes them
    along with boundary & param results.
    """
    curve_params = exponential_curve_fixture.params

    with patch.object(ExponentialCurveValidator, "validate_params", return_value={
        "errors": [],
        "warnings": [],
        "info": {}
    }) as mock_params, \
        patch.object(ExponentialCurveValidator, "boundary_tests", return_value={
            "errors": [],
            "warnings": [],
            "info": {}
        }) as mock_boundary, \
        patch.object(ExponentialCurveValidator, "scenario_tests", return_value={
            "errors": ["buy(100)=> negative cost", "sell(50)=> exception"],
            "warnings": [],
            "info": {}
        }) as mock_scenario:
        results = ExponentialCurveValidator.run_all_validations(exponential_curve_fixture, curve_params)
    # aggregator => these scenario errors appear in final
    assert len(results["errors"]) == 2
    assert "buy(100)=> negative cost" in results["errors"]
    assert "sell(50)=> exception" in results["errors"]


def test_run_all_validations_boundary_warn(exponential_curve_fixture):
    """
    If boundary tests => warnings => aggregator merges them too
    """
    curve_params = exponential_curve_fixture.params

    with patch.object(ExponentialCurveValidator, "validate_params", return_value={
        "errors": [],
        "warnings": [],
        "info": {}
    }), patch.object(ExponentialCurveValidator, "boundary_tests", return_value={
        "errors": [],
        "warnings": ["Exception calling calculate_purchase_cost(10000): capacity exceeded"],
        "info": {}
    }) as mock_bound, patch.object(ExponentialCurveValidator, "scenario_tests", return_value={
        "errors": [],
        "warnings": [],
        "info": {}
    }) as mock_scen:
        results = ExponentialCurveValidator.run_all_validations(exponential_curve_fixture, curve_params)
    # aggregator => merges boundary warn
    assert results["warnings"] == ["Exception calling calculate_purchase_cost(10000): capacity exceeded"]
