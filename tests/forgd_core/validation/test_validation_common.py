import pytest

from decimal import Decimal
from unittest.mock import patch

from forgd_core.common.model import (
    BondingCurveParams,
    BondingCurveState,
    Liquidity,
    Token,
    TransactionResult,
    StepConfig
)
from forgd_core.common.enums import BondingCurveType
from forgd_core.curves.single.linear import LinearBondingCurve
from forgd_core.validation.common_validator import CommonValidator


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
def valid_linear_params():
    """
    Returns a BondingCurveParams for a linear curve with valid initial_price>0, slope>=0.
    """
    return BondingCurveParams(
        curve_type=BondingCurveType.LINEAR,
        initial_price=Decimal("1.0"),
        slope=Decimal("0.1"),
    )


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


def test_validate_linear_params_all_valid(valid_linear_params, valid_options):
    """
    If all parameters are valid => no errors, no warnings, info includes param_summary.
    """
    result = CommonValidator.validate_params(valid_linear_params, valid_options)
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
def test_validate_linear_params_invalid_params(field, value, expected_msg, valid_linear_params, valid_options):
    """
    If certain fields in BondingCurveParams are invalid => we get an error message.
    """
    setattr(valid_linear_params, field, value)  # override that field
    result = CommonValidator.validate_params(valid_linear_params, valid_options)
    assert expected_msg in result["errors"], f"Expected error '{expected_msg}', got {result['errors']}"


@pytest.mark.parametrize(
    "opt_key, opt_value, expected_msg",
    [
        # max_supply<0 => error
        ("max_supply", Decimal("-10"), "Curve: 'max_supply' cannot be negative."),
        # max_liquidity<0 => error
        ("max_liquidity", Decimal("-1"), "Curve: 'max_liquidity' cannot be negative."),
        # txn_fee_rate<0 => error
        ("txn_fee_rate", Decimal("-0.01"), "Curve: 'txn_fee_rate' cannot be negative."),
    ]
)
def test_validate_linear_params_invalid_options(opt_key, opt_value, expected_msg, valid_linear_params, valid_options):
    """
    If certain fields in the options are invalid => we get an error message.
    """
    valid_options[opt_key] = opt_value  # override
    result = CommonValidator.validate_params(valid_linear_params, valid_options)
    assert expected_msg in result["errors"], f"Expected error '{expected_msg}', got {result['errors']}"


def test_validate_linear_params_missing_options(valid_linear_params):
    """
    If max_supply, max_liquidity, or other required fields are missing => we see the relevant error.
    We can pass an empty dict or incomplete dict for options.
    """
    incomplete_options = {}  # missing everything
    result = CommonValidator.validate_params(valid_linear_params, incomplete_options)

    # Expect "max_supply cannot be negative" and "max_liquidity cannot be negative" because it's None => <0 check
    assert "Curve: 'max_supply' is required." in result["errors"]
    assert "Curve: 'max_liquidity' is not set." in result["errors"]

    # txn_fee_rate is by default 0 if missing, so no error unless negative.


def test_validate_linear_params_multiple_errors(valid_linear_params, valid_options):
    """
    If multiple fields are invalid at once => accumulate errors in the list.
    """
    valid_linear_params.initial_price = Decimal("-5")  # invalid
    valid_options["max_liquidity"] = Decimal("-1")  # invalid

    result = CommonValidator.validate_params(valid_linear_params, valid_options)
    assert len(result["errors"]) == 2, f"Expected 2 errors, got {result['errors']}"
    assert "LinearCurve: 'initial_price' must be > 0." in result["errors"]
    assert "Curve: 'max_liquidity' cannot be negative." in result["errors"]


def test_validate_stepwise_params_all_valid(valid_stepwise_params, valid_options):
    """
    If steps are non-empty, strictly ascending, prices>=0,
    and options do not contain negative fields => no errors/warnings.
    """
    result = CommonValidator.validate_params(valid_stepwise_params, valid_options)
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


def test_validate_stepwise_params_negative_supply_threshold(valid_stepwise_params, valid_options):
    """
    If any step has a negative supply_threshold => error.
    """
    valid_stepwise_params.steps[1].supply_threshold = Decimal("-1")  # negative
    result = CommonValidator.validate_params(valid_stepwise_params, valid_options)
    assert "Step 1 has negative supply_threshold -1." in result["errors"]


def test_validate_stepwise_params_negative_price(valid_stepwise_params, valid_options):
    """
    If any step has a negative price => error.
    """
    valid_stepwise_params.steps[0].price = Decimal("-0.5")
    result = CommonValidator.validate_params(valid_stepwise_params, valid_options)
    assert "Step 0 has negative price -0.5." in result["errors"]


def test_validate_stepwise_params_non_strict_ascending(valid_stepwise_params, valid_options):
    """
    If steps[i].supply_threshold <= steps[i-1].supply_threshold => error about ascending thresholds.
    """
    # Make step1 threshold == 100 => not strictly ascending from step0(100)
    valid_stepwise_params.steps[1].supply_threshold = Decimal("100")
    result = CommonValidator.validate_params(valid_stepwise_params, valid_options)
    assert any("Steps must be strictly ascending." in e for e in result["errors"])


@pytest.mark.parametrize(
    "opt_key,opt_value,expected_msg",
    [
        ("max_supply", Decimal("-10"), "Curve: 'max_supply' cannot be negative."),
        ("max_liquidity", Decimal("-5"), "Curve: 'max_liquidity' cannot be negative."),
        ("txn_fee_rate", Decimal("-0.01"), "Curve: 'txn_fee_rate' cannot be negative."),
    ]
)
def test_validate_params_stepwise_negative_options(
    valid_stepwise_params, valid_options, opt_key, opt_value, expected_msg
):
    """
    If max_supply<0 or max_liquidity<0 or txn_fee_rate<0 => error.
    """
    valid_options[opt_key] = opt_value
    result = CommonValidator.validate_params(valid_stepwise_params, valid_options)
    assert expected_msg in result["errors"], f"Expected error: {expected_msg}, got {result['errors']}"


def test_validate_params_all_valid(valid_exponential_params, valid_options):
    """
    If p0>0, alpha>=0, and no negative fields in options => no errors/warnings.
    """
    result = CommonValidator.validate_params(valid_exponential_params, valid_options)
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
    result = CommonValidator.validate_params(valid_exponential_params, valid_options)
    assert "ExponentialCurve: 'p0' (base price) must be > 0." in result["errors"]

    # 2) p0<0
    valid_exponential_params.initial_price = Decimal("-1")
    result2 = CommonValidator.validate_params(valid_exponential_params, valid_options)
    assert "ExponentialCurve: 'p0' (base price) must be > 0." in result2["errors"]


def test_validate_params_alpha_negative(valid_exponential_params, valid_options):
    """
    If alpha<0 => "ExponentialCurve: 'alpha' must be >=0"
    """
    valid_exponential_params.exponential = Decimal("-0.01")
    result = CommonValidator.validate_params(valid_exponential_params, valid_options)
    assert "ExponentialCurve: 'alpha' must be >= 0" in result["errors"][0]


@pytest.mark.parametrize(
    "opt_key, opt_value, expected_msg",
    [
        ("max_supply", Decimal("-1"), "Curve: 'max_supply' cannot be negative."),
        ("max_liquidity", Decimal("-10"), "Curve: 'max_liquidity' cannot be negative."),
        ("txn_fee_rate", Decimal("-0.05"), "Curve: 'txn_fee_rate' cannot be negative."),
    ]
)
def test_validate_params_negative_options(valid_exponential_params, valid_options, opt_key, opt_value, expected_msg):
    """
    If the options contain negative max_supply, max_liquidity, or txn_fee_rate => error.
    """
    valid_options[opt_key] = opt_value
    result = CommonValidator.validate_params(valid_exponential_params, valid_options)
    assert expected_msg in result["errors"], f"Expected error '{expected_msg}', got: {result['errors']}"
