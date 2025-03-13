from decimal import Decimal
from typing import Any, Dict, List

from forgd_core.common.enums import OrderSide
from forgd_core.common.model import (
    BondingCurveParams,
    TransactionRequest,
    Token
)
from forgd_core.curves.single.linear import LinearBondingCurve


class LinearCurveValidator:
    """
    Specialized validator for the LinearBondingCurve.
    Performs:
      1) Param checks (initial_price, slope, etc.)
      2) Boundary tests (spot price at 0, negative cost checks, etc.)
      3) Scenario tests (small buy/sell sequence)

    Each step returns a dict with:
      {
        "errors": [str...],
        "warnings": [str...],
        "info": {...}
      }
    and 'run_all_validations' aggregates them into a single result.
    """

    @staticmethod
    def validate_params(curve_params: 'BondingCurveParams', options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks that the linear curve's parameters are valid:
          - initial_price > 0
          - slope >= 0
        Also checks some common or optional fields in 'options' (like max_supply >= 0).
        """
        errors: List[str] = []
        warnings: List[str] = []
        info: Dict[str, Any] = {}

        # Basic param checks from BondingCurveParams
        initial_price = getattr(curve_params, "initial_price", None)
        if initial_price is None or initial_price <= 0:
            errors.append("LinearCurve: 'initial_price' must be > 0.")

        slope = getattr(curve_params, "slope", None)
        if slope is None or slope < 0:
            errors.append("LinearCurve: 'slope' must be >= 0 (cannot be negative).")

        # Check fields in 'options'
        max_supply = options.get("max_supply", None)
        if max_supply is None or max_supply < 0:
            errors.append("LinearCurve: 'max_supply' cannot be negative.")

        max_liquidity = options.get("max_liquidity", None)
        if max_liquidity is None or max_liquidity < 0:
            errors.append("LinearCurve: 'max_liquidity' cannot be negative.")

        txn_fee_rate = options.get("txn_fee_rate", Decimal("0"))
        if txn_fee_rate < 0:
            errors.append("LinearCurve: 'txn_fee_rate' cannot be negative.")

        info["param_summary"] = {
            "initial_price": str(initial_price),
            "slope": str(slope),
            "max_supply": str(max_supply),
            "max_liquidity": str(max_liquidity),
            "txn_fee_rate": str(txn_fee_rate),
        }

        return {
            "errors": errors,
            "warnings": warnings,
            "info": info
        }

    @staticmethod
    def boundary_tests(curve: 'LinearBondingCurve') -> Dict[str, Any]:
        """
        Calls a few boundary conditions on the linear curve:
          - get_spot_price(0)
          - calculate_purchase_cost(0)
          - large supply checks, etc.

        Returns a dict of errors/warnings/info.
        """
        errors: List[str] = []
        warnings: List[str] = []
        info: Dict[str, Any] = {}

        # 1) Spot price at supply=0
        try:
            price_at_zero = curve.get_spot_price(Decimal("0"))
            if price_at_zero < 0:
                errors.append("Spot price is negative at supply=0.")
        except Exception as e:
            errors.append(f"Exception calling get_spot_price(0): {e}")

        # 2) Cost to buy 0 tokens => should be 0
        try:
            cost_zero = curve.calculate_purchase_cost(Decimal("0"))
            if cost_zero != 0:
                warnings.append(
                    f"Cost to buy 0 tokens is not zero: got {cost_zero}"
                )
        except Exception as e:
            errors.append(f"Exception calling calculate_purchase_cost(0): {e}")

        # 3) Spot price for a large supply (e.g. 1e6)
        #    Just an example to see if we get negative or weird values
        try:
            large_supply_price = curve.get_spot_price(Decimal("1000000"))
            if large_supply_price < 0:
                errors.append("Spot price is negative at supply=1e6.")
        except Exception as e:
            warnings.append(f"Exception calling get_spot_price(1e6): {e}")

        # Could add more boundary checks (like a very large buy cost, or
        # partial fill scenario). The complexity is up to you.

        info["boundary_tests_run"] = True
        return {
            "errors": errors,
            "warnings": warnings,
            "info": info
        }

    @staticmethod
    def scenario_tests(curve: 'LinearBondingCurve') -> Dict[str, Any]:
        """
        Runs a small scenario:
          1) buy(100)
          2) buy(200)
          3) sell(50)
        Checks for negative cost, negative supply, etc.

        Return 'errors', 'warnings', 'info'.
        """
        errors: List[str] = []
        warnings: List[str] = []
        info: Dict[str, Any] = {}

        # SCENARIO STEPS:
        # Step 1: buy(100)
        try:
            req_buy100 = TransactionRequest(
                token=Token("TEST", "Test Token", 18),
                order_type=OrderSide.BUY,
                amount=Decimal("100")
            )
            result_buy100 = curve.buy(req_buy100)
            if result_buy100.total_cost < 0:
                errors.append("Buying 100 tokens => negative total cost.")
            if result_buy100.new_supply < 0:
                errors.append("Supply is negative after buy(100).")
        except Exception as e:
            errors.append(f"Exception in scenario step buy(100): {e}")

        # Step 2: buy(200)
        try:
            req_buy200 = TransactionRequest(
                token=Token("TEST", "Test Token", 18),
                order_type=OrderSide.BUY,
                amount=Decimal("200")
            )
            result_buy200 = curve.buy(req_buy200)
            if result_buy200.total_cost < 0:
                errors.append("Buying 200 tokens => negative total cost.")
            if result_buy200.new_supply < 0:
                errors.append("Supply is negative after buy(200).")
        except Exception as e:
            errors.append(f"Exception in scenario step buy(200): {e}")

        # Step 3: sell(50)
        try:
            req_sell50 = TransactionRequest(
                token=Token("TEST", "Test Token", 18),
                order_type=OrderSide.SELL,
                amount=Decimal("50")
            )
            result_sell50 = curve.sell(req_sell50)
            if result_sell50.total_cost < 0:
                errors.append("Selling 50 tokens => negative total cost (return?).")
            if result_sell50.new_supply < 0:
                errors.append("Supply is negative after sell(50).")
        except Exception as e:
            errors.append(f"Exception in scenario step sell(50): {e}")

        # Record final supply, for example
        info["final_supply_after_scenario"] = str(curve.current_supply)

        return {
            "errors": errors,
            "warnings": warnings,
            "info": info
        }

    @staticmethod
    def run_all_validations(curve: 'LinearBondingCurve', curve_params: 'BondingCurveParams') -> Dict[str, Any]:
        """
        Aggregates:
          - param check
          - boundary tests
          - scenario tests
        Returns a dict with keys: errors, warnings, info
        """
        if not isinstance(curve, LinearBondingCurve):
            raise ValueError("Invalid curve type for LinearCurveValidator.")

        results = {
            "errors": [],
            "warnings": [],
            "info": {}
        }
        options = getattr(curve, "options", {})

        # 1) Param checks
        param_check = LinearCurveValidator.validate_params(curve_params, options)
        results["errors"].extend(param_check["errors"])
        results["warnings"].extend(param_check["warnings"])
        results["info"].update(param_check["info"])

        # 2) Boundary tests
        boundary = LinearCurveValidator.boundary_tests(curve)
        results["errors"].extend(boundary["errors"])
        results["warnings"].extend(boundary["warnings"])
        results["info"].update(boundary["info"])

        # 3) Scenario tests
        scenario = LinearCurveValidator.scenario_tests(curve)
        results["errors"].extend(scenario["errors"])
        results["warnings"].extend(scenario["warnings"])
        results["info"].update(scenario["info"])

        return results
