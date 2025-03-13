from decimal import Decimal
from typing import Any, Dict, List

from forgd_core.common.enums import OrderSide
from forgd_core.common.model import (
    BondingCurveParams, Token
)
from forgd_core.curves.single.exponential import ExponentialBondingCurve


class ExponentialCurveValidator:
    """
    Validator for an ExponentialBondingCurve. Performs:
      1) Parameter checks (p0 > 0, alpha >= 0, etc.)
      2) Boundary tests (spot price at 0, negative cost checks, large supply checks)
      3) Scenario tests (a small buy/sell sequence).
    """

    @staticmethod
    def validate_params(curve_params: "BondingCurveParams", options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks that the exponential curve's parameters are valid:
          - p0 > 0
          - alpha >= 0
        Also checks certain optional fields in 'options' if relevant
        (e.g., max_supply >= 0, max_liquidity >= 0, txn_fee_rate >= 0).
        """
        errors: List[str] = []
        warnings: List[str] = []
        info: Dict[str, Any] = {}

        # Pull out p0 & alpha from curve_params
        p0 = getattr(curve_params, "initial_price", None)
        if p0 is None or p0 <= 0:
            errors.append("ExponentialCurve: 'p0' (base price) must be > 0.")

        alpha = getattr(curve_params, "exponential", None)
        if alpha is None or alpha < 0:
            errors.append("ExponentialCurve: 'alpha' must be >= 0 (no negative growth in this version).")

        # Check some optional fields in 'options' if your curve uses them
        max_supply = options.get("max_supply", None)
        if max_supply is not None and max_supply < 0:
            errors.append("ExponentialCurve: 'max_supply' cannot be negative.")

        max_liquidity = options.get("max_liquidity", None)
        if max_liquidity is not None and max_liquidity < 0:
            errors.append("ExponentialCurve: 'max_liquidity' cannot be negative.")

        txn_fee_rate = options.get("txn_fee_rate", Decimal("0"))
        if txn_fee_rate < 0:
            errors.append("ExponentialCurve: 'txn_fee_rate' cannot be negative.")

        # Possibly record a summary in 'info'
        info["param_summary"] = {
            "p0": str(p0),
            "alpha": str(alpha),
            "max_supply": str(max_supply),
            "max_liquidity": str(max_liquidity),
            "txn_fee_rate": str(txn_fee_rate),
            # add more as needed
        }

        return {
            "errors": errors,
            "warnings": warnings,
            "info": info
        }

    @staticmethod
    def boundary_tests(curve: "ExponentialBondingCurve") -> Dict[str, Any]:
        """
        Minimal checks for exponential curve:
          - get_spot_price(0) => should be p0 (unless alpha or time decay logic changes it)
          - calculate_purchase_cost(0) => should be 0
          - large supply => ensure no negative price or overflow
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

        # 2) Cost to buy 0 => typically 0
        try:
            cost_zero = curve.calculate_purchase_cost(Decimal("0"))
            if cost_zero != 0:
                warnings.append(
                    f"Cost to buy 0 tokens is not zero: got {cost_zero}"
                )
        except Exception as e:
            errors.append(f"Exception calling calculate_purchase_cost(0): {e}")

        # 3) Large supply check (e.g. 1000000)
        #    Just to see if we get overflow or negative
        try:
            large_supply_price = curve.get_spot_price(Decimal("1000000"))
            if large_supply_price < 0:
                errors.append("Price is negative at supply=1e6.")
        except Exception as e:
            warnings.append(f"Exception calling get_spot_price(1e6): {e}")

        info["boundary_tests_run"] = True
        return {
            "errors": errors,
            "warnings": warnings,
            "info": info
        }

    @staticmethod
    def scenario_tests(curve: "ExponentialBondingCurve") -> Dict[str, Any]:
        """
        A small buy/sell sequence:
          - buy(100)
          - buy(200)
          - sell(50)

        Checking for negative cost, negative supply, etc.
        Possibly crossing into large exponent or partial fill scenarios.
        """
        errors: List[str] = []
        warnings: List[str] = []
        info: Dict[str, Any] = {}

        from forgd_core.common.model import TransactionRequest

        # Step 1: buy(100)
        try:
            req_buy100 = TransactionRequest(
                token=Token("TEST", "Test Token", 6),
                order_type=OrderSide.BUY,
                amount=Decimal("100")
            )
            result_buy100 = curve.buy(req_buy100)
            if result_buy100.total_cost < 0:
                errors.append("buy(100) => negative total cost.")
            if result_buy100.new_supply < 0:
                errors.append("Supply is negative after buy(100).")
        except Exception as e:
            errors.append(f"Exception in scenario step buy(100): {e}")

        # Step 2: buy(200)
        try:
            req_buy200 = TransactionRequest(
                token=Token("TEST", "Test Token", 6),
                order_type=OrderSide.BUY,
                amount=Decimal("200")
            )
            result_buy200 = curve.buy(req_buy200)
            if result_buy200.total_cost < 0:
                errors.append("buy(200) => negative total cost.")
            if result_buy200.new_supply < 0:
                errors.append("Supply is negative after buy(200).")
        except Exception as e:
            errors.append(f"Exception in scenario step buy(200): {e}")

        # Step 3: sell(50)
        try:
            req_sell50 = TransactionRequest(
                token=Token("TEST", "Test Token", 6),
                order_type=OrderSide.SELL,
                amount=Decimal("50")
            )
            result_sell50 = curve.sell(req_sell50)
            if result_sell50.total_cost < 0:
                errors.append("sell(50) => negative total cost (return?).")
            if result_sell50.new_supply < 0:
                errors.append("Supply is negative after sell(50).")
        except Exception as e:
            errors.append(f"Exception in scenario step sell(50): {e}")

        # Possibly record final supply or final price
        info["final_supply_after_scenario"] = str(curve.current_supply)

        return {
            "errors": errors,
            "warnings": warnings,
            "info": info
        }

    @staticmethod
    def run_all_validations(
        curve: "ExponentialBondingCurve",
        curve_params: "BondingCurveParams"
    ) -> Dict[str, Any]:
        """
        Aggregates:
          - param check
          - boundary tests
          - scenario tests
        Returns a dict with keys: errors, warnings, info
        """
        results = {
            "errors": [],
            "warnings": [],
            "info": {}
        }

        # get 'options' from the curve if that's how you store them
        options = getattr(curve, "options", {})

        # 1) Param checks
        param_check = ExponentialCurveValidator.validate_params(curve_params, options)
        results["errors"].extend(param_check["errors"])
        results["warnings"].extend(param_check["warnings"])
        results["info"].update(param_check["info"])

        # 2) Boundary tests
        boundary = ExponentialCurveValidator.boundary_tests(curve)
        results["errors"].extend(boundary["errors"])
        results["warnings"].extend(boundary["warnings"])
        results["info"].update(boundary["info"])

        # 3) Scenario tests
        scenario = ExponentialCurveValidator.scenario_tests(curve)
        results["errors"].extend(scenario["errors"])
        results["warnings"].extend(scenario["warnings"])
        results["info"].update(scenario["info"])

        return results
