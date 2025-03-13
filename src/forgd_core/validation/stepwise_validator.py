from decimal import Decimal
from typing import Any, Dict, List

from forgd_core.common.enums import OrderSide
from forgd_core.common.model import (
    BondingCurveParams, Token
)
from forgd_core.curves.single.stepwise import StepwiseBondingCurve


class StepwiseCurveValidator:
    """
        Validator for a stepwise bonding curve. Performs:
          1) Parameter checks (validate that steps are sorted, each step has a valid price, etc.)
          2) Boundary tests (checking for negative costs, partial fill, etc.)
          3) Scenario tests (a small buy/sell sequence).
        """

    @staticmethod
    def validate_params(curve_params: "BondingCurveParams", options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks that the stepwise curve's parameters are valid:
          - steps is non-empty
          - sorted ascending thresholds (or rely on StepwiseCurveHelper.validate_and_sort_steps)
          - each step's price > 0 (if required)
        Also checks optional fields in 'options' (e.g., max_supply >= 0).
        """
        errors: List[str] = []
        warnings: List[str] = []
        info: Dict[str, Any] = {}

        # Typically, you'd store steps in curve_params.other_params or something similar
        steps = getattr(curve_params, "steps", None)
        if steps is None or not isinstance(steps, list) or len(steps) == 0:
            errors.append("StepwiseCurve: 'steps' must be a non-empty list.")
        else:
            # Validate each step individually
            for i, step in enumerate(steps):
                if step.supply_threshold < 0:
                    errors.append(f"Step {i} has negative supply_threshold {step.supply_threshold}.")
                if step.price < 0:
                    errors.append(f"Step {i} has negative price {step.price}.")

            # Optionally, we can check if steps are strictly ascending:
            # You might rely on the code that calls StepwiseCurveHelper.validate_and_sort_steps,
            # but let's do a basic check here as well:
            for i in range(1, len(steps)):
                if steps[i].supply_threshold <= steps[i - 1].supply_threshold:
                    errors.append(
                        f"Steps must be strictly ascending. Step {i} threshold "
                        f"{steps[i].supply_threshold} <= previous {steps[i - 1].supply_threshold}"
                    )

        # Check some optional fields in 'options'
        max_supply = options.get("max_supply", None)
        if max_supply is not None and max_supply < 0:
            errors.append("StepwiseCurve: 'max_supply' cannot be negative.")

        max_liquidity = options.get("max_liquidity", None)
        if max_liquidity is not None and max_liquidity < 0:
            errors.append("StepwiseCurve: 'max_liquidity' cannot be negative.")

        txn_fee_rate = options.get("txn_fee_rate", Decimal("0"))
        if txn_fee_rate < 0:
            errors.append("StepwiseCurve: 'txn_fee_rate' cannot be negative.")

        info["param_summary"] = {
            "num_steps": len(steps) if steps else 0,
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
    def boundary_tests(curve: "StepwiseBondingCurve") -> Dict[str, Any]:
        """
        Calls a few boundary conditions on the stepwise curve:
          - get_spot_price(0) if the curve implements a spot price method
          - calculate_purchase_cost(0) => should be 0
          - large buy or negative checks
        """
        errors: List[str] = []
        warnings: List[str] = []
        info: Dict[str, Any] = {}

        # If your StepwiseBondingCurve has get_spot_price:
        try:
            price_at_zero = curve.get_spot_price(Decimal("0"))
            if price_at_zero < 0:
                errors.append("Spot price is negative at supply=0.")
        except Exception as e:
            warnings.append(f"Exception calling get_spot_price(0): {e}")

        # cost to buy 0 tokens => 0
        try:
            cost_zero = curve.calculate_purchase_cost(Decimal("0"))
            if cost_zero != 0:
                warnings.append(
                    f"Cost to buy 0 tokens is not zero: got {cost_zero}"
                )
        except Exception as e:
            errors.append(f"Exception calling calculate_purchase_cost(0): {e}")

        # Possibly check a large buy that crosses multiple tiers
        try:
            cost_large = curve.calculate_purchase_cost(Decimal("10000"))
            if cost_large < 0:
                errors.append("Cost to buy 10k tokens is negative.")
        except Exception as e:
            warnings.append(f"Exception calling calculate_purchase_cost(10000): {e}")

        info["boundary_tests_run"] = True
        return {
            "errors": errors,
            "warnings": warnings,
            "info": info
        }

    @staticmethod
    def scenario_tests(curve: "StepwiseBondingCurve") -> Dict[str, Any]:
        """
        Runs a small scenario:
          1) buy(100)
          2) buy(200)
          3) sell(50)

        Checking if partial fill or multiple tiers are triggered,
        ensures no negative cost or supply, etc.
        """
        from forgd_core.common.model import TransactionRequest
        errors: List[str] = []
        warnings: List[str] = []
        info: Dict[str, Any] = {}

        # Step 1: buy 100
        try:
            req_buy100 = TransactionRequest(
                token=Token("TEST", "Test Token", 18),
                order_type=OrderSide.BUY,
                amount=Decimal("100")
            )
            result_buy100 = curve.buy(req_buy100)
            if result_buy100.total_cost < 0:
                errors.append("buy(100) => negative total cost.")
            if result_buy100.new_supply < 0:
                errors.append("new_supply is negative after buy(100).")
        except Exception as e:
            errors.append(f"Exception in scenario step buy(100): {e}")

        # Step 2: buy 200
        try:
            req_buy200 = TransactionRequest(
                token=Token("TEST", "Test Token", 18),
                order_type=OrderSide.BUY,
                amount=Decimal("200")
            )
            result_buy200 = curve.buy(req_buy200)
            if result_buy200.total_cost < 0:
                errors.append("buy(200) => negative total cost.")
            if result_buy200.new_supply < 0:
                errors.append("new_supply is negative after buy(200).")
        except Exception as e:
            errors.append(f"Exception in scenario step buy(200): {e}")

        # Step 3: sell 50
        try:
            req_sell50 = TransactionRequest(
                token=Token("TEST", "Test Token", 18),
                order_type=OrderSide.SELL,
                amount=Decimal("50")
            )
            result_sell50 = curve.sell(req_sell50)
            if result_sell50.total_cost < 0:
                errors.append("sell(50) => negative total cost/return?")
            if result_sell50.new_supply < 0:
                errors.append("new_supply is negative after sell(50).")
        except Exception as e:
            errors.append(f"Exception in scenario step sell(50): {e}")

        # Possibly record final supply
        info["final_supply_after_scenario"] = str(curve.current_supply)

        return {
            "errors": errors,
            "warnings": warnings,
            "info": info
        }

    @staticmethod
    def run_all_validations(
        curve: "StepwiseBondingCurve",
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

        # 1) We'll get the 'options' from the curve if that's how you've structured it
        #    or from the curve_params if you store them there
        options = getattr(curve, "options", {})

        # Param checks
        param_check = StepwiseCurveValidator.validate_params(curve_params, options)
        results["errors"].extend(param_check["errors"])
        results["warnings"].extend(param_check["warnings"])
        results["info"].update(param_check["info"])

        # Boundary tests
        boundary = StepwiseCurveValidator.boundary_tests(curve)
        results["errors"].extend(boundary["errors"])
        results["warnings"].extend(boundary["warnings"])
        results["info"].update(boundary["info"])

        # Scenario tests
        scenario = StepwiseCurveValidator.scenario_tests(curve)
        results["errors"].extend(scenario["errors"])
        results["warnings"].extend(scenario["warnings"])
        results["info"].update(scenario["info"])

        return results
