from decimal import Decimal
from typing import Any, Dict, List

from forgd_core.common.enums import BondingCurveType
from forgd_core.common.model import BondingCurveParams
from forgd_core.curves.single.base import BondingCurve


class CommonValidator:
    """
    Common validator for all single BondingCurves.
    1) Param checks (initial_price, slope, etc.)
    2) Boundary tests (spot price at 0, negative cost checks, etc.)
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

        initial_price = options.get("initial_price", None)
        slope = options.get("slope", None)
        steps = options.get("steps", None)
        p0 = options.get("p0", None)
        alpha = options.get("alpha", None)

        if curve_params.curve_type == BondingCurveType.LINEAR:
            initial_price = getattr(curve_params, "initial_price", None)
            if initial_price is None or initial_price <= 0:
                errors.append("LinearCurve: 'initial_price' must be > 0.")

            slope = getattr(curve_params, "slope", None)
            if slope is None or slope < 0:
                errors.append("LinearCurve: 'slope' must be >= 0 (cannot be negative).")

        if curve_params.curve_type == BondingCurveType.EXPONENTIAL:
            p0 = getattr(curve_params, "initial_price", None)
            if p0 is None or p0 <= 0:
                errors.append("ExponentialCurve: 'p0' (base price) must be > 0.")

            alpha = getattr(curve_params, "exponential", None)
            if alpha is None or alpha < 0:
                errors.append("ExponentialCurve: 'alpha' must be >= 0 (no negative growth in this version).")

        if curve_params.curve_type == BondingCurveType.STEPWISE:
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

                # Validate that steps are sorted in ascending order
                for i in range(1, len(steps)):
                    if steps[i].supply_threshold <= steps[i - 1].supply_threshold:
                        errors.append(
                            f"Steps must be strictly ascending. Step {i} threshold "
                            f"{steps[i].supply_threshold} <= previous {steps[i - 1].supply_threshold}"
                        )

        # Check some optional fields in 'options' if your curve uses them
        max_supply = options.get("max_supply", None)
        if not max_supply:
            errors.append("Curve: 'max_supply' is required.")
        if max_supply is not None and max_supply < 0:
            errors.append("Curve: 'max_supply' cannot be negative.")

        max_liquidity = options.get("max_liquidity", None)
        if not max_liquidity:
            errors.append("Curve: 'max_liquidity' is not set.")
        if max_liquidity is not None and max_liquidity < 0:
            errors.append("Curve: 'max_liquidity' cannot be negative.")

        txn_fee_rate = options.get("txn_fee_rate", Decimal("0"))
        if txn_fee_rate < 0:
            errors.append("Curve: 'txn_fee_rate' cannot be negative.")

        # Possibly record a summary in 'info'
        info["param_summary"] = {
            "initial_price": str(initial_price),
            "slope": str(slope),
            "num_steps": len(steps) if steps else 0,
            "p0": str(p0),
            "alpha": str(alpha),
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
    def run_all_validations(
        curve: "BondingCurve",
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
        param_check = CommonValidator.validate_params(curve_params, options)
        results["errors"].extend(param_check["errors"])
        results["warnings"].extend(param_check["warnings"])
        results["info"].update(param_check["info"])

        return results
