from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from forgd_core.common.model import (
    BondingCurveState,
    BondingCurveParams,
    TransactionRequest,
    TransactionResult,
    StepConfig
)
from forgd_core.curves.single.base import BondingCurve
from forgd_core.curves.helpers.common import CommonCurveHelper as common_helper
from forgd_core.curves.helpers.stepwise import StepwiseCurveHelper as stepwise_helper


class StepwiseCurve(BondingCurve):
    """
    A stepwise bonding curve with multiple discrete tiers. Each tier has a range of
    supply [prev_threshold ... supply_threshold) at a constant price. If a buy or sell
    crosses multiple tiers, we sum the cost or return across each tier portion.
    """

    def __init__(
        self,
        params: BondingCurveParams,
        state: Optional[BondingCurveState] = None,
        **kwargs
    ):
        """
        :param params: BondingCurveParams object with standard fields like initial_price, slope, etc.
                      (Not all fields may be relevant if you rely on tiers for pricing.)
        :param state: optional existing BondingCurveState
        :param steps: list of StepConfig objects describing the tiers
        :param kwargs: advanced flags, e.g.
            - allow_buy
            - allow_sell
            - max_supply
            - max_liquidity
            - slippage_tolerance
            - txn_fee_rate
            - time_decay_rate
            - risk_profile
            - partial_fill_approach
            - time_decay_approach
            ...
        """
        super().__init__(params, state)

        # Store or initialize a BondingCurveState
        if self._state.last_timestamp is None:
            self._state.last_timestamp = datetime.now()

        # Default options, similar to your linear approach
        self.options = {
            "allow_buy": True,
            "allow_sell": True,
            "max_supply": None,
            "max_liquidity": None,
            "slippage_tolerance": Decimal("0"),
            "txn_fee_rate": Decimal("0"),
            "time_decay_rate": Decimal("0"),
            "risk_profile": None,
            "partial_fill_approach": 1,
            "time_decay_approach": 1,
            "pro_rata": False,  # or 'allow_partial_fill' for stepwise
        }
        for k, v in kwargs.items():
            if k in self.options:
                self.options[k] = v
            else:
                if not hasattr(self, "custom_options"):
                    self.custom_options = {}
                self.custom_options[k] = v

        # Validate & store the steps
        self._steps = self._params.steps or []
        self._steps = stepwise_helper.validate_and_sort_steps(self._steps)

    def _apply_time_decay_to_steps(self):
        tdr = self.options["time_decay_rate"]
        if tdr and tdr < 0:
            raise ValueError("Negative time_decay_rate is not allowed.")

        if tdr and tdr > 0:
            # existing logic calling apply_time_decay_to_steps
            self._steps, new_ts = stepwise_helper.apply_time_decay_to_steps(
                self._steps,
                self._state.last_timestamp,
                tdr,
                approach=self.options["time_decay_approach"]
            )
            self._state.last_timestamp = new_ts

    def _update_state_after_buy(self, amount: Decimal, total_cost: Decimal):
        """
        minted tokens = +amount, liquidity += cost.
        Adjust as needed if partial fill or fees go elsewhere.
        """
        self._state.current_supply += amount
        if self._state.liquidity:
            self._state.liquidity.amount += total_cost

    def _update_state_after_sell(self, amount: Decimal, total_return: Decimal):
        """
        burned tokens = -amount, liquidity -= return.
        Adjust if partial fill or fee distribution is different.
        """
        self._state.current_supply -= amount
        if self._state.liquidity:
            self._state.liquidity.amount -= total_return

    def get_spot_price(self, supply: Decimal) -> Decimal:
        """
        Return the spot price at 'supply' by finding which tier that supply belongs to.
        IMPORTANT: This also applies time decay on each call, so the tier prices are
        always up-to-date.

        If supply is beyond the last tier, we return the last tier's price.
        """
        # 1) Apply time decay so _steps is updated
        self._apply_time_decay_to_steps()

        # 2) If no tiers, default to zero or some fallback
        if not self._steps:
            return Decimal("0")

        # 3) Find tier that includes 'supply'
        idx = 0
        while idx < len(self._steps) and supply >= self._steps[idx].supply_threshold:
            idx += 1

        if idx >= len(self._steps):
            # beyond last tier => return last tier's price
            return self._steps[-1].price
        else:
            return self._steps[idx].price

    def calculate_purchase_cost(self, amount: Decimal) -> Decimal:
        """
        Uses stepwise_cost_for_purchase from the helper, applying time decay first if desired.
        """
        self._apply_time_decay_to_steps()

        cost = stepwise_helper.stepwise_cost_for_purchase(
            current_supply=self._state.current_supply,
            amount=amount,
            steps=self._steps,
            allow_partial_fill=self.options["pro_rata"]
        )
        return cost

    def calculate_sale_return(self, amount: Decimal) -> Decimal:
        """
        Uses stepwise_return_for_sale from the helper, applying time decay first if desired.
        """
        self._apply_time_decay_to_steps()

        ret = stepwise_helper.stepwise_return_for_sale(
            current_supply=self._state.current_supply,
            amount=amount,
            steps=self._steps,
            allow_partial_fill=self.options["pro_rata"]
        )
        return ret

    def buy(self, request: TransactionRequest) -> TransactionResult:
        """
        Buys 'request.amount' tokens, referencing stepwise tiers for the cost,
        then applying risk profile, fees, slippage, partial fills, etc.
        """
        if not self.options["allow_buy"]:
            raise ValueError("Buys are disabled on this stepwise bonding curve.")

        original_amount = request.amount
        final_amount = original_amount

        # 1) Check max_supply if relevant
        max_supply = self.options["max_supply"]
        if max_supply is not None:
            remaining_capacity = max_supply - self._state.current_supply
            if remaining_capacity <= 0:
                if self.options["pro_rata"]:
                    final_amount = Decimal("0")
                else:
                    raise ValueError("Max supply reached; cannot buy more tokens.")
            elif final_amount > remaining_capacity:
                if self.options["pro_rata"]:
                    final_amount = common_helper.partial_fill(final_amount, remaining_capacity)
                else:
                    raise ValueError("Requested amount exceeds max supply. Reverting.")

        if final_amount == 0:
            # trivial transaction result
            return TransactionResult(
                executed_amount=Decimal("0"),
                total_cost=Decimal("0"),
                average_price=Decimal("0"),
                new_supply=self._state.current_supply,
                timestamp=datetime.now()
            )

        # 2) Calculate raw cost from stepwise tiers
        raw_cost = self.calculate_purchase_cost(final_amount)

        # 3) Apply risk profile (markup or discount)
        raw_cost = common_helper.apply_risk_profile(
            raw_cost,
            self.options["risk_profile"],
            is_buy=True
        )

        # 4) Check max_liquidity
        max_liquidity = self.options["max_liquidity"]
        if max_liquidity is not None:
            current_liquid = self._state.liquidity.amount
            if (current_liquid + raw_cost) > max_liquidity:
                if self.options["pro_rata"]:
                    # partial fill approach
                    available_liq = max_liquidity - current_liquid
                    if available_liq < 0:
                        # No liquidity left at all
                        final_amount = Decimal("0")
                        raw_cost = Decimal("0")
                    else:
                        fraction = available_liq / raw_cost if raw_cost != 0 else Decimal("0")
                        final_amount = final_amount * fraction
                        raw_cost = available_liq
                else:
                    raise ValueError("Max liquidity reached; reverting buy.")

        # 5) Slippage tolerance
        slip = self.options["slippage_tolerance"]
        if slip > 0:
            # Baseline cost ignoring risk markup
            baseline_cost = stepwise_helper.stepwise_cost_for_purchase(
                current_supply=self._state.current_supply,
                amount=final_amount,
                steps=self._steps,
                allow_partial_fill=self.options["pro_rata"]
            )
            final_cost, revert_flag = common_helper.scale_for_slippage(
                raw_cost,
                baseline_cost,
                slip,
                is_buy=True,
                approach=self.options["partial_fill_approach"]
            )
            if revert_flag:
                raise ValueError("Slippage tolerance exceeded on buy. Reverting.")
        else:
            final_cost = raw_cost

        # 6) Transaction fee
        final_cost = common_helper.apply_transaction_fee(
            final_cost,
            self.options["txn_fee_rate"],
            is_buy=True
        )

        # 7) Update state after buy
        self._update_state_after_buy(final_amount, final_cost)

        # 8) Return transaction result
        return TransactionResult(
            executed_amount=final_amount,
            total_cost=final_cost,
            average_price=(final_cost / final_amount) if final_amount != 0 else Decimal("0"),
            new_supply=self._state.current_supply,
            timestamp=datetime.now()
        )

    def sell(self, request: TransactionRequest) -> TransactionResult:
        """
        Sells 'request.amount' tokens using stepwise tiers, then applies risk profile,
        partial fills, fees, slippage, etc.
        """
        if not self.options["allow_sell"]:
            raise ValueError("Sells are disabled on this stepwise bonding curve.")

        original_amount = request.amount
        final_amount = original_amount

        # 1) Calculate raw return
        raw_return = self.calculate_sale_return(final_amount)

        # 2) Apply risk profile
        raw_return = common_helper.apply_risk_profile(
            raw_return,
            self.options["risk_profile"],
            is_buy=False
        )

        # 3) Check liquidity coverage
        current_liquid = self._state.liquidity.amount
        if raw_return > current_liquid:
            if self.options["pro_rata"]:
                fraction = common_helper.partial_fill(raw_return, current_liquid)
                if raw_return > 0:
                    fraction_of_tokens = fraction / raw_return
                else:
                    fraction_of_tokens = Decimal("0")
                final_amount = final_amount * fraction_of_tokens
                raw_return = fraction
            else:
                raise ValueError("Insufficient liquidity to cover this sell. Reverting.")

        # 4) Slippage tolerance
        slip = self.options["slippage_tolerance"]
        if slip > 0:
            baseline_ret = stepwise_helper.stepwise_return_for_sale(
                current_supply=self._state.current_supply,
                amount=final_amount,
                steps=self._steps,
                allow_partial_fill=self.options["pro_rata"]
            )
            final_ret, revert_flag = common_helper.scale_for_slippage(
                raw_return,
                baseline_ret,
                slip,
                is_buy=False,
                approach=self.options["partial_fill_approach"]
            )
            if revert_flag:
                raise ValueError("Slippage tolerance not met on sell. Reverting.")
            raw_return = final_ret
        else:
            final_ret = raw_return

        # 5) Transaction fee
        final_return = common_helper.apply_transaction_fee(
            raw_return,
            self.options["txn_fee_rate"],
            is_buy=False
        )

        # 6) Update state after sell
        self._update_state_after_sell(final_amount, final_return)

        # 7) Return transaction result
        return TransactionResult(
            executed_amount=final_amount,
            total_cost=final_return,
            average_price=(final_return / final_amount) if final_amount != 0 else Decimal("0"),
            new_supply=self._state.current_supply,
            timestamp=datetime.now()
        )
