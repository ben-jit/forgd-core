from datetime import datetime
from decimal import Decimal
from typing import Optional

from forgd_core.common.model import (
    BondingCurveState,
    BondingCurveParams,
    TransactionRequest,
    TransactionResult,
)
from forgd_core.curves.single.base import BondingCurve
from forgd_core.curves.helpers.common import CommonCurveHelper as common_helper
from forgd_core.curves.helpers.exponential import ExponentialCurveHelper as exponential_helper


class ExponentialBondingCurve(BondingCurve):
    """
    A bonding curve where the price function is modeled as:
        price(s) = p0 * exp(alpha * s)

    This class:
      - Validates p0 > 0, alpha >= 0
      - Optionally applies time decay
      - Calculates cost/return via ExponentialCurveHelper
      - Handles partial fills, slippage, risk, fees (via CommonCurveHelper)
      - Updates supply & liquidity in BondingCurveState
    """

    def __init__(
        self,
        params: BondingCurveParams,
        state: Optional[BondingCurveState] = None,
        **kwargs
    ):
        """
                :param params: BondingCurveParams with fields for p0, alpha, etc.
                              (You can store them in params.other_params or directly in dedicated fields.)
                :param state: existing BondingCurveState, or None => new state
                :param kwargs: advanced options such as:
                  - allow_buy, allow_sell
                  - max_supply
                  - max_liquidity
                  - slippage_tolerance
                  - txn_fee_rate
                  - risk_profile
                  - partial_fill_approach
                  - time_decay_rate
                  - time_decay_approach
                  - pro_rata (bool) for partial fill logic
                  - clamp_prices (bool) whether to clamp in the cost/return calculations
                  - min_price, max_price for clamping
                """
        super().__init__(params, state)

        # Ensure we have a valid last_timestamp in state
        if self._state.last_timestamp is None:
            self._state.last_timestamp = datetime.now()

        # Default options
        self.options = {
            "allow_buy": True,
            "allow_sell": True,
            "max_supply": None,
            "max_liquidity": None,
            "slippage_tolerance": Decimal("0"),
            "txn_fee_rate": Decimal("0"),
            "risk_profile": None,
            "partial_fill_approach": 1,
            "time_decay_rate": Decimal("0"),
            "time_decay_approach": 1,
            "pro_rata": False,
            "clamp_prices": False,  # if True => use min_price/max_price in integral
            "min_price": None,
            "max_price": None,
        }
        for k, v in kwargs.items():
            if k in self.options:
                self.options[k] = v
            else:
                # store custom keys if needed
                if not hasattr(self, "custom_options"):
                    self.custom_options = {}
                self.custom_options[k] = v

        # (Adjust if you store them differently, e.g. in params.other_params)
        p0 = getattr(params, "initial_price", None)
        alpha = getattr(params, "exponential", None)

        # Validate presence
        if p0 is None or alpha is None:
            raise ValueError("ExponentialBondingCurve requires p0 & alpha in params.")

        exponential_helper.validate_exponential_params(p0, alpha)

        # store them in state or local fields
        self._p0 = p0
        self._alpha = alpha

    @property
    def current_p0(self) -> Decimal:
        return self._p0

    @property
    def current_alpha(self) -> Decimal:
        return self._alpha

    def _apply_time_decay(self):
        """
        Optionally updates self._p0, self._alpha based on time_decay_rate.
        Called before we compute cost/return or spot price, so the
        parameters remain up-to-date with the current time.
        """
        tdr = self.options["time_decay_rate"]
        if tdr > 0:
            now_p0, now_alpha, new_ts = exponential_helper.apply_time_decay_simple(
                self._p0,
                self._alpha,
                self._state.last_timestamp,
                tdr,
                approach=self.options["time_decay_approach"]
            )
            self._p0 = now_p0
            self._alpha = now_alpha
            self._state.last_timestamp = new_ts

    def get_spot_price(self, supply: Decimal) -> Decimal:
        """
        Returns the spot price at 'supply':
            price(s) = current_p0 * exp(current_alpha * s)
        Also applies time decay (if time_decay_rate > 0) before calculating.
        If clamp_prices is True, we also clamp the single-spot price
        to [min_price, max_price].
        """
        # 1) Possibly apply time decay
        self._apply_time_decay()

        # 2) Compute raw price
        raw_price = self._p0 * (self._alpha * supply).exp() if self._alpha != 0 else self._p0

        if self.options["clamp_prices"]:
            min_p = self.options["min_price"]
            max_p = self.options["max_price"]
            if min_p is not None and raw_price < min_p:
                raw_price = min_p
            if max_p is not None and raw_price > max_p:
                raw_price = max_p

        return raw_price

    def calculate_purchase_cost(self, amount: Decimal) -> Decimal:
        """
        Integrates from [current_supply.. current_supply + amount] for
        p0 * exp(alpha * x), optionally clamped & after time decay.
        """
        if amount <= 0:
            return Decimal("0")

        # apply time decay
        self._apply_time_decay()

        return exponential_helper.exponential_cost_for_purchase(
            current_supply=self._state.current_supply,
            amount=amount,
            p0=self._p0,
            alpha=self._alpha,
            clamped=self.options["clamp_prices"],
            min_price=self.options["min_price"],
            max_price=self.options["max_price"]
        )

    def calculate_sale_return(self, amount: Decimal) -> Decimal:
        """
        Integrates from [current_supply - amount .. current_supply]
        p0 * exp(alpha * x), optionally clamped & after time decay.
        """
        if amount <= 0:
            return Decimal("0")

        # apply time decay
        self._apply_time_decay()

        return exponential_helper.exponential_return_for_sale(
            current_supply=self._state.current_supply,
            amount=amount,
            p0=self._p0,
            alpha=self._alpha,
            clamped=self.options["clamp_prices"],
            min_price=self.options["min_price"],
            max_price=self.options["max_price"]
        )

    def buy(self, request: TransactionRequest) -> TransactionResult:
        """
        Executes a buy operation, referencing the exponential integral for cost,
        plus partial fill, slippage tolerance, risk profile, fees, etc.
        """
        if not self.options["allow_buy"]:
            raise ValueError("Buys are disabled on this exponential bonding curve.")

        original_amount = request.amount
        final_amount = original_amount

        # 1) max_supply check
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
                    final_amount = common_helper.partial_fill(
                        final_amount, remaining_capacity,
                        approach=self.options["partial_fill_approach"]
                    )
                else:
                    raise ValueError("Requested amount exceeds max supply. Reverting.")

        if final_amount == Decimal("0"):
            # trivial transaction result
            return TransactionResult(
                executed_amount=Decimal("0"),
                total_cost=Decimal("0"),
                average_price=Decimal("0"),
                new_supply=self._state.current_supply,
                timestamp=datetime.now()
            )

        # 2) raw cost
        raw_cost = self.calculate_purchase_cost(final_amount)

        # 3) apply risk profile
        raw_cost = common_helper.apply_risk_profile(
            raw_cost,
            self.options["risk_profile"],
            is_buy=True
        )

        # 4) check max_liquidity
        max_liquidity = self.options["max_liquidity"]
        if max_liquidity is not None:
            current_liquid = self._state.liquidity.amount
            if (current_liquid + raw_cost) > max_liquidity:
                if self.options["pro_rata"]:
                    overrun = (current_liquid + raw_cost) - max_liquidity
                    fraction = (raw_cost - overrun) / raw_cost if raw_cost != 0 else Decimal("0")
                    final_amount = final_amount * fraction
                    raw_cost = raw_cost * fraction
                else:
                    raise ValueError("Max liquidity reached; reverting buy.")

        # 5) slippage tolerance
        slip = self.options["slippage_tolerance"]
        if slip > 0:
            # baseline cost ignoring risk markup
            baseline_cost = exponential_helper.exponential_cost_for_purchase(
                current_supply=self._state.current_supply,
                amount=final_amount,
                p0=self._p0,
                alpha=self._alpha,
                clamped=self.options["clamp_prices"],
                min_price=self.options["min_price"],
                max_price=self.options["max_price"]
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
            raw_cost = final_cost

        else:
            final_cost = raw_cost

        # 6) transaction fee
        final_cost = common_helper.apply_transaction_fee(
            final_cost,
            self.options["txn_fee_rate"],
            is_buy=True
        )

        # 7) update state
        self._update_state_after_buy(final_amount, final_cost)

        # 8) result
        return TransactionResult(
            executed_amount=final_amount,
            total_cost=final_cost,
            average_price=(final_cost / final_amount) if final_amount != 0 else Decimal("0"),
            new_supply=self._state.current_supply,
            timestamp=datetime.now()
        )

    def sell(self, request: TransactionRequest) -> TransactionResult:
        """
        Executes a sell operation, referencing the exponential integral for return,
        plus partial fill, slippage, risk, fees, etc.
        """
        if not self.options["allow_sell"]:
            raise ValueError("Sells are disabled on this exponential bonding curve.")

        original_amount = request.amount
        final_amount = original_amount

        # 1) raw return
        raw_return = self.calculate_sale_return(final_amount)

        # 2) apply risk profile
        raw_return = common_helper.apply_risk_profile(
            raw_return,
            self.options["risk_profile"],
            is_buy=False
        )

        # 3) check liquidity coverage
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

        # 4) slippage tolerance
        slip = self.options["slippage_tolerance"]
        if slip > 0:
            baseline_ret = exponential_helper.exponential_return_for_sale(
                current_supply=self._state.current_supply,
                amount=final_amount,
                p0=self._p0,
                alpha=self._alpha,
                clamped=self.options["clamp_prices"],
                min_price=self.options["min_price"],
                max_price=self.options["max_price"]
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

        # 5) transaction fee (subtract from user proceeds)
        final_return = common_helper.apply_transaction_fee(
            raw_return,
            self.options["txn_fee_rate"],
            is_buy=False
        )

        # 6) update state
        self._update_state_after_sell(final_amount, final_return)

        return TransactionResult(
            executed_amount=final_amount,
            total_cost=final_return,
            average_price=(final_return / final_amount) if final_amount != 0 else Decimal("0"),
            new_supply=self._state.current_supply,
            timestamp=datetime.now()
        )
