from datetime import datetime
from decimal import Decimal
from typing import Optional


from forgd_core.common.model import (
    BondingCurveState,
    BondingCurveParams,
    TransactionResult,
    TransactionRequest
)
from forgd_core.curves.single.base import BondingCurve
from forgd_core.curves.utils.linear_curve_helper import LinearCurveHelper as helper


class LinearBondingCurve(BondingCurve):
    """
        A linear bonding curve with extra flags and features, including:
          - Pro-Rata partial fills
          - Max supply
          - Max liquidity
          - Allow buy/sell toggles
          - Slippage tolerance
          - Transaction fees
          - Time decay
          - Risk profiles (Conservative/Moderate/Aggressive)

        The base formula for cost is:
          price(s) = initial_price + slope * s

        The integral for purchase from supply s to s+Δs is:
          cost(Δs) = ∫(s to s+Δs) [initial_price + slope*x] dx
                    = initial_price*Δs + (slope / 2)*[(s+Δs)² - s²]
    """

    def __init__(self, params: BondingCurveParams, state: Optional[BondingCurveState] = None, **kwargs):
        super().__init__(params, state)
        if self._state and self._state.last_timestamp is None:
            self._state.last_timestamp = datetime.now()

        # Todo SCOTT Default options - Change as needed:
        self.options = {
            "pro_rata": False,
            "max_supply": None,
            "max_liquidity": None,
            "allow_buy": True,
            "allow_sell": True,
            "slippage_tolerance": Decimal("0"),
            "txn_fee_rate": Decimal("0"),
            "time_decay_rate": Decimal("0"),
            "risk_profile": None,
            "partial_fill_approach": 1,
            "time_decay_approach": 1,
            "risk_approach": 1,
        }

        for k, v in kwargs.items():
            if k in self.options:
                self.options[k] = v
            else:
                # Todo TEAM Store rando params in a sub-dict - decide how to handle later.
                if "custom" not in self.options:
                    self.options["custom"] = {}
                self.options["custom"][k] = v

    def _apply_time_decay(self) -> (Decimal, Decimal):
        """
        Retrieves current i, m from params, then calls helper to apply time decay.
        Updates state.last_timestamp as needed.
        """
        # Todo SCOTT - this currently is in pull mode (i.e. each time someone calls a function like buy or sell,
        #  we compute how much time has elapsed and adjust the slope/price accordingly).
        #  We can change this to push or make it purely mathematical if you want a truly continuous decay
        current_i = self.params.initial_price
        current_m = self.params.slope
        last_ts = self._state.last_timestamp

        new_i, new_m, new_ts = helper.apply_time_decay(
            current_i,
            current_m,
            last_ts,
            self.options["time_decay_rate"],
            approach=self.options["time_decay_approach"]
        )
        self._state.last_timestamp = new_ts
        return new_i, new_m

    def _update_state_after_buy(self, amount: Decimal, total_cost: Decimal):
        """
        Updates the internal state after a buy.
        Needs expansion to check treasury if required.
        """
        self._state.current_supply += amount
        if self._state.liquidity:
            self._state.liquidity.amount += total_cost
        else:
            # Todo SCOTT - If liquidity is None, decide how to handle
            pass

    def _update_state_after_sell(self, amount: Decimal, total_return: Decimal):
        """
        Updates the internal state after a sell.
        """
        self._state.current_supply -= amount
        if self._state.liquidity:
            self._state.liquidity.amount -= total_return
        else:
            # Todo SCOTT - If liquidity is None, decide how to handle
            pass

    def get_spot_price(self, supply: Decimal) -> Decimal:
        """
        Return the spot price at the given 'supply', factoring in possible time decay.
        """
        i, m = self._apply_time_decay()
        return i + (m * supply)

    def calculate_purchase_cost(self, amount: Decimal) -> Decimal:
        """
        Integrates price from current_supply to current_supply + amount to find the total cost for 'amount' tokens.
        """
        i, m = self._apply_time_decay()
        start_supply = self._state.current_supply
        end_supply = start_supply + amount
        return helper.cost_between(start_supply, end_supply, i, m)

    def calculate_sale_return(self, amount: Decimal) -> Decimal:
        """
        Integrates price from (current_supply - amount) to current_supply.
        """
        if amount > self._state.current_supply:
            raise ValueError("Cannot sell more tokens than current supply.")

        i, m = self._apply_time_decay()
        start_supply = self._state.current_supply - amount
        end_supply = self._state.current_supply
        return helper.cost_between(start_supply, end_supply, i, m)

    def buy(self, request: TransactionRequest) -> TransactionResult:
        """
        Buys 'request.amount' tokens, respecting:
          - allow_buy
          - pro_rata partial fills or revert if max_supply is exceeded
          - time decay
          - risk profile
          - max_liquidity
          - slippage tolerance
          - transaction fees
        Updates supply & liquidity in state, returns a TransactionResult.
        """
        # 0) Check if buying is allowed
        if not self.options["allow_buy"]:
            raise ValueError("Buys are disabled for this bonding curve.")

        # 1) Check max_supply and possibly partial fill
        original_amount = request.amount
        final_amount = original_amount

        max_supply = self.options["max_supply"]
        if max_supply is not None:
            remaining = max_supply - self._state.current_supply
            if remaining <= Decimal("0"):
                # no supply left
                if self.options["pro_rata"]:
                    final_amount = Decimal("0")
                else:
                    raise ValueError("Max supply reached; cannot buy more tokens.")
            elif original_amount > remaining:
                if self.options["pro_rata"]:
                    final_amount = helper.partial_fill(
                        original_amount,
                        remaining,
                        approach=self.options["partial_fill_approach"]
                    )
                else:
                    raise ValueError("Requested amount exceeds max supply. Reverting.")

        if final_amount == Decimal("0"):
            # Return a trivial result if no tokens can be bought
            return TransactionResult(
                executed_amount=Decimal("0"),
                total_cost=Decimal("0"),
                average_price=Decimal("0"),
                new_supply=self._state.current_supply,
                timestamp=datetime.now()
            )

        # 2) Calculate raw cost
        raw_cost = self.calculate_purchase_cost(final_amount)

        # 3) Apply risk profile (mark up the cost if "conservative", discount if "aggressive", etc.)
        raw_cost = helper.apply_risk_profile(
            raw_cost,
            self.options["risk_profile"],
            is_buy=True
        )

        # 4) Check max_liquidity (partial fill or revert)
        max_liquidity = self.options["max_liquidity"]
        if max_liquidity is not None:
            current_liquid = self._state.liquidity.amount
            if (current_liquid + raw_cost) > max_liquidity:
                if self.options["pro_rata"]:
                    overrun = (current_liquid + raw_cost) - max_liquidity
                    # Simple approach: scale down final_amount proportionally
                    fraction = (raw_cost - overrun) / raw_cost if raw_cost != 0 else Decimal("0")
                    final_amount = final_amount * fraction
                    raw_cost = raw_cost * fraction
                else:
                    raise ValueError("Max liquidity reached; reverting buy.")

        # 5) Slippage tolerance
        slip = self.options["slippage_tolerance"]
        if slip > 0:
            # For demonstration, let's define "expected_cost" as cost without the risk markup:
            baseline_cost = super().calculate_purchase_cost(final_amount)
            final_cost, revert_flag = helper.scale_for_slippage(
                raw_cost,
                baseline_cost,
                slip,
                is_buy=True,
                approach=self.options["partial_fill_approach"]
            )
            if revert_flag:
                raise ValueError("Slippage tolerance exceeded. Reverting buy.")
            raw_cost = final_cost

        # 6) Transaction fee
        final_cost = helper.apply_transaction_fee(
            raw_cost,
            self.options["txn_fee_rate"],
            is_buy=True
        )

        # 7) Update state
        self._update_state_after_buy(final_amount, final_cost)

        # 8) Return TransactionResult
        return TransactionResult(
            executed_amount=final_amount,
            total_cost=final_cost,
            average_price=(final_cost / final_amount) if final_amount != 0 else Decimal("0"),
            new_supply=self._state.current_supply,
            timestamp=datetime.now()
        )

    def sell(self, request: TransactionRequest) -> TransactionResult:
        """
        Sells 'request.amount' tokens, respecting:
          - allow_sell
          - pro_rata partial fills or revert if insufficient liquidity
          - time decay
          - risk profile
          - slippage tolerance
          - transaction fees
        Updates supply & liquidity in state, returns a TransactionResult.
        """
        if not self.options["allow_sell"]:
            raise ValueError("Sells are disabled for this bonding curve.")

        original_amount = request.amount
        final_amount = original_amount
        raw_return = self.calculate_sale_return(final_amount)

        # 1) Apply risk profile (often a discount for conservative sellers, etc.)
        raw_return = helper.apply_risk_profile(
            raw_return,
            self.options["risk_profile"],
            is_buy=False
        )

        # 2) Check liquidity for the return
        current_liquid = self._state.liquidity.amount
        if raw_return > current_liquid:
            if self.options["pro_rata"]:
                fraction = current_liquid / raw_return if raw_return != 0 else Decimal("0")
                final_amount = final_amount * fraction
                raw_return = current_liquid
            else:
                raise ValueError("Insufficient liquidity. Reverting sell.")

        # 3) Slippage tolerance
        slip = self.options["slippage_tolerance"]
        if slip > 0:
            # baseline_return ignoring risk markup
            baseline_ret = super().calculate_sale_return(final_amount)
            final_ret, revert_flag = helper.scale_for_slippage(
                raw_return,
                baseline_ret,
                slip,
                is_buy=False,
                approach=self.options["partial_fill_approach"]
            )
            if revert_flag:
                raise ValueError("Slippage tolerance not met. Reverting sell.")
            raw_return = final_ret

        # 4) Transaction fees (subtract from user proceeds)
        final_return = helper.apply_transaction_fee(
            raw_return,
            self.options["txn_fee_rate"],
            is_buy=False
        )

        # 5) Update state
        self._update_state_after_sell(final_amount, final_return)

        # 6) Return transaction result
        return TransactionResult(
            executed_amount=final_amount,
            total_cost=final_return,  # "total_cost" is how the base model labels it
            average_price=(final_return / final_amount) if final_amount != 0 else Decimal("0"),
            new_supply=self._state.current_supply,
            timestamp=datetime.now()
        )
