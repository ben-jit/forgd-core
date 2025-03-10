from decimal import Decimal
from datetime import datetime
from typing import Union, Optional

from forgd_core.common.enums import BondingCurveDistribution


class LinearCurveHelper:
    """A separate helper class for shared or complex logic used by LinearCurve."""

    @staticmethod
    def cost_between(start: Decimal, end: Decimal, i: Decimal, m: Decimal) -> Decimal:
        """
        Computes the integral of the linear price function from 'start' to 'end':
            cost = i*(end - start) + (m/2)*(end^2 - start^2).
        """
        return (i * (end - start)) + (m / Decimal("2")) * (end ** 2 - start ** 2)

    @staticmethod
    def apply_time_decay(
        current_i: Decimal,
        current_m: Decimal,
        last_timestamp: datetime,
        time_decay_rate: Decimal,
        approach: int = 1,
    ) -> (Decimal, Decimal, datetime):
        """
        Adjusts 'current_i' (initial_price) or 'current_m' (slope) based on time decay,
        if 'time_decay_rate' != 0. Returns (new_i, new_m, new_timestamp).

        :param current_i: current initial price
        :param current_m: current slope
        :param last_timestamp: last time the curve was updated
        :param time_decay_rate: decimal rate to apply per day
        :param approach: demonstration of different ways to handle time decay
        """
        now = datetime.now()
        elapsed_days = (now - last_timestamp).total_seconds() / 86400
        if time_decay_rate != 0 and elapsed_days > 0:
            additional_slope = Decimal(elapsed_days) * time_decay_rate
            # Todo SCOTT - decide which approach or both
            if approach == 1:
                # Approach #1: just add the slope increment
                new_m = current_m + additional_slope
                new_i = current_i
            elif approach == 2:
                # Approach #2: multiply slope by (1 + additional_slope)
                new_m = current_m * (Decimal("1") + additional_slope)
                new_i = current_i
            else:
                # NOTE: For future logic, maybe we also adjust initial price
                new_m = current_m
                new_i = current_i
            new_timestamp = now
        else:
            # No changes
            new_i = current_i
            new_m = current_m
            new_timestamp = last_timestamp
        return new_i, new_m, new_timestamp

    @staticmethod
    def partial_fill(
        user_requested_amount: Decimal,
        available_amount: Decimal,
        approach: int = 1,
    ) -> Decimal:
        """
        Returns the final (partial) amount that can be filled if 'user_requested_amount'
        exceeds 'available_amount'. The approach can vary:
          - approach == 1: return exactly 'available_amount' (hard cap).
          - approach == 2: some proportion or some other logic.

        :param user_requested_amount: the user’s requested buy/sell quantity
        :param available_amount: how many tokens or how much supply is actually available
        :param approach: demonstration of different ways to handle partial fills
        """
        if user_requested_amount <= available_amount:
            return user_requested_amount
        else:
            # Todo SCOTT - choose approach
            if approach == 1:
                return available_amount  # Hard cap
            elif approach == 2:
                # NOTE: If there's a reason to do a ratio-based fill, we might do:
                fraction = available_amount / user_requested_amount if user_requested_amount != 0 else Decimal("0")
                return fraction * user_requested_amount
            # default fallback
            return available_amount

    @staticmethod
    def scale_for_slippage(
        raw_cost_or_return: Decimal,
        expected_cost_or_return: Decimal,
        slippage_tolerance: Decimal,
        is_buy: bool,
        approach: int = 1,
    ) -> (Decimal, bool):
        """
        Compares raw cost/return vs. expected cost/return, applying slippage_tolerance.
        If we exceed tolerance, we either partial fill or revert, based on approach.
        Returns (final_value, reverted_flag).
          - final_value: either the same as raw_cost_or_return or scaled for partial fill
          - reverted_flag: True if we decide to revert (and do no fill)

        :param raw_cost_or_return: raw cost/return
        :param expected_cost_or_return: expected cost/return
        :param slippage_tolerance: slippage tolerance
        :param is_buy: True if buy, so we check "raw_cost > expected*(1+slip)"
                       False if sell, we check "raw_return < expected*(1-slip)"
        :param approach: demonstration of different ways to handle slippage
        """
        if slippage_tolerance <= 0:
            return raw_cost_or_return, False  # no slip logic

        if is_buy:
            # If raw_cost > expected * (1 + slip) => too high
            threshold = expected_cost_or_return * (Decimal("1") + slippage_tolerance)
            if raw_cost_or_return > threshold:
                # Todo SCOTT - choose approach.
                if approach == 1:
                    # partial fill approach => reduce cost proportionally
                    fraction = threshold / raw_cost_or_return if raw_cost_or_return != 0 else Decimal("0")
                    new_value = raw_cost_or_return * fraction
                    return new_value, False
                elif approach == 2:
                    # revert
                    return Decimal("0"), True
        else:
            # selling => raw_return < expected*(1 - slip) => too low
            threshold = expected_cost_or_return * (Decimal("1") - slippage_tolerance)
            if raw_cost_or_return < threshold:
                # Todo SCOTT - choose approach.
                if approach == 1:
                    fraction = threshold / raw_cost_or_return if raw_cost_or_return != 0 else Decimal("0")
                    new_value = raw_cost_or_return * fraction
                    return new_value, False
                elif approach == 2:
                    return Decimal("0"), True

        return raw_cost_or_return, False

    @staticmethod
    def apply_risk_profile(
        raw_value: Decimal,
        risk_profile: Optional[BondingCurveDistribution] = None,
        is_buy: bool = True,
    ) -> Decimal:
        """
        Adjusts raw_cost (if is_buy = True) or raw_return (if is_sell) based on risk_profile.

        :param raw_value: raw value of the curve
        :param risk_profile: e.g. "conservative", "moderate", "aggressive"
        :param is_buy: True if buying, so a markup => user pays more
        """
        if not risk_profile:
            return raw_value

        if risk_profile == BondingCurveDistribution.CONSERVATIVE:
            if is_buy:
                raw_value *= Decimal("1.20")  # SCOTT - CHANGE AS NEEDED
            else:
                raw_value *= Decimal("0.85")  # SCOTT - CHANGE AS NEEDED

        elif risk_profile == BondingCurveDistribution.MODERATE:
            if is_buy:
                raw_value *= Decimal("1.10")  # SCOTT - CHANGE AS NEEDED
            else:
                raw_value *= Decimal("0.9")  # SCOTT - CHANGE AS NEEDED

        elif risk_profile == BondingCurveDistribution.AGGRESSIVE:
            if is_buy:
                raw_value *= Decimal("0.95")  # SCOTT - CHANGE AS NEEDED
            else:
                raw_value *= Decimal("1.01")  # SCOTT - CHANGE AS NEEDED

        return raw_value

    @staticmethod
    def apply_transaction_fee(
        raw_value: Decimal,
        fee_rate: Decimal,
        is_buy: bool
    ) -> Decimal:
        """
        Applies a transaction fee (fee_rate = 0.01 => 1%).
        For a buy, user pays raw_value + fee.
        For a sell, user receives raw_value - fee.

        :param raw_value: raw value of the curve
        :param fee_rate: fee rate
        :param is_buy: True if buying, so a markup => user pays more
        """
        if fee_rate <= 0:
            return raw_value

        fee_amount = raw_value * fee_rate
        if is_buy:
            return raw_value + fee_amount
        else:
            return raw_value - fee_amount
