from datetime import datetime
from decimal import Decimal
from typing import List

from forgd_core.common.model import StepConfig


class StepwiseCurveHelper:
    """
    A helper class for stepwise bonding curve logic, including:
      - Time decay for tier prices
      - Calculations for buy (stepwise_cost_for_purchase) & sell (stepwise_cost_for_sale)
      - Partial fill / pro-rata logic
      - Slippage checks
      - Risk profile adjustments
      - Transaction fees
    """

    @staticmethod
    def validate_and_sort_steps(steps: List[StepConfig]) -> List[StepConfig]:
        """
        Sorts the given list of StepConfig by ascending supply_threshold
        and validates that each threshold is strictly greater than the previous.

        :param steps: list of StepConfig
        :return: a new list of sorted, validated StepConfig
        :raises ValueError: if thresholds are not strictly ascending
        """
        if not steps:
            return steps  # no tiers => nothing to sort or validate

        # Sort by supply_threshold
        sorted_steps = sorted(steps, key=lambda s: s.supply_threshold)

        # Validate strictly ascending thresholds
        for i in range(1, len(sorted_steps)):
            if sorted_steps[i].supply_threshold <= sorted_steps[i - 1].supply_threshold:
                raise ValueError(
                    f"Stepwise tiers must have strictly ascending supply_threshold. "
                    f"Found {sorted_steps[i].supply_threshold} <= {sorted_steps[i - 1].supply_threshold}"
                )

        # Check if any threshold is negative (we may not let this happen by design):
        for step in sorted_steps:
            if step.supply_threshold < 0:
                raise ValueError("Negative thresholds are not allowed in stepwise tiers.")

        return sorted_steps

    @staticmethod
    def apply_time_decay_to_steps(
        steps: List[StepConfig],
        last_timestamp: datetime,
        time_decay_rate: Decimal,
        approach: int = 1
    ) -> (List[StepConfig], datetime):
        """
        Adjusts each tier's price based on how much time has elapsed since last_timestamp.
        Example approaches:
          - approach == 1: price *= (1 + time_decay_rate * elapsed_days)
          - approach == 2: do something else

        :param steps: current step configs
        :param last_timestamp: when we last updated
        :param time_decay_rate: how much to multiply or add per day
        :param approach: demonstration of different ways to handle time-based tier updates
        :return: (updated_steps, new_timestamp)
        """
        now = datetime.now()
        elapsed_days = (now - last_timestamp).total_seconds() / 86400
        if time_decay_rate == 0 or elapsed_days <= 0:
            # no change
            return steps, last_timestamp

        new_steps = []
        for step in steps:
            new_price = step.price
            additional_factor = Decimal(elapsed_days) * time_decay_rate
            # Todo SCOTT - choose approach
            if approach == 1:
                # Approach #1: price *= (1 + additional_factor)
                new_price = step.price * (Decimal("1") + additional_factor)
            elif approach == 2:
                # Approach #2: maybe additive => new_price = step.price + (time_decay_rate * elapsed_days)
                new_price = step.price + (time_decay_rate * Decimal(elapsed_days))

            # store the updated StepConfig
            new_steps.append(
                StepConfig(step.supply_threshold, new_price)
            )

        return new_steps, now

    @staticmethod
    def stepwise_cost_for_purchase(
        current_supply: Decimal,
        amount: Decimal,
        steps: List[StepConfig],
        allow_partial_fill: bool = True
    ) -> Decimal:
        """
        Calculates total cost to buy 'amount' of tokens starting from 'current_supply'.
        The user moves through tiers until:
          - They purchase 'amount' tokens
          - Or they reach the final tier capacity
          - If 'allow_partial_fill' = False and they exceed final tier capacity, raise ValueError

        :param current_supply: The current supply of tokens before purchase
        :param amount: How many tokens the user wants to buy
        :param steps: A sorted list of StepConfig (ascending supply_threshold)
        :param allow_partial_fill: if False => revert if request exceeds final tier capacity
        :return: total cost as a Decimal
        """
        if amount <= Decimal("0"):
            return Decimal("0")

        total_cost = Decimal("0")
        remaining_to_buy = amount

        # We'll track the "previous_threshold" to define each tier range [previous_threshold ... step.supply_threshold)
        prev_threshold = Decimal("0")

        # 1) Find which tier the current supply is in
        idx = 0
        while idx < len(steps) and current_supply >= steps[idx].supply_threshold:
            prev_threshold = steps[idx].supply_threshold
            idx += 1

        # 2) Traverse tiers until we fill the requested amount or exhaust tiers
        while remaining_to_buy > Decimal("0") and idx < len(steps):
            tier = steps[idx]
            if tier.supply_threshold <= prev_threshold:
                # If thresholds are not strictly ascending, or we've progressed beyond it
                idx += 1
                continue

            # tokens in this tier: [prev_threshold ... tier.supply_threshold)
            tier_capacity = tier.supply_threshold - max(prev_threshold, current_supply)
            if tier_capacity <= Decimal("0"):
                # This means current_supply is beyond this tier, move on
                prev_threshold = tier.supply_threshold
                idx += 1
                continue

            # Number of tokens the user can buy in this tier
            tokens_in_this_tier = min(remaining_to_buy, tier_capacity)

            if tokens_in_this_tier > Decimal("0"):
                total_cost += tokens_in_this_tier * tier.price
                remaining_to_buy -= tokens_in_this_tier
                current_supply += tokens_in_this_tier
                prev_threshold = tier.supply_threshold

            # Move to next tier if needed
            idx += 1

        # 3) If we still have tokens to buy after exhausting all tiers
        if remaining_to_buy > Decimal("0"):
            if allow_partial_fill:
                # Partial fill means the user only gets what was available
                # The leftover is unfilled, but we do NOT raise an error
                # total_cost is for the partial portion we've successfully bought
                pass  # Do nothing: partial fill is effectively complete
            else:
                # revert => the user wanted the full amount, but there's no capacity
                raise ValueError("Requested amount exceeds final tier capacity. Reverting.")

        return total_cost

    @staticmethod
    def stepwise_return_for_sale(
        current_supply: Decimal,
        amount: Decimal,
        steps: List[StepConfig],
        allow_partial_fill: bool = True
    ) -> Decimal:
        """
        Calculates how much capital is returned if a user sells 'amount' of tokens
        from the current supply. We go *downwards* through tiers.

        If 'allow_partial_fill' = False and the sale cannot be fully accounted for
        (i.e., user tries to sell more tokens than exist in the relevant tiers),
        we raise ValueError. If partial fill is allowed, we sell as much as possible
        and ignore the rest.
        """
        if amount <= Decimal("0"):
            return Decimal("0")

        if not steps:
            # No tiers => either partial fill => 0 or revert => error
            if allow_partial_fill:
                return Decimal("0")
            else:
                raise ValueError("Exceeded final tier capacity. Reverting.")

        total_return = Decimal("0")
        remaining_to_sell = amount

        # 1) Find the tier that includes current_supply (move down from highest threshold)
        idx = len(steps) - 1
        while idx >= 0 and current_supply >= steps[idx].supply_threshold:
            idx -= 1

        if idx < 0:
            idx = 0

        # 2) Move downward until we sell everything or run out of tiers
        while remaining_to_sell > Decimal("0") and idx >= 0:
            tier = steps[idx]
            # The tier covers supply range [previous_threshold .. tier.supply_threshold)
            prev_threshold = steps[idx - 1].supply_threshold if idx > 0 else Decimal("0")

            # how many tokens are "in this tier" from supply perspective
            # e.g., if current_supply=7200, tier.supply_threshold=5000 =>
            # the tier covers [5000...7200). So tokens_in_tier= 7200 - 5000=2200
            in_this_tier = current_supply - max(prev_threshold, Decimal("0"))
            if in_this_tier < Decimal("0"):
                in_this_tier = Decimal("0")

            # we can sell up to 'in_this_tier' tokens at tier.price
            can_sell_here = min(in_this_tier, remaining_to_sell)
            if can_sell_here > Decimal("0"):
                total_return += can_sell_here * tier.price
                remaining_to_sell -= can_sell_here
                current_supply -= can_sell_here

            idx -= 1  # move to lower tier

        # 3) If there's still tokens to sell but no more tiers
        if remaining_to_sell > Decimal("0"):
            if allow_partial_fill:
                # We sold as much as possible; leftover remains unsold
                pass
            else:
                raise ValueError("Requested sale amount exceeds tier coverage. Reverting.")

        return total_return
