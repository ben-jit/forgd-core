from decimal import Decimal
from datetime import datetime


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
