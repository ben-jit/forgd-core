from decimal import Decimal
from datetime import datetime
from typing import Tuple, Optional


class ExponentialCurveHelper:
    """
    A helper class for exponential bonding curve logic, including:
      - Parameter validation
      - Time decay for p0 or alpha
      - Cost for purchase (integral from s...s+Δs)
      - Return for sale (integral from s-Δs...s)
    """

    @staticmethod
    def validate_exponential_params(p0: Decimal, alpha: Decimal):
        """
        Validates that:
          - p0 > 0
          - alpha >= 0 (for now we ignore negative alpha, which would be a decreasing curve)
        Raises ValueError if invalid.
        """
        if p0 <= Decimal("0"):
            raise ValueError("Exponential curve requires p0 > 0.")
        if alpha < Decimal("0"):
            raise ValueError("Negative alpha is not supported in this version.")

    @staticmethod
    def apply_time_decay_simple(
        current_p0: Decimal,
        current_alpha: Decimal,
        last_timestamp: datetime,
        time_decay_rate: Decimal,
        approach: int = 1,
    ) -> Tuple[Decimal, Decimal, datetime]:
        """
        A simpler time decay that modifies *one* parameter over time, either p0 or alpha.

        approach 1: multiply alpha by (1 + rate * elapsed_days)
        approach 2: multiply p0 by (1 + rate * elapsed_days)

        Return (new_p0, new_alpha, new_timestamp).
        """

        now = datetime.now()
        elapsed_days = Decimal((now - last_timestamp).total_seconds()) / Decimal("86400")
        if time_decay_rate <= 0 or elapsed_days <= 0:
            # no changes
            return current_p0, current_alpha, last_timestamp

        additional_factor = time_decay_rate * elapsed_days
        new_p0 = current_p0
        new_alpha = current_alpha

        # Todo SCOTT - can choose approach.
        if approach == 1:
            # approach #1 => alpha = alpha * (1 + additional_factor)
            new_alpha = current_alpha * (Decimal("1") + additional_factor)
        elif approach == 2:
            # approach #2 => p0 = p0 * (1 + additional_factor)
            new_p0 = current_p0 * (Decimal("1") + additional_factor)
        else:
            # no-op, or define other logic
            pass

        return new_p0, new_alpha, now

    @staticmethod
    def apply_time_decay_advanced(
        current_p0: Decimal,
        current_alpha: Decimal,
        last_timestamp: datetime,
        time_decay_rate: Decimal,
        approach: int = 1
    ) -> Tuple[Decimal, Decimal, datetime]:
        """
        An example "advanced" function that adjusts both p0 and alpha over time.

        approach 1:
          - alpha = alpha * (1 + rate * elapsed_days)
          - p0    = p0 * (1 + rate * elapsed_days)   # i.e. both scale up

        approach 2:
          - Add a small offset to alpha each day
          - Multiply p0 by a factor, etc.
        """
        now = datetime.now()
        elapsed_days = Decimal((now - last_timestamp).total_seconds()) / Decimal("86400")
        if time_decay_rate <= 0 or elapsed_days <= 0:
            return current_p0, current_alpha, last_timestamp

        additional_factor = time_decay_rate * elapsed_days
        new_p0 = current_p0
        new_alpha = current_alpha

        if approach == 1:
            # both scale up multiplicatively
            scale = (Decimal("1") + additional_factor)
            new_alpha = current_alpha * scale
            new_p0 = current_p0 * scale
        elif approach == 2:
            # example: alpha += rate * elapsed_days, p0 *= (1 + some fraction)
            new_alpha = current_alpha + additional_factor
            new_p0 = current_p0 * (Decimal("1") + (additional_factor / Decimal("2")))

        return new_p0, new_alpha, now

    @staticmethod
    def _clamped_exponential_integral(
        start: Decimal,
        end: Decimal,
        p0: Decimal,
        alpha: Decimal,
        min_price: Optional[Decimal],
        max_price: Optional[Decimal]
    ) -> Decimal:
        """
        Computes ∫ from x=start..end of clamp( p0 * e^(alpha*x), [min_price, max_price] ) dx,
        i.e., piecewise integration:
          - if p0*e^(alpha*x) < min_price => integrate min_price
          - if p0*e^(alpha*x) > max_price => integrate max_price
          - else => integrate p0*e^(alpha*x)

        Returns the total area as a Decimal.

        If alpha=0 => price is constant p0 => clamp that to [min_price, max_price].
        """
        if start >= end:
            return Decimal("0")
        if alpha == 0:
            # Price is constant = p0
            clamped_price = p0
            if min_price is not None and clamped_price < min_price:
                clamped_price = min_price
            if max_price is not None and clamped_price > max_price:
                clamped_price = max_price
            return clamped_price * (end - start)

        # We'll potentially have up to two "breakpoints":
        #   xMin = (1/alpha)*ln(min_price/p0) if min_price>0
        #   xMax = (1/alpha)*ln(max_price/p0) if max_price>0
        # We'll place them in ascending order with start and end => piecewise sub-intervals
        intervals = [start]
        # compute xMin if min_price is set and >0
        if min_price is not None and min_price > 0:
            # xMin = (1/alpha)*ln(min_price / p0)
            # only if p0 < min_price => that means there's a region where e^(alpha*x) < min_price/p0
            # i.e., e^(alpha*x) < (min_price/p0). This might or might not intersect [start,end].
            if min_price > p0:
                # only then is there some region where p0*e^(alpha*x) transitions from < min_price to >= min_price
                try:
                    xMin = (min_price / p0).ln() / alpha
                    # We'll only care if xMin is in (start, end)
                    if start < xMin < end:
                        intervals.append(xMin)
                except (ValueError, OverflowError):
                    pass

        # compute xMax if max_price is set and >0
        if max_price is not None and max_price > 0:
            if max_price > p0:
                # There's a region in [start,end] where we transition from <max_price to >max_price
                # i.e., e^(alpha*x) crosses (max_price/p0).
                try:
                    xMax = (max_price / p0).ln() / alpha
                    if start < xMax < end:
                        intervals.append(xMax)
                except (ValueError, OverflowError):
                    pass
            else:
                # if max_price <= p0 => p0 e^(alpha*x) might always be >= max_price for x>0
                # We'll handle that in the piecewise logic below.
                pass

        intervals.append(end)
        intervals = sorted(list(set(intervals)))  # unique, sorted

        total_area = Decimal("0")
        for i in range(len(intervals) - 1):
            xA = intervals[i]
            xB = intervals[i + 1]
            if xA >= xB:
                continue  # skip degenerate

            # We'll pick the "midpoint" to see what the exponential is in that subrange
            # to figure out which portion we are in. Or we do a more direct approach:
            # We know that on [xA... xB], p0 e^(alpha*x) is monotonic increasing if alpha>0.
            # We'll check the price at xA and xB to see if it crosses min_price or max_price.

            priceA = p0 * (alpha * xA).exp()
            priceB = p0 * (alpha * xB).exp()

            # We have a few cases:
            # 1) If priceB < min_price => entire sub-interval is below min_price => integrate min_price
            # 2) If priceA > max_price => entire sub-interval is above max_price => integrate max_price
            # 3) Otherwise => we do the actual exponential integral, but might still intersect min/max in the interior

            # But because we put breakpoints at xMin/xMax, the function won't cross min_price or max_price
            # inside this subinterval, so we can treat the entire subinterval as either fully below, within, or above.

            if (min_price is not None) and (priceB < min_price):
                # entire subrange is below min_price
                total_area += min_price * (xB - xA)
            elif (max_price is not None) and (priceA > max_price):
                # entire subrange is above max_price
                total_area += max_price * (xB - xA)
            else:
                # partial or fully in normal exponential range
                # clamp the actual integral at min_price or max_price, but we've bracketed so
                # we know in [xA...xB], price is between [min_price, max_price].
                # We'll do an integral of p0 e^(alpha*x) but also clamp if it doesn't exceed bounds.

                # We'll do a simpler approach:
                # Because xA...xB was chosen so we don't cross min/max inside it,
                # we can just do the integral of p0 e^(alpha*x).
                # Then, if the entire subrange is above min_price and below max_price, that is correct.

                sub_area = (p0 / alpha) * ((alpha * xB).exp() - (alpha * xA).exp())

                # However, if min_price is not None and priceA < min_price, or if max_price is not None and priceB>max_price,
                # we do partial piecewise? Actually, we "cut" the interval at xMin/xMax so we wouldn't have crossing inside.
                total_area += sub_area

        return total_area

    @staticmethod
    def exponential_cost_for_purchase(
        current_supply: Decimal,
        amount: Decimal,
        p0: Decimal,
        alpha: Decimal,
        clamped: bool = False,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None
    ) -> Decimal:
        """
        Computes the integral from s...(s+Δs) of p0 * e^(alpha*x) dx.
        If alpha != 0:
          cost = (p0 / alpha) * [ e^(alpha*(s+Δs)) - e^(alpha*s) ]
        If alpha == 0:
          cost = p0 * amount  (constant price = p0)

        :param current_supply: the current supply s
        :param amount: number of tokens to buy Δs
        :param p0: initial price factor
        :param alpha: exponent rate
        :param clamped: if True, use a clamped exponential integral
        :param min_price: minimum price for clamped integral
        :param max_price: maximum price for clamped integral
        :return: total cost as Decimal
        """
        if amount <= 0:
            return Decimal("0")

        start = current_supply
        end = current_supply + amount

        if clamped:
            return ExponentialCurveHelper._clamped_exponential_integral(
            start, end, p0, alpha, min_price, max_price
        )
        else:
            if alpha == 0:
                # cost = p0 * amount
                return p0 * amount
            else:
                # cost = (p0 / alpha) * [ e^(alpha*end) - e^(alpha*start) ]
                return (p0 / alpha) * ((alpha * end).exp() - (alpha * start).exp())

    @staticmethod
    def exponential_return_for_sale(
        current_supply: Decimal,
        amount: Decimal,
        p0: Decimal,
        alpha: Decimal,
        clamped: bool = False,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None
    ) -> Decimal:
        """
        Computes the integral from (s - Δs)...s of p0 * e^(alpha*x) dx.
        If alpha != 0:
          return = (p0 / alpha) * [ e^(alpha*s) - e^(alpha*(s - Δs)) ]
        If alpha == 0:
          return = p0 * amount  (constant price).

        :param current_supply: the current supply s
        :param amount: number of tokens to sell Δs
        :param p0: initial price factor
        :param alpha: exponent rate
        :param clamped: if True, use a clamped exponential integral
        :param min_price: minimum price for clamped integral
        :param max_price: maximum price for clamped integral
        :return: total return as Decimal
        """
        if amount <= 0:
            return Decimal("0")

        start = current_supply - amount
        end = current_supply

        if start < 0:
            # optional: raise an error or let partial fill logic happen externally
            # We'll just do the integral from 0...end.
            # For now, raise ValueError to avoid negative supply integrals:
            raise ValueError("Cannot sell more tokens than the current supply.")

        if clamped:
            return ExponentialCurveHelper._clamped_exponential_integral(
            start, end, p0, alpha, min_price, max_price
        )
        else:
            if alpha == 0:
                # return = p0 * amount
                return p0 * amount
            else:
                # return = (p0 / alpha) * [ e^(alpha*end) - e^(alpha*start) ]
                return (p0 / alpha) * ((alpha * end).exp() - (alpha * start).exp())
