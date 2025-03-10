from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional

from forgd_core.common.model import BondingCurveState, BondingCurveParams, TransactionRequest, TransactionResult


class BondingCurve(ABC):
    """Abstract base class defining the interface for any bonding curve implementation."""
    def __init__(self, params: 'BondingCurveParams', state: Optional['BondingCurveState'] = None):
        """
        Initializes the bonding curve with parameters and an optional existing state.

        :param params: BondingCurveParameters - defines curve configuration
        :param state: BondingCurveState - optional initial state
        """
        self._params = params
        self._state = state or BondingCurveState()

    @property
    def params(self) -> 'BondingCurveParams':
        """Returns the bonding curve parameters."""
        return self._params

    @property
    def current_supply(self) -> Decimal:
        """Returns the current supply from the state."""
        return self._state.current_supply

    @abstractmethod
    def get_spot_price(self, supply: Decimal) -> Decimal:
        """
        Returns the current spot price for a given supply.

        :param supply: Decimal - Current supply of tokens.
        :return: Decimal: The price at given supply.
        """
        pass

    @abstractmethod
    def calculate_purchase_cost(self, amount: Decimal) -> Decimal:
        """
        Calculates how much it costs to buy a specified 'amount' of tokens from the current state of the bonding curve.
        This may involve incremental pricing if the curve is not perfectly linear (e.g., exponential, stepwise).

        :param amount: Decimal - Number of tokens the user wants to purchase.
        :return: Total cost to purchase 'amount' of tokens.
        """
        pass

    @abstractmethod
    def calculate_sale_return(self, amount: Decimal) -> Decimal:
        """
        Calculates how much capital (e.g., collateral) is returned if a user sells a specified 'amount' of tokens
        back into the bonding curve.

        :param amount: Decimal - Number of tokens the user wants to sell.
        :return: Total return for selling 'amount' of tokens.
        """
        pass

    @abstractmethod
    def buy(self, request: 'TransactionRequest') -> 'TransactionResult':
        """
        Executes a buy operation along the bonding curve, updating the internal state (e.g., supply, liquidity)
        if needed, and returns a TransactionResult.

        :param request: TransactionRequest
        :return: A TransactionResult detailing executed amount, total cost, new supply, etc.
        """
        pass

    @abstractmethod
    def sell(self, request: 'TransactionRequest') -> 'TransactionResult':
        """
        Executes a sell operation along the bonding curve, updating the internal state (e.g., supply, liquidity)
        if needed, and returns a TransactionResult.

        :param request: TransactionRequest
        :return: A TransactionResult detailing executed amount, total return, new supply, etc.
        """
        pass

    def _update_state_after_buy(self, amount: Decimal, total_cost: Decimal):
        """
        Updates the internal state of the bonding curve after buying 'amount' of tokens.

        :param amount: Decimal - Number of tokens the user wants to buy.
        :param total_cost: Decimal - Total cost to purchase 'amount' of tokens.
        """
        self._state.current_supply = self.current_supply + amount
        self._state.liquidity.amount += total_cost

    def _update_state_after_sell(self, amount: Decimal, total_cost: Decimal):
        """
        Updates the internal state of the bonding curve after selling 'amount' of tokens.

        :param amount: Decimal - Number of tokens the user wants to sell.
        :param total_cost: Decimal - Total cost to purchase 'amount' of tokens.
        """
        self._state.current_supply = self.current_supply - amount
        self._state.liquidity.amount -= total_cost
