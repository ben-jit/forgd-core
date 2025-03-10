from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Union

from forgd_core.common.enums import ChainType, BondingCurveType, OrderSide


@dataclass
class Token:
    """Represents a particular token whose price is determined by the bonding curve."""
    name: str
    symbol: str
    decimals: int
    total_supply: Optional[Decimal] = Decimal('0.0')


@dataclass
class ChainInfo:
    """Stores optional metadata about the chain."""
    chain_type: ChainType = ChainType.UNKNOWN
    chain_id: Optional[Union[int, str]] = None
    name: Optional[str] = None


@dataclass
class Liquidity:
    """Represents liquidity or collateral in the system."""
    token: Token
    amount: Optional[Decimal] = Decimal('0.0')
    amount_usd: Optional[Decimal] = None
    currency_symbol: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class StepConfig:
    """Defines a single step's threshold and the price at or above that threshold."""
    supply_threshold: Decimal
    price: Decimal


@dataclass
class BondingCurveParams:
    """Encapsulates parameters for initializing a specific bonding curve shape."""
    curve_type: BondingCurveType
    initial_price: Optional[Decimal] = Decimal('0.0')
    slope: Optional[Decimal] = Decimal('0.0')
    steps: Optional[List[StepConfig]] = None
    exponential: Optional[Decimal] = Decimal('0.0')
    max_price: Optional[Decimal] = Decimal('0.0')
    other_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.initial_price < Decimal('0.0'):
            raise ValueError("Initial price must be non-negative.")
        if self.curve_type not in [BondingCurveType.LINEAR, BondingCurveType.EXPONENTIAL, BondingCurveType.STEPWISE]:
            raise ValueError("Invalid curve type.")
        if self.curve_type == BondingCurveType.LINEAR and self.slope < Decimal('0.0'):
            raise ValueError("Slope must be non-negative.")
        if self.curve_type == BondingCurveType.EXPONENTIAL and self.exponential < Decimal('0.0'):
            raise ValueError("Exponential must be non-negative.")
        if self.curve_type == BondingCurveType.STEPWISE:
            if not self.steps:
                raise ValueError("Stepwise curve requires steps.")
            if len(self.steps) < 2:
                raise ValueError("Stepwise curve requires at least two steps.")
            for i in range(len(self.steps) - 1):
                if self.steps[i].supply_threshold >= self.steps[i + 1].supply_threshold:
                    raise ValueError(f"Step {i} threshold must be less than step {i + 1} threshold.")


@dataclass
class BondingCurveState:
    """Tracks the runtime state of an instantiated bonding curve."""
    current_supply: Optional[Decimal] = Decimal('0.0')
    current_price: Optional[Decimal] = Decimal('0.0')
    liquidity: Optional[Liquidity] = None
    last_timestamp: Optional[datetime] = None


@dataclass
class TransactionRequest:
    """Represents a discrete purchase or sale request."""
    token: Token
    order_type: OrderSide
    amount: Optional[Decimal] = Decimal('0.0')
    user_id: Optional[str] = None


@dataclass
class TransactionResult:
    """Outcome of a transaction."""
    executed_amount: Decimal
    total_cost: Decimal
    average_price: Decimal
    new_supply: Decimal
    timestamp: datetime


@dataclass
class SimulationResult:
    """Holds the aggregated results of simulating multiple transactions."""
    transactions: List[TransactionResult] = field(default_factory=list)
    final_supply: Optional[Decimal] = Decimal('0.0')
    final_price: Optional[Decimal] = Decimal('0.0')
    average_purchase_price: Optional[Decimal] = Decimal('0.0')
    average_sale_price: Optional[Decimal] = Decimal('0.0')
    metadata: Dict = field(default_factory=dict)
