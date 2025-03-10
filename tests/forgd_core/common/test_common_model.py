import pytest
from datetime import datetime
from decimal import Decimal

from forgd_core.common.enums import (
    ChainType,
    BondingCurveType,
    OrderSide
)

from forgd_core.common.model import (
    Token,
    ChainInfo,
    Liquidity,
    StepConfig,
    BondingCurveParams,
    BondingCurveState,
    TransactionRequest,
    TransactionResult,
    SimulationResult,
)


class TestToken:
    def test_token_defaults(self):
        """Test Token instantiation with default total_supply."""
        token = Token(name="Test Token", symbol="TT", decimals=18)
        assert token.name == "Test Token"
        assert token.symbol == "TT"
        assert token.decimals == 18
        assert token.total_supply == Decimal("0.0")

    def test_token_custom_supply(self):
        """Test Token with custom total_supply."""
        token = Token(
            name="Custom Supply Token",
            symbol="CST",
            decimals=18,
            total_supply=Decimal("1000.5")
        )
        assert token.total_supply == Decimal("1000.5")


class TestChainInfo:
    def test_chain_info_defaults(self):
        """Test ChainInfo with default fields."""
        chain_info = ChainInfo()
        assert chain_info.chain_type == ChainType.UNKNOWN
        assert chain_info.chain_id is None
        assert chain_info.name is None

    def test_chain_info_custom(self):
        """Test ChainInfo with custom fields."""
        chain_info = ChainInfo(
            chain_type=ChainType.EVM,
            chain_id=1,
            name="Ethereum Mainnet"
        )
        assert chain_info.chain_type == ChainType.EVM
        assert chain_info.chain_id == 1
        assert chain_info.name == "Ethereum Mainnet"

    def test_chain_info_chain_id_as_string(self):
        """Test chain_id can handle a string."""
        chain_info = ChainInfo(chain_type=ChainType.SOLANA, chain_id="SolanaTestnet")
        assert chain_info.chain_type == ChainType.SOLANA
        assert chain_info.chain_id == "SolanaTestnet"


class TestLiquidity:
    def test_liquidity_defaults(self):
        """Test Liquidity with default fields."""
        token = Token(name="Stablecoin", symbol="USDC", decimals=6)
        liquidity = Liquidity(token=token)
        assert liquidity.token == token
        assert liquidity.amount == Decimal("0.0")
        assert liquidity.amount_usd is None
        assert liquidity.currency_symbol is None
        assert liquidity.updated_at is None

    def test_liquidity_custom(self):
        """Test Liquidity with custom amounts and metadata."""
        token = Token(name="Test Token", symbol="TT", decimals=18)
        liquidity = Liquidity(
            token=token,
            amount=Decimal("100"),
            amount_usd=Decimal("200.50"),
            currency_symbol="USD",
            updated_at="2025-03-08T10:30:00"
        )
        assert liquidity.amount == Decimal("100")
        assert liquidity.amount_usd == Decimal("200.50")
        assert liquidity.currency_symbol == "USD"
        assert liquidity.updated_at == "2025-03-08T10:30:00"


class TestStepConfig:
    def test_step_config_creation(self):
        """Test valid StepConfig creation."""
        step_config = StepConfig(
            supply_threshold=Decimal("1000"),
            price=Decimal("2.5")
        )
        assert step_config.supply_threshold == Decimal("1000")
        assert step_config.price == Decimal("2.5")


class TestBondingCurveParams:
    def test_linear_valid_params(self):
        """Test valid linear BondingCurveParams."""
        params = BondingCurveParams(
            curve_type=BondingCurveType.LINEAR,
            initial_price=Decimal("1.0"),
            slope=Decimal("0.5"),
            max_price=Decimal("10.0")
        )
        assert params.curve_type == BondingCurveType.LINEAR
        assert params.initial_price == Decimal("1.0")
        assert params.slope == Decimal("0.5")
        assert params.exponential == Decimal("0.0")
        assert params.steps is None
        assert params.max_price == Decimal("10.0")

    def test_exponential_valid_params(self):
        """Test valid exponential BondingCurveParams."""
        params = BondingCurveParams(
            curve_type=BondingCurveType.EXPONENTIAL,
            initial_price=Decimal("2.0"),
            exponential=Decimal("1.1"),
            max_price=Decimal("50.0")
        )
        assert params.curve_type == BondingCurveType.EXPONENTIAL
        assert params.initial_price == Decimal("2.0")
        assert params.exponential == Decimal("1.1")
        assert params.slope == Decimal("0.0")
        assert params.steps is None
        assert params.max_price == Decimal("50.0")

    def test_stepwise_valid_params(self):
        """Test valid stepwise BondingCurveParams."""
        steps = [
            StepConfig(supply_threshold=Decimal("1000"), price=Decimal("2.0")),
            StepConfig(supply_threshold=Decimal("2000"), price=Decimal("3.0")),
        ]
        params = BondingCurveParams(
            curve_type=BondingCurveType.STEPWISE,
            initial_price=Decimal("1.0"),
            steps=steps
        )
        assert params.curve_type == BondingCurveType.STEPWISE
        assert params.steps == steps
        assert params.initial_price == Decimal("1.0")

    @pytest.mark.parametrize("curve_type", [BondingCurveType.LINEAR, BondingCurveType.EXPONENTIAL])
    def test_initial_price_negative_raises(self, curve_type):
        """Test that negative initial_price raises ValueError."""
        with pytest.raises(ValueError) as exc:
            BondingCurveParams(curve_type=curve_type, initial_price=Decimal("-1.0"))
        assert "Initial price must be non-negative." in str(exc.value)

    def test_invalid_curve_type_raises(self):
        """Test that curve_type=BondingCurveType.CUSTOM raises ValueError."""
        with pytest.raises(ValueError) as exc:
            BondingCurveParams(curve_type=BondingCurveType.CUSTOM)
        assert "Invalid curve type" in str(exc.value)

    def test_linear_negative_slope_raises(self):
        """Test that a negative slope for LINEAR raises ValueError."""
        with pytest.raises(ValueError) as exc:
            BondingCurveParams(
                curve_type=BondingCurveType.LINEAR,
                slope=Decimal("-0.1")
            )
        assert "Slope must be non-negative." in str(exc.value)

    def test_exponential_negative_raises(self):
        """Test that negative exponential for EXPONENTIAL raises ValueError."""
        with pytest.raises(ValueError) as exc:
            BondingCurveParams(
                curve_type=BondingCurveType.EXPONENTIAL,
                exponential=Decimal("-0.5")
            )
        assert "Exponential must be non-negative." in str(exc.value)

    def test_stepwise_no_steps_raises(self):
        """Test that STEPWISE curve with no steps raises ValueError."""
        with pytest.raises(ValueError) as exc:
            BondingCurveParams(curve_type=BondingCurveType.STEPWISE, steps=[])
        assert "Stepwise curve requires steps." in str(exc.value)

    def test_stepwise_single_step_raises(self):
        """Test that STEPWISE curve with only one step raises ValueError."""
        with pytest.raises(ValueError) as exc:
            BondingCurveParams(
                curve_type=BondingCurveType.STEPWISE,
                steps=[StepConfig(supply_threshold=Decimal("1000"), price=Decimal("2.0"))]
            )
        assert "Stepwise curve requires at least two steps." in str(exc.value)

    def test_stepwise_non_increasing_raises(self):
        """Test that STEPWISE curve with non-increasing thresholds raises ValueError."""
        steps = [
            StepConfig(supply_threshold=Decimal("2000"), price=Decimal("2.0")),
            StepConfig(supply_threshold=Decimal("1000"), price=Decimal("3.0")),
        ]
        with pytest.raises(ValueError) as exc:
            BondingCurveParams(curve_type=BondingCurveType.STEPWISE, steps=steps)
        assert "Step 0 threshold must be less than step 1 threshold." in str(exc.value)


class TestBondingCurveState:
    def test_bonding_curve_state_defaults(self):
        """Test BondingCurveState defaults."""
        state = BondingCurveState()
        assert state.current_supply == Decimal("0.0")
        assert state.current_price == Decimal("0.0")
        assert state.liquidity is None
        assert state.last_timestamp is None

    def test_bonding_curve_state_custom(self):
        """Test BondingCurveState with custom values."""
        liquidity = Liquidity(Token("Test", "TST", 18), amount=Decimal("1000"))
        now = datetime.now()
        state = BondingCurveState(
            current_supply=Decimal("500.0"),
            current_price=Decimal("2.5"),
            liquidity=liquidity,
            last_timestamp=now
        )
        assert state.current_supply == Decimal("500.0")
        assert state.current_price == Decimal("2.5")
        assert state.liquidity == liquidity
        assert state.last_timestamp == now


class TestTransactionRequest:
    def test_transaction_request_defaults(self):
        """Test TransactionRequest defaults."""
        token = Token(name="Test Token", symbol="TT", decimals=18)
        request = TransactionRequest(token=token, order_type=OrderSide.BUY)
        assert request.token == token
        assert request.order_type == OrderSide.BUY
        assert request.amount == Decimal("0.0")
        assert request.user_id is None

    def test_transaction_request_custom(self):
        """Test TransactionRequest with custom values."""
        token = Token(name="Test Token", symbol="TT", decimals=18)
        request = TransactionRequest(
            token=token,
            order_type=OrderSide.SELL,
            amount=Decimal("10.0"),
            user_id="user123"
        )
        assert request.order_type == OrderSide.SELL
        assert request.amount == Decimal("10.0")
        assert request.user_id == "user123"


class TestTransactionResult:
    def test_transaction_result(self):
        """Test TransactionResult creation."""
        now = datetime.now()
        result = TransactionResult(
            executed_amount=Decimal("10"),
            total_cost=Decimal("25"),
            average_price=Decimal("2.5"),
            new_supply=Decimal("1010"),
            timestamp=now
        )
        assert result.executed_amount == Decimal("10")
        assert result.total_cost == Decimal("25")
        assert result.average_price == Decimal("2.5")
        assert result.new_supply == Decimal("1010")
        assert result.timestamp == now


class TestSimulationResult:
    def test_simulation_result_defaults(self):
        """Test SimulationResult defaults."""
        sim_result = SimulationResult()
        assert sim_result.transactions == []
        assert sim_result.final_supply == Decimal("0.0")
        assert sim_result.final_price == Decimal("0.0")
        assert sim_result.average_purchase_price == Decimal("0.0")
        assert sim_result.average_sale_price == Decimal("0.0")
        assert sim_result.metadata == {}

    def test_simulation_result_with_transactions(self):
        """Test SimulationResult with transaction results."""
        now = datetime.now()
        tx1 = TransactionResult(
            executed_amount=Decimal("5"),
            total_cost=Decimal("10"),
            average_price=Decimal("2"),
            new_supply=Decimal("1005"),
            timestamp=now
        )
        tx2 = TransactionResult(
            executed_amount=Decimal("10"),
            total_cost=Decimal("30"),
            average_price=Decimal("3"),
            new_supply=Decimal("1015"),
            timestamp=now
        )
        sim_result = SimulationResult(
            transactions=[tx1, tx2],
            final_supply=Decimal("1015"),
            final_price=Decimal("3.0"),
            average_purchase_price=Decimal("2.5"),
            average_sale_price=Decimal("2.75"),
            metadata={"info": "test"}
        )
        assert sim_result.transactions == [tx1, tx2]
        assert sim_result.final_supply == Decimal("1015")
        assert sim_result.final_price == Decimal("3.0")
        assert sim_result.average_purchase_price == Decimal("2.5")
        assert sim_result.average_sale_price == Decimal("2.75")
        assert sim_result.metadata["info"] == "test"
