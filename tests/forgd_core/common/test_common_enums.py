import pytest
from forgd_core.common.enums import (
    ChainType,
    BondingCurveType,
    OrderSide,
    BondingCurveDistribution,
)


class TestChainType:
    @pytest.mark.parametrize(
        "input_str, expected_enum",
        [
            ("SOLANA", ChainType.SOLANA),
            ("solana", ChainType.SOLANA),
            ("EVM", ChainType.EVM),
            ("evm", ChainType.EVM),
        ],
    )
    def test_from_str_valid(self, input_str, expected_enum):
        """Test that from_str returns the correct ChainType for valid inputs."""
        assert ChainType.from_str(input_str) == expected_enum

    @pytest.mark.parametrize("input_str", ["", "UNKNOWN", "solanaa", "evmm"])
    def test_from_str_invalid(self, input_str):
        """Test that from_str raises NotImplementedError for invalid inputs."""
        with pytest.raises(NotImplementedError):
            ChainType.from_str(input_str)

    def test_str(self):
        """Test that __str__ returns the enum name."""
        assert str(ChainType.EVM) == "EVM"
        assert str(ChainType.SOLANA) == "SOLANA"

    def test_repr(self):
        """Test that __repr__ returns the enum name (same as __str__)."""
        assert repr(ChainType.EVM) == "EVM"
        assert repr(ChainType.SOLANA) == "SOLANA"


class TestBondingCurveType:
    @pytest.mark.parametrize(
        "input_str, expected_enum",
        [
            ("LINEAR", BondingCurveType.LINEAR),
            ("linear", BondingCurveType.LINEAR),
            ("EXPONENTIAL", BondingCurveType.EXPONENTIAL),
            ("exponential", BondingCurveType.EXPONENTIAL),
            ("STEPWISE", BondingCurveType.STEPWISE),
            ("stepwise", BondingCurveType.STEPWISE),
            ("CUSTOM", BondingCurveType.CUSTOM),
            ("custom", BondingCurveType.CUSTOM),
        ],
    )
    def test_from_str_valid(self, input_str, expected_enum):
        """Test that from_str returns the correct BondingCurveType for valid inputs."""
        assert BondingCurveType.from_str(input_str) == expected_enum

    @pytest.mark.parametrize("input_str", ["", "lin", "exponentialx", "unknown"])
    def test_from_str_invalid(self, input_str):
        """Test that from_str raises NotImplementedError for invalid inputs."""
        with pytest.raises(NotImplementedError):
            BondingCurveType.from_str(input_str)

    def test_hashable(self):
        """Test that BondingCurveType is hashable and can be used in sets/dicts."""
        test_set = {BondingCurveType.LINEAR, BondingCurveType.EXPONENTIAL}
        assert BondingCurveType.LINEAR in test_set
        assert BondingCurveType.EXPONENTIAL in test_set

        test_dict = {BondingCurveType.STEPWISE: "stepwise_value"}
        assert test_dict[BondingCurveType.STEPWISE] == "stepwise_value"

    def test_str(self):
        """Test that __str__ returns the enum name."""
        assert str(BondingCurveType.LINEAR) == "LINEAR"
        assert str(BondingCurveType.CUSTOM) == "CUSTOM"

    def test_repr(self):
        """Test that __repr__ returns the enum name (same as __str__)."""
        assert repr(BondingCurveType.LINEAR) == "LINEAR"
        assert repr(BondingCurveType.CUSTOM) == "CUSTOM"


class TestOrderSide:
    @pytest.mark.parametrize(
        "input_str, expected_enum",
        [
            ("BUY", OrderSide.BUY),
            ("buy", OrderSide.BUY),
            ("SELL", OrderSide.SELL),
            ("sell", OrderSide.SELL),
        ],
    )
    def test_from_str_valid(self, input_str, expected_enum):
        """Test that from_str returns the correct OrderSide for valid inputs."""
        assert OrderSide.from_str(input_str) == expected_enum

    @pytest.mark.parametrize("input_str", ["", "buu", "selling", "invalid"])
    def test_from_str_invalid(self, input_str):
        """Test that from_str raises NotImplementedError for invalid inputs."""
        with pytest.raises(NotImplementedError):
            OrderSide.from_str(input_str)

    def test_str(self):
        """Test that __str__ returns the enum name."""
        assert str(OrderSide.BUY) == "BUY"
        assert str(OrderSide.SELL) == "SELL"

    def test_repr(self):
        """Test that __repr__ returns the enum name (same as __str__)."""
        assert repr(OrderSide.BUY) == "BUY"
        assert repr(OrderSide.SELL) == "SELL"


class TestBondingCurveDistribution:
    @pytest.mark.parametrize(
        "input_str, expected_enum",
        [
            ("CONSERVATIVE", BondingCurveDistribution.CONSERVATIVE),
            ("conservative", BondingCurveDistribution.CONSERVATIVE),
            ("MODERATE", BondingCurveDistribution.MODERATE),
            ("moderate", BondingCurveDistribution.MODERATE),
            ("AGGRESSIVE", BondingCurveDistribution.AGGRESSIVE),
            ("aggressive", BondingCurveDistribution.AGGRESSIVE),
        ],
    )
    def test_from_str_valid(self, input_str, expected_enum):
        """Test that from_str returns the correct BondingCurveDistribution for valid inputs."""
        assert BondingCurveDistribution.from_str(input_str) == expected_enum

    @pytest.mark.parametrize("input_str", ["", "conservativ", "moderate_", "aggressivex", "unknown"])
    def test_from_str_invalid(self, input_str):
        """Test that from_str raises NotImplementedError for invalid inputs."""
        with pytest.raises(NotImplementedError):
            BondingCurveDistribution.from_str(input_str)

    def test_str(self):
        """Test that __str__ returns the enum name."""
        assert str(BondingCurveDistribution.CONSERVATIVE) == "CONSERVATIVE"
        assert str(BondingCurveDistribution.MODERATE) == "MODERATE"
        assert str(BondingCurveDistribution.AGGRESSIVE) == "AGGRESSIVE"

    def test_repr(self):
        """Test that __repr__ returns the enum name (same as __str__)."""
        assert repr(BondingCurveDistribution.CONSERVATIVE) == "CONSERVATIVE"
        assert repr(BondingCurveDistribution.MODERATE) == "MODERATE"
        assert repr(BondingCurveDistribution.AGGRESSIVE) == "AGGRESSIVE"
