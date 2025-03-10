from enum import Enum


class ChainType(Enum):
    EVM = "EVM"
    SOLANA = "SOLANA"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, side_str: str) -> "ChainType":
        """
        Convert a string to a ChainType enum.
        :param side_str: str
        :return: ChainType or NotImplementedError
        """
        if side_str.upper() == ChainType.SOLANA.name:
            return ChainType.SOLANA
        elif side_str.upper() == ChainType.EVM.name:
            return ChainType.EVM
        else:
            raise NotImplementedError(f"No order side enum for {side_str}")

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class BondingCurveType(Enum):
    LINEAR = "LINEAR"
    EXPONENTIAL = "EXPONENTIAL"
    STEPWISE = "STEPWISE"
    CUSTOM = "CUSTOM"

    @classmethod
    def from_str(cls, side_str):
        if side_str.upper() == BondingCurveType.LINEAR.name:
            return BondingCurveType.LINEAR
        elif side_str.upper() == BondingCurveType.EXPONENTIAL.name:
            return BondingCurveType.EXPONENTIAL
        elif side_str.upper() == BondingCurveType.STEPWISE.name:
            return BondingCurveType.STEPWISE
        elif side_str.upper() == BondingCurveType.CUSTOM.name:
            return BondingCurveType.CUSTOM
        else:
            raise NotImplementedError(f"No order side enum for {side_str}")

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

    @classmethod
    def from_str(cls, side_str):
        if side_str.upper() == OrderSide.BUY.name:
            return OrderSide.BUY
        elif side_str.upper() == OrderSide.SELL.name:
            return OrderSide.SELL
        else:
            raise NotImplementedError(f"No order side enum for {side_str}")

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class BondingCurveDistribution(Enum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


    @classmethod
    def from_str(cls, dist_str):
        if dist_str.upper() == BondingCurveDistribution.CONSERVATIVE.name:
            return BondingCurveDistribution.CONSERVATIVE
        elif dist_str.upper() == BondingCurveDistribution.MODERATE.name:
            return BondingCurveDistribution.MODERATE
        elif dist_str.upper() == BondingCurveDistribution.AGGRESSIVE.name:
            return BondingCurveDistribution.AGGRESSIVE
        else:
            raise NotImplementedError(f"No Bonding Curve Distribution enum for {dist_str}")

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()
