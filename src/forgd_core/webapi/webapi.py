from enum import Enum
from typing import Dict
from pydantic import BaseModel, Field

from flask import jsonify
from flask_openapi3 import Info, Tag
from flask_openapi3 import OpenAPI

from decimal import Decimal

from forgd_core.curves.single.linear import LinearBondingCurve
from forgd_core.common.model import (
    BondingCurveParams,
    BondingCurveState,
    Token,
    TransactionRequest,
)
from forgd_core.common.enums import BondingCurveType, OrderSide


info = Info(title="Bonding Curve API", version="1.0.0")
app = OpenAPI(__name__, info=info)


class CurveTransactionAction(Enum):
    buy = "buy"
    sell = "sell"


class CurveType(Enum):
    linear = "linear"


class CurveTransactionRequest(BaseModel):
    curve_type: CurveType = Field(description="The bonding curve type to use")
    curve_params: Dict = Field(None)
    allocated_supply: Decimal = Field(description="Already allocated token supply")
    action: CurveTransactionAction = Field(description="API action to perform")
    amount: Decimal = Field(description="amount to buy / sell")


curve_action_tag = Tag(
    name="Bonding Curve Transaction",
    description="Perform a buy or sell transaction on a curve and get the execution information",
)


@app.post("/curve/transaction", summary="Curve Transaction", tags=[curve_action_tag])
def transaction(query: CurveTransactionRequest):
    """
    Handles a buy or sell operation on a curve
    """
    if query.curve_type != CurveType.linear:
        return jsonify("err")

    params = BondingCurveParams(
        curve_type=BondingCurveType.from_str(query.curve_type.value)
    )
    state = BondingCurveState(current_supply=Decimal(query.allocated_supply))

    curve = LinearBondingCurve(params, state)
    dummy_token = Token("dummy", "dummy", 8)

    side = OrderSide.from_str(query.action.name)

    req_inner = TransactionRequest(dummy_token, side, query.amount)

    result = curve.buy(req_inner)
    return jsonify(result)


class CurveStatusRequest(BaseModel):
    curve_type: CurveType = Field(description="The bonding curve type to use")
    curve_params: Dict = Field(None)
    allocated_supply: Decimal = Field(description="Already allocated token supply")


curve_status_tag = Tag(
    name="Bonding Curve Status",
    description="Get the shape of a bonding curve for plotting and additional info",
)

@app.get("/curve/status", summary="Curve Status", tags=[curve_status_tag])
def status(query: CurveStatusRequest):
    """
    Return a representation of the curve which can be plotted visually by the caller.
    Return the 'midprice' of the curve based on the allocated amount specified by the caller.
    """
    return jsonify("TODO")


if __name__ == "__main__":
    app.run(debug=True)
