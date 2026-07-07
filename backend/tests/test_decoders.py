from decimal import Decimal
from typing import Any

from hexbytes import HexBytes

from app.blockchain.decoders import ORDER_FILLED_TOPIC, address_topic, decode_order_filled
from app.blockchain.normalize import fill_to_trade, utc_from_unix

MAKER = "0x" + "aa" * 20
TAKER = "0x" + "bb" * 20
TOKEN_ID = 12345678901234567890


def make_log(
    maker_asset_id: int,
    taker_asset_id: int,
    maker_amount: int,
    taker_amount: int,
    log_index: int = 3,
) -> dict[str, Any]:
    data = b"".join(
        w.to_bytes(32) for w in (maker_asset_id, taker_asset_id, maker_amount, taker_amount, 0)
    )
    return {
        "topics": [
            HexBytes(ORDER_FILLED_TOPIC),
            HexBytes(b"\x11" * 32),
            HexBytes(address_topic(MAKER)),
            HexBytes(address_topic(TAKER)),
        ],
        "data": HexBytes(data),
        "transactionHash": HexBytes("0x" + "cd" * 32),
        "logIndex": log_index,
        "blockNumber": 45_000_000,
    }


def test_decode_buy_fill() -> None:
    # maker pays 50 USDC (asset 0) for 100 outcome tokens -> BUY at 0.50
    fill = decode_order_filled(make_log(0, TOKEN_ID, 50_000_000, 100_000_000))
    assert fill is not None
    assert fill.maker == MAKER
    assert fill.taker == TAKER
    trade = fill_to_trade(fill, utc_from_unix(1_700_000_000), {})
    assert trade["side"] == "BUY"
    assert trade["token_id"] == str(TOKEN_ID)
    assert trade["price"] == Decimal("0.5")
    assert trade["size"] == Decimal("100")
    assert trade["usdc_amount"] == Decimal("50")
    assert trade["wallet_address"] == MAKER


def test_decode_sell_fill() -> None:
    # maker sells 200 outcome tokens for 150 USDC -> SELL at 0.75
    fill = decode_order_filled(make_log(TOKEN_ID, 0, 200_000_000, 150_000_000))
    assert fill is not None
    trade = fill_to_trade(fill, utc_from_unix(1_700_000_000), {str(TOKEN_ID): "0xcond"})
    assert trade["side"] == "SELL"
    assert trade["price"] == Decimal("0.75")
    assert trade["size"] == Decimal("200")
    assert trade["market_id"] == "0xcond"


def test_non_order_filled_log_is_skipped() -> None:
    log = make_log(0, TOKEN_ID, 1, 1)
    log["topics"][0] = HexBytes(b"\x00" * 32)
    assert decode_order_filled(log) is None
