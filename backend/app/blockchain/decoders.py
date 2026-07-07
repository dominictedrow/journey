"""Decode CTF Exchange / NegRisk Exchange `OrderFilled` logs.

Both exchange generations emit the same event. One event is emitted per filled order
(maker side), so indexing by the `maker` topic captures every wallet's fills exactly once.

    OrderFilled(bytes32 indexed orderHash, address indexed maker, address indexed taker,
                uint256 makerAssetId, uint256 takerAssetId,
                uint256 makerAmountFilled, uint256 takerAmountFilled, uint256 fee)

Asset id 0 is the collateral token (USDC/pUSD); a nonzero id is a CTF outcome token.
"""

from dataclasses import dataclass
from typing import Any

from web3 import Web3

ORDER_FILLED_TOPIC: str = Web3.keccak(
    text="OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)"
).to_0x_hex()


def _bytes(value: Any) -> bytes:
    return bytes(value) if not isinstance(value, str) else bytes.fromhex(value.removeprefix("0x"))


def address_topic(address: str) -> str:
    """Left-pad an address to a 32-byte log topic (for maker filters)."""
    return "0x" + "00" * 12 + address.lower().removeprefix("0x")


@dataclass(frozen=True)
class Fill:
    tx_hash: str
    log_index: int
    block_number: int
    maker: str
    taker: str
    maker_asset_id: int
    taker_asset_id: int
    maker_amount: int
    taker_amount: int
    fee: int


def decode_order_filled(log: Any) -> Fill | None:
    """Decode one raw log; returns None if it is not an OrderFilled event."""
    topics = log["topics"]
    if len(topics) != 4 or _bytes(topics[0]) != _bytes(ORDER_FILLED_TOPIC):
        return None
    data = _bytes(log["data"])
    words = [int.from_bytes(data[i : i + 32]) for i in range(0, 160, 32)]
    return Fill(
        tx_hash="0x" + _bytes(log["transactionHash"]).hex(),
        log_index=int(log["logIndex"]),
        block_number=int(log["blockNumber"]),
        maker="0x" + _bytes(topics[2])[-20:].hex(),
        taker="0x" + _bytes(topics[3])[-20:].hex(),
        maker_asset_id=words[0],
        taker_asset_id=words[1],
        maker_amount=words[2],
        taker_amount=words[3],
        fee=words[4],
    )
