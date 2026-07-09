"""On-chain collateral balance — the source of truth for live equity.

Live mode has no play-money starting cash: what the account can actually deploy is its real
USDC/pUSD collateral balance on Polygon. Reading it here lets the risk gate's %-of-equity caps
and percentage sizing scale to real capital with zero human input (the autonomy target), instead
of a static config figure that would be 0 by default (blocking every live BUY) and would drift
from reality as trades, settlements, and redemptions move collateral.

Fails open: any RPC/encoding error (or an unconfigured pool/key) returns `None`, and the caller
falls back to the `risk_live_starting_equity` config value. Advisory-safe — never raises.
"""

import logging
from decimal import Decimal
from typing import Any

from eth_account import Account
from web3 import Web3

from app.blockchain.client import RpcPool
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# USDC and the current pUSD collateral proxy on Polygon are both 6-decimal ERC-20s.
_COLLATERAL_SCALE = Decimal(10) ** 6

_ERC20_BALANCE_ABI: list[dict[str, Any]] = [
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    }
]

# ABI encoding is pure/offline — no provider needed, just used as a codec.
_codec = Web3()


def collateral_owner(settings: Settings) -> str | None:
    """The address that actually holds the account's collateral.

    EOA scheme (`clob_signature_type == 0`): the signing key's own address. Proxy schemes
    (1 = email/magic, 2 = Gnosis Safe): the funder proxy in `clob_funder_address`."""
    if settings.clob_signature_type != 0 and settings.clob_funder_address:
        return settings.clob_funder_address
    if settings.wallet_private_key:
        return Account.from_key(settings.wallet_private_key).address
    return settings.clob_funder_address or None


async def read_collateral_balance(settings: Settings | None = None) -> Decimal | None:
    """Current on-chain collateral balance of the live account, or `None` on any failure
    (unconfigured pool/key, RPC error) so the caller can fall back to config."""
    settings = settings or get_settings()
    owner = collateral_owner(settings)
    if not owner or not settings.polygon_rpc_url_list or not settings.usdc_address:
        return None
    try:
        data = _codec.eth.contract(
            address=Web3.to_checksum_address(settings.usdc_address), abi=_ERC20_BALANCE_ABI
        ).encode_abi("balanceOf", args=[Web3.to_checksum_address(owner)])
        async with RpcPool(settings.polygon_rpc_url_list) as pool:
            raw = await pool.call(to=settings.usdc_address, data=data)
        return Decimal(int.from_bytes(raw, "big")) / _COLLATERAL_SCALE
    except Exception:  # noqa: BLE001 — fail open; live equity falls back to config
        logger.warning("live collateral balance read failed; using config fallback", exc_info=True)
        return None
