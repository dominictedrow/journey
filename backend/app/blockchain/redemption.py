"""Module E — on-chain redemption of settled live positions.

Two contracts redeem CTF positions on Polymarket depending on the market type:

  * standard binary markets — the Conditional Tokens contract's `redeemPositions(collateralToken,
    parentCollectionId, conditionId, indexSets)`. No amount is passed: it redeems the caller's
    *entire* balance in each index set automatically (verified against Gnosis's canonical
    `ConditionalTokens.sol`, which Polymarket's CTF deployment is, on 2026-07-08).
  * negative-risk (multi-outcome) markets route through the Neg Risk Adapter instead —
    `redeemPositions(bytes32 conditionId, uint256[] amounts)` — verified against its on-chain
    source (`Polymarket/neg-risk-ctf-adapter`) on 2026-07-08, since the published docs page
    disagreed with the deployed contract's actual parameter types.

Only the EOA signature scheme (`clob_signature_type == 0`, the default) is supported: the
signing key both holds the CTF token balance and pays gas, so a direct signed call redeems it.
Proxy schemes (1 = email/magic, 2 = Gnosis Safe) hold the balance in a *different* address than
the signing key — redeeming there needs a Safe/proxy meta-transaction, which this module does
not build. Those positions are flagged (`redemption_status="failed"` + a `RiskEvent`) for manual
redemption instead of silently no-op'ing or attempting a call that would redeem nothing.

On-chain balance is the source of truth for whether there is anything left to redeem, not our
internal books: a position closed by a normal CLOB sell already transferred its CTF tokens away,
so its balance is already zero by the time this sweep looks at it, and that's treated as success
with no transaction sent — idempotent by construction, not by a status flag alone.

Retry/failure handling mirrors `LiveExecutor` (`copy_trading/executors/live.py`): bounded
retries with linear backoff, a terminal failure is logged (not raised) so a stuck redemption
never crashes the settlement beat and is retried on the next sweep.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from eth_account import Account
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from web3 import Web3

from app.blockchain.client import RpcPool
from app.config import Settings, get_settings
from app.database.models import Market, PortfolioPosition, RiskEvent

logger = logging.getLogger(__name__)

ZERO_BYTES32 = "0x" + "00" * 32

_ERC1155_BALANCE_ABI: list[dict[str, Any]] = [
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}, {"name": "id", "type": "uint256"}],
        "outputs": [{"name": "", "type": "uint256"}],
    }
]

_CTF_REDEEM_ABI: list[dict[str, Any]] = [
    {
        "name": "redeemPositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "outputs": [],
    }
]

_NEG_RISK_REDEEM_ABI: list[dict[str, Any]] = [
    {
        "name": "redeemPositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "_conditionId", "type": "bytes32"},
            {"name": "_amounts", "type": "uint256[]"},
        ],
        "outputs": [],
    }
]

# ABI encoding is pure/offline — no provider needed, just used as a codec.
_codec = Web3()


def _owner_address(settings: Settings) -> str:
    return Account.from_key(settings.wallet_private_key).address


async def _balance_of(pool: RpcPool, contract: str, owner: str, token_id: str) -> int:
    data = _codec.eth.contract(
        address=Web3.to_checksum_address(contract), abi=_ERC1155_BALANCE_ABI
    ).encode_abi("balanceOf", args=[owner, int(token_id)])
    raw = await pool.call(to=contract, data=data)
    return int.from_bytes(raw, "big")


async def _submit(pool: RpcPool, settings: Settings, *, to: str, data: str) -> str:
    """Sign, broadcast, and wait for confirmation — bounded retries with linear backoff on
    submission failures, mirroring `LiveExecutor`'s retry shape. Raises on final failure; the
    caller marks the position failed rather than letting this crash the beat."""
    backoff = float(settings.redemption_retry_backoff_seconds)
    last_error: Exception | None = None
    for attempt in range(1, settings.redemption_max_retries + 1):
        try:
            tx_hash = await pool.send_signed_transaction(
                to=to,
                data=data,
                private_key=settings.wallet_private_key,
                chain_id=settings.polygon_chain_id,
                gas_limit=settings.redemption_gas_limit,
            )
            receipt = await pool.wait_for_receipt(
                tx_hash, timeout=settings.redemption_receipt_timeout_seconds
            )
            if not receipt.get("status"):
                raise RuntimeError(f"transaction {tx_hash} reverted")
            return tx_hash
        except Exception as exc:  # noqa: BLE001 — RPC/network errors are retryable
            last_error = exc
            if attempt < settings.redemption_max_retries:
                await asyncio.sleep(backoff * attempt)
    assert last_error is not None
    raise last_error


def _mark(
    positions: list[PortfolioPosition], status: str, tx_hash: str | None, now: datetime
) -> None:
    for position in positions:
        position.redemption_status = status
        position.redemption_tx_hash = tx_hash
        if status == "redeemed":
            position.redeemed_at = now


def _event(now: datetime, market_id: str, note: str, tx_hash: str | None) -> RiskEvent:
    return RiskEvent(
        timestamp=now,
        type="redemption",
        severity="info" if tx_hash else "warning",
        context={"market_id": market_id, "note": note, "tx_hash": tx_hash},
        action_taken="redeemed" if tx_hash else "flagged",
    )


async def redeem_market(
    session: AsyncSession,
    pool: RpcPool,
    market: Market,
    positions: list[PortfolioPosition],
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> None:
    """Redeem every held outcome token of one resolved market on-chain, then mark every
    `positions` row (every internal row tied to that market) redeemed or failed."""
    settings = settings or get_settings()
    now = now or datetime.now(UTC)

    if settings.clob_signature_type != 0:
        _mark(positions, "failed", None, now)
        note = "manual redemption required (non-EOA signature scheme)"
        session.add(_event(now, market.id, note, None))
        return

    token_ids = [str(t) for t in (market.token_ids or [])]
    if not token_ids:
        _mark(positions, "failed", None, now)
        session.add(_event(now, market.id, "market has no token_ids on record", None))
        return

    owner = _owner_address(settings)
    ctf = settings.conditional_tokens_address
    try:
        balances = [await _balance_of(pool, ctf, owner, tid) for tid in token_ids]
    except Exception:
        logger.exception("redemption balance check failed for market %s", market.id)
        _mark(positions, "failed", None, now)
        session.add(_event(now, market.id, "balance check failed", None))
        return

    if not any(balances):
        # Nothing left on-chain — already redeemed, or exited via a normal CLOB sell.
        _mark(positions, "redeemed", None, now)
        return

    try:
        if market.neg_risk:
            adapter = settings.neg_risk_adapter_address
            data = _codec.eth.contract(
                address=Web3.to_checksum_address(adapter), abi=_NEG_RISK_REDEEM_ABI
            ).encode_abi("redeemPositions", args=[market.id, balances])
            tx_hash = await _submit(pool, settings, to=adapter, data=data)
        else:
            index_sets = [1 << i for i in range(len(token_ids))]
            data = _codec.eth.contract(
                address=Web3.to_checksum_address(ctf), abi=_CTF_REDEEM_ABI
            ).encode_abi(
                "redeemPositions",
                args=[settings.usdc_address, ZERO_BYTES32, market.id, index_sets],
            )
            tx_hash = await _submit(pool, settings, to=ctf, data=data)
    except Exception:
        logger.exception("redemption transaction failed for market %s", market.id)
        _mark(positions, "failed", None, now)
        session.add(_event(now, market.id, "redemption transaction failed", None))
        return

    _mark(positions, "redeemed", tx_hash, now)
    session.add(_event(now, market.id, "redeemed", tx_hash))


async def redeem_settled_positions(
    session: AsyncSession, *, settings: Settings | None = None, now: datetime | None = None
) -> list[str]:
    """Redeem every resolved live-mode position not yet redeemed; returns the market ids
    processed. Commits per market so one failure doesn't roll back another market's success."""
    settings = settings or get_settings()
    if settings.execution_mode != "live":
        return []
    now = now or datetime.now(UTC)

    rows = (
        await session.execute(
            select(PortfolioPosition, Market)
            .join(Market, Market.id == PortfolioPosition.market_id)
            .where(
                PortfolioPosition.mode == "live",
                PortfolioPosition.is_open.is_(False),
                Market.closed.is_(True),
                or_(
                    PortfolioPosition.redemption_status.is_(None),
                    PortfolioPosition.redemption_status != "redeemed",
                ),
            )
        )
    ).all()

    by_market: dict[str, tuple[Market, list[PortfolioPosition]]] = {}
    for position, market in rows:
        by_market.setdefault(market.id, (market, []))[1].append(position)

    processed: list[str] = []
    async with RpcPool(settings.polygon_rpc_url_list) as pool:
        for market_id, (market, positions) in by_market.items():
            try:
                await redeem_market(session, pool, market, positions, settings=settings, now=now)
                await session.commit()
                processed.append(market_id)
            except Exception:  # noqa: BLE001 — one market's failure must not sink the sweep
                logger.exception("redemption sweep failed for market %s", market_id)
                await session.rollback()
    return processed
