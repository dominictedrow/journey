"""Historical backfill: walk block ranges for a wallet's fills with a resumable checkpoint.

Checkpoints live in `indexer_state` (one stream per wallet backfill); each chunk commits
trades + checkpoint together, so a killed indexer resumes where it stopped and the
(tx_hash, log_index) unique constraint absorbs any overlap.
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from web3 import Web3

from app.blockchain.client import RpcPool
from app.blockchain.decoders import ORDER_FILLED_TOPIC, address_topic, decode_order_filled
from app.blockchain.normalize import (
    fill_to_trade,
    insert_trades,
    load_token_market_map,
    rebuild_positions,
    utc_from_unix,
)
from app.config import get_settings
from app.database.base import async_session_factory
from app.database.models import IndexerState


async def get_checkpoint(session: AsyncSession, stream: str) -> int | None:
    state = await session.scalar(select(IndexerState).where(IndexerState.stream == stream))
    return state.last_block if state else None


async def set_checkpoint(session: AsyncSession, stream: str, block: int) -> None:
    """Monotonic: never moves the cursor backwards (safe under overlapping runs)."""
    state = await session.scalar(select(IndexerState).where(IndexerState.stream == stream))
    if state is None:
        session.add(IndexerState(stream=stream, last_block=block, updated_at=datetime.now(UTC)))
    elif block > state.last_block:
        state.last_block = block
        state.updated_at = datetime.now(UTC)


async def fetch_trade_rows(
    pool: RpcPool,
    session: AsyncSession,
    from_block: int,
    to_block: int,
    maker_topics: list[str],
    token_map: dict[str, str],
) -> list[dict]:
    """eth_getLogs one chunk of OrderFilled events for the given makers -> trades rows."""
    settings = get_settings()
    logs = await pool.get_logs(
        {
            "fromBlock": from_block,
            "toBlock": to_block,
            "address": [Web3.to_checksum_address(a) for a in settings.exchange_address_list],
            "topics": [ORDER_FILLED_TOPIC, None, maker_topics],
        }
    )
    fills = [f for f in (decode_order_filled(log) for log in logs) if f is not None]
    timestamps: dict[int, datetime] = {}
    for number in {f.block_number for f in fills}:
        block = await pool.get_block(number)
        timestamps[number] = utc_from_unix(block["timestamp"])
    return [fill_to_trade(f, timestamps[f.block_number], token_map) for f in fills]


async def backfill_wallet(address: str, from_block: int | None = None) -> int:
    """Backfill one wallet's full fill history, then rebuild its positions."""
    settings = get_settings()
    address = address.lower()
    stream = f"backfill:{address}"
    inserted = 0
    async with RpcPool(settings.polygon_rpc_url_list) as pool:
        head = await pool.get_block_number() - settings.chain_confirmations
        async with async_session_factory() as session:
            checkpoint = await get_checkpoint(session, stream)
            if from_block is not None:
                start = from_block
            elif checkpoint is not None:
                start = checkpoint + 1
            else:
                start = settings.backfill_start_block
            token_map = await load_token_market_map(session)
            for lo in range(start, head + 1, settings.rpc_log_chunk_size):
                hi = min(lo + settings.rpc_log_chunk_size - 1, head)
                rows = await fetch_trade_rows(
                    pool, session, lo, hi, [address_topic(address)], token_map
                )
                inserted += await insert_trades(session, rows)
                await set_checkpoint(session, stream, hi)
                await session.commit()
            await rebuild_positions(session, address)
            await session.commit()
    return inserted
