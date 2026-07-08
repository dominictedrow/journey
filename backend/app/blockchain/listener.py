"""Live tail: poll new blocks for tracked wallets' fills, from the shared "live" checkpoint.

Runs as a short periodic Celery task (beat) rather than a long-lived WS subscription —
simpler crash recovery, and the checkpoint + dedup constraint make overlapping runs safe.
"""

import logging

from sqlalchemy import select

from app.blockchain.client import RpcPool
from app.blockchain.decoders import address_topic
from app.blockchain.indexer import fetch_trade_rows, get_checkpoint, set_checkpoint
from app.blockchain.normalize import insert_trades, load_token_market_map, rebuild_positions
from app.config import get_settings
from app.database.base import async_session_factory
from app.database.models import Wallet

LIVE_STREAM = "live"
logger = logging.getLogger(__name__)


async def poll_live_tail() -> int:
    """Process blocks (checkpoint, head-confirmations]; returns trades inserted."""
    settings = get_settings()
    if not settings.polygon_rpc_url_list:
        logger.warning("POLYGON_RPC_URLS not configured; live tail is idle")
        return 0
    async with RpcPool(settings.polygon_rpc_url_list) as pool:
        head = await pool.get_block_number() - settings.chain_confirmations
        inserted = 0
        async with async_session_factory() as session:
            last = await get_checkpoint(session, LIVE_STREAM)
            if last is None or head <= last:
                # First run starts at the current head; nothing historical to scan here.
                await set_checkpoint(session, LIVE_STREAM, head)
                await session.commit()
                return 0
            tracked = list(await session.scalars(select(Wallet.address).where(Wallet.is_tracked)))
            if not tracked:
                await set_checkpoint(session, LIVE_STREAM, head)
                await session.commit()
                return 0
            token_map = await load_token_market_map(session)
            maker_topics = [address_topic(a) for a in tracked]
            touched: set[str] = set()
            for lo in range(last + 1, head + 1, settings.rpc_log_chunk_size):
                hi = min(lo + settings.rpc_log_chunk_size - 1, head)
                rows = await fetch_trade_rows(pool, session, lo, hi, maker_topics, token_map)
                inserted += await insert_trades(session, rows)
                touched.update(r["wallet_address"] for r in rows)
                await set_checkpoint(session, LIVE_STREAM, hi)
                await session.commit()
            for address in touched:
                await rebuild_positions(session, address)
            await session.commit()
    return inserted
