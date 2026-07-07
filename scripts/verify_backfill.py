"""Cross-check a backfilled wallet's on-chain positions against the Data-API.

Usage (inside the worker/api container, after `chain.backfill_wallet` ran):
    python /app/../scripts/verify_backfill.py 0xWALLET [tolerance]

Prints per-token size deltas and exits nonzero if any exceed the tolerance (default 1.0).
"""

import asyncio
import sys
from decimal import Decimal

from sqlalchemy import select

from app.database.base import async_session_factory, engine
from app.database.models import Position
from app.ingest.data_api import get_positions


async def main(address: str, tolerance: Decimal) -> int:
    address = address.lower()
    async with async_session_factory() as session:
        ours = {
            p.token_id: p.net_size
            for p in await session.scalars(
                select(Position).where(Position.wallet_address == address, Position.is_open)
            )
        }
    theirs = {
        str(p["asset"]): Decimal(str(p.get("size", 0)))
        for p in await get_positions(address)
        if Decimal(str(p.get("size", 0))) != 0
    }
    await engine.dispose()

    failures = 0
    for token in sorted(ours.keys() | theirs.keys()):
        delta = abs(ours.get(token, Decimal(0)) - theirs.get(token, Decimal(0)))
        flag = "" if delta <= tolerance else "  <-- MISMATCH"
        if flag:
            failures += 1
        print(f"{token}: ours={ours.get(token, 0)} data-api={theirs.get(token, 0)}{flag}")
    print(f"\n{len(ours)} open on-chain, {len(theirs)} per Data-API, {failures} beyond tolerance")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tol = Decimal(sys.argv[2]) if len(sys.argv) > 2 else Decimal("1.0")
    sys.exit(asyncio.run(main(sys.argv[1], tol)))
