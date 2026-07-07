"""Module 2: discovery expands the candidate pool idempotently and isolates source failures."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Market, Wallet
from app.wallet_discovery.discovery import run_discovery, upsert_candidates
from app.wallet_discovery.sources.base import Candidate
from app.wallet_discovery.sources.leaderboard import leaderboard_source
from app.wallet_discovery.sources.market_holders import market_holders_source

A1 = "0x" + "11" * 20
A2 = "0x" + "22" * 20
A3 = "0x" + "33" * 20


async def source_one(session: AsyncSession) -> list[Candidate]:
    return [Candidate(A1, "one"), Candidate(A2, "one")]


async def source_two(session: AsyncSession) -> list[Candidate]:
    return [Candidate(A2, "two"), Candidate(A3, "two")]


async def source_broken(session: AsyncSession) -> list[Candidate]:
    raise RuntimeError("api down")


async def test_run_discovery_upserts_and_attributes(session: AsyncSession) -> None:
    results = await run_discovery(session, sources=[source_one, source_two])
    assert results == {"source_one": 2, "source_two": 1}  # A2 already claimed by source_one
    wallets = {w.address: w for w in (await session.scalars(select(Wallet))).all()}
    assert wallets[A1].discovery_source == "one"
    assert wallets[A2].discovery_source == "one"
    assert wallets[A3].discovery_source == "two"
    assert all(w.discovered_at is not None for w in wallets.values())
    assert all(w.is_tracked is False for w in wallets.values())


async def test_run_discovery_is_idempotent(session: AsyncSession) -> None:
    await run_discovery(session, sources=[source_one])
    results = await run_discovery(session, sources=[source_one])
    assert results == {"source_one": 0}
    assert len((await session.scalars(select(Wallet))).all()) == 2


async def test_failing_source_does_not_block_others(session: AsyncSession) -> None:
    results = await run_discovery(session, sources=[source_broken, source_one])
    assert results == {"source_one": 2}  # broken source logged and skipped


async def test_existing_attribution_is_never_overwritten(session: AsyncSession) -> None:
    session.add(Wallet(address=A1, discovery_source="chain_index"))
    await session.commit()
    await upsert_candidates(session, [Candidate(A1, "leaderboard")], datetime.now(UTC))
    await session.commit()
    wallet = await session.get(Wallet, A1)
    assert wallet is not None and wallet.discovery_source == "chain_index"


async def test_leaderboard_source_maps_rows(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_leaderboard(order_by: str = "PNL", time_period: str = "MONTH", **_: object):
        return [{"proxyWallet": A1.upper(), "userName": "x"}, {"userName": "no-wallet"}]

    monkeypatch.setattr(
        "app.wallet_discovery.sources.leaderboard.get_leaderboard", fake_leaderboard
    )
    candidates = await leaderboard_source(session)
    assert len(candidates) == 4  # one wallet per sweep, wallet-less rows dropped
    assert candidates[0].address == A1  # lowercased
    assert candidates[0].source == "leaderboard:PNL:MONTH"


async def test_market_holders_source_scans_top_markets(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session.add_all(
        [
            Market(id="m1", question="q1", active=True, closed=False, volume=1000),
            Market(id="m2", question="q2", active=True, closed=False, volume=500),
            Market(id="m3", question="q3", active=False, closed=True, volume=9999),
        ]
    )
    await session.commit()
    seen: list[list[str]] = []

    async def fake_holders(condition_ids: list[str], limit: int = 20) -> list[dict]:
        seen.append(condition_ids)
        return [{"token": "t", "holders": [{"proxyWallet": A2}, {"proxyWallet": A3}]}]

    monkeypatch.setattr("app.wallet_discovery.sources.market_holders.get_holders", fake_holders)
    candidates = await market_holders_source(session)
    assert seen == [["m1", "m2"]]  # closed m3 excluded, volume-ordered
    assert {c.address for c in candidates} == {A2, A3}
    assert all(c.source == "market_holders" for c in candidates)
