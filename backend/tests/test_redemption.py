"""Module E — on-chain redemption of settled live positions, against a stub Web3 client (no
real RPC, mirrors how test_live_executor.py fakes the CLOB client)."""

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import app.blockchain.redemption as redemption_mod
from app.blockchain.redemption import redeem_market, redeem_settled_positions
from app.config import Settings
from app.database.models import Market, PortfolioPosition, RiskEvent

NOW = datetime(2026, 7, 8, 12, 0, tzinfo=UTC)
PRIVATE_KEY = "0x" + "11" * 32
COND_ID = "0x" + "ab" * 32  # condition_id is a real bytes32 hex on-chain, not a slug


def _settings(**over) -> Settings:
    base = dict(
        execution_mode="live",
        wallet_private_key=PRIVATE_KEY,
        redemption_retry_backoff_seconds=Decimal("0"),  # no real sleeps in tests
        redemption_max_retries=3,
    )
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


def _market(market_id: str = COND_ID, *, neg_risk: bool = False, closed: bool = True) -> Market:
    return Market(
        id=market_id,
        question="resolved?",
        token_ids=["111", "222"],  # CTF position ids are large integers, not slugs
        outcome_prices=["1", "0"],
        closed=closed,
        neg_risk=neg_risk,
    )


def _position(market_id: str = COND_ID, **kw) -> PortfolioPosition:
    base = dict(
        mode="live",
        market_id=market_id,
        token_id="111",
        size=Decimal("10"),
        is_open=False,
        closed_at=NOW,
    )
    base.update(kw)
    return PortfolioPosition(**base)


class FakePool:
    """Stand-in for RpcPool: scripted balances + send/receipt outcomes."""

    def __init__(self, balances: dict[str, int], send_outcomes: list[object] | None = None):
        self.balances = balances
        self.send_outcomes = list(send_outcomes or [])
        self.sent: list[dict] = []

    async def call(self, *, to: str, data: str) -> bytes:
        # Last 64 hex chars of the calldata encode the uint256 token id argument.
        token_id = str(int(data[-64:], 16))
        return self.balances.get(token_id, 0).to_bytes(32, "big")

    async def send_signed_transaction(self, *, to: str, data: str, **kw) -> str:
        self.sent.append({"to": to, "data": data})
        outcome = self.send_outcomes.pop(0) if self.send_outcomes else "0xhash"
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def wait_for_receipt(self, tx_hash: str, *, timeout: float) -> dict:
        return {"status": 1}

    async def aclose(self) -> None:
        pass

    async def __aenter__(self) -> "FakePool":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


async def test_zero_balance_marks_redeemed_without_transaction(session: AsyncSession) -> None:
    market = _market()
    positions = [_position()]
    pool = FakePool(balances={})  # nothing held on-chain

    await redeem_market(session, pool, market, positions, settings=_settings(), now=NOW)
    await session.commit()

    assert positions[0].redemption_status == "redeemed"
    assert positions[0].redemption_tx_hash is None
    assert pool.sent == []
    assert await session.scalar(select(RiskEvent)) is None


async def test_standard_ctf_redemption_success(session: AsyncSession) -> None:
    market = _market(neg_risk=False)
    positions = [_position()]
    pool = FakePool(balances={"111": 10_000_000, "222": 0})
    settings = _settings()

    await redeem_market(session, pool, market, positions, settings=settings, now=NOW)
    await session.commit()

    assert positions[0].redemption_status == "redeemed"
    assert positions[0].redemption_tx_hash == "0xhash"
    assert pool.sent[0]["to"] == settings.conditional_tokens_address
    event = await session.scalar(select(RiskEvent).where(RiskEvent.type == "redemption"))
    assert event is not None and event.action_taken == "redeemed"


async def test_neg_risk_market_routes_to_adapter(session: AsyncSession) -> None:
    market = _market(neg_risk=True)
    positions = [_position()]
    pool = FakePool(balances={"111": 10_000_000, "222": 0})
    settings = _settings()

    await redeem_market(session, pool, market, positions, settings=settings, now=NOW)

    assert positions[0].redemption_status == "redeemed"
    assert pool.sent[0]["to"] == settings.neg_risk_adapter_address


async def test_transient_submission_failure_then_retry_succeeds(session: AsyncSession) -> None:
    market = _market()
    positions = [_position()]
    pool = FakePool(
        balances={"111": 10_000_000, "222": 0},
        send_outcomes=[RuntimeError("nonce too low"), "0xretried"],
    )

    await redeem_market(session, pool, market, positions, settings=_settings(), now=NOW)

    assert positions[0].redemption_status == "redeemed"
    assert positions[0].redemption_tx_hash == "0xretried"
    assert len(pool.sent) == 2


async def test_gives_up_after_max_retries_marks_failed(session: AsyncSession) -> None:
    market = _market()
    positions = [_position()]
    pool = FakePool(
        balances={"111": 10_000_000, "222": 0},
        send_outcomes=[RuntimeError("boom")] * 5,
    )
    settings = _settings(redemption_max_retries=3)

    await redeem_market(session, pool, market, positions, settings=settings, now=NOW)
    await session.commit()

    assert positions[0].redemption_status == "failed"
    assert positions[0].redemption_tx_hash is None
    assert len(pool.sent) == 3  # bounded, not unbounded retries
    event = await session.scalar(select(RiskEvent).where(RiskEvent.type == "redemption"))
    assert event is not None and event.action_taken == "flagged"


async def test_non_eoa_signature_scheme_flags_manual_redemption(session: AsyncSession) -> None:
    market = _market()
    positions = [_position()]
    pool = FakePool(balances={"111": 10_000_000, "222": 0})
    settings = _settings(clob_signature_type=2)  # Gnosis Safe proxy

    await redeem_market(session, pool, market, positions, settings=settings, now=NOW)

    assert positions[0].redemption_status == "failed"
    assert pool.sent == []  # never even checked balance — the owner address would be wrong


async def test_sweep_noops_in_paper_mode(session: AsyncSession, monkeypatch) -> None:
    def _boom(urls: list[str]) -> None:
        raise AssertionError("RpcPool must not be constructed in paper mode")

    monkeypatch.setattr(redemption_mod, "RpcPool", _boom)

    result = await redeem_settled_positions(session, settings=_settings(execution_mode="paper"))

    assert result == []


async def test_sweep_groups_by_market_and_is_idempotent(
    session: AsyncSession, monkeypatch
) -> None:
    market = _market()
    session.add(market)
    session.add_all(
        [
            _position(source_wallet=None, token_id="111"),
            _position(source_wallet=None, token_id="222"),
        ]
    )
    await session.commit()

    pool = FakePool(balances={"111": 10_000_000, "222": 0})
    monkeypatch.setattr(redemption_mod, "RpcPool", lambda urls: pool)

    settings = _settings()
    processed = await redeem_settled_positions(session, settings=settings, now=NOW)

    assert processed == [COND_ID]
    positions = (await session.scalars(select(PortfolioPosition))).all()
    assert all(p.redemption_status == "redeemed" for p in positions)
    assert len(pool.sent) == 1  # one redemption call covers every internal row for the market

    # A second sweep finds nothing left to redeem — already-redeemed rows are excluded.
    processed_again = await redeem_settled_positions(session, settings=settings, now=NOW)
    assert processed_again == []
    assert len(pool.sent) == 1  # no new transaction submitted
