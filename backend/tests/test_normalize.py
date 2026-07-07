from datetime import UTC, datetime
from decimal import Decimal

from app.blockchain.normalize import compute_positions
from app.database.models import Trade

WALLET = "0x" + "aa" * 20


def make_trade(
    side: str, price: str, size: str, token_id: str = "t1", block: int = 1, log_index: int = 0
) -> Trade:
    return Trade(
        tx_hash="0x" + "00" * 32,
        log_index=log_index,
        wallet_address=WALLET,
        token_id=token_id,
        side=side,
        price=Decimal(price),
        size=Decimal(size),
        block_number=block,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_round_trip_realizes_pnl_and_closes() -> None:
    positions = compute_positions(
        [
            make_trade("BUY", "0.40", "100", block=1),
            make_trade("BUY", "0.60", "100", block=2),
            make_trade("SELL", "0.70", "200", block=3),
        ]
    )
    assert len(positions) == 1
    pos = positions[0]
    assert pos["net_size"] == 0
    assert pos["avg_entry_price"] == Decimal("0.5")
    assert pos["realized_pnl"] == Decimal("40")  # (0.70 - 0.50) * 200
    assert pos["is_open"] is False
    assert pos["closed_at"] is not None


def test_partial_sell_stays_open() -> None:
    positions = compute_positions(
        [
            make_trade("BUY", "0.50", "100", block=1),
            make_trade("SELL", "0.80", "40", block=2),
        ]
    )
    pos = positions[0]
    assert pos["net_size"] == Decimal("60")
    assert pos["realized_pnl"] == Decimal("12")  # (0.80 - 0.50) * 40
    assert pos["is_open"] is True


def test_oversell_clamps_at_zero() -> None:
    positions = compute_positions(
        [
            make_trade("BUY", "0.50", "10", block=1),
            make_trade("SELL", "0.60", "25", block=2),
        ]
    )
    pos = positions[0]
    assert pos["net_size"] == 0
    assert pos["realized_pnl"] == Decimal("1")  # only the held 10 matched


def test_tokens_tracked_independently() -> None:
    positions = compute_positions(
        [
            make_trade("BUY", "0.30", "50", token_id="t1", block=1),
            make_trade("BUY", "0.90", "10", token_id="t2", block=2),
        ]
    )
    assert {p["token_id"] for p in positions} == {"t1", "t2"}
    assert all(p["is_open"] for p in positions)
