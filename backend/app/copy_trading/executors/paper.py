"""Paper executor (Module 9, default).

Simulates a fill against the live CLOB order book: a BUY walks the asks, a SELL walks the
bids, filling up to the requested size and reporting the size-weighted average price plus the
slippage versus the source (intent) price. When the book is unavailable (offline, tests, or an
empty market) it degrades to a full fill at the intent price moved by a fixed fallback
slippage, so paper trading never stalls on connectivity. `delay_ms` records how stale the copy
was at decision time, feeding the copy-quality score.
"""

from collections.abc import Awaitable, Callable
from decimal import Decimal

from app.copy_trading.executors.base import ExecutionResult, Executor
from app.ingest.clob_api import get_book
from app.risk.models import TradeIntent

QUANTUM = Decimal("0.000001")
PRICE_FLOOR = Decimal("0.001")
PRICE_CEIL = Decimal("0.999")

BookFetcher = Callable[[str], Awaitable[dict | None]]


def _dec(value: Decimal) -> Decimal:
    return value.quantize(QUANTUM)


async def _default_book_fetcher(token_id: str) -> dict | None:
    try:
        return await get_book(token_id)
    except Exception:  # noqa: BLE001 — offline/thin market is expected; fall back to intent price
        return None


def _levels(book: dict | None, key: str) -> list[tuple[Decimal, Decimal]]:
    rows = (book or {}).get(key) or []
    out: list[tuple[Decimal, Decimal]] = []
    for row in rows:
        try:
            out.append((Decimal(str(row["price"])), Decimal(str(row["size"]))))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _walk(
    levels: list[tuple[Decimal, Decimal]], size: Decimal, ascending: bool
) -> tuple[Decimal, Decimal] | None:
    """Consume `size` across price levels (asks ascending / bids descending); returns
    (filled_size, vwap) or None when the book is empty."""
    if not levels:
        return None
    ordered = sorted(levels, key=lambda lvl: lvl[0], reverse=not ascending)
    remaining = size
    cost = Decimal(0)
    for price, avail in ordered:
        take = min(remaining, avail)
        if take <= 0:
            continue
        cost += take * price
        remaining -= take
        if remaining <= 0:
            break
    filled = size - remaining
    if filled <= 0:
        return None
    return _dec(filled), _dec(cost / filled)


class PaperExecutor(Executor):
    mode = "paper"

    def __init__(
        self,
        book_fetcher: BookFetcher | None = None,
        fallback_slippage_pct: Decimal | None = None,
    ) -> None:
        self._fetch = book_fetcher or _default_book_fetcher
        self._fallback_slippage = (
            fallback_slippage_pct if fallback_slippage_pct is not None else Decimal("0.005")
        )

    async def execute(self, intent: TradeIntent, size: Decimal) -> ExecutionResult:
        delay_ms = int(intent.delay_seconds * 1000) if intent.delay_seconds is not None else 0
        if size <= 0:
            return ExecutionResult("rejected", Decimal(0), None, None, delay_ms, "zero size")

        is_buy = intent.side == "BUY"
        walked = _walk(
            _levels(await self._fetch(intent.token_id), "asks" if is_buy else "bids"),
            size,
            ascending=is_buy,
        )
        if walked is not None:
            filled, fill_price = walked
        else:
            # No book: fill fully at the intent price, moved adversely by the fallback slippage.
            move = 1 + self._fallback_slippage if is_buy else 1 - self._fallback_slippage
            filled, fill_price = size, _dec(intent.price * move)

        fill_price = min(max(fill_price, PRICE_FLOOR), PRICE_CEIL)
        slippage = None
        if intent.price > 0:
            raw = (fill_price - intent.price) / intent.price
            slippage = _dec(raw if is_buy else -raw)  # adverse move is positive for both sides
        status = "filled" if filled >= size else "partial"
        return ExecutionResult(status, filled, fill_price, slippage, delay_ms)
