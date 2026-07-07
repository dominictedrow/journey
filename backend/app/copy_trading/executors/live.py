"""Live executor (Module 9, gated live path — Phase 8).

Submits real Polymarket CLOB orders through `py-clob-client`. It is disabled by default and
guarded three ways, all of which must clear before a single share trades:

  1. `EXECUTION_MODE=live` + `LIVE_TRADING_ARMED=true` — a deliberate, deploy-time arming.
     This executor re-checks the arming itself (defence in depth), so even if it is wired in
     by mistake it refuses to place an order while disarmed.
  2. The global kill switch — enforced by `RiskEngine.check` before the executor is ever
     reached, halting all execution (paper and live) durably across processes.
  3. The full risk gate — the same size/exposure/slippage/delay rules paper trading passes.

A live copy is a *marketable* limit priced to cross the resting book by
`live_marketable_buffer_pct`, submitted Fill-And-Kill: it fills what it can immediately
(fully or partially) and cancels the remainder rather than resting as a stale maker order
chasing a source we can no longer track. Transient placement failures (nonce/network) are
retried with linear backoff on the still-unfilled size; a rejection with no order id is
surfaced as-is. The order id is returned on the result so a post-crash sweep can reconcile
orders that were submitted but whose fill we never recorded (see `copy_trading.recovery`).

The `py-clob-client` SDK is synchronous and imported lazily inside the adapter, so neither
the default paper path nor the test suite needs the dependency or live credentials: tests
inject a fake `LiveClient`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from app.config import Settings, get_settings
from app.copy_trading.executors.base import ExecutionResult, Executor
from app.risk.models import Side, TradeIntent

QUANTUM = Decimal("0.000001")
PRICE_QUANTUM = Decimal("0.001")  # CLOB price tick; the SDK re-rounds to the market's own tick
PRICE_FLOOR = Decimal("0.001")
PRICE_CEIL = Decimal("0.999")
DUST = Decimal("0.000001")


def _dec(value: Decimal) -> Decimal:
    return value.quantize(QUANTUM)


@dataclass
class OrderPlacement:
    """Normalized outcome of one CLOB placement attempt.

    `order_id` is None only when the order never reached the book (a transient error worth
    retrying). `filled_size`/`avg_price` describe what actually matched immediately."""

    order_id: str | None
    filled_size: Decimal
    avg_price: Decimal | None
    error: str | None = None


@dataclass
class OrderState:
    """The current state of a live order, as reported by the CLOB during reconciliation.

    `status` is normalized to the copy-order vocabulary: filled | partial | pending | canceled
    | rejected."""

    order_id: str
    token_id: str | None
    side: Side | None
    status: str
    filled_size: Decimal
    avg_price: Decimal | None


class LiveClient(Protocol):
    """The order-placement surface the executor needs; `PyClobLiveClient` adapts the SDK to it
    and tests provide a fake. `price`/`size` are in shares and probability price (0, 1)."""

    async def place_order(
        self, *, token_id: str, side: Side, price: Decimal, size: Decimal
    ) -> OrderPlacement: ...


class ReconcilableClient(LiveClient, Protocol):
    """A `LiveClient` that can also enumerate the account's recent orders, so a post-crash
    sweep can match them against our records (see `copy_trading.recovery`)."""

    async def list_recent_orders(self) -> list[OrderState]: ...


class LiveExecutor(Executor):
    mode = "live"

    def __init__(self, client: LiveClient, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or get_settings()

    def _armed(self) -> bool:
        return self._settings.execution_mode == "live" and self._settings.live_trading_armed

    def _marketable_price(self, intent: TradeIntent) -> Decimal:
        """Price to cross the book by the configured buffer (up for a BUY, down for a SELL)."""
        buffer = self._settings.live_marketable_buffer_pct / Decimal(100)
        move = (1 + buffer) if intent.side == "BUY" else (1 - buffer)
        price = (intent.price * move).quantize(PRICE_QUANTUM)
        return min(max(price, PRICE_FLOOR), PRICE_CEIL)

    async def execute(self, intent: TradeIntent, size: Decimal) -> ExecutionResult:
        delay_ms = int(intent.delay_seconds * 1000) if intent.delay_seconds is not None else 0
        if not self._armed():
            # Defence in depth: refuse even if wired in while disarmed. No order is placed.
            return ExecutionResult(
                "rejected", Decimal(0), None, None, delay_ms, "live execution disarmed"
            )
        if size <= 0:
            return ExecutionResult("rejected", Decimal(0), None, None, delay_ms, "zero size")

        price = self._marketable_price(intent)
        backoff = float(self._settings.live_retry_backoff_seconds)
        last_error = "no order placed"
        for attempt in range(1, self._settings.live_max_retries + 1):
            try:
                placement = await self._client.place_order(
                    token_id=intent.token_id, side=intent.side, price=price, size=size
                )
            except Exception as exc:  # noqa: BLE001 — SDK/network errors are retryable
                last_error = str(exc) or exc.__class__.__name__
                if attempt < self._settings.live_max_retries:
                    await asyncio.sleep(backoff * attempt)
                    continue
                return ExecutionResult("rejected", Decimal(0), None, None, delay_ms, last_error)

            if placement.order_id is None and placement.filled_size <= 0:
                # Never reached the book — a transient failure; retry the full size.
                last_error = placement.error or "placement failed"
                if attempt < self._settings.live_max_retries:
                    await asyncio.sleep(backoff * attempt)
                    continue
                return ExecutionResult("rejected", Decimal(0), None, None, delay_ms, last_error)

            # The order reached the book; Fill-And-Kill means this is the final fill — a
            # partial means the book was too thin at our price, so retrying would only chase.
            return self._result(intent, size, placement, delay_ms)

        return ExecutionResult("rejected", Decimal(0), None, None, delay_ms, last_error)

    @staticmethod
    def _result(
        intent: TradeIntent, size: Decimal, placement: OrderPlacement, delay_ms: int
    ) -> ExecutionResult:
        filled = _dec(placement.filled_size) if placement.filled_size > 0 else Decimal(0)
        if filled <= DUST or placement.avg_price is None:
            return ExecutionResult(
                "rejected",
                Decimal(0),
                None,
                None,
                delay_ms,
                placement.error or "unmatched",
                external_order_id=placement.order_id,
            )
        fill_price = min(max(_dec(placement.avg_price), PRICE_FLOOR), PRICE_CEIL)
        slippage = None
        if intent.price > 0:
            raw = (fill_price - intent.price) / intent.price
            slippage = _dec(raw if intent.side == "BUY" else -raw)  # adverse move positive
        status = "filled" if filled >= size - DUST else "partial"
        return ExecutionResult(
            status, filled, fill_price, slippage, delay_ms, external_order_id=placement.order_id
        )


class PyClobLiveClient:
    """Adapts the synchronous `py-clob-client` SDK to the async `LiveClient` protocol.

    The SDK is imported lazily and each blocking call is run in a worker thread so the async
    event loop is never stalled. Built from settings by `build_live_client`; constructing it
    requires the signing key and CLOB API credentials, so it is only ever created in live mode.
    """

    def __init__(self, client: object) -> None:
        self._client = client  # a py_clob_client.client.ClobClient

    @classmethod
    def from_settings(cls, settings: Settings) -> PyClobLiveClient:
        from py_clob_client.client import ClobClient  # lazy: live path only
        from py_clob_client.clob_types import ApiCreds

        creds = ApiCreds(
            api_key=settings.polymarket_api_key,
            api_secret=settings.polymarket_api_secret,
            api_passphrase=settings.polymarket_api_passphrase,
        )
        client = ClobClient(
            host=settings.clob_api_url,
            key=settings.wallet_private_key,
            chain_id=settings.polygon_chain_id,
            creds=creds,
            signature_type=settings.clob_signature_type,
            funder=settings.clob_funder_address or None,
        )
        return cls(client)

    async def place_order(
        self, *, token_id: str, side: Side, price: Decimal, size: Decimal
    ) -> OrderPlacement:
        return await asyncio.to_thread(self._place_sync, token_id, side, price, size)

    def _place_sync(
        self, token_id: str, side: Side, price: Decimal, size: Decimal
    ) -> OrderPlacement:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL

        args = OrderArgs(
            token_id=token_id,
            price=float(price),
            size=float(size),
            side=BUY if side == "BUY" else SELL,
        )
        signed = self._client.create_order(args)  # type: ignore[attr-defined]
        resp = self._client.post_order(signed, OrderType.FAK)  # type: ignore[attr-defined]
        return _parse_response(resp)

    async def list_recent_orders(self) -> list[OrderState]:
        return await asyncio.to_thread(self._list_sync)

    def _list_sync(self) -> list[OrderState]:
        orders = self._client.get_orders()  # type: ignore[attr-defined]
        return [_parse_order_state(o) for o in (orders or [])]


def _parse_response(resp: dict) -> OrderPlacement:
    """Map the CLOB post-order response to an `OrderPlacement`.

    A `success: false` (or an error message) with no order id is a transient failure the
    executor retries; a returned id with `matchedAmount == 0` is an accepted-but-unmatched
    order the executor treats as a rejection."""
    resp = resp or {}
    error = resp.get("errorMsg") or resp.get("error") or None
    order_id = resp.get("orderID") or resp.get("orderId") or resp.get("id")
    if resp.get("success") is False and not order_id:
        return OrderPlacement(order_id=None, filled_size=Decimal(0), avg_price=None, error=error)
    matched = resp.get("makingAmount") or resp.get("matchedAmount") or resp.get("sizeMatched") or 0
    avg = resp.get("price") or resp.get("avgPrice")
    filled = Decimal(str(matched))
    avg_price = Decimal(str(avg)) if avg not in (None, "", 0, "0") else None
    return OrderPlacement(
        order_id=str(order_id) if order_id else None,
        filled_size=filled,
        avg_price=avg_price,
        error=error,
    )


_CLOB_STATUS = {
    "matched": "filled",
    "filled": "filled",
    "live": "pending",
    "open": "pending",
    "delayed": "pending",
    "unmatched": "canceled",
    "canceled": "canceled",
    "cancelled": "canceled",
    "rejected": "rejected",
}


def _parse_order_state(o: dict) -> OrderState:
    """Map one CLOB order record to a normalized `OrderState`."""
    o = o or {}
    size = Decimal(str(o.get("size") or o.get("originalSize") or 0))
    matched = Decimal(str(o.get("sizeMatched") or o.get("makingAmount") or 0))
    raw_status = str(o.get("status") or "").lower()
    status = _CLOB_STATUS.get(raw_status, "pending")
    if status == "filled" and 0 < matched < size:
        status = "partial"
    avg = o.get("price") or o.get("avgPrice")
    side = o.get("side")
    return OrderState(
        order_id=str(o.get("orderID") or o.get("id") or ""),
        token_id=(str(o["asset_id"]) if o.get("asset_id") else o.get("token_id")),
        side="BUY" if str(side).upper() == "BUY" else ("SELL" if side else None),
        status=status,
        filled_size=matched,
        avg_price=Decimal(str(avg)) if avg not in (None, "", 0, "0") else None,
    )


def build_live_client(settings: Settings | None = None) -> PyClobLiveClient:
    """Construct the real CLOB client from settings (live mode only)."""
    return PyClobLiveClient.from_settings(settings or get_settings())
